import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution

# region 文本特征提取
class TextEncoder(nn.Module):
    def __init__(self, 
                 text_channels: int,
                 channels: list[int],
                 spatial_dims: int = 1,):
        super().__init__()

        # 1.Input Convolution
        self.conv_in = Convolution(
            spatial_dims=spatial_dims, 
            in_channels=text_channels, 
            out_channels=channels[0],
            padding=1,)
        
        # 2.多层次特征提取
        self.conv_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            in_ch = channels[i]
            out_ch = channels[i+1]
            conv = Convolution(
                spatial_dims=spatial_dims, 
                in_channels=in_ch, 
                out_channels=out_ch,
                padding=1,
                )
            self.conv_blocks.append(conv)

    def forward(self, text:torch.Tensor)-> torch.Tensor:
        text_feat_list = []

        t = self.conv_in(text)
        text_feat_list.append(t)

        for conv in self.conv_blocks:
            t = conv(t)
            text_feat_list.append(t)

        return text_feat_list
    
# # ============ 测试部分 ============
# if __name__ == "__main__":

#     import sys
#     sys.path.append("/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/")
#     # 参数
#     spatial_dims = 1        # 支持 3D 输入 (B, C, D, H, W)
#     text_channels = 256       # 输入的通道数 (如 token embedding 通道数)
#     channels = [32,64,128,256,512] # 每层通道数

#     # 随机输入 (batch=2, text_channels=4, depth=8, H=32, W=32)
#     x = torch.randn(2, text_channels, 21)

#     # 构建 TextEncoder
#     model = TextEncoder(spatial_dims, text_channels, channels)

#     # 前向传播
#     text_feats = model(x)

#     # 打印输出特征列表
#     print(f"输入 shape: {x.shape}")
#     for i, feat in enumerate(text_feats):
#         print(f"特征[{i}] shape: {feat.shape}")
## endregion

# region 图像特征提取
import sys
sys.path.append("/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/")
from Model.Net.CPC_BM_Net.down_block import DownSample
from Model.Net.CPC_BM_Net.resnet_block import ResnetBlock

class CondImageEncoder(nn.Module):
    def __init__(self,
                 spatial_dims: int,
                 cond_channels: int,
                 channels: list[int],):
        super().__init__()

        # Input conv
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=cond_channels,
            out_channels=channels[0],
            padding=1
        )

        # DownSample
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            in_ch = channels[i]
            out_ch = channels[i+1]
            if i == (len(channels)-2):
                only_down_HW = True
            else:
                only_down_HW = False
            res_block = ResnetBlock(
                spatial_dims=spatial_dims,
                in_channels=in_ch,
                out_channels=in_ch
            )
            self.down_blocks.append(res_block)
            down_block = DownSample(
                spatial_dims=spatial_dims,
                in_channels=in_ch,
                out_channels=out_ch,
                only_down_HW=only_down_HW
            )
            self.down_blocks.append(down_block)

    def forward(self, cond_feat:torch.Tensor)-> torch.Tensor:
        cond_feat_list = []
        c = self.conv_in(cond_feat)
        cond_feat_list.append(c)

        for down_block in self.down_blocks:
            c = down_block(c)
            if isinstance(down_block, DownSample):
                cond_feat_list.append(c)

        return cond_feat_list
    
# # ============ 测试部分 ============
# if __name__ == "__main__":

#     # 参数
#     spatial_dims = 3        # 支持 3D 输入 (B, C, D, H, W)
#     cond_channels = 3       # 输入的通道数 (如 token embedding 通道数)
#     channels = [32,64,128,256,512] # 每层通道数

#     # 随机输入 (batch=2, text_channels=4, depth=8, H=32, W=32)
#     x = torch.randn(2, cond_channels, 8, 256, 256)

#     # 构建 TextEncoder
#     model = CondImageEncoder(spatial_dims, cond_channels, channels)

#     # 前向传播
#     cond_feat = model(x)

#     # 打印输出特征列表
#     print(f"输入 shape: {x.shape}")
#     for i, feat in enumerate(cond_feat):
#         print(f"特征[{i}] shape: {feat.shape}")

# region 文本-图像 特征对齐
from Model.Net.CPC_BM_Net.condition_function import compute_cost_matrix
    
class T2I_OT_AdapGating_Fusion(nn.Module):
    def __init__(self, epsilon, niter, spatial_dims, channels, use_sigmoid=False):
        super().__init__()
        # //? OT参数
        self.epsilon = epsilon
        self.niter = niter

        # //? AdaptiveGating
        self.use_sigmoid = use_sigmoid
        ConvNd = {1:nn.Conv1d, 2:nn.Conv2d, 3:nn.Conv3d}[spatial_dims]
        self.gate_conv = nn.Sequential(
            ConvNd(channels*2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            ConvNd(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def optimal_transport(self, text_feats, image_feats):
        """
        使用最优传输算法计算文本特征 A 和 图像特征 B 之间的对齐。
        
        参数：
        - text_feats: 文本特征，形状 [L, N]
        - image_feats: 图像特征，形状 [L, M]
        - epsilon: Sinkhorn 算法的正则化因子
        - max_iter: Sinkhorn 算法的最大迭代次数
        
        返回：
        - aligned_A: 对齐后的文本特征，形状与 B 相同 [L, M]
        """
        L1, N = text_feats.shape
        L2, M = image_feats.shape
        assert L1 == L2, "Batch size does not match."
        
        # 计算代价矩阵
        cost_matrix = compute_cost_matrix(text_feats, image_feats, cost_function="euclidean")
        exp_term = torch.exp(-cost_matrix / self.epsilon)  # 转换成 Sinkhorn 算法的形式
        
        # Sinkhorn 算法：计算最优传输矩阵
        u = torch.ones(cost_matrix.shape[0], 1).to(text_feats.device)  # 初始化 u
        v = torch.ones(cost_matrix.shape[1], 1).to(text_feats.device)  # 初始化 v

        prev_u, prev_v = u.clone(), v.clone()
        
        for _ in range(self.niter):
            u = 1.0 / (torch.sum(exp_term * v.T, dim=1, keepdim=True)+ 1e2)  # 更新 u_i
            v = 1.0 / (torch.sum(exp_term.T * u.T, dim=1, keepdim=True)+ 1e2)  # 更新 v_j
            # 计算u和v的变化量
            u_diff = torch.norm(u - prev_u, p=2)
            v_diff = torch.norm(v - prev_v, p=2)
            # print(f"Sinkhorn算法迭代 {_+1}, u_diff: {u_diff}, v_diff: {v_diff}")
            if u_diff < 0.01 and v_diff < 0.01:
                # print(f"Sinkhorn算法收敛，迭代次数: {_+1}")
                break
            # 更新前一次的 u 和 v
            prev_u, prev_v = u.clone(), v.clone()
        
        # 计算最优传输矩阵
        transport_matrix = torch.exp(-cost_matrix * self.epsilon)*u*v.T   # 计算最优传输矩阵
        
        # 将最优传输矩阵应用于 A，以对齐 A 到 B
        aligned_A = torch.matmul(image_feats, transport_matrix.T)  # 对齐特征 A 到 B
        aligned_B = torch.matmul(text_feats, transport_matrix)  # 对齐特征 B 到 A
        
        return aligned_A, aligned_B
    
    def OT(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        对文本特征和图像特征进行最优传输对齐
        参数：
            text_feats: [B, N, D] 文本特征
            image_feats: [B, C, H, W] 或 [B, C, D_s, H, W] 图像特征
        返回：
            对齐后的文本特征
        """

        B, C1, N = text.shape
        if image.dim() == 4:  # 处理二维图像特征 [B, C, H, W]
            B2, C2, H, W = image.shape
            M = H * W
        elif image.dim() == 5:  # 处理三维图像特征 [B, C, D_s, H, W]
            B2, C2, D, H, W = image.shape
            M = D * H * W
                
        assert B == B2 and C1 == C2, "Batch size or channel dimensions do not match."

        text_feats = text.view(-1, N) # [B, C1, N] -> [B*C1, N]

        image_feats = image.view(B, C2, -1)  # [B, C2, H, W] -> [B, C2, H*W] or [B, C2, D_s, H, W] -> [B, C2, D_s*H*W]
        image_feats = image_feats.view(-1, M)  # [B, C2, M] -> [B*C2, M]

        aligned_text, aligned_image = self.optimal_transport(text_feats, image_feats)  # [B*C1, M] -> [B*C1, N]

        aligned_text = aligned_text.view(B, C1, N)  # [B*C1, N] -> [B, C1, N]

        aligned_image = aligned_image.view(B, C2, M)  # [B*C2, M] -> [B, C2, M]
        aligned_image = aligned_image.view(B, C2, *image.shape[2:])

        return aligned_text, aligned_image
    
    def AGF(self, text, image):
        B, C1, D, H, W = image.shape
        _, C2, N = text.shape
        assert C1 == C2  # 确保数据维度相同

        # 1. 把 text_feat reshape / 插值到和 image_feat 相同的空间维
        text_feat_reshape = text.view(B, C, 1, 1, -1)  # (B, C, 1, 1, N)
        text_feat_reshape = F.interpolate(
            text_feat_reshape, size=(D, H, W), mode="trilinear", align_corners=False
        )  # (B, C, D, H, W)

        # 计算门控权重: [B, C_t] -> [B, C_i]
        gate = self.gate_conv(torch.cat([text_feat_reshape, image], dim=1))
        if self.use_sigmoid:
            gate = torch.sigmoid(gate)  # [B, C, D, H, W]

        fusion_feat = image * (1 - gate) + text_feat_reshape * gate

        return fusion_feat
    
    def forward(self, text, image):
        """文本图像特征最优化传输对齐后自适应门控融合

        Args:
            text: [B,C,N]
            image: [B,C,D,H,W]

        Returns:
            fusion: [B,C,D,H,W]
        """

        t_alig, _ = self.OT(text, image)
        fusion_feat = self.AGF(t_alig, image)

        return fusion_feat
    
# #################################################################
# # test
# import sys
# sys.path.append("/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/")

# import torch

# B, C, N = 4, 256, 21
# D, H, W = 1, 32, 32
# gpu = 7

# t = torch.randn(B,C,N)
# c = torch.randn(B,C,D,H,W)

# model = T2I_OT_AdapGating_Fusion(epsilon=0.1, niter=50,
#                                  spatial_dims=3,
#                                  channels=256)

# fusion = model(t, c)

# print("The shape of fusion feature is", fusion.shape)