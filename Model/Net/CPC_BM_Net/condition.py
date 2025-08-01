import sys
sys.path.append("/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/")

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

class T2I_OTGatedFusion(nn.Module):
    def __init__(self, epsilon=0.1, niter=50, spatial_dims=3, channels=64, use_sigmoid=False):
        super().__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.spatial_dims = spatial_dims
        self.use_sigmoid = use_sigmoid

        # Conv 选择 (2D or 3D)
        ConvNd = {2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        self.gate_conv = nn.Sequential(
            ConvNd(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=False),
            ConvNd(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def compute_cost_matrix(self, A, B):
        """欧几里得距离代价矩阵"""
        L1, N = A.shape
        L2, M = B.shape
        assert L1 == L2 , "Features must have the same dimension."
        A_sq = A.pow(2).sum(dim=1, keepdim=True)
        B_sq = B.pow(2).sum(dim=1, keepdim=True).T
        cost_matrix = torch.zeros(N, M).to(A.device)
        cost_matrix = torch.cdist(A.T, B.T, p=2)  # 计算欧几里得距离
        return cost_matrix

    def compute_ot(self, text_feats, image_feats):
        """Sinkhorn OT 对齐 (text_feats: [L,N], image_feats: [L,M])"""
        L1, N = text_feats.shape
        L2, M = image_feats.shape
        assert L1 == L2, "Batch size mismatch."

        cost_matrix = self.compute_cost_matrix(text_feats, image_feats)
        exp_term = torch.exp(-cost_matrix / self.epsilon)

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

    def align(self, text, image):
        """
        text:  [B,C,N]
        image: [B,C,H,W] or [B,C,D,H,W]
        """
        B, C, N = text.shape

        # 图像展开
        if image.dim() == 4:     # 2D: [B,C,H,W]
            B2, C2, H, W = image.shape
            M = H * W
            image_shape = (B, C, H, W)
        elif image.dim() == 5:   # 3D: [B,C,D,H,W]
            B2, C2, D, H, W = image.shape
            M = D * H * W
            image_shape = (B, C, D, H, W)
        else:
            raise ValueError("Image must be 4D or 5D tensor.")
        assert B == B2 and C == C2, "Batch/Channel mismatch."

        text_feats = text.view(-1, N)                  # [B*C,N]
        image_feats = image.view(B, C, -1).view(-1, M) # [B*C,M]

        aligned_text, aligned_image = self.compute_ot(text_feats, image_feats)

        aligned_text = aligned_text.view(B, C, N)
        aligned_image = aligned_image.view(*image_shape)
        return aligned_text, aligned_image

    def gated_fusion(self, text, image):
        """
        自适应门控融合
        text:  [B,C,N]  (OT 对齐后的文本特征)
        image: [B,C,H,W] or [B,C,D,H,W]
        """
        B, C, N = text.shape

        # 文本特征 reshape 到和图像特征同维度
        if image.dim() == 4:
            _, _, H, W = image.shape
            text_reshape = text.view(B, C, 1, 1, N) if self.spatial_dims == 3 else text.view(B, C, 1, N)
            text_reshape = F.interpolate(
                text_reshape, size=(H, W), mode="bilinear", align_corners=False
            )
        elif image.dim() == 5:
            _, _, D, H, W = image.shape
            text_reshape = text.view(B, C, 1, 1, N)
            text_reshape = F.interpolate(
                text_reshape, size=(D, H, W), mode="trilinear", align_corners=False
            )
        else:
            raise ValueError("Image must be 4D or 5D tensor.")

        # 拼接 + 门控卷积
        fused_input = torch.cat([text_reshape, image], dim=1)  # [B,2C,...]
        gate = self.gate_conv(fused_input)
        if self.use_sigmoid:
            gate = torch.sigmoid(gate)

        # 融合
        fusion_feat = image * (1 - gate) + text_reshape * gate
        return fusion_feat

    def forward(self, text, image):
        """
        text:  [B,C,N]
        image: [B,C,H,W] or [B,C,D,H,W]
        return: fusion_feat: [B,C,H,W] or [B,C,D,H,W]
        """
        aligned_text, _ = self.align(text, image)
        fusion_feat = self.gated_fusion(aligned_text, image)
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

class T2I_OTF(nn.Module):
    def __init__(self, 
                 epsilon: float, niter: int,  # //* Sinkhorn算法参数
                 spatial_dims: int, channels: list[int], use_sigmoid: bool=False
                 ):
        super().__init__()
        self.fusion_blocks = nn.ModuleList()
        self.length = len(channels)
        for i in range(len(channels)):
            ch = channels[i]
            fusion_block = T2I_OTGatedFusion(epsilon=epsilon, niter=niter,
                                             spatial_dims=spatial_dims, 
                                             channels=ch)
            self.fusion_blocks.append(fusion_block)

    def forward(self,
                text_feat_list,
                cond_feat_list):
        "多尺度文本特征和图像特征OT对齐+门控融合"
        fusion_feat_list = []
        assert len(text_feat_list) == len(cond_feat_list) == self.length, "Mismatch in number of features"

        for i, fusion_block in enumerate(self.fusion_blocks):
            t = text_feat_list[i]
            c = cond_feat_list[i]
            fusion_feat = fusion_block(t, c)
            fusion_feat_list.append(fusion_feat)

        return fusion_feat_list

# if __name__ == "__main__":
#     from typing import List

#     # 参数
#     epsilon = 0.1
#     niter = 50
#     spatial_dims = 3
#     channels = [16, 32, 64]   # 多尺度通道数
#     B = 2                     # batch size

#     # 构建 T2I_OTF 模型
#     model = T2I_OTF(epsilon=epsilon, niter=niter,
#                     spatial_dims=spatial_dims, channels=channels)
#     model.eval()

#     # 构造多尺度假数据
#     text_feat_list=[
#         torch.randn(B,16,21),
#         torch.randn(B,32,21),
#         torch.randn(B,64,21),
#     ]
#     cond_feat_list=[
#         torch.randn(B,16,8,256,256),
#         torch.randn(B,32,4,128,128),
#         torch.randn(B,64,2,64,64),
#     ]

#     # 前向传播
#     with torch.no_grad():
#         fusion_feat_list = model(text_feat_list, cond_feat_list)

#     # 打印结果 shape
#     print(f"输入尺度数: {len(channels)}")
#     for i, feat in enumerate(fusion_feat_list):
#         print(f"Fusion [{i}] shape: {feat.shape}")


# region 条件特征融合

class DynamicConvFiLM(nn.Module):
    def __init__(self, channels,
                 kernel_size=3, K=4, spatial_dims=3):
        """
        动态卷积 + FiLM 条件融合模块

        Args:
            channels: 网络特征通道数
            kernel_size: 动态卷积核大小
            K: 多核加权动态卷积的核数
            spatial_dims: 2 or 3
        """
        super().__init__()
        self.K = K
        self.channels = channels
        self.kernel_size = kernel_size
        ConvNd = {2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]
        PoolNd = {2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}[spatial_dims]

        # 多核卷积核参数: K 个卷积核共享
        self.weight = nn.Parameter(
            torch.randn(K, channels, channels, *(kernel_size,) * spatial_dims)
        )
        self.bias = nn.Parameter(torch.zeros(K, channels))

        # 条件特征 -> 动态权重 α_i
        self.alpha_gen = nn.Sequential(
            PoolNd(1),
            ConvNd(channels, K, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # 条件特征 -> FiLM (γ, β)
        self.gamma_gen = ConvNd(channels, channels, kernel_size=1)
        self.beta_gen = ConvNd(channels, channels, kernel_size=1)

    def forward(self, net_feat, cond_feat):
        """
        net_feat: [B, C_in, D, H, W] or [B, C_in, H, W]
        cond_feat: [B, C_cond, D, H, W] or [B, C_cond, H, W]
        """
        assert net_feat.shape[1] == cond_feat.shape[1] == self.channels, "Mismatch of fusion feature dimensions"

        # 1. 动态卷积 (多核加权)
        alpha = self.alpha_gen(cond_feat)  # [B, K, 1, 1, (1)]
        out = 0
        for i in range(self.K):
            out += alpha[:, i:i+1] * F.conv3d(
                net_feat, self.weight[i], self.bias[i], padding=self.kernel_size // 2
            ) if net_feat.dim() == 5 else alpha[:, i:i+1] * F.conv2d(
                net_feat, self.weight[i], self.bias[i], padding=self.kernel_size // 2
            )

        # 2. FiLM 条件调制
        gamma = self.gamma_gen(cond_feat)  # [B, C_out, ...]
        beta = self.beta_gen(cond_feat)
        out = gamma * out + beta

        return out

# if __name__ == "__main__":
#     from typing import List

#     # 参数
#     n = torch.randn(4,32,8,256,256)
#     c = torch.randn(4,32,8,256,256)

#     # 构建 T2I_OTF 模型
#     model = DynamicConvFiLM(channels=32)
#     model.eval()

#     # 前向传播
#     with torch.no_grad():
#         feat = model(n, c)

    
#     print(f"Fusion shape: {feat.shape}")