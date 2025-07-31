import torch
import torch.nn as nn
from monai.networks.blocks import Convolution


# region 最优传输对齐
def compute_cost_matrix(A, B, cost_function="euclidean"):
    """
    计算文本特征 A 和 图像特征 B 之间的代价矩阵，形状为 [L, L]。
    
    参数：
    - A: 文本特征，形状 [L, N]，L 是样本数，N 是特征维度
    - B: 图像特征，形状 [L, M]，L 是样本数，M 是特征维度
    - cost_function: 选择代价函数类型 ("euclidean" | "cosine")
    
    返回：
    - cost_matrix: 代价矩阵，形状为 [N, M]，每对文本和图像特征之间的代价
    """
    L1, N = A.shape
    L2, M = B.shape
    assert L1 == L2  # "Features must have the same dimension."
    cost_matrix = torch.zeros(N, M).to(A.device)
    cost_matrix = torch.cdist(A.T, B.T, p=2)  # 计算欧几里得距离

    return cost_matrix





class I2I_OptimalTransportAligner(nn.Module):
    """
    图像间熵正则化最优传输对齐：
      - 图像1特征：B × C × H × W 或 B × C × D_s × H × W
      - 图像2特征：B × C × H × W 或 B × C × D_s × H × W
    会把图像2特征展平为 B × M × C，再计算 OT，将图像1特征对齐到图像2空间。
    """

    def __init__(self, eps: float = 0.1, n_iters: int = 50):
        super().__init__()
        self.eps = eps
        self.n_iters = n_iters

    def optimal_transport(self, image_feat1, image_feat2):
        """
        使用最优传输算法计算文本特征 A 和 图像特征 B 之间的对齐。
        
        参数：
        - image_feat1: 文本特征，形状 [L, N]
        - image_feat2: 图像特征，形状 [L, M]
        - epsilon: Sinkhorn 算法的正则化因子
        - max_iter: Sinkhorn 算法的最大迭代次数
        
        返回：
        - aligned_A: 对齐后的文本特征，形状与 B 相同 [L, M]
        """
        L1, N = image_feat1.shape
        L2, M = image_feat2.shape
        assert L1 == L2, "Batch size does not match."
        
        # 计算代价矩阵
        cost_matrix = compute_cost_matrix(image_feat1, image_feat2, cost_function="euclidean")
        exp_term = torch.exp(-cost_matrix / self.epsilon)  # 转换成 Sinkhorn 算法的形式
        
        # Sinkhorn 算法：计算最优传输矩阵
        u = torch.ones(cost_matrix.shape[0], 1).to(image_feat1.device)  # 初始化 u
        v = torch.ones(cost_matrix.shape[1], 1).to(image_feat1.device)  # 初始化 v

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
        aligned_A = torch.matmul(image_feat2, transport_matrix.T)  # 对齐特征 A 到 B
        aligned_B = torch.matmul(image_feat1, transport_matrix)  # 对齐特征 B 到 A
        
        return aligned_A, aligned_B


    def forward(
        self,
        image1: torch.Tensor,  # [B, C, H, W] or [B, C, D_s, H, W]
        image2: torch.Tensor   # [B, C, H, W] or [B, C, D_s, H, W]
    ) -> torch.Tensor:
        if image1.dim() == 4:  # 处理二维图像特征 [B, C, H, W]
            B1, C1, H1, W1 = image1.shape
            B2, C2, H2, W2 = image2.shape
            M1 = H1 * W1
            M2 = H2 * W2
        elif image1.dim() == 5:  # 处理三维图像特征 [B, C, D_s, H, W]
            B1, C1, D1, H1, W1 = image1.shape
            B2, C2, D2, H2, W2 = image2.shape
            M1 = D1 * H1 * W1
            M2 = D2 * H2 * W2
                
        assert B1 == B2 and C1 == C2, "Batch size or channel dimensions do not match."


        image_feats1 = image1.view(B1, C1, -1)  # [B, C2, H, W] -> [B, C2, H*W] or [B, C2, D_s, H, W] -> [B, C2, D_s*H*W]
        image_feats1 = image_feats1.view(-1, M1)  # [B, C2, M] -> [B*C2, M]

        image_feats2 = image2.view(B2, C2, -1)  # [B, C2, H, W] -> [B, C2, H*W] or [B, C2, D_s, H, W] -> [B, C2, D_s*H*W]
        image_feats2 = image_feats2.view(-1, M2)  # [B, C2, M] -> [B*C2, M]

        aligned_image1, aligned_image2 = self.optimal_transport(image_feats1, image_feats2)  # [B*C1, M] -> [B*C1, N]

        aligned_image1 = aligned_image1.view(B1, C1, M1)  # [B*C1, M] -> [B, C1, M]
        aligned_image1 = aligned_image1.view(B1, C1, *image1.shape[2:])

        aligned_image2 = aligned_image2.view(B2, C2, M2)  # [B*C2, M] -> [B, C2, M]
        aligned_image2 = aligned_image2.view(B2, C2, *image2.shape[2:])

        return aligned_image1, aligned_image2
    
# endregion

# region 特征融合
class T2I_GatedCrossAttentionFusion(nn.Module):
    """
    Cross-Attention 实现文本和图像的特征对齐 + Adaptive Gating 实现特征融合
    输入:
        text_feat: [B, N, C_t]
        img_feat:  [B, C_i, D, H, W]
    输出:
        fused_feat: [B, C_i, D, H, W]
    """
    def __init__(self, text_dim, img_dim, num_heads=4, hidden_dim=128):
        super(T2I_GatedCrossAttentionFusion, self).__init__()
        
        # ========== Cross-Attention 部分 ==========
        self.q_proj = nn.Linear(text_dim, img_dim)
        self.k_proj = nn.Linear(img_dim, img_dim)
        self.v_proj = nn.Linear(img_dim, img_dim)
        self.attn = nn.MultiheadAttention(embed_dim=img_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(img_dim, img_dim)

        # ========== Adaptive Gating 部分 ==========
        self.text_pool = nn.AdaptiveAvgPool1d(1)
        self.gating = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, img_dim)
        )

    def forward(self, text_feat, img_feat):
        """
        text_feat: [B, N, C_t]
        img_feat:  [B, C_i, D, H, W]
        """
        B, C_i, D, H, W = img_feat.shape
        _, N, C_t = text_feat.shape
        
        # ========== Cross-Attention ==========
        q = self.q_proj(text_feat)  # [B, N, C_i]
        img_seq = img_feat.view(B, C_i, -1).permute(0, 2, 1)  # [B, L, C_i]
        k = self.k_proj(img_seq)
        v = self.v_proj(img_seq)
        # 文本和图像通过cross-attention实现特征对齐： 建立文本特征与图像特征之间的细粒度关联
        attn_out, _ = self.attn(q, k, v)  # [B, N, C_i]
        attn_out = self.out_proj(attn_out)

        # 将 Cross-Attention 上下文变成全局图像上下文
        attn_context = attn_out.mean(dim=1).view(B, C_i, 1, 1, 1)  # [B, C_i, 1, 1, 1]
        attn_context = attn_context.expand(-1, -1, D, H, W)

        # ========== Adaptive Gating ==========
        text_global = self.text_pool(text_feat.permute(0, 2, 1)).squeeze(-1)  # [B, C_t]
        gate = torch.sigmoid(self.gating(text_global))  # [B, C_i]
        gate = gate.view(B, C_i, 1, 1, 1)

        # ========== 融合 ==========
        fused_feat = img_feat * (1 - gate) + attn_context * gate

        return fused_feat
    

class I2I_CrossAttentionWithGateFusion(nn.Module):
    """
    使用 Cross-Attention 对齐两个图像特征，并利用 Adaptive Gating 实现特征融合
    输入：
        feat1: [B, C, D, H, W] (图像1特征)
        feat2: [B, C, D, H, W] (图像2特征)
    输出：
        fused_feat: [B, C, D, H, W] (融合后的特征)
    """
    def __init__(self, channels, num_heads=4):
        super(I2I_CrossAttentionWithGateFusion, self).__init__()
        
        # Cross-Attention 部分
        self.q_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)

        # Adaptive Gating 部分
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, feat1, feat2):
        B, C, D, H, W = feat1.shape
        
        # ========== Cross-Attention 计算对齐 ==========
        q = self.q_proj(feat1).view(B, self.num_heads, C // self.num_heads, -1)  # [B, heads, C_head, L]
        k = self.k_proj(feat2).view(B, self.num_heads, C // self.num_heads, -1)
        v = self.v_proj(feat2).view(B, self.num_heads, C // self.num_heads, -1)
        
        attn = torch.softmax((q.transpose(-2, -1) @ k) * self.scale, dim=-1)  # [B, heads, L, L]
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # [B, heads, C_head, L]
        out = out.reshape(B, C, D, H, W)  # [B, C, D, H, W]
        
        # Cross-Attention 输出
        attn_out = self.out_proj(out)
        
        # ========== Adaptive Gating ==========
        # 拼接两个特征
        gated_input = torch.cat([attn_out, feat1], dim=1)  # [B, 2C, D, H, W]
        
        # 计算门控权重
        gate = self.gate_conv(gated_input)  # [B, C, D, H, W]
        
        # 特征融合
        fused_feat = gate * feat1 + (1 - gate) * attn_out
        
        return fused_feat

# endregion

# region 自适应门控特征融合
class T2I_AdaptiveGatingFusion(nn.Module):
    """
    自适应门控特征融合模块：
    输入:
        text_feat: [B, N, C_t]
        img_feat:  [B, C_i, D, H, W]
    输出:
        fused_feat: [B, C_i, D, H, W]
    """
    def __init__(self, text_dim, img_dim, hidden_dim=128, use_sigmoid=True):
        super(T2I_AdaptiveGatingFusion, self).__init__()
        
        # 将文本特征聚合成全局特征
        self.text_pool = nn.AdaptiveAvgPool1d(1)  # [B, C_t, 1]
        
        # 门控网络，将文本特征映射到图像通道维度
        self.gating = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, img_dim)
        )
        
        self.use_sigmoid = use_sigmoid

        # 将文本特征映射到图像特征的维度 (用于直接融合)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, img_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, text_feat, img_feat):
        """
        text_feat: [B, N, C_t]
        img_feat:  [B, C_i, D, H, W]
        """
        B, C_i, D, H, W = img_feat.shape
        _, N, C_t = text_feat.shape

        # [B, N, C_t] -> [B, C_t, N]
        text_feat_t = text_feat.permute(0, 2, 1)

        # 聚合文本特征: [B, C_t, N] -> [B, C_t, 1] -> [B, C_t]
        text_global = self.text_pool(text_feat_t).squeeze(-1)

        # 计算门控权重: [B, C_t] -> [B, C_i]
        gate = self.gating(text_global)
        if self.use_sigmoid:
            gate = torch.sigmoid(gate)  # [B, C_i]
        gate = gate.view(B, C_i, 1, 1, 1)  # 广播到图像维度

        # 将文本特征投影到图像维度并扩展到[D,H,W]
        text_proj = self.text_proj(text_global)  # [B, C_i]
        text_proj = text_proj.view(B, C_i, 1, 1, 1).expand(-1, -1, D, H, W)

        # 融合: 图像特征 + 自适应门控文本特征
        fused_feat = img_feat * (1 - gate) + text_proj * gate

        return fused_feat
    
class I2I_AdaptiveGatingFusion(nn.Module):
    """
    Adaptive Gating 融合 [B,C,D,H,W]
    """
    def __init__(self, channels):
        super(I2I_AdaptiveGatingFusion, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        # 拼接计算门控权重
        gate = self.gate_conv(torch.cat([feat1, feat2], dim=1))  # [B,C,D,H,W]
        fused = gate * feat1 + (1 - gate) * feat2
        return fused

# endregion

# region 最优传输融合
class I2I_OTFeaturesFusion(nn.Module):
    """
    基于熵正则化最优传输（OT）与自适应门控融合的图像特征融合。

    - 输入 feat1, feat2: B×C×H×W (2D) 或 B×C×D_s×H×W (3D)
    - 输出 fused:      B×C×H×W 或 B×C×D_s×H×W
    """
    def __init__(self, in_channels: int, eps: float = 0.1, n_iters: int = 50):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.n_iters = n_iters
        # 门控融合模块，通道数与输入一致
        self.gated_fuser = I2I_AdaptiveGatingFusion(in_channels)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        # feat1, feat2 必须同形状
        if feat1.shape != feat2.shape:
            raise ValueError(f"Shapes must match, got {feat1.shape} vs {feat2.shape}")
        B, C, *spatial = feat1.shape

        # 展平空间维为序列: B×C×M → B×M×C
        M = int(torch.prod(torch.tensor(spatial, device=feat1.device)))
        seq1 = feat1.reshape(B, C, M).permute(0, 2, 1)  # [B, M, C]
        seq2 = feat2.reshape(B, C, M).permute(0, 2, 1)  # [B, M, C]

        # 1. 代价矩阵 C_mat[b,i,j] = ||seq1[b,i] - seq2[b,j]||^2
        C_mat = torch.cdist(seq1, seq2, p=2).pow(2)     # [B, M, M]

        # 2. 构建 log_K 并做 Sinkhorn
        log_K = -C_mat / self.eps                      # [B, M, M]
        for _ in range(self.n_iters):
            log_K = log_K - torch.logsumexp(log_K, dim=2, keepdim=True)
            log_K = log_K - torch.logsumexp(log_K, dim=1, keepdim=True)
        P = torch.exp(log_K)                           # [B, M, M]

        # 3. 重心投影：得到 OT 融合特征
        fused_seq = torch.bmm(P, seq2)                # [B, M, C]
        fused_ot = fused_seq.permute(0, 2, 1).view_as(feat1)

        # 4. 自适应门控加权融合 原始 feat1 与 OT 融合结果
        out = self.gated_fuser(feat1, fused_ot)
        return out
    
# endregion

