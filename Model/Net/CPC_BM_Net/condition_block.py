
# region ImageEncode
"版本：1.0.0"
# region 
# class ImageEncode(nn.Module):
#     def __init__(self, 
#                  spatial_dims: int,
#                  in_channels: int,
#                  out_channels: int,
#                  num_res_blocks: int,
#                  norm_num_groups: int,
#                  norm_eps: float,
#                  resblock_updown: bool,
#                  num_head_channels: int,
#                  with_attn: bool = False,
#                  add_downsample: bool = True,
#                  downsample_padding: int = 1,
#                  include_fc: bool = True,
#                  use_combined_linear: bool = False,
#                  use_flash_attention: bool = False,):
#         super().__init__()
#         self.resblock_updown = resblock_updown
#         self.with_attn = with_attn
#         # 按照 num_res_blocks 次数构建平行的 ResNet+Attention 列表
#         resnets = []
#         for i in range(num_res_blocks):
#             resnets.append(
#                 ConditionalUNetResnetBlock(
#                     spatial_dims=spatial_dims,
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     norm_num_groups=norm_num_groups,
#                     norm_eps=norm_eps,
#                 )
#             )
#         # 将列表封装为 ModuleList，保证参数注册
#         self.resnets = nn.ModuleList(resnets)

#         attentions = []
#         for i in range(num_res_blocks):
#             if with_attn:
#                 # 2) 空间注意力模块
#                 attentions.append(
#                     SpatialAttentionBlock(
#                         spatial_dims=spatial_dims,
#                         num_channels=out_channels,
#                         num_head_channels=num_head_channels,
#                         norm_num_groups=norm_num_groups,
#                         norm_eps=norm_eps,
#                         include_fc=include_fc,
#                         use_combined_linear=use_combined_linear,
#                         use_flash_attention=use_flash_attention,
#                     )
#                 )
#         self.attentions = nn.ModuleList(attentions)
#         # 下采样模块：可选添加
#         self.downsampler = None
#         if add_downsample:
#             if resblock_updown:
#                 # 使用 ResNet Block（down=True）来做下采样
#                 self.downsampler = ConditionalUNetResnetBlock(
#                     spatial_dims=spatial_dims,
#                     in_channels=out_channels,
#                     out_channels=out_channels,
#                     norm_num_groups=norm_num_groups,
#                     norm_eps=norm_eps,
#                     down=True,                # 指定下采样行为
#                 )
#             else:
#                 # 使用专门的 Conv/Pool 下采样
#                 self.downsampler = ConditionalUnetDownsample(
#                     spatial_dims=spatial_dims,
#                     num_channels=out_channels,
#                     use_conv=True,            # 用卷积方式下采样
#                     out_channels=out_channels,
#                     padding=downsample_padding,
#                 )
    
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#     ) -> tuple[torch.Tensor, list[torch.Tensor]]:
#         """
#         Args:
#             hidden_states: [B, C, ...]，输入特征图
#             context:       可选上下文（此处忽略，仅为接口一致）
#         Returns:
#             hidden_states:   经所有模块处理后的特征图
#             output_states:   包含每次 ResNet+Attention 后（及最终下采样后）的中间特征列表
#         """
#         output_states = []

#         # 1) 对每个 ResNet Block + Attention 作前向，并收集输出
#         for resnet in self.resnets:
#             hidden_states = resnet(hidden_states)    # 注入时间嵌入
#             if self.with_attn:
#                 for attn in self.attentions:
#                     hidden_states = attn(hidden_states).contiguous()
#                     output_states.append(hidden_states)
#             else:
#                 output_states.append(hidden_states)

#         # 2) 下采样（若配置了）
#         if self.downsampler is not None:
#             hidden_states = self.downsampler(hidden_states)
#             output_states.append(hidden_states)

#         # 返回最终特征与所有中间特征
#         return hidden_states, output_states
# endregion

"version：1.0.1"   # //? 只采用n个残差块提取条件图像特征然后通过下采样到对应维度 /
class ImageEncode(nn.Module):
    def __init__(self, 
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 num_res_blocks: int,
                 norm_num_groups: int,
                 norm_eps: float,
                 add_downsample: bool = True,
                 downsample_padding: int = 1,):
        super().__init__()
        # 按照 num_res_blocks 次数构建平行的 ResNet列表
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(
                ResNetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
        # 将列表封装为 ModuleList，保证参数注册
        self.resnets = nn.ModuleList(resnets)

        # 使用专门的 Conv/Pool 下采样
        self.downsampler = None
        if add_downsample:
            self.downsampler = ConditionalUnetDownsample(
                spatial_dims=spatial_dims,
                num_channels=in_channels,
                use_conv=True,            # 用卷积方式下采样
                out_channels=out_channels,
                padding=downsample_padding,
            )
        else: 
            self.downsampler = ConditionalUnetDownsample(
                spatial_dims=spatial_dims,
                num_channels=in_channels,
                use_conv=True,            # 用卷积方式下采样
                out_channels=out_channels,
                strides=(1, 2, 2),          # //?只在H和W维度下采样
                padding=downsample_padding,
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, C, ...]，输入特征图
        Returns:
            hidden_states:   经所有模块处理后的特征图
        """

        # 1) 对每个 ResNet Block + Attention 作前向，并收集输出
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)  

        # 2) 下采样
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
        

        # 返回最终特征与所有中间特征
        return hidden_states
# endregion

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

    # for i in range(N):
    #     for j in range(M):
    #         if cost_function == "euclidean":
    #             cost_matrix[i, j] = torch.norm(A[:, i] - B[:, j], p=2)  # 计算欧几里得距离

    return cost_matrix


class T2I_OptimalTransportAligner(nn.Module):
    """
    熵正则化最优传输对齐：
      - 文本特征：B × N × D
      - 图像特征：B × C × H × W 或  B × C × D_s × H × W
    会把图像特征展平为 B × M × C，再计算 OT，将文本特征对齐到图像空间。
    """

    def __init__(self, epsilon=0.1, niter=50):  # 增大 eps
        super().__init__()
        self.epsilon = epsilon
        self.niter = niter
    
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


    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        对文本特征和图像特征进行最优传输对齐
        参数：
            text_feats: [B, N, D] 文本特征
            image_feats: [B, C, H, W] 或 [B, C, D_s, H, W] 图像特征
        返回：
            对齐后的文本特征
        """

        B, N, C1 = text.shape
        if image.dim() == 4:  # 处理二维图像特征 [B, C, H, W]
            B2, C2, H, W = image.shape
            M = H * W
        elif image.dim() == 5:  # 处理三维图像特征 [B, C, D_s, H, W]
            B2, C2, D, H, W = image.shape
            M = D * H * W
                
        assert B == B2 and C1 == C2, "Batch size or channel dimensions do not match."

        text_feats = text.permute(0, 2, 1)  # [B, N, C1] -> [B, C1, N]
        text_feats = text_feats.view(-1, N) # [B, C1, N] -> [B*C1, N]

        image_feats = image.view(B, C2, -1)  # [B, C2, H, W] -> [B, C2, H*W] or [B, C2, D_s, H, W] -> [B, C2, D_s*H*W]
        image_feats = image_feats.view(-1, M)  # [B, C2, M] -> [B*C2, M]

        aligned_text, aligned_image = self.optimal_transport(text_feats, image_feats)  # [B*C1, M] -> [B*C1, N]

        aligned_text = aligned_text.view(B, C1, N)  # [B*C1, N] -> [B, C1, N]
        aligned_text = aligned_text.permute(0, 2, 1)  # [B, C1, N] -> [B, N, C1]

        aligned_image = aligned_image.view(B, C2, M)  # [B*C2, M] -> [B, C2, M]
        aligned_image = aligned_image.view(B, C2, *image.shape[2:])

        return aligned_text, aligned_image


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

# region 文本特征映射
class TextMapping(nn.Module):
    """
    文本特征映射：
      - 输入 feat: B×N×D_text
      - 输出 mapped: B×N×D_mapped
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # 每个block 用一个 Sequential 包两层 Convolution
            block = nn.Sequential(
                Convolution(
                    spatial_dims=1,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                ),
                Convolution(
                    spatial_dims=1,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                ),
            )
            self.layers.append(block)
            # 下一轮的 in_channels 要变成 out_channels
            in_channels = out_channels

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, N, D_text] → [B, D_text, N]
        x = feat.permute(0, 2, 1)
        for block in self.layers:
            x = block(x)
        # [B, D_mapped, N] → [B, N, D_mapped]
        return x.permute(0, 2, 1)