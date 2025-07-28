
import torch
import torch.nn as nn
from Model.Net.BDBM_CC_Net.zero_module import zero_module
from monai.networks.blocks import Convolution, CrossAttentionBlock, MLPBlock, SABlock

class DiffusionUNetTransformerBlock(nn.Module):
    """
    Diffusion U-Net 中的 Transformer 块：
      - 先做 Self-Attention + 残差
      - 再做 Cross-Attention (可选) + 残差
      - 最后做 Feed-Forward (MLP) + 残差
      - 每一步前都有 LayerNorm，遵循 Pre-Norm 设计
    """

    def __init__(
        self,
        num_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
    ) -> None:
        """
        Args:
            num_channels:           输入和输出的特征维度 C
            num_attention_heads:    注意力头数 H
            num_head_channels:      每个头的通道数 d_h，故 embed_dim = H * d_h
            dropout:                注意力和 MLP 中的 dropout 比例
            cross_attention_dim:    Cross-Attention 的 context 维度 (None 则只做 Self-Attn)
            upcast_attention:       是否在注意力计算中使用更高精度 (float32)
            use_flash_attention:    是否启用 FlashAttention 优化
            include_fc:             在 Self-Attn/Cross-Attn 后是否包含前馈层（SABlock 内部选项）
            use_combined_linear:    是否在 SABlock 中合并 QKV 线性层以少量胶囊化优化
        """
        super().__init__()
        embed_dim = num_attention_heads * num_head_channels

        # 1) Self-Attention Block
        #    - 输入： [B, N, C] （在使用时，x 可能需要先 reshape）
        #    - hidden_input_size = C, hidden_size = embed_dim
        self.attn1 = SABlock(
            hidden_size=embed_dim,
            hidden_input_size=num_channels,
            num_heads=num_attention_heads,
            dim_head=num_head_channels,
            dropout_rate=dropout,
            attention_dtype=torch.float if upcast_attention else None,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        # 2) Feed-Forward MLP
        #    - 通道数翻倍再回到 C (GEGLU 激活)
        self.ff = MLPBlock(
            hidden_size=num_channels,
            mlp_dim=num_channels * 4,
            act="GEGLU",
            dropout_rate=dropout
        )
        # 3) Cross-Attention Block
        #    - Query 来自 x，Key/Value 来自 context
        self.attn2 = CrossAttentionBlock(
            hidden_size=embed_dim,
            hidden_input_size=num_channels,
            context_input_size=cross_attention_dim,
            num_heads=num_attention_heads,
            dim_head=num_head_channels,
            dropout_rate=dropout,
            attention_dtype=torch.float if upcast_attention else None,
            use_flash_attention=use_flash_attention,
        )

        # LayerNorm for Pre-Norm
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.norm3 = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:       [B, N, C] 输入特征，N 可能是序列长度或扁平化空间维度
            context: [B, M, C_ctx] 可选的跨注意力上下文（文本、编码器输出等）
        Returns:
            out:     [B, N, C] 通过 Self-Attn、Cross-Attn、MLP 后的残差输出
        """
        # 1️⃣ Self-Attention + 残差
        #    norm1 → SABlock → + x
        x = self.attn1(self.norm1(x)) + x

        # 2️⃣ Cross-Attention + 残差（如果提供 context，否则等价恒等）
        #    norm2 → CrossAttentionBlock → + x
        x = self.attn2(self.norm2(x), context=context) + x

        # 3️⃣ Feed-Forward (MLP) + 残差
        #    norm3 → MLPBlock → + x
        x = self.ff(self.norm3(x)) + x

        return x

class SpatialTransformer(nn.Module):
    """
    空间维度上的跨模态 Transformer 模块，支持 2D/3D 特征图的 Self-/Cross-Attention 与前馈处理。

    1. 先做 GroupNorm + 1×1 卷积投影到注意力内维度
    2. 按空间展平，送入若干层 TransformerBlock（包含 Self-Attn、Cross-Attn、MLP）
    3. 重塑回空间形状，1×1 卷积投影回原始通道数
    4. 残差连接
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:         空间维度，2 或 3
            in_channels:          输入特征图的通道数 C
            num_attention_heads:  注意力头数 H
            num_head_channels:    每个头的通道维度 d_h，inner_dim = H * d_h
            num_layers:           TransformerBlock 层数
            dropout:              注意力与 MLP 的 dropout 比例
            norm_num_groups:      GroupNorm 的组数
            norm_eps:             GroupNorm epsilon
            cross_attention_dim:  跨注意力上下文的特征维度（None 则省略 Cross-Attn）
            upcast_attention:     是否在注意力计算中使用 float32 精度
            include_fc:           TransformerBlock 中是否包含前馈层
            use_combined_linear:  在注意力中是否合并 QKV 线性变换以优化
            use_flash_attention:  是否启用 FlashAttention 加速
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels

        # 内部注意力嵌入维度 = H * d_h
        inner_dim = num_attention_heads * num_head_channels

        # 1. 归一化层：GroupNorm 适用 2D/3D
        self.norm = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=norm_eps,
            affine=True
        )

        # 2. 投影到注意力维度的 1×1 卷积 (conv_only=True: 仅卷积，无 Norm/Act)
        self.proj_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=inner_dim,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )

        # 3. 若干层 TransformerBlock (包含 Self-Attn, Cross-Attn, MLP)
        self.transformer_blocks = nn.ModuleList([
            DiffusionUNetTransformerBlock(
                num_channels=inner_dim,
                num_attention_heads=num_attention_heads,
                num_head_channels=num_head_channels,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )
            for _ in range(num_layers)
        ])

        # 4. 输出投影回原始通道数的 1×1 卷积，zero_module 使其初始为零，以便初期近似恒等映射
        self.proj_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inner_dim,
                out_channels=in_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:       输入特征图 tensor
                     - 2D: [B, C, H, W]
                     - 3D: [B, C, D, H, W]
            context: 可选的跨注意力上下文 (sequence 或扁平化空间特征)
        Returns:
            输出与输入相同形状，但经过 (Norm→Proj_in→Transformer→Proj_out) 后再加上残差 x
        """
        # 保留残差分支
        residual = x

        # 1) 归一化
        x = self.norm(x)

        # 2) 投影到内维
        x = self.proj_in(x)

        # 3) 展平空间维度到序列长度 N
        B = x.shape[0]
        if self.spatial_dims == 2:
            # [B, inner_dim, H, W] → [B, H*W, inner_dim]
            _, inner_dim, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, inner_dim)
        else:  # spatial_dims == 3
            # [B, inner_dim, D, H, W] → [B, D*H*W, inner_dim]
            _, inner_dim, D, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, inner_dim)

        # 4) 通过每层 TransformerBlock
        for block in self.transformer_blocks:
            x = block(x, context=context)

        # 5) 重塑回空间并投影回原始通道
        if self.spatial_dims == 2:
            x = x.reshape(B, H, W, inner_dim).permute(0, 3, 1, 2).contiguous()
        else:
            x = x.reshape(B, D, H, W, inner_dim).permute(0, 4, 1, 2, 3).contiguous()

        # 6) 输出投影并加上残差
        x = self.proj_out(x)
        return x + residual
