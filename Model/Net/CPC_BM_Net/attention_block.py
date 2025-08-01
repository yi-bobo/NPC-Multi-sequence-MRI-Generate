import sys
sys.path.append("/data1/weiyibo/NPC-MRI/Code/NPC-Multi-sequence-MRI-Generate/")
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, SABlock

import torch
import torch.nn as nn
from typing import Optional
from einops.layers.torch import Rearrange


class SpatialAttentionBlock(nn.Module):
    """优化版 Spatial Self-Attention 模块，支持 1D/2D/3D"""

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        attention_dtype: Optional[torch.dtype] = None,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        assert spatial_dims in [1, 2, 3], "spatial_dims must be 1, 2, or 3"
        self.spatial_dims = spatial_dims

        # GroupNorm
        self.norm = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=num_channels,
            eps=norm_eps,
            affine=True,
        )

        # 计算多头注意力的head数量
        if num_head_channels is None:
            num_heads = 1
        else:
            if num_channels % num_head_channels != 0:
                raise ValueError(f"num_channels ({num_channels}) 必须能被 num_head_channels ({num_head_channels}) 整除")
            num_heads = num_channels // num_head_channels

        # 构建注意力核心
        self.attn = SABlock(
            hidden_size=num_channels,
            num_heads=num_heads,
            qkv_bias=True,
            attention_dtype=attention_dtype,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        # Rearrange 输入输出形状 (提前定义)
        if spatial_dims == 1:
            self.rearrange_input = Rearrange("b c h -> b h c")
            self.rearrange_output = Rearrange("b h c -> b c h")
        elif spatial_dims == 2:
            self.rearrange_input = Rearrange("b c h w -> b (h w) c")
            self.rearrange_output = lambda x, h, w: x.view(x.shape[0], -1, h, w)
        else:  # 3D
            self.rearrange_input = Rearrange("b c h w d -> b (h w d) c")
            self.rearrange_output = lambda x, h, w, d: x.view(x.shape[0], -1, h, w, d)

    def forward(self, x: torch.Tensor):
        residual = x
        b, c, *spatial = x.shape

        x = self.norm(x)
        x = self.rearrange_input(x)  # B x N x C

        # 注意力计算
        x = self.attn(x)

        # 恢复形状
        if self.spatial_dims == 1:
            x = self.rearrange_output(x)
        elif self.spatial_dims == 2:
            h, w = spatial
            x = self.rearrange_output(x, h, w)
        else:
            h, w, d = spatial
            x = self.rearrange_output(x, h, w, d)

        return x + residual

class MultiHeadAttention(nn.Module):
    def __init__(self, spatial_dims, channels, num_heads=4, norm_groups=32, cross_attention=False):
        """
        通用 Multi-Head Attention (支持 Self-Attention 和 Cross-Attention)
        使用 MONAI 的 Convolution 模块替代原生 ConvNd
        Args:
            channels: 输入通道数 (Q通道数)
            num_heads: 注意力头数
            norm_groups: GroupNorm组数
            spatial_dims: 支持1/2/3维
            cross_attention: 是否启用 Cross-Attention (Q≠K/V)
        """
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} 必须能被 num_heads {num_heads} 整除"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.cross_attention = cross_attention

        # Q的GroupNorm和投影
        self.norm_q = nn.GroupNorm(num_groups=norm_groups, num_channels=channels)
        self.q_proj = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            conv_only=True,  # 不加Norm和激活
        )

        if cross_attention:
            # Cross-Attention: context 用于 K 和 V
            self.norm_kv = nn.GroupNorm(num_groups=norm_groups, num_channels=channels)
            self.k_proj = Convolution(spatial_dims, channels, channels, 1, conv_only=True)
            self.v_proj = Convolution(spatial_dims, channels, channels, 1, conv_only=True)
        else:
            # Self-Attention: 共享 qkv 投影
            self.qkv_proj = Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels,
                out_channels=channels * 3,
                kernel_size=1,
                conv_only=True,
            )

        # 输出投影
        self.proj_out = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            conv_only=True,
        )

    def forward(self, x, context=None):
        """
        x: Query输入 [B, C, ...]
        context: Cross-Attention时提供的上下文特征 (K/V来源)
        """
        B, C, *spatial = x.shape
        num_tokens = torch.prod(torch.tensor(spatial, device=x.device)).item()

        if self.cross_attention:
            assert context is not None, "Cross-Attention 模式必须提供 context"
            q = self.q_proj(self.norm_q(x))
            k = self.k_proj(self.norm_kv(context))
            v = self.v_proj(self.norm_kv(context))
        else:
            qkv = self.qkv_proj(self.norm_q(x))
            q, k, v = torch.chunk(qkv, 3, dim=1)

        # 调整为 [B, heads, head_dim, N]
        q = q.reshape(B, self.num_heads, self.head_dim, num_tokens)
        k = k.reshape(B, self.num_heads, self.head_dim, num_tokens)
        v = v.reshape(B, self.num_heads, self.head_dim, num_tokens)

        # 计算注意力权重 [B, heads, Nq, Nk]
        attn = torch.einsum("b h c n, b h c m -> b h n m", q * self.scale, k)
        attn = F.softmax(attn, dim=-1)

        # 聚合输出 [B, heads, head_dim, N]
        out = torch.einsum("b h n m, b h c m -> b h c n", attn, v)
        out = out.reshape(B, C, *spatial)

        return x + self.proj_out(out)



