
import torch
import torch.nn as nn

from Model.Net.BDBM_CC_Net.transformer_block import SpatialTransformer
from Model.Net.BDBM_CC_Net.resnet import DiffusionUNetResnetBlock, ConditionalUNetResnetBlock

from monai.networks.blocks import SpatialAttentionBlock


class AttnMidBlock(nn.Module):
    """
    中间（Bridge）块：用于在 U-Net 中心位置插入 ResNet + Attention + ResNet 模块。
    结构：
      1. ResNet Block（注入时间嵌入 temb）
      2. 空间注意力（Spatial Attention）
      3. ResNet Block（注入时间嵌入 temb）
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:         2 或 3，决定使用 2D/3D 操作
            in_channels:          输入和输出的特征通道数
            temb_channels:        时间/条件嵌入向量的维度
            norm_num_groups:      GroupNorm 的组数
            norm_eps:             GroupNorm 的 epsilon 值
            num_head_channels:    每个注意力头的通道数
            include_fc:           是否在 SpatialAttentionBlock 中包含前馈层
            use_combined_linear:  是否合并 QKV 线性层以优化
            use_flash_attention:  是否启用 FlashAttention 加速
        """
        super().__init__()

        # 1️⃣ 第一个 ResNet Block
        #    - 输入/输出通道均为 in_channels
        #    - 注入时间嵌入 temb，用于扩散模型的时序条件
        self.resnet_1 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

        # 2️⃣ 空间注意力模块
        #    - 在空间维度上为每个位置分配注意力权重
        #    - 可以选择是否包含前馈层、是否合并 QKV、是否使用 FlashAttention
        self.attention = SpatialAttentionBlock(
            spatial_dims=spatial_dims,
            num_channels=in_channels,
            num_head_channels=num_head_channels[-1],
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        # 3️⃣ 第二个 ResNet Block
        #    - 与第一个 ResNet 相同，但用于在注意力后再做一次特征变换
        self.resnet_2 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, C, ...]  输入特征图
            temb:          [B, temb_channels] 时间/条件嵌入
            context:       未使用，仅为接口兼容
        Returns:
            hidden_states: 经过 ResNet → Attention → ResNet 后的特征图
        """
        # 丢弃未使用的 context
        del context

        # 1) ResNet Block + 时间嵌入
        hidden_states = self.resnet_1(hidden_states, temb)

        # 2) 空间注意力
        hidden_states = self.attention(hidden_states).contiguous()

        # 3) 再次 ResNet Block + 时间嵌入
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states

class ConditionalAttnMidBlock(nn.Module):
    """
    中间（Bridge）块：用于在 U-Net 中心位置插入 ResNet + Attention + ResNet 模块。
    结构：
      1. ResNet Block（注入时间嵌入 temb）
      2. 空间注意力（Spatial Attention）
      3. ResNet Block（注入时间嵌入 temb）
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims:         2 或 3，决定使用 2D/3D 操作
            in_channels:          输入和输出的特征通道数
            temb_channels:        时间/条件嵌入向量的维度
            norm_num_groups:      GroupNorm 的组数
            norm_eps:             GroupNorm 的 epsilon 值
            num_head_channels:    每个注意力头的通道数
            include_fc:           是否在 SpatialAttentionBlock 中包含前馈层
            use_combined_linear:  是否合并 QKV 线性层以优化
            use_flash_attention:  是否启用 FlashAttention 加速
        """
        super().__init__()

        # 1️⃣ 第一个 ResNet Block
        #    - 输入/输出通道均为 in_channels
        #    - 注入时间嵌入 temb，用于扩散模型的时序条件
        self.resnet_1 = ConditionalUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

        # 2️⃣ 空间注意力模块
        #    - 在空间维度上为每个位置分配注意力权重
        #    - 可以选择是否包含前馈层、是否合并 QKV、是否使用 FlashAttention
        self.attention = SpatialAttentionBlock(
            spatial_dims=spatial_dims,
            num_channels=in_channels,
            num_head_channels=num_head_channels[-1],
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )

        # 3️⃣ 第二个 ResNet Block
        #    - 与第一个 ResNet 相同，但用于在注意力后再做一次特征变换
        self.resnet_2 = ConditionalUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, C, ...]  输入特征图
            context:       未使用，仅为接口兼容
        Returns:
            hidden_states: 经过 ResNet → Attention → ResNet 后的特征图
        """
        # 丢弃未使用的 context
        del context

        # 1) ResNet Block + 时间嵌入
        hidden_states = self.resnet_1(hidden_states)

        # 2) 空间注意力
        hidden_states = self.attention(hidden_states).contiguous()

        # 3) 再次 ResNet Block + 时间嵌入
        hidden_states = self.resnet_2(hidden_states)

        return hidden_states

class CrossAttnMidBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        self.resnet_1 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialTransformer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_attention_heads=in_channels // num_head_channels,
            num_head_channels=num_head_channels,
            num_layers=transformer_num_layers,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.resnet_2 = DiffusionUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states, context=context)
        hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states
    
class ConditionalCrossAttnMidBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        num_head_channels: int = 1,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        self.resnet_1 = ConditionalUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )
        self.attention = SpatialTransformer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_attention_heads=in_channels // num_head_channels,
            num_head_channels=num_head_channels,
            num_layers=transformer_num_layers,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            dropout=dropout_cattn,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
        )
        self.resnet_2 = ConditionalUNetResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
        )

    def forward(
        self, hidden_states: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = self.resnet_1(hidden_states)
        hidden_states = self.attention(hidden_states, context=context)
        hidden_states = self.resnet_2(hidden_states)

        return hidden_states