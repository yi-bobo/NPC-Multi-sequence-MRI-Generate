
import torch
import torch.nn as nn

from Model.Net.CPC_BM_Net.transformer_block import SpatialTransformer
from Model.Net.CPC_BM_Net.resnet_block import DiffusionResnetBlock

from monai.networks.blocks import Convolution, SpatialAttentionBlock, Upsample

class WrappedUpsample(Upsample):
    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        upsampled: torch.Tensor = super().forward(x)
        return upsampled

class UpSample(nn.Module):
    """
    反卷积上采样块（与 DownSample 对称）

    Args:
        spatial_dims: 2 或 3，指定空域维度
        in_channels: 输入通道数
        out_channels: 输出通道数
        only_hw: 仅在 H/W 维度上上采样（只针对 3D 数据）
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        only_hw: bool = False
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.only_hw = only_hw

        # 根据 only_hw 和 spatial_dims 设置 stride, kernel_size, padding
        if self.only_hw and self.spatial_dims == 3:
            # 只在 H/W 上采样，D 维度保持不变
            stride = (1, 2, 2)
            kernel_size = (1, 3, 3)
            padding = (0, 1, 1)
        else:
            # 在所有维度上均等采样
            stride = (2,) * self.spatial_dims
            kernel_size = (3,) * self.spatial_dims
            padding = (1,) * self.spatial_dims

        # 构建转置卷积（反卷积）层
        self.up = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding=padding,
            conv_only=True,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        del emb
        # 通道数检查
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"输入通道 ({x.shape[1]}) 不等于预期 in_channels ({self.in_channels})"
            )
        # 执行反卷积上采样
        return self.up(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        only_hw: bool,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown
        self.num_res_blocks = num_res_blocks
        resnets = []

        for i in range(num_res_blocks):
            in_ch = in_channels if i == (num_res_blocks - 1) else in_channels+prev_output_channel
            out_ch = out_channels if i == (num_res_blocks - 1) else in_channels

            resnets.append(
                DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.upsampler: nn.Module | None
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = UpSample(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    only_hw=only_hw
                )

        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context

        if self.upsampler is not None:
            # pop res hidden states
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = self.upsampler(hidden_states, temb)

        for i, resnet in enumerate(self.resnets):
            if i != self.num_res_blocks-1:
                # pop res hidden states
                res_hidden_states = res_hidden_states_list[-1]
                res_hidden_states_list = res_hidden_states_list[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        return hidden_states

class DiffusionUpSample(nn.Module):
    """
    对应 DownSample 的反卷积上采样块
    stride/kernel_size/padding 设置应与 DownSample 保持一致：
      - (2,2,2)/(3,3,3)/(1,1,1)：D/H/W 全维度上采样
      - (1,2,2)/(1,3,3)/(0,1,1)：只在 H/W 上采样
    """
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 stride: tuple[int] = (2,2,2),
                 kernel_size: tuple[int] = (3,3,3),
                 padding: tuple[int] = (1,1,1),
                 only_up_HW: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        if only_up_HW:
            stride, kernel_size, padding = (1, 2, 2), (1, 3, 3), (0, 1, 1)
        self.up = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding=padding,
            conv_only=True,
            is_transposed=True   # 关键参数：启用反卷积
        )

    def forward(self, x: torch.Tensor, emb:torch.Tensor):
        del emb
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels ({x.shape[1]}) 不等于定义的 in_channels {self.in_channels}"
            )
        return self.up(x)

class AttnUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        only_hw: bool,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.resblock_updown = resblock_updown
        self.num_res_blocks = num_res_blocks

        # 1) 先根据 only_hw / spatial_dims 构造好 upsampler
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:
                self.upsampler = UpSample(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    only_hw=only_hw)
        else:
            self.upsampler = None

        # 2) 再动态根据 num_res_blocks 构造 resnets + attentions
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_res_blocks):
            # 计算“来自 encoder 的 skip 通道数”
            skip_ch = in_channels
            # 计算当前 Resnet 块的输入通道数
            # 第一个 block: 来自上一级 upsampler 输出
            # 后续 block: 来自上一个 block 输出 + skip 通道
            in_ch = prev_output_channel if (i == num_res_blocks - 1) else (in_channels + skip_ch)
            out_ch = out_channels if (i == num_res_blocks - 1) else in_channels

            self.resnets.append(
                DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            self.attentions.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_ch,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        # 1) pop 本层 skip for the first resnet (it's at the end of the list)
        #    注意：pop 顺序要和 init 中 skip_ch 一致
        res_skip = res_hidden_states_list.pop()
        # 2) upsample
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        # 3) 走每个 Resnet + Attention
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            # 再 pop 下一个 skip
            if i != (self.num_res_blocks-1):
                res_skip = res_hidden_states_list.pop()
                # cat skip
                hidden_states = torch.cat([hidden_states, res_skip], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states).contiguous()

        return hidden_states

# class AttnUpBlock(nn.Module):
#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         prev_output_channel: int,
#         out_channels: int,
#         temb_channels: int,
#         only_hw: bool,
#         num_res_blocks: int = 1,
#         norm_num_groups: int = 32,
#         norm_eps: float = 1e-6,
#         add_upsample: bool = True,
#         resblock_updown: bool = False,
#         num_head_channels: int = 1,
#         include_fc: bool = True,
#         use_combined_linear: bool = False,
#         use_flash_attention: bool = False,
#     ) -> None:
#         super().__init__()
#         self.resblock_updown = resblock_updown

#         resnets = []
#         attentions = []

#         for i in range(num_res_blocks):
#             res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
#             resnet_in_channels = prev_output_channel if i == 0 else out_channels

#             resnets.append(
#                 DiffusionResnetBlock(
#                     spatial_dims=spatial_dims,
#                     in_channels=resnet_in_channels + res_skip_channels,
#                     out_channels=out_channels,
#                     temb_channels=temb_channels,
#                     norm_num_groups=norm_num_groups,
#                     norm_eps=norm_eps,
#                 )
#             )
#             attentions.append(
#                 SpatialAttentionBlock(
#                     spatial_dims=spatial_dims,
#                     num_channels=out_channels,
#                     num_head_channels=num_head_channels,
#                     norm_num_groups=norm_num_groups,
#                     norm_eps=norm_eps,
#                     include_fc=include_fc,
#                     use_combined_linear=use_combined_linear,
#                     use_flash_attention=use_flash_attention,
#                 )
#             )

#         self.resnets = nn.ModuleList(resnets)
#         self.attentions = nn.ModuleList(attentions)

#         self.upsampler: nn.Module | None
#         if add_upsample:
#             if resblock_updown:
#                 self.upsampler = DiffusionResnetBlock(
#                     spatial_dims=spatial_dims,
#                     in_channels=out_channels,
#                     out_channels=out_channels,
#                     temb_channels=temb_channels,
#                     norm_num_groups=norm_num_groups,
#                     norm_eps=norm_eps,
#                     up=True,
#                 )
#             else:
#                 post_conv = Convolution(
#                     spatial_dims=spatial_dims,
#                     in_channels=out_channels,
#                     out_channels=out_channels,
#                     strides=1,
#                     kernel_size=3,
#                     padding=1,
#                     conv_only=True,
#                 )
#                 self.upsampler = UpSample(
#                     spatial_dims=spatial_dims,
#                     in_channels=out_channels,
#                     out_channels=out_channels,
#                     only_hw=only_hw
#                 )
#         else:
#             self.upsampler = None

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         res_hidden_states_list: list[torch.Tensor],
#         temb: torch.Tensor,
#         context: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         del context
#         # pop res hidden states
#         res_hidden_states = res_hidden_states_list[-1]
#         res_hidden_states_list = res_hidden_states_list[:-1]
        
#         if self.upsampler is not None:
#             hidden_states = self.upsampler(hidden_states, temb)
        
#         for resnet, attn in zip(self.resnets, self.attentions):
#             # pop res hidden states
#             res_hidden_states = res_hidden_states_list[-1]
#             res_hidden_states_list = res_hidden_states_list[:-1]
#             hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

#             hidden_states = resnet(hidden_states, temb)
#             hidden_states = attn(hidden_states).contiguous()

#         # if self.upsampler is not None:
#         #     hidden_states = self.upsampler(hidden_states, temb)

#         return hidden_states


class CrossAttnUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        only_hw:bool,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_upsample: bool = True,
        resblock_updown: bool = False,
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
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    num_layers=transformer_num_layers,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    dropout=dropout_cattn,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.upsampler: nn.Module | None
        if add_upsample:
            if resblock_updown:
                self.upsampler = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    up=True,
                )
            else:

                post_conv = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
                self.upsampler = UpSample(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    only_hw=only_hw
                )
        else:
            self.upsampler = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_list: list[torch.Tensor],
        temb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states