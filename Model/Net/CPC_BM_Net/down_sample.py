import torch
import torch.nn as nn

from Model.Net.CPC_BM_Net.transformer_block import SpatialTransformer
from Model.Net.CPC_BM_Net.resnet_block import DiffusionResnetBlock, ResnetBlock

from monai.networks.blocks import Convolution, SpatialAttentionBlock
from monai.networks.layers.factories import Pool

class DiffusionUnetDownsample(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        use_conv: bool,
        out_channels: int | None = None,
        padding: int = 1,
        only_hw: bool = False
    ) -> None:
        """
        Args:
            spatial_dims: 2 or 3, spatial dimensions of the data.
            num_channels: input channels.
            use_conv: whether to use convolutional downsampling.
            out_channels: channels after downsampling (if None, = num_channels).
            padding: padding size for conv downsample.
            downsample_hw_only: if True and spatial_dims==3, only downsample H and W dimensions,
                                keep D (depth) unchanged.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        self.only_hw = only_hw
        if only_hw:
            strides, kernel_size, paddings = (1, 2, 2), (1, 3, 3), (0, padding, padding)
        else:
            strides, kernel_size, paddings = 2, 3, padding

        if use_conv:
            # Only downsample H and W: stride (1,2,2), kernel_size (1,3,3), pad (0,p,p)
            self.op = Convolution(
                spatial_dims=3,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=strides,
                kernel_size=kernel_size,
                padding=paddings,
                conv_only=True,
            )
        else:
            # Pooling-based downsampling
            if self.num_channels != self.out_channels:
                raise ValueError(
                    "num_channels and out_channels must match when use_conv=False"
                )
            else:
                # Uniform pooling downsample on all dims
                self.op = Pool[Pool.AVG, self.spatial_dims](
                    kernel_size=kernel_size,
                    stride=strides,
                )

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        # emb not used
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input channels ({x.shape[1]}) != expected ({self.num_channels})"
            )
        # Apply the selected downsampling operation
        return self.op(x)


class Downsample(nn.Module):
    def __init__(
        self, spatial_dims: int, num_channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1, only_hw:bool=False,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if only_hw:
            strides, kernel_size, paddings = (1, 2, 2), (1, 3, 3), (0, padding, padding)
        else:
            strides, kernel_size, paddings = 2, 3, padding
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=strides,
                kernel_size=kernel_size,
                padding=paddings,
                conv_only=True,
            )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError("num_channels and out_channels must be equal when use_conv=False")
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=kernel_size, stride=strides)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels "
                f"({self.num_channels})"
            )
        output: torch.Tensor = self.op(x)
        return output

class DownBlock(nn.Module):
    
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        only_hw:bool,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsampler: nn.Module | None
            if resblock_updown:
                self.downsampler = DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    down=True,
                )
            else:
                self.downsampler = DiffusionUnetDownsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                    only_hw=only_hw,
                )
        else:
            self.downsampler = None

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states

class DiffusionDownSample(nn.Module):
    """
    stride=(2,2,2)&kernel_size=(3,3,3)&padding=(1,1,1)表示在D H W三个维度上进行下采样；
    stride=(1,2,2)&kernel_size=(1,3,3)&padding=(0,1,1)表示只在 H W两个维度上进行下采样；
    """
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 stride: list[int] = (2,2,2),
                 kernel_size: list[int] = (3,3,3),
                 padding: list[int] = (1,1,1),
                 only_hw: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        if only_hw:
            stride, kernel_size, padding = (1, 2, 2), (1, 3, 3), (0, 1, 1)
        self.down = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding=padding,
            # conv_only=True
        )

    def forward(self, x:torch.Tensor, emb:torch.Tensor):
        del emb
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels {self.in_channels}")
        else:
            out = self.down(x)
        return out
    
class AttnDownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        only_hw=bool,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = True,
        downsample_padding: int = 1,
        num_head_channels: int = 1,
        include_fc: bool = True,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                )
            )
            attentions.append(
                SpatialAttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    include_fc=include_fc,
                    use_combined_linear=use_combined_linear,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsampler: nn.Module | None
        if resblock_updown:
            self.downsampler = DiffusionResnetBlock(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                only_hw=only_hw,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                down=True,
            )
        else:
            self.downsampler = DiffusionUnetDownsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
                only_hw=only_hw,
            )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states).contiguous()
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        only_hw=bool,
        num_res_blocks: int = 1,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        add_downsample: bool = True,
        resblock_updown: bool = False,
        downsample_padding: int = 1,
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
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                DiffusionResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
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
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsampler: nn.Module | None
        if resblock_updown:
            self.downsampler = DiffusionResnetBlock(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                only_hw=only_hw,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                down=True,
            )
        else:
            self.downsampler = DiffusionUnetDownsample(
                spatial_dims=spatial_dims,
                num_channels=out_channels,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
                only_hw=only_hw,
            )

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context).contiguous()
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states
