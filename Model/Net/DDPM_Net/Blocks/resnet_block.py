
import torch
import torch.nn as nn

from .module import zero_module

from monai.networks.blocks import Convolution


class DiffusionUNetResnetBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        out_channels: int | None = None,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.upsample = self.downsample = None
        if self.up:
            from .up_sample import WrappedUpsample
            self.upsample = WrappedUpsample(
                spatial_dims=spatial_dims,
                mode="nontrainable",
                in_channels=in_channels,
                out_channels=in_channels,
                interp_mode="nearest",
                scale_factor=2.0,
                align_corners=None,
            )
        elif down:
            from .down_sample import DiffusionUnetDownsample
            self.downsample = DiffusionUnetDownsample(spatial_dims, in_channels, use_conv=False)

        self.time_emb_proj = nn.Linear(temb_channels, self.out_channels)

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )
        self.skip_connection: nn.Module
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if self.spatial_dims == 2:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
        else:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
        h = h + temb

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        output: torch.Tensor = self.skip_connection(x) + h
        return output
    
class ResnetBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int | None = None,
        up: bool = False,
        down: bool = False,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.upsample = self.downsample = None
        if self.up:
            from .up_sample import WrappedUpsample
            self.upsample = WrappedUpsample(
                spatial_dims=spatial_dims,
                mode="nontrainable",
                in_channels=in_channels,
                out_channels=in_channels,
                interp_mode="nearest",
                scale_factor=2.0,
                align_corners=None,
            )
        elif down:
            from .down_sample import DiffusionUnetDownsample
            self.downsample = DiffusionUnetDownsample(spatial_dims, in_channels, use_conv=False)


        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )
        self.skip_connection: nn.Module
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)
        output: torch.Tensor = self.skip_connection(x) + h
        return output