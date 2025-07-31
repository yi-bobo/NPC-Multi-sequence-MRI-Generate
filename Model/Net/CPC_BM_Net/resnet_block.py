import torch
import torch.nn as nn

from monai.networks.blocks import Convolution

class ResnetBlock(nn.Module):
    "conv(conv(x) + x) 实现残差块"
    def __init__(self, 
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 norm_num_groups: int = 32,
                 norm_eps: float = 1e-6) -> None:
        super(ResnetBlock, self).__init__()

        # //? 参数
        self.spatial_dims = spatial_dims

        # //? 激活函数
        self.nonlinearity = nn.SiLU()
        # //? 第一个卷积
        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups, 
            num_channels=in_channels, 
            eps=norm_eps, 
            affine=True
            )
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            conv_only=True
        )
        #//? 第2个卷积
        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups, 
            num_channels=out_channels, 
            eps=norm_eps, 
            affine=True
            )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            conv_only=True
        )

        # //? 跳跃连接
        self.skip_connection = nn.Module
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                padding=0,
                conv_only=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1️⃣归一化+激活+卷积
        x1 = self.norm1(x)
        x1 = self.nonlinearity(x1)
        x1 = self.conv1(x1)

        # 2️⃣归一化+激活+卷积
        x2 = self.norm2(x1)
        x2 = self.nonlinearity(x2)
        x2 = self.conv2(x2)

        # 43️⃣跳跃连接
        out = self.skip_connection(x) + x2
        return out


class DiffusionResnetBlock(nn.Module):
    "conv(conv(x) + temb) + x 实现时间步嵌入残差块"
    def __init__(self, 
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 temb_channels: int,
                 norm_num_groups: int = 32,
                 norm_eps: float = 1e-6) -> None:
        super(DiffusionResnetBlock, self).__init__()

        # //? 参数
        self.spatial_dims = spatial_dims

        # //? 激活函数
        self.nonlinearity = nn.SiLU()
        # //? 第一个卷积
        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups, 
            num_channels=in_channels, 
            eps=norm_eps, 
            affine=True
            )
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            conv_only=True
        )
        #//? 第2个卷积
        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups, 
            num_channels=out_channels, 
            eps=norm_eps, 
            affine=True
            )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            strides=1,
            padding=1,
            conv_only=True
        )
        # //? 时间步嵌入
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        # //? 跳跃连接
        self.skip_connection = nn.Module
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                strides=1,
                padding=0,
                conv_only=True
            )

    def forward(self, 
                x: torch.Tensor,
                emb: torch.Tensor) -> torch.Tensor:
        # 1️⃣归一化+激活+卷积
        x1 = self.norm1(x)
        x1 = self.nonlinearity(x1)
        x1 = self.conv1(x1)
        # 2️⃣时间步嵌入
        if self.spatial_dims == 2:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
        elif self.spatial_dims == 3:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
        else:
            raise ValueError(f"Unsupported spatial dimensions -- {self.spatial_dims} in DiffusionResnetBlock")
        
        x1 = x1 + temb

        # 3️⃣归一化+激活+卷积
        x2 = self.norm2(x1)
        x2 = self.nonlinearity(x2)
        x2 = self.conv2(x2)

        # 4️⃣跳跃连接
        out = self.skip_connection(x) + x2
        return out
    
class ConditionalResnetBlock(nn.Module):
    "conv(conc(x) + context) + x 实现条件嵌入"
    def __init__(self, 
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 norm_num_groups: int = 32,
                 norm_eps: float = 1e-6) -> None:
        super(ResnetBlock, self).__init__()

        # //? 参数
        self.spatial_dims = spatial_dims

        # //? 激活函数
        self.nonlinearity = nn.SiLU()
        # //? 第一个卷积
        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups, 
            num_channels=in_channels, 
            eps=norm_eps, 
            affine=True
            )
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_only=True
        )
        #//? 第2个卷积
        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups, 
            num_channels=out_channels, 
            eps=norm_eps, 
            affine=True
            )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_only=True
        )

        # //? 跳跃连接
        self.skip_connection = nn.Module
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_only=True
            )

    def forward(self,
                x:torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        # 1️⃣归一化+激活+卷积
        x1 = self.norm1(x)
        x1 = self.nonlinearity(x1)
        x1 = self.conv1(x1)

        # 2️⃣条件嵌入
        x1 = x1 + context

        # 3️⃣归一化+激活+卷积
        x2 = self.norm2(x1)
        x2 = self.nonlinearity(x2)
        x2 = self.conv2(x2)

        # 4️⃣跳跃连接
        out = self.skip_connection(x) + x2
        return out
    
class ConditionalDiffusionResnetBlock(nn.Module):
    "conv(conv(x) + context + temb)+x 实现条件嵌入"
    def __init__(self, 
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int,
                 temb_channels: int,
                 norm_num_groups: int = 32,
                 norm_eps: float = 1e-6) -> None:
        super(ConditionalDiffusionResnetBlock, self).__init__()

        # //? 参数
        self.spatial_dims = spatial_dims

        # //? 激活函数
        self.nonlinearity = nn.SiLU()
        # //? 第一个卷积
        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups, 
            num_channels=in_channels, 
            eps=norm_eps, 
            affine=True
            )
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_only=True
        )
        #//? 第2个卷积
        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups, 
            num_channels=out_channels, 
            eps=norm_eps, 
            affine=True
            )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_only=True
        )
        # //? 时间步嵌入
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        # //? 跳跃连接
        self.skip_connection = nn.Module
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_only=True
            )

    def forward(self, 
                x: torch.Tensor,
                emb: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        # 1️⃣归一化+激活+卷积
        x1 = self.norm1(x)
        x1 = self.nonlinearity(x1)
        x1 = self.conv1(x1)
        # 2️⃣时间步嵌入
        if self.spatial_dims == 2:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
        elif self.spatial_dims == 3:
            temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
        else:
            raise ValueError(f"Unsupported spatial dimensions -- {self.spatial_dims} in DiffusionResnetBlock")
        
        x1 = x1 + temb + context

        # 3️⃣归一化+激活+卷积
        x2 = self.norm2(x1)
        x2 = self.nonlinearity(x2)
        x2 = self.conv2(x2)

        # 4️⃣跳跃连接
        out = self.skip_connection(x) + x2
        return out