import torch
import torch.nn as nn
from monai.networks.blocks import Convolution

class UpSample(nn.Module):
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
                 padding: tuple[int] = (1,1,1)):
        super().__init__()
        self.in_channels = in_channels
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

    def forward(self, x: torch.Tensor):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels ({x.shape[1]}) 不等于定义的 in_channels {self.in_channels}"
            )
        return self.up(x)

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