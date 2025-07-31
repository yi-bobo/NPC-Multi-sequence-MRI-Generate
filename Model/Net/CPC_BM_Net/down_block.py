import torch
import torch.nn as nn

from monai.networks.blocks import Convolution

class DownSample(nn.Module):
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
                 padding: list[int] = (1,1,1)) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.down = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding=padding,
            conv_only=True
        )

    def forward(self, x:torch.Tensor):
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels {self.in_channels}")
        else:
            out = self.down(x)
        return out
    
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
                 padding: list[int] = (1,1,1)) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.down = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=stride,
            kernel_size=kernel_size,
            padding=padding,
            conv_only=True
        )

    def forward(self, x:torch.Tensor, emb:torch.Tensor):
        del emb
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels {self.in_channels}")
        else:
            out = self.down(x)
        return out