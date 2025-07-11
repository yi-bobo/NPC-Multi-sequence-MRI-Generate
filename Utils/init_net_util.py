import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm

def init_weights(net: nn.Module,
                 init_type: str = 'normal',
                 init_gain: float = 0.02) -> None:
    """
    通用网络权重初始化，兼容 Conv（含转置）、Linear、BatchNorm(可能 affine=False)。
    """
    def _init(m: nn.Module):
        # Conv 和 Linear 一律初始化
        if isinstance(m, (_ConvNd, nn.Linear)):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise ValueError(f"Unsupported init_type {init_type}")
            # bias 可能为 None
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0.0)

        # BatchNorm1d/2d/3d（注意可能 affine=False）
        elif isinstance(m, _BatchNorm):
            if getattr(m, 'weight', None) is not None:
                nn.init.normal_(m.weight, 1.0, init_gain)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0.0)

    net.apply(_init)
    print(f"[init_weights] initialized network with {init_type}, gain={init_gain}")

def init_net(net: nn.Module,
             init_type: str = 'normal',
             init_gain: float = 0.02,
             gpu_ids: list[int] | None = None) -> nn.Module:
    """
    如果提供 gpu_ids，则做 DataParallel 并迁移到 cuda。
    然后调用 init_weights。
    """
    if gpu_ids:
        net = nn.DataParallel(net, device_ids=gpu_ids)
        net.to(f'cuda:{gpu_ids[0]}')
    init_weights(net, init_type, init_gain)
    return net
