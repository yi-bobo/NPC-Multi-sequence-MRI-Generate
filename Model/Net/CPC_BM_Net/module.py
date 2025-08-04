
import torch.nn as nn

def zero_module(module: nn.Module) -> nn.Module:

    for p in module.parameters():
        p.detach().zero_()
    return module