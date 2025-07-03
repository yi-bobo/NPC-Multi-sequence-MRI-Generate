import torch.nn as nn

def build_conv(dims, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=1):
    if dims == 1:
        conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
    elif dims == 2:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
    elif dims == 3:
        conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
    else:
        raise ValueError('Unsupported dimensions: {}'.format(dims))
    return conv

class ResnetBlock(nn.Module):
    """
    两个卷积层和归一化层，处理后的结果与输入直接相加
    input: x
    output: out = x + conv_block(x)
    """
    def __init__(self, dims, in_channels, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dims, in_channels, norm_layer, activation, use_dropout)

    def build_conv_block(self, dims, in_channels, norm_layer, activation, use_dropout):
        conv_block = []
        p = 1

        conv_block += [build_conv(dims, in_channels, in_channels, kernel_size=3, padding=p),
                       norm_layer(in_channels),
                       activation]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [build_conv(dims, in_channels, in_channels, kernel_size=3, padding=p),
                       norm_layer(in_channels)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out