
import torch.nn as nn

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator for 1D, 2D, and 3D data."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, dims=3):
        """Construct a PatchGAN discriminator.
        
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer (default: nn.BatchNorm3d)
            dims            -- the number of dimensions in the input data (1D, 2D, or 3D)
        """
        super(NLayerDiscriminator, self).__init__()

        self.dims = dims
        norm_layer = self._get_norm_layer()

        kw = 3  # Kernel size
        padw = (kw - 1) // 2  # Padding based on kernel size

        # Define the initial conv layer and the subsequent layers
        layers = []

        # Initial convolution layer (input -> first conv)
        layers.append(self._conv_layer(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                self._conv_layer(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult, affine=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # Last conv layer with stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            self._conv_layer(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult, affine=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Final convolution to output 1 channel prediction map
        layers.append(self._conv_layer(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))

        self.model = nn.Sequential(*layers)

    def _get_norm_layer(self):
        """根据dims返回相应的归一化层"""
        if self.dims == 1:
            return nn.BatchNorm1d
        elif self.dims == 2:
            return nn.BatchNorm2d
        elif self.dims == 3:
            return nn.BatchNorm3d
        else:
            raise ValueError("dims must be 1, 2, or 3")

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """根据dims返回相应的卷积层"""
        if self.dims == 1:
            return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        elif self.dims == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        elif self.dims == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            raise ValueError("dims must be 1, 2, or 3")

    def forward(self, input, isDetach):
        """Forward pass through the discriminator."""
        return self.model(input)

