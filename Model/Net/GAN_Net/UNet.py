
from Model.Net.GAN_Net.Blocks.resnet_block import ResnetBlock, build_conv
import torch.nn as nn
import torch

bias_setting = False

class UNet(nn.Module):
    def __init__(self, dims, input_nc, output_nc, ngf=16, n_downsampling=3, n_blocks=9, norm_layer=nn.InstanceNorm3d,
                 padding_type='zero', resblock_type='smoothdilated', upsample_type='nearest', skip_connection=True):
        assert (n_blocks >= 0)
        super(UNet, self).__init__()

        self.skip_connection = skip_connection
        activation = nn.ReLU(True)
        conv0 = [build_conv(dims, input_nc, ngf, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=bias_setting), norm_layer(ngf),
                 activation]
        self.conv0 = nn.Sequential(*conv0)

        ### downsample
        mult = 1
        conv_down1 = [build_conv(dims, ngf * mult, ngf * mult * 2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                                bias=bias_setting),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down1 = nn.Sequential(*conv_down1)

        mult = 2
        conv_down2 = [build_conv(dims, ngf * mult, ngf * mult * 2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                                bias=bias_setting),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down2 = nn.Sequential(*conv_down2)

        mult = 4
        conv_down3 = [
            build_conv(dims, ngf * mult, ngf * mult * 2, kernel_size=3, stride=(2, 2, 2), padding=1, bias=bias_setting),
            norm_layer(ngf * mult * 2), activation]
        self.conv_down3 = nn.Sequential(*conv_down3)

        mult = 8
        self.resnetBlock1 = ResnetBlock(dims, ngf * mult,
                                          activation=activation,
                                          norm_layer=norm_layer)
        self.resnetBlock2 = ResnetBlock(dims, ngf * mult,
                                          activation=activation,
                                          norm_layer=norm_layer)

        ### upsample
        mult = 8
        convt_up3 = [nn.Upsample(scale_factor=(2, 2, 2), mode=upsample_type),
                     build_conv(dims, ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up3 = nn.Sequential(*convt_up3)

        mult = 4
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv3 = [build_conv(dims, in_channels, ngf * mult, kernel_size=3, stride=1, padding=1, bias=bias_setting),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv3 = nn.Sequential(*decoder_conv3)

        mult = 4
        convt_up2 = [nn.Upsample(scale_factor=(2, 2, 2), mode=upsample_type),
                     build_conv(dims, ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up2 = nn.Sequential(*convt_up2)

        mult = 2
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv2 = [build_conv(dims, in_channels, ngf * mult, kernel_size=5, stride=1, padding=2, bias=bias_setting),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv2 = nn.Sequential(*decoder_conv2)

        mult = 2
        convt_up1 = [nn.Upsample(scale_factor=(2, 2, 2), mode=upsample_type),
                     build_conv(dims, ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up1 = nn.Sequential(*convt_up1)

        if skip_connection:
            in_channels = ngf * 2
        else:
            in_channels = ngf
        decoder_conv1 = [build_conv(dims, in_channels, output_nc, kernel_size=3, stride=1, padding=1, bias=True), nn.Tanh()]
        self.decoder_conv1 = nn.Sequential(*decoder_conv1)

    def forward(self, input):
        x0 = self.conv0(input) # torch.Size([2, 32, 16, 224, 360])
        x1 = self.conv_down1(x0) # torch.Size([2, 64, 8, 112, 180])
        x2 = self.conv_down2(x1) # torch.Size([2, 128, 4, 56, 90])
        x3 = self.conv_down3(x2) # torch.Size([2, 256, 2, 28, 45])

        x3 = self.resnetBlock1(x3)
        x3 = self.resnetBlock2(x3)

        x4 = self.convt_up3(x3)
        if self.skip_connection:
            x4 = torch.cat((x4, x2), dim=1)  # batchsize*channnel*z*x*y
        x4 = self.decoder_conv3(x4)

        x5 = self.convt_up2(x4)
        if self.skip_connection:
            x5 = torch.cat( (x5, x1), dim=1)
        x5 = self.decoder_conv2(x5)

        x6 = self.convt_up1(x5)
        if self.skip_connection:
            x6 = torch.cat((x6, x0), dim=1)
        out = self.decoder_conv1(x6)
        return out
