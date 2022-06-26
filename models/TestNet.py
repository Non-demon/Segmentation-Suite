# -*- coding=utf-8 -*-
# @Time: 2021/11/27 22:15
# @Author: N
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F
from .comm_modules import Conv


# todo
# tie specified recv field feature to pined channel
# SeNet
# align feature map with various resolutions by digging

class SelfAttn(nn.Module):
    def __init__(self, in_channels, mode = 'hw'):
        super(SelfAttn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kernel_size = (1, 1))
        self.key_conv = Conv(in_channels, in_channels // 8, kernel_size = (1, 1))
        self.value_conv = Conv(in_channels, in_channels, kernel_size = (1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


class SqueezeExciteBlock(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(ResDoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.norm_with_activate = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.conv1(x) + self.double_conv(x)
        return self.norm_with_activate(x)


class ResDoubleConvWithAttnAndSE(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(ResDoubleConvWithAttnAndSE, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
        )
        self.h_attn = SelfAttn(out_channels, mode = 'h')
        self.w_attn = SelfAttn(out_channels, mode = 'w')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.activate_with_se = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            SqueezeExciteBlock(out_channels)
        )

    def forward(self, x):
        x0 = self.conv1(x)
        x = self.double_conv(x)
        x1 = self.h_attn(x)
        x2 = self.w_attn(x)
        return self.activate_with_se(x0 + x1 + x2)

class ResDoubleConvWithSE(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(ResDoubleConvWithSE, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.activate_with_se = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            SqueezeExciteBlock(out_channels)
        )

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.double_conv(x)
        return self.activate_with_se(x0 + x1)


class Inception(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels // 8
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size = 1) if in_channels != out_channels else nn.Identity()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, padding = 1),
        )
        self.double_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, padding = 1),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size = (1, 7), padding = (0, 3)),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size = (7, 1), padding = (3, 0)),
        )
        self.out_transformer = nn.Sequential(
            nn.Conv2d(mid_channels * 3, out_channels, kernel_size = 1),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.batch_norm(x)
        x0 = self.conv1(x)
        x1 = self.conv3(x)
        x2 = self.double_conv3(x)
        x3 = self.conv7(x)
        x = torch.cat([x1, x2, x3], dim = 1)
        x = self.out_transformer(x) + x0
        return x


class Block(ResDoubleConvWithAttnAndSE):
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(Block, self).__init__(in_channels, out_channels, mid_channels)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = Block(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = Block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class TestNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = True):
        super(TestNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = Block(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
