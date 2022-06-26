# -*- coding=utf-8 -*-
# @Time: 2021/10/11 17:56
# @Author: N
# @Software: PyCharm

# Original code and checkpoints by Hang Zhang
# https://github.com/zhanghang1989/PyTorch-Encoding


import math

import cv2
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .beit import beit_base_patch16_224_8k_vocab
from torch.nn import functional as F

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['resnet18', 'BasicBlock', 'Bottleneck']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet50s-a75c83cf.zip',
    'resnet101': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet101s-03a0f310.zip',
    'resnet152': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet152s-36670e8b.zip'
}


def conv3x3(in_planes, out_planes, stride = 1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, dilation = 1, downsample = None, previous_dilation = 1,
                 norm_layer = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = stride,
                               padding = dilation, dilation = dilation, bias = False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1,
                               padding = previous_dilation, dilation = previous_dilation, bias = False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, dilation = 1,
                 downsample = None, previous_dilation = 1, norm_layer = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size = 3, stride = stride,
            padding = dilation, dilation = dilation, bias = False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size = 1, bias = False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetWthBeit(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." CVPR. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, block, layers, dilated = True,
                 deep_base = True, norm_layer = nn.BatchNorm2d):
        self.inplanes = 128 if deep_base else 64
        super(ResNetWthBeit, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1, bias = False),
                norm_layer(64),
                nn.ReLU(inplace = True),
                nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                norm_layer(64),
                nn.ReLU(inplace = True),
                nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, bias = False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3,
                                   bias = False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer = norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2, norm_layer = norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride = 1,
                                           dilation = 2, norm_layer = norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride = 2,
                                           norm_layer = norm_layer)
        self.layer4 = beit_base_patch16_224_8k_vocab(drop_path_rate = 0.1, num_classes = -1, init_values = 0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride = 1, dilation = 1, norm_layer = None, multi_grid = False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size = 1, stride = stride, bias = False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation = multi_dilations[0],
                                downsample = downsample, previous_dilation = dilation, norm_layer = norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation = 1,
                                downsample = downsample, previous_dilation = dilation, norm_layer = norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation = 2,
                                downsample = downsample, previous_dilation = dilation, norm_layer = norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation = multi_dilations[i],
                                    previous_dilation = dilation, norm_layer = norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation = dilation, previous_dilation = dilation,
                                    norm_layer = norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x.clone()
        x1 = F.interpolate(x1, (224, 224), mode = 'nearest')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ret = []

        x = self.layer1(x)
        ret.append(x)

        x = self.layer2(x)
        ret.append(x)

        x = self.layer3(x)
        ret.append(x)

        x1 = self.layer4(x1, torch.zeros(196), return_all_tokens = True).permute(0, 2, 1).view(-1, 4096, 14, 14)

        ret.append(x1)

        return ret

    def load4beit(self, path):
        state = torch.load(path)
        self.layer4.load_state_dict(state['state_dict'])
        print('pretrained beit has been loaded')


def resnet18(pretrained = False, **kwargs):
    """Constructs a ResNet-18 models.

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNetWthBeit(BasicBlock, [2, 2, 2, 2], deep_base = False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
