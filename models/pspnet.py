import torch
from itertools import chain

import torch
import torch.nn.functional as F
from models import resnet
from torch import nn
from utils import initialize_weights,set_trainable


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s, norm_layer) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(torch.nn.Module):
    def __init__(self, num_classes, in_channels=3, backbone='resnet152', pretrained=True, use_aux=True, freeze_bn=False, freeze_backbone=False, mk_conf=False):
        if mk_conf: self.config_dict=locals().copy()
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer)
        m_out_sz = model.fc.in_features
        self.use_aux = use_aux 

        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.master_branch = nn.Sequential(
            _PSPModule(m_out_sz, bin_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(m_out_sz//2, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        output = output[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
                   self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    def get_config_dict(self):
        for v in ['self','mk_conf','__class__','num_classes']:
            try:
                del self.config_dict[v]
                print(v)
            except: pass
        return self.config_dict

if __name__ == '__main__':
    model=PSPNet(32,backbone='resnet18',pretrained=False)
    x=torch.rand(2,3,500,500)
    for name,md in list(model.named_children())[:-1]:
        x=md(x)
        print(name,x.size(),sep='\n')