#import encoding.nn as nn
#import encoding.functions as F
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools

import sys, os

from utils.pyt_utils import load_model

from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

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

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class RRB(nn.Module):

    def __init__(self, features, out_features=512):
        super(RRB, self).__init__()

        self.unify = nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False)
        self.residual = nn.Sequential(nn.Conv2d(out_features, out_features//4, kernel_size=3, padding=1, dilation=1, bias=False),
                                    InPlaceABNSync(out_features//4),
                                    nn.Conv2d(out_features//4, out_features, kernel_size=3, padding=1, dilation=1, bias=False))
        self.norm = InPlaceABNSync(out_features)

    def forward(self, feats):
        feats = self.unify(feats)
        residual = self.residual(feats)
        feats = self.norm(feats + residual)
        return feats

# class CAB(nn.Module):
#     def __init__(self, features):
#         super(CAB, self).__init__()

#     def forward(self, low_stage, high_stage):
#         h, w = low_stage.size(2), low_stage.size(3)
#         high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
#         high_stage += low_stage
#         return high_stage

class CAB(nn.Module):
    def __init__(self, features):
        super(CAB, self).__init__()

        self.delta_gen = nn.Sequential(
                        nn.Conv2d(features*2, features, kernel_size=1, bias=False),
                        InPlaceABNSync(features),
                        nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
                        )
        self.delta_gen[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        norm = torch.tensor([[[[w, h]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage_up = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)
        concat = torch.cat((low_stage, high_stage_up), 1)
        delta = self.delta_gen(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta)
        high_stage += low_stage
        return high_stage

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, recurrence=1):
        super(RCCAModule, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                                   InPlaceABNSync(out_channels))
        self.cca = PAM_Module(out_channels)
        

    def forward(self, x):
        output = self.conva(x)
        output = self.cca(output)

        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, criterion):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.gap = RCCAModule(2048, 256)
        self.gap = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                        nn.Conv2d(2048, 256, kernel_size=1, bias=False),
                        InPlaceABNSync(256))
        self.RRB4a = RRB(2048, 256)
        self.CAB4 = CAB(256)
        self.RRB4b = RRB(256, 256)
        self.RRB3a = RRB(1024, 256)
        self.CAB3 = CAB(256)
        self.RRB3b = RRB(256, 256)
        self.RRB2a = RRB(512, 256)
        self.CAB2 = CAB(256)
        self.RRB2b = RRB(256, 256)
        self.RRB1a = RRB(256, 256)
        self.CAB1 = CAB(256)
        self.RRB1b = RRB(256, 256)

        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.criterion = criterion

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x1 = self.layer1(x)
        d1 = self.RRB1a(x1)

        x2 = self.layer2(x1)
        d2 = self.RRB2a(x2)
        d2 = self.CAB2(d1, d2)
        d2 = self.RRB2b(d2)

        x3 = self.layer3(x2)
        d3 = self.RRB3a(x3)
        d3 = self.CAB3(d2, d3)
        d3 = self.RRB3b(d3)

        x4 = self.layer4(x3)
        d4 = self.RRB4a(x4)
        d4 = self.CAB4(d3, d4)
        d4 = self.RRB4b(d4)

        d5 = self.gap(x4)
        d5 = self.CAB1(d4, d5)
        d5 = self.RRB1b(d5)

        out = self.head(d5)

        outs = [out]
        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels)
        else:
            return outs

        return [out]

    def init(self, restore_from):
        saved_state_dict = torch.load(restore_from)
        new_params = self.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0]=='fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i] 
        
        self.load_state_dict(new_params)


def Seg_Model(num_classes, criterion=None, pretrained_model=None):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, criterion)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)

    return model

