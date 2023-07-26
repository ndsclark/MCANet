import torch.nn as nn
from .mca_module import *

__all__ = ['wide_resnet16_8', 'wide_resnet28_10']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_attention=False):
        super(PreActBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if use_attention:
            self.attention = MCALayer(planes)
        else:
            self.attention = None

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.attention is not None:
            out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class WideResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, widen_factor=8, att_type=False):
        super(WideResNet, self).__init__()

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 16 * widen_factor, layers[0], stride=1, att_type=att_type)
        self.layer2 = self._make_layer(block, 32 * widen_factor, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 64 * widen_factor, layers[2], stride=2, att_type=att_type)

        self.bn = nn.BatchNorm2d(64 * widen_factor * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64 * widen_factor * block.expansion, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_attention=att_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_attention=att_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def wide_resnet16_8(**kwargs):
    """Constructs a wide_resnet16_8 model.
    """
    depth = 16
    assert ((depth - 4) % 6 == 0), 'depth should be 6n+4'
    n = int((depth - 4) / 6)

    kwargs['widen_factor'] = 8
    model = WideResNet(PreActBasicBlock, [n, n, n], **kwargs)
    return model


def wide_resnet28_10(**kwargs):
    """Constructs a wide_resnet28_10 model.
    """
    depth = 28
    assert ((depth - 4) % 6 == 0), 'depth should be 6n+4'
    n = int((depth - 4) / 6)

    kwargs['widen_factor'] = 10
    model = WideResNet(PreActBasicBlock, [n, n, n], **kwargs)
    return model



