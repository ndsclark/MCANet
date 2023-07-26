from torch import nn
from .mca_module import *

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_attention=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if use_attention:
            self.attention = MCALayer(planes)
        else:
            self.attention = None

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.attention is not None:
            out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, att_type=False):
        super(ResNet, self).__init__()

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, att_type=att_type)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, att_type=att_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_attention=att_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_attention=att_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def resnet20(**kwargs):
    """Constructs a ResNet-20 model.
    """
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32(**kwargs):
    """Constructs a ResNet-32 model.
    """
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44(**kwargs):
    """Constructs a ResNet-44 model.
    """
    model = ResNet(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56(**kwargs):
    """Constructs a ResNet-56 model.
    """
    model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110(**kwargs):
    """Constructs a ResNet-110 model.
    """
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202(**kwargs):
    """Constructs a ResNet-1202 model.
    """
    model = ResNet(BasicBlock, [200, 200, 200], **kwargs)
    return model
