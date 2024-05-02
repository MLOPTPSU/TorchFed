# -*- coding: utf-8 -*-
import math
import torch.nn as nn

from ..model_builder import MODEL

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes, out_channels=out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

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
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes,
            kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_planes)

        self.conv2 = nn.Conv2d(
            in_channels=out_planes, out_channels=out_planes,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = nn.Conv2d(
            in_channels=out_planes, out_channels=out_planes * 4,
            kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
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

        out += residual
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0, std=0.01)
            #     m.bias.data.zero_()

    def _make_block(self, block_fn, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block_fn.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_fn.expansion),
            )

        layers = []
        layers.append(block_fn(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_fn.expansion

        for i in range(1, block_num):
            layers.append(block_fn(self.inplanes, planes))
        return nn.Sequential(*layers)

@MODEL.register_module()
class ResNet_imagenet(ResNetBase):
    def __init__(self, dataset_config, resnet_size):
        super(ResNet_imagenet, self).__init__()
        for p in ['num_classes']:
            if not hasattr(dataset_config, p):
                raise ValueError("'{}' should be specified in the dataset config for a ResNet model!".format(p))

        # define model param.
        model_params = {
            18: {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
            34: {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
            50: {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
            101: {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
            152: {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
        }
        block_fn = model_params[resnet_size]['block']
        block_nums = model_params[resnet_size]['layers']

        # decide the num of classes.
        self.num_classes = dataset_config.num_classes

        # define layers.
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn, planes=64, block_num=block_nums[0])
        self.layer2 = self._make_block(
            block_fn=block_fn, planes=128, block_num=block_nums[1], stride=2)
        self.layer3 = self._make_block(
            block_fn=block_fn, planes=256, block_num=block_nums[2], stride=2)
        self.layer4 = self._make_block(
            block_fn=block_fn, planes=512, block_num=block_nums[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(
            in_features=512 * block_fn.expansion,
            out_features=self.num_classes
        )

        # weight initialization based on layer type.
        self._weight_initialization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@MODEL.register_module()
class ResNet_cifar(ResNetBase):
    def __init__(self, dataset_config, resnet_size):
        super(ResNet_cifar, self).__init__()
        for p in ['num_classes']:
            if not hasattr(dataset_config, p):
                raise ValueError("'{}' should be specified in the dataset config for a ResNet model!".format(p))
        # define model.
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

        # decide the num of classes.
        self.num_classes = dataset_config.num_classes

        # define layers.
        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn, planes=16, block_num=block_nums)
        self.layer2 = self._make_block(
            block_fn=block_fn, planes=32, block_num=block_nums, stride=2)
        self.layer3 = self._make_block(
            block_fn=block_fn, planes=64, block_num=block_nums, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(
            in_features=64 * block_fn.expansion, out_features=self.num_classes)

        # weight initialization based on layer type.
        self._weight_initialization()

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


