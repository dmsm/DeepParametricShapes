import string

import torch as th
import torch.nn as nn

from torchvision.models.resnet import conv3x3, conv1x1, BasicBlock


class CurvesModel(nn.Module):
    def __init__(self, n_curves):
        super(CurvesModel, self).__init__()

        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=256, n_z=len(string.ascii_uppercase))
        self.curves = nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_curves*4),
                nn.Sigmoid()
            )
        self.strokes = nn.Sequential(
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_curves),
                nn.Sigmoid()
            )

    def forward(self, x, z=None):
        code = self.resnet18(x, z)
        return { 'curves': self.curves(code), 'strokes': self.strokes(code) }


class ResNet(nn.Module):
    # modification of torchvision.models.resnet.ResNet to support z conditioning and single channel input

    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, n_z=0):
        super(ResNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64+n_z
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1+n_z, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, z=None):
        x = add_z(x, z)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = th.flatten(x, 1)
        x = self.fc(x)

        return x


def add_z(x, z):
    if z is not None:
        z = z[:,:,None,None].expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x = th.cat([x, z], dim=1)
    return x
