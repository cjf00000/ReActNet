import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from quantize import BinaryActivation, HardBinaryConv, MultiBitBinaryActivation, MultibitActivation, LSQ, LSQConv

__all__ = ['birealnet18', 'birealnet34']
debug = False
cnt = 0


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, qa='fp', qw='fp'):
        super(BasicBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        if qa == 'b':
            self.binary_activation = BinaryActivation()
        elif qa == 'fp':
            self.binary_activation = lambda x: x
        elif qa[0] == 'q':
            bits = int(qa[1:])
            self.binary_activation = MultiBitBinaryActivation(bits)
        elif qa[0] == 'l':
            bits = int(qa[1:])
            self.binary_activation = LSQ(bits)
        else:
            bits = int(qa[1:])
            self.binary_activation = MultibitActivation(bits)

        if qw == 'b':
            self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        elif qw == 'fp':
            self.binary_conv = conv3x3(inplanes, planes, stride=stride)
        else:
            bits = int(qw[1:])
            self.binary_conv = LSQConv(inplanes, planes, stride=stride, num_bits=bits)

        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        if debug:
            global cnt
            cnt += 1
            data = [out, self.binary_conv.weight, self.binary_conv.bias]
            torch.save(data, 'layers/layer_{}.pth.tar'.format(cnt))

        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out


class BiRealNet(nn.Module):
    def __init__(self, block, layers, num_channels=64, num_classes=1000, zero_init_residual=False,
                 qa='fp', qw='fp'):
        super(BiRealNet, self).__init__()
        self.inplanes = num_channels
        self.qa = qa
        self.qw = qw
        if num_classes == 1000:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
            self.maxpool = lambda x: x

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.layer1 = self._make_layer(block, num_channels, layers[0])
        self.layer2 = self._make_layer(block, num_channels*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_channels*4, layers[2], stride=2)

        if len(layers) == 4:
            self.layer4 = self._make_layer(block, num_channels*8, layers[3], stride=2)
            self.fc = nn.Linear(num_channels*8, num_classes)
        else:
            self.layer4 = lambda x: x
            self.fc = nn.Linear(num_channels*4, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, qa=self.qa, qw=self.qw))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, qa=self.qa, qw=self.qw))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def birealnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def birealnet20(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [6, 6, 6], **kwargs)
    return model


def birealnet32(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [10, 10, 10], **kwargs)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model


def birealnet68(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 46, 6], **kwargs)
    return model


def get_model(arch, num_classes, num_channels, qa='fp', qw='fp'):
    models = {'resnet18': birealnet18,
              'resnet20': birealnet20,
              'resnet32': birealnet32,
              'resnet34': birealnet34,
              'resnet68': birealnet68}
    return models[arch](num_classes=num_classes, num_channels=num_channels, qa=qa, qw=qw)
