import torch
from torch import nn
from torch import Tensor
from torchvision.models.vgg import vgg19
from torchvision.models._utils import IntermediateLayerGetter


class Squeeze_Excite(nn.Module):

    def __init__(self, channel, reduction):
        super(Squeeze_Excite, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class conv_block(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.SE = Squeeze_Excite(out_channels, 8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.SE(out)

        return (out)


def output_block():
    Layer = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1)),
                          nn.Sigmoid())
    return Layer


class encoder1(nn.Module):
    def __init__(self):
        super(encoder1, self).__init__()
        self.skip_connnections = []
        self.model = IntermediateLayerGetter(vgg19().features,
                                             {'2': 'conv_block1', '7': 'conv_block2', '16': 'conv_block3',
                                              '25': 'conv_block4', '34': 'conv_block5'})

    def forward(self, x):
        out = self.model(x)

        names = ["conv_block1", "conv_block2", "conv_block3", "conv_block4"]
        for name in names:
            self.skip_connnections.append(out[name])

        output = out['conv_block5']

        # out->[-1, 512, 32, 32] skip_1->[[64, 512, 512], [128, 256, 256], [256, 128, 128], [512, 64, 64]]
        return output, self.skip_connnections


class decoder1(nn.Module):
    def __init__(self):
        super(decoder1, self).__init__()
        output_channels = [256, 128, 64, 32]
        input_channels = [64+512, 512, 256, 128]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_blocks = nn.ModuleList([conv_block(input_channels[no], i, i) for no, i in enumerate(output_channels)])

    def forward(self, x, skip_connections):
        skip_connections.reverse()
        for i, convblock in enumerate(self.conv_blocks):
            x = self.up(x)
            x = torch.cat([x, skip_connections[i]], 1)
            x = convblock(x)
        return x


class encoder2(nn.Module):
    def __init__(self):
        super(encoder2, self).__init__()
        self.num_filters = [32, 64, 128, 256]
        input_channels = [3, 32, 64, 128]
        self.skip_connections = []
        self.conv_blocks = nn.ModuleList(
            [conv_block(input_channels[no], i, i) for no, i in enumerate(self.num_filters)])
        self.max_pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        for convblock in self.conv_blocks:
            x = convblock(x)
            self.skip_connections.append(x)
            x = self.max_pool(x)

        # x->[256,32,32] skip_2->[[32,512,512],[64,256,256],[128,128,128],[256,64,64]]
        return x, self.skip_connections


class decoder2(nn.Module):
    def __init__(self):
        super(decoder2, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        output_channels = [256, 128, 64, 32]
        input_channels = [64+512+256, 256+128+256, 128+64+128, 64+32+64]
        self.conv_blocks = nn.ModuleList([conv_block(input_channels[no], i, i) for no, i in enumerate(output_channels)])

    def forward(self, x, skip_1, skip_2):
        skip_2.reverse()

        for i, convblock in enumerate(self.conv_blocks):
            x = self.up(x)
            x = torch.cat([x, skip_1[i], skip_2[i]], 1)
            x = convblock(x)

        return x


class ASPP(nn.Module):
    def __init__(self, input_channel, output_channel, shape):
        super(ASPP, self).__init__()

        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, padding="same")
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Upsample(shape, mode='bilinear', align_corners=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1, dilation=1, padding="same", bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, dilation=6, padding="same", bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, dilation=12, padding="same", bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, dilation=18, padding="same", bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(output_channel*5, output_channel, kernel_size=(1, 1), dilation=1, padding="same", bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.mean(x)
        x1 = self.conv1(x1)
        x1 = self.relu(self.bn(x1))
        x1 = self.up(x1)

        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        y = torch.cat([x1, x2, x3, x4, x5], 1)
        y = self.conv6(y)

        return y


class DB_UNet(nn.Module):
    def __init__(self):
        super(DB_UNet, self).__init__()
        self.encoder1 = encoder1()
        self.aspp1 = ASPP(512, 64, (32, 32))
        self.decoder1 = decoder1()

        self.out = output_block()

        self.encoder2 = encoder2()
        self.aspp2 = ASPP(256, 64, (32, 32))
        self.decoder2 = decoder2()

    def forward(self, input):
        x = input
        x, skip_1 = self.encoder1(x)
        x = self.aspp1(x)
        x = self.decoder1(x, skip_1)
        out1 = self.out(x)

        x = input * out1
        x, skip_2 = self.encoder2(x)
        x = self.aspp2(x)
        x = self.decoder2(x, skip_1, skip_2)
        out2 = self.out(x)

        return torch.cat([out1, out2], 1)


if __name__ == '__main__':
    model = DB_UNet()
    img = torch.rand((2, 3, 512, 512))
    ans = model(img)