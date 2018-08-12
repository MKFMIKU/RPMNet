import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduce=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduce),
            nn.ReLU(),
            nn.Linear(channel // reduce, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = self.avg_pool(x).view(b, c)
        excitation = self.fc(squeeze).view(b, c, 1, 1)
        out = x * excitation
        return out


class DDBlock(nn.Module):
    def __init__(self, channels, use_se=True):
        super(DDBlock, self).__init__()
        self.use_se = use_se
        self.rate = 2
        self.plane = 1
        if self.use_se:
            self.se = SELayer(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels * 3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(channels * 4, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottle = nn.Conv2d(channels * 5, channels, kernel_size=1, stride=1, bias=False)

        self.up_bottle = nn.Conv2d(channels, self.rate ** 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(self.rate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)

    def forward(self, x, lr):
        output = F.relu(self.conv1(x))
        x = torch.cat([x, output], 1)
        output = F.relu(self.conv2(x))
        x = torch.cat([x, output], 1)
        output = F.relu(self.conv3(x))
        x = torch.cat([x, output], 1)
        output = F.relu(self.conv4(x))
        x = torch.cat([x, output], 1)
        output = self.bottle(x)
        if self.use_se:
            output = self.se(output)
        up = torch.add(lr, output)
        up = self.up_bottle(up)
        up = self.pixel_shuffle(up)
        return output, up


class Net(nn.Module):
    def __init__(self, plane=1, channels=32, rese_blocks=16, use_se=True):
        super(Net, self).__init__()
        self.use_se = use_se
        self.rese_blocks = rese_blocks
        self.conv_input = nn.Conv2d(in_channels=plane, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                    bias=False)

        self.conv_F = nn.ModuleList()
        for i in range(self.rese_blocks):
            self.conv_F.append(DDBlock(channels=channels, use_se=use_se))

        self.up_bottle = nn.Conv2d(rese_blocks, plane, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)

    def forward(self, x):
        output = self.conv_input(x)
        residual = output
        features = []
        for i in range(self.rese_blocks):
            output, up = self.conv_F[i](output, residual)
            features.append(up)
        output = torch.cat(features, 1)

        output = self.up_bottle(output)

        return output
