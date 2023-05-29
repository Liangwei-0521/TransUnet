import torch
import torch.nn as nn


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """change the channel into the target size"""
        super(double_conv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super(Down, self).__init__()
        """缩小两倍"""
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            """scale_factor 放大的倍数"""
            self.up = nn.Upsample(scale_factor=2)
            self.conv = double_conv(in_channels, out_channels)
        else:
            pass

    def forward(self, x1, x2):
        x_ = self.up(x1)
        x = torch.concat([x_, x2], dim=1)
        return self.conv(x)
