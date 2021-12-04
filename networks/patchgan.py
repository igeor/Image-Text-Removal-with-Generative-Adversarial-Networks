import torch.nn as nn
import torch 

class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x

class PatchGAN(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128, batchnorm=True)
        self.d3 = DownSampleConv(128, 256, batchnorm=True)
        self.d4 = DownSampleConv(256, 512, strides=1, batchnorm=True)
        self.final = nn.Conv2d(512, 1, kernel_size=2, stride=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)#;print(x.shape)
        x0 = self.d1(x)#;print(x0.shape)
        x1 = self.d2(x0)#;print(x1.shape)
        x2 = self.d3(x1)#;print(x2.shape)
        x3 = self.d4(x2)#;print(x3.shape)
        xn = self.final(x3)#;print(xn.shape)
        return xn