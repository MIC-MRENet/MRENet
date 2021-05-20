""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels1, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels1, out_channels2, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return  self.single_conv(x)



class SingleDown(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        return self.maxpool(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels1, out_channels2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class SingleUp(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = SingleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x= torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels1, out_channels2, trilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels1, out_channels2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffZ = torch.tensor([x2.size()[4] - x1.size()[4]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels,32, 32)
        self.down1 = Down(32, 64, 64)
        self.down2 = Down(64, 128, 128)
        self.down3 = Down(128, 128, 256)
        self.down4 = Down(256, 256, 384)
        self.up0 = Up(640, 256, 256)
        self.up1 = Up(384, 128, 128, trilinear)
        self.up2 = Up(192, 64, 64, trilinear)
        self.up3 = Up(96, 32, 32, trilinear)

        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda:0')
    # x = torch.rand((1,5,160,160,160),device=device) # [bsize,channels,Height,Width,Depth]
    model = UNet(n_channels=5, n_classes=1)
    model.cuda(device)
    # y = model(x)
    # print(y.shape)
    flops,params = get_model_complexity_info(model,(5,160,160,160),as_strings=True,print_per_layer_stat=True)
    print("%s |flops: %s |param: %s" % ('UNet',flops,params))
