""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, is_bilinear=False, is_leaky=False, is_avg=False
    ):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.is_bilinear = is_bilinear
        self.is_leaky = is_leaky
        self.is_avg = is_avg

        self.inc = DoubleConv(in_channels, 4, is_leaky)
        self.down1 = Down(4, 8, is_avg, is_leaky)
        self.down2 = Down(8, 16, is_avg, is_leaky)
        self.down3 = Down(16, 32, is_avg, is_leaky)
        factor = 2 if self.is_bilinear else 1
        self.down4 = Down(32, 64 // factor, is_avg, is_leaky)
        self.up1 = Up(64, 32 // factor, is_bilinear, is_leaky)
        self.up2 = Up(32, 16 // factor, is_bilinear, is_leaky)
        self.up3 = Up(16, 8 // factor, is_bilinear, is_leaky)
        self.up4 = Up(8, 4, is_bilinear, is_leaky)
        self.outc = OutConv(4, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet512(nn.Module):
    def __init__(
        self, in_channels, out_channels, is_bilinear=False, is_leaky=False, is_avg=False
    ):
        super(UNet512, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.is_bilinear = is_bilinear
        self.is_leaky = is_leaky
        self.is_avg = is_avg

        self.inc = DoubleConv(in_channels, 2, is_leaky)
        self.down1 = Down(2, 4, is_avg, is_leaky)
        self.down2 = Down(4, 8, is_avg, is_leaky)
        self.down3 = Down(8, 16, is_avg, is_leaky)
        self.down4 = Down(16, 32, is_avg, is_leaky)
        self.down5 = Down(32, 64, is_avg, is_leaky)
        self.down6 = Down(64, 128, is_avg, is_leaky)
        factor = 2 if self.is_bilinear else 1
        self.down7 = Down(128, 256 // factor, is_avg, is_leaky)
        self.up1 = Up(256, 128 // factor, is_bilinear, is_leaky)
        self.up2 = Up(128, 64 // factor, is_bilinear, is_leaky)
        self.up3 = Up(64, 32 // factor, is_bilinear, is_leaky)
        self.up4 = Up(32, 16 // factor, is_bilinear, is_leaky)
        self.up5 = Up(16, 8 // factor, is_bilinear, is_leaky)
        self.up6 = Up(8, 4 // factor, is_bilinear, is_leaky)
        self.up7 = Up(4, 2, is_bilinear, is_leaky)
        self.outc = OutConv(2, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)

        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)

        logits = self.outc(x)
        return logits


class UNet512m(nn.Module):
    def __init__(
        self, in_channels, out_channels, is_bilinear=False, is_leaky=False, is_avg=False
    ):
        super(UNet512m, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.is_bilinear = is_bilinear
        self.is_leaky = is_leaky
        self.is_avg = is_avg

        self.inc = DoubleConv(in_channels, 4, is_leaky)
        self.down1 = Down(4, 8, is_avg, is_leaky)
        self.down2 = Down(8, 16, is_avg, is_leaky)
        self.down3 = Down(16, 32, is_avg, is_leaky)
        self.down4 = Down(32, 64, is_avg, is_leaky)
        self.down5 = Down(64, 128, is_avg, is_leaky)
        self.down6 = Down(128, 256, is_avg, is_leaky)
        factor = 2 if is_bilinear else 1
        self.down7 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, is_bilinear, is_leaky)
        self.up2 = Up(256, 128 // factor, is_bilinear, is_leaky)
        self.up3 = Up(128, 64 // factor, is_bilinear, is_leaky)
        self.up4 = Up(64, 32 // factor, is_bilinear, is_leaky)
        self.up5 = Up(32, 16 // factor, is_bilinear, is_leaky)
        self.up6 = Up(16, 8 // factor, is_bilinear, is_leaky)
        self.up7 = Up(8, 4, is_bilinear, is_leaky)
        self.outc = OutConv(4, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)

        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)

        logits = self.outc(x)
        return logits


class UNet1024(nn.Module):
    def __init__(
        self, in_channels, out_channels, is_bilinear=False, is_leaky=False, is_avg=False
    ):
        super(UNet1024, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.is_bilinear = is_bilinear
        self.is_leaky = is_leaky
        self.is_avg = is_avg

        self.inc = DoubleConv(in_channels, 2, is_leaky)
        self.down1 = Down(2, 4, is_avg, is_leaky)
        self.down2 = Down(4, 8, is_avg, is_leaky)
        self.down3 = Down(8, 8, is_avg, is_leaky)
        self.down4 = Down(8, 16, is_avg, is_leaky)
        self.down5 = Down(16, 32, is_avg, is_leaky)
        self.down6 = Down(32, 64, is_avg, is_leaky)
        self.down7 = Down(64, 128, is_avg, is_leaky)
        factor = 2 if is_bilinear else 1
        self.down8 = Down(128, 256 // factor, is_avg, is_leaky)
        self.up1 = Up(256, 128 // factor, is_bilinear, is_leaky)
        self.up2 = Up(128, 64 // factor, is_bilinear, is_leaky)
        self.up3 = Up(64, 32 // factor, is_bilinear, is_leaky)
        self.up4 = Up(32, 16 // factor, is_bilinear, is_leaky)
        self.up5 = Up(16, 8 // factor, is_bilinear, is_leaky)
        self.up6 = Up(12, 8 // factor, is_bilinear, is_leaky)
        self.up7 = Up(8, 4 // factor, is_bilinear, is_leaky)
        self.up8 = Up(4, 2, is_bilinear, is_leaky)
        self.outc = OutConv(2, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)

        x = self.up1(x9, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)

        logits = self.outc(x)
        return logits


class UNet1024Avg300k(nn.Module):
    def __init__(
        self, in_channels, out_channels, is_bilinear=False, is_leaky=False, is_avg=False
    ):
        super(UNet1024Avg300k, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.is_bilinear = is_bilinear
        self.is_leaky = is_leaky
        self.is_avg = is_avg

        self.inc = DoubleConv(in_channels, 2, is_leaky)
        self.down1 = Down(2, 4, is_avg, is_leaky)
        self.down2 = Down(4, 8, is_avg, is_leaky)
        self.down3 = Down(8, 8, is_avg, is_leaky)
        self.down4 = Down(8, 8, is_avg, is_leaky)
        self.down5 = Down(8, 16, is_avg, is_leaky)
        self.down6 = Down(16, 32, is_avg, is_leaky)
        self.down7 = Down(32, 64, is_avg, is_leaky)
        factor = 2 if is_bilinear else 1
        self.down8 = Down(64, 128 // factor, is_avg, is_leaky)
        self.up1 = Up(128, 64 // factor, is_bilinear, is_leaky)
        self.up2 = Up(64, 32 // factor, is_bilinear, is_leaky)
        self.up3 = Up(32, 16 // factor, is_bilinear, is_leaky)
        self.up4 = Up(16, 8 // factor, is_bilinear, is_leaky)
        self.up5 = Up(12, 8 // factor, is_bilinear, is_leaky)
        self.up6 = Up(12, 8 // factor, is_bilinear, is_leaky)
        self.up7 = Up(8, 4 // factor, is_bilinear, is_leaky)
        self.up8 = Up(4, 2, is_bilinear, is_leaky)
        self.outc = OutConv(2, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)

        x = self.up1(x9, x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)
        x = self.up8(x, x1)

        logits = self.outc(x)
        return logits


class UNet2048Avg850k(nn.Module):
    def __init__(
        self, in_channels, out_channels, is_bilinear=False, is_leaky=False, is_avg=False
    ):
        super(UNet2048Avg850k, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.is_bilinear = is_bilinear
        self.is_leaky = is_leaky
        self.is_avg = is_avg

        self.inc = DoubleConv(in_channels, 2)
        self.down1 = Down(2, 4, is_avg, is_leaky)
        self.down2 = Down(4, 8, is_avg, is_leaky)
        self.down3 = Down(8, 8, is_avg, is_leaky)
        self.down4 = Down(8, 8, is_avg, is_leaky)
        self.down5 = Down(8, 16, is_avg, is_leaky)
        self.down6 = Down(16, 16, is_avg, is_leaky)
        self.down7 = Down(16, 32, is_avg, is_leaky)
        self.down8 = Down(32, 64, is_avg, is_leaky)
        self.down9 = Down(64, 128, is_avg, is_leaky)
        factor = 2 if is_bilinear else 1
        self.down10 = Down(128, 256 // factor, is_avg)
        self.up1 = Up(256, 128 // factor, is_bilinear, is_leaky)
        self.up2 = Up(128, 64 // factor, is_bilinear, is_leaky)
        self.up3 = Up(64, 32 // factor, is_bilinear, is_leaky)
        self.up4 = Up(32, 16 // factor, is_bilinear, is_leaky)
        self.up5 = Up(24, 16 // factor, is_bilinear, is_leaky)
        self.up6 = Up(16, 8 // factor, is_bilinear, is_leaky)
        self.up7 = Up(12, 8 // factor, is_bilinear, is_leaky)
        self.up8 = Up(12, 8 // factor, is_bilinear, is_leaky)
        self.up9 = Up(8, 4 // factor, is_bilinear, is_leaky)
        self.up10 = Up(4, 2, is_bilinear, is_leaky)
        self.outc = OutConv(2, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x10 = self.down9(x9)
        x11 = self.down10(x10)

        x = self.up1(x11, x10)
        x = self.up2(x, x9)
        x = self.up3(x, x8)
        x = self.up4(x, x7)
        x = self.up5(x, x6)
        x = self.up6(x, x5)
        x = self.up7(x, x4)
        x = self.up8(x, x3)
        x = self.up9(x, x2)
        x = self.up10(x, x1)

        logits = self.outc(x)
        return logits


class UNet2048Avg300k(nn.Module):
    def __init__(
        self, in_channels, out_channels, is_bilinear=False, is_leaky=False, is_avg=False
    ):
        super(UNet2048Avg300k, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.is_bilinear = is_bilinear
        self.is_leaky = is_leaky
        self.is_avg = is_avg

        self.inc = DoubleConv(in_channels, 2)
        self.down1 = Down(2, 4, is_avg, is_leaky)
        self.down2 = Down(4, 8, is_avg, is_leaky)
        self.down3 = Down(8, 8, is_avg, is_leaky)
        self.down4 = Down(8, 8, is_avg, is_leaky)
        self.down5 = Down(8, 16, is_avg, is_leaky)
        self.down6 = Down(16, 16, is_avg, is_leaky)
        self.down7 = Down(16, 32, is_avg, is_leaky)
        self.down8 = Down(32, 32, is_avg, is_leaky)
        self.down9 = Down(32, 64, is_avg, is_leaky)
        factor = 2 if is_bilinear else 1
        self.down10 = Down(64, 128 // factor, is_avg, is_leaky)
        self.up1 = Up(128, 64 // factor, is_bilinear, is_leaky)
        self.up2 = Up(64, 64 // factor, is_bilinear, is_leaky)
        self.up3 = Up(64, 32 // factor, is_bilinear, is_leaky)
        self.up4 = Up(32, 16 // factor, is_bilinear, is_leaky)
        self.up5 = Up(24, 16 // factor, is_bilinear, is_leaky)
        self.up6 = Up(16, 8 // factor, is_bilinear, is_leaky)
        self.up7 = Up(12, 8 // factor, is_bilinear, is_leaky)
        self.up8 = Up(12, 8 // factor, is_bilinear, is_leaky)
        self.up9 = Up(8, 4 // factor, is_bilinear, is_leaky)
        self.up10 = Up(4, 2, is_bilinear, is_leaky)
        self.outc = OutConv(2, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)
        x10 = self.down9(x9)
        x11 = self.down10(x10)

        x = self.up1(x11, x10)
        x = self.up2(x, x9)
        x = self.up3(x, x8)
        x = self.up4(x, x7)
        x = self.up5(x, x6)
        x = self.up6(x, x5)
        x = self.up7(x, x4)
        x = self.up8(x, x3)
        x = self.up9(x, x2)
        x = self.up10(x, x1)

        logits = self.outc(x)
        return logits
