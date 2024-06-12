""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 64 // factor)
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16 // factor, bilinear)
        self.up3 = Up(16, 8 // factor, bilinear)
        self.up4 = Up(8, 4, bilinear)
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


class UNetAvg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNetAvg, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 4)
        self.down1 = Down(4, 8, avg=True)
        self.down2 = Down(8, 16, avg=True)
        self.down3 = Down(16, 32, avg=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 64 // factor, avg=True)
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16 // factor, bilinear)
        self.up3 = Up(16, 8 // factor, bilinear)
        self.up4 = Up(8, 4, bilinear)
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


class UNetLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNetLeaky, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 4, leaky=True)
        self.down1 = Down(4, 8, leaky=True)
        self.down2 = Down(8, 16, leaky=True)
        self.down3 = Down(16, 32, leaky=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 64 // factor, leaky=True)
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16 // factor, bilinear)
        self.up3 = Up(16, 8 // factor, bilinear)
        self.up4 = Up(8, 4, bilinear)
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


class UNetLeakyAvg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNetLeakyAvg, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 4, leaky=True)
        self.down1 = Down(4, 8, avg=True, leaky=True)
        self.down2 = Down(8, 16, avg=True, leaky=True)
        self.down3 = Down(16, 32, avg=True, leaky=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 64 // factor, avg=True, leaky=True)
        self.up1 = Up(64, 32 // factor, bilinear)
        self.up2 = Up(32, 16 // factor, bilinear)
        self.up3 = Up(16, 8 // factor, bilinear)
        self.up4 = Up(8, 4, bilinear)
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


class NestedUNetTransposed(nn.Module):
    def __init__(
        self, in_channels, out_channels, filters, deep_supervision=False, **kwargs
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, filters[0], filters[0])
        self.conv1_0 = VGGBlock(filters[0], filters[1], filters[1])

        self.up1_0 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)

        self.conv2_0 = VGGBlock(filters[1], filters[2], filters[2])

        self.up2_0 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)

        self.conv3_0 = VGGBlock(filters[2], filters[3], filters[3])
        self.up3_0 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=2, stride=2)
        self.conv4_0 = VGGBlock(filters[3], filters[4], filters[4])
        self.up4_0 = nn.ConvTranspose2d(filters[4], filters[4], kernel_size=2, stride=2)

        self.conv0_1 = VGGBlock(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlock(filters[1] + filters[2], filters[1], filters[1])

        self.up1_1 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)

        self.conv2_1 = VGGBlock(filters[2] + filters[3], filters[2], filters[2])
        self.up2_1 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        self.conv3_1 = VGGBlock(filters[3] + filters[4], filters[3], filters[3])
        self.up3_1 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=2, stride=2)
        self.conv0_2 = VGGBlock(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = VGGBlock(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.up1_2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)
        self.conv2_2 = VGGBlock(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.up2_2 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        self.conv0_3 = VGGBlock(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = VGGBlock(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.up1_3 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)

        self.conv0_4 = VGGBlock(filters[0] * 4 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class NestedUNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, filters, deep_supervision=False, **kwargs
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, filters[0], filters[0])
        self.conv1_0 = VGGBlock(filters[0], filters[1], filters[1])
        self.conv2_0 = VGGBlock(filters[1], filters[2], filters[2])
        self.conv3_0 = VGGBlock(filters[2], filters[3], filters[3])
        self.conv4_0 = VGGBlock(filters[3], filters[4], filters[4])

        self.conv0_1 = VGGBlock(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlock(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = VGGBlock(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = VGGBlock(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = VGGBlock(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = VGGBlock(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = VGGBlock(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = VGGBlock(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = VGGBlock(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = VGGBlock(filters[0] * 4 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class NestedUNetAvg(nn.Module):
    def __init__(
        self, in_channels, out_channels, filters, deep_supervision=False, **kwargs
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, filters[0], filters[0])
        self.conv1_0 = VGGBlock(filters[0], filters[1], filters[1])
        self.conv2_0 = VGGBlock(filters[1], filters[2], filters[2])
        self.conv3_0 = VGGBlock(filters[2], filters[3], filters[3])
        self.conv4_0 = VGGBlock(filters[3], filters[4], filters[4])

        self.conv0_1 = VGGBlock(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlock(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = VGGBlock(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = VGGBlock(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = VGGBlock(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = VGGBlock(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = VGGBlock(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = VGGBlock(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = VGGBlock(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = VGGBlock(filters[0] * 4 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class NestedUNetLeaky(nn.Module):
    def __init__(
        self, in_channels, out_channels, filters, deep_supervision=False, **kwargs
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlockLeaky(in_channels, filters[0], filters[0])
        self.conv1_0 = VGGBlockLeaky(filters[0], filters[1], filters[1])
        self.conv2_0 = VGGBlockLeaky(filters[1], filters[2], filters[2])
        self.conv3_0 = VGGBlockLeaky(filters[2], filters[3], filters[3])
        self.conv4_0 = VGGBlockLeaky(filters[3], filters[4], filters[4])

        self.conv0_1 = VGGBlockLeaky(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlockLeaky(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = VGGBlockLeaky(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = VGGBlockLeaky(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = VGGBlockLeaky(
            filters[0] * 2 + filters[1], filters[0], filters[0]
        )
        self.conv1_2 = VGGBlockLeaky(
            filters[1] * 2 + filters[2], filters[1], filters[1]
        )
        self.conv2_2 = VGGBlockLeaky(
            filters[2] * 2 + filters[3], filters[2], filters[2]
        )

        self.conv0_3 = VGGBlockLeaky(
            filters[0] * 3 + filters[1], filters[0], filters[0]
        )
        self.conv1_3 = VGGBlockLeaky(
            filters[1] * 3 + filters[2], filters[1], filters[1]
        )

        self.conv0_4 = VGGBlockLeaky(
            filters[0] * 4 + filters[1], filters[0], filters[0]
        )

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class NestedUNetLeakyAvg(nn.Module):
    def __init__(
        self, in_channels, out_channels, filters, deep_supervision=False, **kwargs
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlockLeaky(in_channels, filters[0], filters[0])
        self.conv1_0 = VGGBlockLeaky(filters[0], filters[1], filters[1])
        self.conv2_0 = VGGBlockLeaky(filters[1], filters[2], filters[2])
        self.conv3_0 = VGGBlockLeaky(filters[2], filters[3], filters[3])
        self.conv4_0 = VGGBlockLeaky(filters[3], filters[4], filters[4])

        self.conv0_1 = VGGBlockLeaky(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlockLeaky(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = VGGBlockLeaky(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = VGGBlockLeaky(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = VGGBlockLeaky(
            filters[0] * 2 + filters[1], filters[0], filters[0]
        )
        self.conv1_2 = VGGBlockLeaky(
            filters[1] * 2 + filters[2], filters[1], filters[1]
        )
        self.conv2_2 = VGGBlockLeaky(
            filters[2] * 2 + filters[3], filters[2], filters[2]
        )

        self.conv0_3 = VGGBlockLeaky(
            filters[0] * 3 + filters[1], filters[0], filters[0]
        )
        self.conv1_3 = VGGBlockLeaky(
            filters[1] * 3 + filters[2], filters[1], filters[1]
        )

        self.conv0_4 = VGGBlockLeaky(
            filters[0] * 4 + filters[1], filters[0], filters[0]
        )

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class NestedUNet512(nn.Module):
    def __init__(
        self, in_channels, out_channels, filters, deep_supervision=False, **kwargs
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, filters[0], filters[0])
        self.conv1_0 = VGGBlock(filters[0], filters[1], filters[1])
        self.conv2_0 = VGGBlock(filters[1], filters[2], filters[2])
        self.conv3_0 = VGGBlock(filters[2], filters[3], filters[3])
        self.conv4_0 = VGGBlock(filters[3], filters[4], filters[4])
        self.conv5_0 = VGGBlock(filters[4], filters[5], filters[5])
        self.conv6_0 = VGGBlock(filters[5], filters[6], filters[6])

        self.conv0_1 = VGGBlock(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlock(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = VGGBlock(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = VGGBlock(filters[3] + filters[4], filters[3], filters[3])
        self.conv4_1 = VGGBlock(filters[4] + filters[5], filters[4], filters[4])
        self.conv5_1 = VGGBlock(filters[5] + filters[6], filters[5], filters[5])

        self.conv0_2 = VGGBlock(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = VGGBlock(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = VGGBlock(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.conv3_2 = VGGBlock(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.conv4_2 = VGGBlock(filters[4] * 2 + filters[5], filters[4], filters[4])

        self.conv0_3 = VGGBlock(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = VGGBlock(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.conv2_3 = VGGBlock(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.conv3_3 = VGGBlock(filters[3] * 3 + filters[4], filters[3], filters[3])

        self.conv0_4 = VGGBlock(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_4 = VGGBlock(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.conv2_4 = VGGBlock(filters[2] * 4 + filters[3], filters[2], filters[2])

        self.conv0_5 = VGGBlock(filters[0] * 5 + filters[1], filters[0], filters[0])
        self.conv1_5 = VGGBlock(filters[1] * 5 + filters[2], filters[1], filters[1])

        self.conv0_6 = VGGBlock(filters[0] * 6 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        x6_0 = self.conv6_0(self.pool(x5_0))
        x5_1 = self.conv5_1(torch.cat([x5_0, self.up(x6_0)], 1))
        x4_2 = self.conv4_2(torch.cat([x4_0, x4_1, self.up(x5_1)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_0, x3_1, x3_2, self.up(x4_2)], 1))
        x2_4 = self.conv2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self.up(x3_3)], 1))
        x1_5 = self.conv1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self.up(x2_4)], 1))
        x0_6 = self.conv0_6(
            torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self.up(x1_5)], 1)
        )

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_6)
            return output


class UNet512(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet512, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 2)
        self.down1 = Down(2, 4)
        self.down2 = Down(4, 8)
        self.down3 = Down(8, 16)
        self.down4 = Down(16, 32)
        self.down5 = Down(32, 64)
        self.down6 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down7 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16 // factor, bilinear)
        self.up5 = Up(16, 8 // factor, bilinear)
        self.up6 = Up(8, 4 // factor, bilinear)
        self.up7 = Up(4, 2, bilinear)
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

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)

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
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet512m, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)
        self.down5 = Down(64, 128)
        self.down6 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down7 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32 // factor, bilinear)
        self.up5 = Up(32, 16 // factor, bilinear)
        self.up6 = Up(16, 8 // factor, bilinear)
        self.up7 = Up(8, 4, bilinear)
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

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)

        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)

        logits = self.outc(x)
        return logits


class UNetAvg512(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNetAvg512, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 4)
        self.down1 = Down(4, 8, avg=True)
        self.down2 = Down(8, 16, avg=True)
        self.down3 = Down(16, 32, avg=True)
        self.down4 = Down(32, 64, avg=True)
        self.down5 = Down(64, 128, avg=True)
        self.down6 = Down(128, 256, avg=True)
        factor = 2 if bilinear else 1
        self.down7 = Down(256, 512 // factor, avg=True)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32 // factor, bilinear)
        self.up5 = Up(32, 16 // factor, bilinear)
        self.up6 = Up(16, 8 // factor, bilinear)
        self.up7 = Up(8, 4, bilinear)
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


class NestedUNet512Avg(nn.Module):
    def __init__(
        self, in_channels, out_channels, filters, deep_supervision=False, **kwargs
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, filters[0], filters[0])
        self.conv1_0 = VGGBlock(filters[0], filters[1], filters[1])
        self.conv2_0 = VGGBlock(filters[1], filters[2], filters[2])
        self.conv3_0 = VGGBlock(filters[2], filters[3], filters[3])
        self.conv4_0 = VGGBlock(filters[3], filters[4], filters[4])
        self.conv5_0 = VGGBlock(filters[4], filters[5], filters[5])
        self.conv6_0 = VGGBlock(filters[5], filters[6], filters[6])

        self.conv0_1 = VGGBlock(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlock(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = VGGBlock(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = VGGBlock(filters[3] + filters[4], filters[3], filters[3])
        self.conv4_1 = VGGBlock(filters[4] + filters[5], filters[4], filters[4])
        self.conv5_1 = VGGBlock(filters[5] + filters[6], filters[5], filters[5])

        self.conv0_2 = VGGBlock(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = VGGBlock(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = VGGBlock(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.conv3_2 = VGGBlock(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.conv4_2 = VGGBlock(filters[4] * 2 + filters[5], filters[4], filters[4])

        self.conv0_3 = VGGBlock(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = VGGBlock(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.conv2_3 = VGGBlock(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.conv3_3 = VGGBlock(filters[3] * 3 + filters[4], filters[3], filters[3])

        self.conv0_4 = VGGBlock(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_4 = VGGBlock(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.conv2_4 = VGGBlock(filters[2] * 4 + filters[3], filters[2], filters[2])

        self.conv0_5 = VGGBlock(filters[0] * 5 + filters[1], filters[0], filters[0])
        self.conv1_5 = VGGBlock(filters[1] * 5 + filters[2], filters[1], filters[1])

        self.conv0_6 = VGGBlock(filters[0] * 6 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        x6_0 = self.conv6_0(self.pool(x5_0))
        x5_1 = self.conv5_1(torch.cat([x5_0, self.up(x6_0)], 1))
        x4_2 = self.conv4_2(torch.cat([x4_0, x4_1, self.up(x5_1)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_0, x3_1, x3_2, self.up(x4_2)], 1))
        x2_4 = self.conv2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self.up(x3_3)], 1))
        x1_5 = self.conv1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self.up(x2_4)], 1))
        x0_6 = self.conv0_6(
            torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self.up(x1_5)], 1)
        )

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_6)
            return output


class UNet1024Avg(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet1024Avg, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 2)
        self.down1 = Down(2, 4, avg=True)
        self.down2 = Down(4, 8, avg=True)
        self.down3 = Down(8, 8, avg=True)
        self.down4 = Down(8, 16, avg=True)
        self.down5 = Down(16, 32, avg=True)
        self.down6 = Down(32, 64, avg=True)
        self.down7 = Down(64, 128, avg=True)
        factor = 2 if bilinear else 1
        self.down8 = Down(128, 256 // factor, avg=True)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16 // factor, bilinear)
        self.up5 = Up(16, 8 // factor, bilinear)
        self.up6 = Up(12, 8 // factor, bilinear)
        self.up7 = Up(8, 4 // factor, bilinear)
        self.up8 = Up(4, 2, bilinear)
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


class NestedUNet1024Avg(nn.Module):
    def __init__(
        self, in_channels, out_channels, filters, deep_supervision=False, **kwargs
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, filters[0], filters[0])
        self.conv1_0 = VGGBlock(filters[0], filters[1], filters[1])
        self.conv2_0 = VGGBlock(filters[1], filters[2], filters[2])
        self.conv3_0 = VGGBlock(filters[2], filters[3], filters[3])
        self.conv4_0 = VGGBlock(filters[3], filters[4], filters[4])
        self.conv5_0 = VGGBlock(filters[4], filters[5], filters[5])
        self.conv6_0 = VGGBlock(filters[5], filters[6], filters[6])

        self.conv7_0 = VGGBlock(filters[6], filters[7], filters[7])

        self.conv0_1 = VGGBlock(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlock(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = VGGBlock(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = VGGBlock(filters[3] + filters[4], filters[3], filters[3])
        self.conv4_1 = VGGBlock(filters[4] + filters[5], filters[4], filters[4])
        self.conv5_1 = VGGBlock(filters[5] + filters[6], filters[5], filters[5])
        self.conv6_1 = VGGBlock(filters[6] + filters[7], filters[6], filters[6])

        self.conv0_2 = VGGBlock(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = VGGBlock(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = VGGBlock(filters[2] * 2 + filters[3], filters[2], filters[2])
        self.conv3_2 = VGGBlock(filters[3] * 2 + filters[4], filters[3], filters[3])
        self.conv4_2 = VGGBlock(filters[4] * 2 + filters[5], filters[4], filters[4])
        self.conv5_2 = VGGBlock(filters[5] * 2 + filters[6], filters[5], filters[5])

        self.conv0_3 = VGGBlock(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = VGGBlock(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.conv2_3 = VGGBlock(filters[2] * 3 + filters[3], filters[2], filters[2])
        self.conv3_3 = VGGBlock(filters[3] * 3 + filters[4], filters[3], filters[3])
        self.conv4_3 = VGGBlock(filters[4] * 3 + filters[5], filters[4], filters[4])

        self.conv0_4 = VGGBlock(filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_4 = VGGBlock(filters[1] * 4 + filters[2], filters[1], filters[1])
        self.conv2_4 = VGGBlock(filters[2] * 4 + filters[3], filters[2], filters[2])
        self.conv3_4 = VGGBlock(filters[3] * 4 + filters[4], filters[3], filters[3])

        self.conv0_5 = VGGBlock(filters[0] * 5 + filters[1], filters[0], filters[0])
        self.conv1_5 = VGGBlock(filters[1] * 5 + filters[2], filters[1], filters[1])
        self.conv2_5 = VGGBlock(filters[2] * 5 + filters[3], filters[2], filters[2])

        self.conv0_6 = VGGBlock(filters[0] * 6 + filters[1], filters[0], filters[0])
        self.conv1_6 = VGGBlock(filters[1] * 6 + filters[2], filters[1], filters[1])

        self.conv0_7 = VGGBlock(filters[0] * 7 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        x6_0 = self.conv6_0(self.pool(x5_0))
        x5_1 = self.conv5_1(torch.cat([x5_0, self.up(x6_0)], 1))
        x4_2 = self.conv4_2(torch.cat([x4_0, x4_1, self.up(x5_1)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_0, x3_1, x3_2, self.up(x4_2)], 1))
        x2_4 = self.conv2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self.up(x3_3)], 1))
        x1_5 = self.conv1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self.up(x2_4)], 1))
        x0_6 = self.conv0_6(
            torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self.up(x1_5)], 1)
        )

        x7_0 = self.conv7_0(self.pool(x6_0))
        x6_1 = self.conv6_1(torch.cat([x6_0, self.up(x7_0)], 1))
        x5_2 = self.conv5_2(torch.cat([x5_0, x5_1, self.up(x6_1)], 1))
        x4_3 = self.conv4_3(torch.cat([x4_0, x4_1, x4_2, self.up(x5_2)], 1))
        x3_4 = self.conv3_4(torch.cat([x3_0, x3_1, x3_2, x3_3, self.up(x4_3)], 1))
        x2_5 = self.conv2_5(torch.cat([x2_0, x2_1, x2_2, x2_3, x2_4, self.up(x3_4)], 1))
        x1_6 = self.conv1_6(
            torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, x1_5, self.up(x2_5)], 1)
        )
        x0_7 = self.conv0_7(
            torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, self.up(x1_6)], 1)
        )

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_7)
            return output


class UNet1024Avg300k(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet1024Avg300k, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 2)
        self.down1 = Down(2, 4, avg=True)
        self.down2 = Down(4, 8, avg=True)
        self.down3 = Down(8, 8, avg=True)
        self.down4 = Down(8, 8, avg=True)
        self.down5 = Down(8, 16, avg=True)
        self.down6 = Down(16, 32, avg=True)
        self.down7 = Down(32, 64, avg=True)
        factor = 2 if bilinear else 1
        self.down8 = Down(64, 128 // factor, avg=True)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8 // factor, bilinear)
        self.up5 = Up(12, 8 // factor, bilinear)
        self.up6 = Up(12, 8 // factor, bilinear)
        self.up7 = Up(8, 4 // factor, bilinear)
        self.up8 = Up(4, 2, bilinear)
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
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet2048Avg850k, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 2)
        self.down1 = Down(2, 4, avg=True)
        self.down2 = Down(4, 8, avg=True)
        self.down3 = Down(8, 8, avg=True)
        self.down4 = Down(8, 8, avg=True)
        self.down5 = Down(8, 16, avg=True)
        self.down6 = Down(16, 16, avg=True)
        self.down7 = Down(16, 32, avg=True)
        self.down8 = Down(32, 64, avg=True)
        self.down9 = Down(64, 128, avg=True)
        factor = 2 if bilinear else 1
        self.down10 = Down(128, 256 // factor, avg=True)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16 // factor, bilinear)
        self.up5 = Up(24, 16 // factor, bilinear)
        self.up6 = Up(16, 8 // factor, bilinear)
        self.up7 = Up(12, 8 // factor, bilinear)
        self.up8 = Up(12, 8 // factor, bilinear)
        self.up9 = Up(8, 4 // factor, bilinear)
        self.up10 = Up(4, 2, bilinear)
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
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet2048Avg300k, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 2)
        self.down1 = Down(2, 4, avg=True)
        self.down2 = Down(4, 8, avg=True)
        self.down3 = Down(8, 8, avg=True)
        self.down4 = Down(8, 8, avg=True)
        self.down5 = Down(8, 16, avg=True)
        self.down6 = Down(16, 16, avg=True)
        self.down7 = Down(16, 32, avg=True)
        self.down8 = Down(32, 32, avg=True)
        self.down9 = Down(32, 64, avg=True)
        factor = 2 if bilinear else 1
        self.down10 = Down(64, 128 // factor, avg=True)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16 // factor, bilinear)
        self.up5 = Up(24, 16 // factor, bilinear)
        self.up6 = Up(16, 8 // factor, bilinear)
        self.up7 = Up(12, 8 // factor, bilinear)
        self.up8 = Up(12, 8 // factor, bilinear)
        self.up9 = Up(8, 4 // factor, bilinear)
        self.up10 = Up(4, 2, bilinear)
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
