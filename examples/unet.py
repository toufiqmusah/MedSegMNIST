import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downsampling: MaxPool2d → DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling: either bilinear or transposed conv → skip-connection → DoubleConv."""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch + out_ch if bilinear else out_ch + out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1×1 convolution to produce the final segmentation map."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """2D U-Net with configurable depth, filter scaling, and upsampling mode.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 1 for grayscale, 3 for RGB).
    n_classes : int
        Number of output segmentation classes.
    base_filters : int
        Number of filters at the first layer; doubled at each downsampling
        level.
    bilinear : bool
        Use bilinear upsampling (``True``) or transposed convolutions.
    depth : int
        Number of down/up sampling blocks.
    """

    def __init__(
        self, in_channels=1, n_classes=2, base_filters=64, bilinear=True, depth=4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        filters = [base_filters * (2**i) for i in range(depth + 1)]

        self.inc = DoubleConv(in_channels, filters[0])
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(filters[i], filters[i + 1]))

        self.bottleneck = DoubleConv(filters[depth], filters[depth] * 2)
        ch = filters[depth] * 2

        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            up_in = ch
            up_out = filters[i]
            self.ups.append(Up(up_in, up_out, bilinear))
            ch = up_out

        self.outc = OutConv(filters[0], n_classes)

    def forward(self, x):
        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[-(i + 2)])
        logits = self.outc(x)
        return logits
