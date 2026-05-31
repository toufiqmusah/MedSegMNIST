import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """(Conv3d → BN → ReLU) × 2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3D(nn.Module):
    """Downsampling: MaxPool3d → DoubleConv3D."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up3D(nn.Module):
    """Upsampling: trilinear or transposed conv → skip-connection → DoubleConv3D."""

    def __init__(self, in_ch, out_ch, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv3D(
            in_ch + out_ch if trilinear else out_ch + out_ch, out_ch
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
                diff_d // 2,
                diff_d - diff_d // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """1×1×1 convolution to produce the final segmentation map."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net with configurable depth, filter scaling, and upsampling mode.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 1).
    n_classes : int
        Number of output segmentation classes.
    base_filters : int
        Number of filters at the first layer; doubled at each downsampling
        level.  Default ``32`` (lower than the 2D default to keep GPU
        memory manageable).
    trilinear : bool
        Use trilinear upsampling (``True``) or transposed convolutions.
    depth : int
        Number of down/up sampling blocks.
    """

    def __init__(
        self, in_channels=1, n_classes=2, base_filters=32, trilinear=True, depth=4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        filters = [base_filters * (2**i) for i in range(depth + 1)]

        self.inc = DoubleConv3D(in_channels, filters[0])
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down3D(filters[i], filters[i + 1]))

        self.bottleneck = DoubleConv3D(filters[depth], filters[depth] * 2)
        ch = filters[depth] * 2

        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            up_in = ch
            up_out = filters[i]
            self.ups.append(Up3D(up_in, up_out, trilinear))
            ch = up_out

        self.outc = OutConv3D(filters[0], n_classes)

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
