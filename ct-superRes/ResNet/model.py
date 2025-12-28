import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

class SRResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_res_blocks=16, n_feats=64, scale_factor=4):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale CT).
            out_channels: Number of output channels (1 for grayscale CT).
            n_res_blocks: Number of residual blocks.
            n_feats: Number of feature maps.
            scale_factor: Upsampling factor (e.g., 2, 4).
        """
        super(SRResNet, self).__init__()

        # Initial Feature Extraction
        self.conv_input = nn.Conv2d(in_channels, n_feats, kernel_size=9, padding=4, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Residual Blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(n_res_blocks)])

        # Mid-Conv
        self.conv_mid = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False)
        self.bn_mid = nn.BatchNorm2d(n_feats)

        # Upsampling
        # Handle scale factors like 4 (two 2x blocks) or 2 (one 2x block)
        upsample_layers = []
        if scale_factor == 4:
            upsample_layers.append(UpsampleBlock(n_feats, 2))
            upsample_layers.append(UpsampleBlock(n_feats, 2))
        elif scale_factor == 2:
            upsample_layers.append(UpsampleBlock(n_feats, 2))
        else:
            # Generic case if needed, but PixelShuffle expects integer square
            upsample_layers.append(UpsampleBlock(n_feats, scale_factor))
        
        self.upsample = nn.Sequential(*upsample_layers)

        # Output Layer
        self.conv_output = nn.Conv2d(n_feats, out_channels, kernel_size=9, padding=4, bias=False)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.res_blocks(out)
        out = self.bn_mid(self.conv_mid(out))
        out = out + residual
        out = self.upsample(out)
        out = self.conv_output(out)
        return out
