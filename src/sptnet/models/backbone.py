"""3D convolutional backbone used by SPTnet."""

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two-layer 3D residual block used in the convolutional backbone.

    Parameters
    ----------
    in_channels:
        Number of channels in the input feature volume.
    out_channels:
        Number of channels produced by the block.
    stride:
        Kept for compatibility with the original architecture. The current
        convolutions use unit stride; a projection shortcut is used when the
        channel count changes.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=True,
        )
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=True,
        )
        self.shortcut = nn.Sequential()
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                    bias=True,
                )
            )

    def forward(self, x):
        """Apply the residual block to a `[B, C, T, H, W]` tensor."""
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(out)


class BackBone(nn.Module):
    """3D CNN feature extractor for SPTnet video inputs.

    The backbone expects normalized movie tensors shaped `[B, 1, T, H, W]`
    and returns feature volumes with 256 channels. Spatial dimensions are
    downsampled by max-pooling while the temporal dimension is preserved.
    """

    def __init__(self):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv3d(
            1,
            16,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=True,
        )
        self.layer1 = self.make_layer(32, 2, stride=(1, 1, 1))
        self.layer2 = self.make_layer(64, 2, stride=(1, 1, 1))
        self.layer3 = self.make_layer(128, 2, stride=(1, 1, 1))
        self.layer4 = self.make_layer(256, 2, stride=(1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d((30, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool3d((30, 4, 4))

    def make_layer(self, out_channels, num_blocks, stride):
        """Build a sequence of residual blocks and update channel state."""
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Extract spatiotemporal features from a batched movie tensor."""
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.pool1(x)
        x = self.layer4(x)
        return self.pool1(x)
