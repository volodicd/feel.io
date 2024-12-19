# src/models/components/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection for improved gradient flow.
    Implements the standard ResNet basic block architecture.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Initialize residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for convolution
        """
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.init_weights()

    def init_weights(self):
        """Initialize layer weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return F.relu(out)
