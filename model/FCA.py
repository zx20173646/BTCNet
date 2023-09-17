import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class FCA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(FCA, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, padding=0, bias=True)
        )
        self.conv2x1 = nn.Conv2d(gate_channels, gate_channels, kernel_size=(2, 1), groups=gate_channels) # Group Conv
    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        cat = torch.cat([max_pool, avg_pool], dim=2)
        channel_vector = self.conv2x1(cat)
        channel_weight = self.mlp(channel_vector)
        channel_weight = torch.sigmoid(channel_weight)

        return channel_weight

