from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleFPN(nn.Module):
    """Small top-down FPN for ResNet C3/C4/C5 features."""

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (128, 256, 512),
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        c3_channels, c4_channels, c5_channels = in_channels

        self.lateral3 = nn.Conv2d(c3_channels, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4_channels, out_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(c5_channels, out_channels, kernel_size=1)

        self.smooth3 = nn.Sequential(
            ConvBNAct(out_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )
        self.smooth4 = nn.Sequential(
            ConvBNAct(out_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )
        self.smooth5 = nn.Sequential(
            ConvBNAct(out_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(
            p5,
            size=c4.shape[-2:],
            mode="nearest",
        )
        p3 = self.lateral3(c3) + F.interpolate(
            p4,
            size=c3.shape[-2:],
            mode="nearest",
        )

        return self.smooth3(p3), self.smooth4(p4), self.smooth5(p5)
