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


class Bottleneck(nn.Module):
    """YOLOv8-style bottleneck used inside C2f blocks."""

    def __init__(
        self,
        channels: int,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3)
        self.conv2 = ConvBNAct(channels, channels, kernel_size=3)
        self.use_shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return x + out if self.use_shortcut else out


class C2f(nn.Module):
    """Compact CSP block used by YOLOv8."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = False,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden_channels * 2, kernel_size=1)
        self.blocks = nn.ModuleList(
            Bottleneck(hidden_channels, shortcut=shortcut)
            for _ in range(num_blocks)
        )
        self.conv2 = ConvBNAct(
            hidden_channels * (2 + num_blocks),
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = list(self.conv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.conv2(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling-fast block from YOLOv8."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int = 5,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
        )
        self.conv2 = ConvBNAct(hidden_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), dim=1))


class YoloV8PAN(nn.Module):
    """YOLOv8-style PAN-FPN neck with C2f fusion blocks."""

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        out_channels: int = 256,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        c3_channels, c4_channels, c5_channels = in_channels

        self.reduce_c5 = ConvBNAct(c5_channels, out_channels, kernel_size=1)
        self.reduce_c4 = ConvBNAct(c4_channels, out_channels, kernel_size=1)
        self.reduce_c3 = ConvBNAct(c3_channels, out_channels, kernel_size=1)

        self.top_p4 = C2f(out_channels * 2, out_channels, num_blocks=num_blocks)
        self.top_p3 = C2f(out_channels * 2, out_channels, num_blocks=num_blocks)
        self.down_p3 = ConvBNAct(out_channels, out_channels, stride=2)
        self.bottom_p4 = C2f(out_channels * 2, out_channels, num_blocks=num_blocks)
        self.down_p4 = ConvBNAct(out_channels, out_channels, stride=2)
        self.bottom_p5 = C2f(out_channels * 2, out_channels, num_blocks=num_blocks)

    @staticmethod
    def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[-2:] == target.shape[-2:]:
            return source
        return F.interpolate(source, size=target.shape[-2:], mode="nearest")

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features
        p3 = self.reduce_c3(c3)
        p4 = self.reduce_c4(c4)
        p5 = self.reduce_c5(c5)

        p4_td = self.top_p4(
            torch.cat((p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")), dim=1)
        )
        p3_out = self.top_p3(
            torch.cat((p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")), dim=1)
        )

        p4_out = self.bottom_p4(
            torch.cat((p4_td, self._resize_like(self.down_p3(p3_out), p4_td)), dim=1)
        )
        p5_out = self.bottom_p5(
            torch.cat((p5, self._resize_like(self.down_p4(p4_out), p5)), dim=1)
        )
        return p3_out, p4_out, p5_out


class WeightedFusion(nn.Module):
    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if len(features) != self.weights.numel():
            raise ValueError(
                f"Expected {self.weights.numel()} features, got {len(features)}."
            )
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        fused = features[0] * weights[0]
        for index in range(1, len(features)):
            fused = fused + features[index] * weights[index]
        return fused


class BiFPNBlock(nn.Module):
    """One 3-level bidirectional feature pyramid block."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.fuse_p4_td = WeightedFusion(2)
        self.fuse_p3_out = WeightedFusion(2)
        self.fuse_p4_out = WeightedFusion(3)
        self.fuse_p5_out = WeightedFusion(2)

        self.p4_td_conv = ConvBNAct(channels, channels)
        self.p3_out_conv = ConvBNAct(channels, channels)
        self.p4_out_conv = ConvBNAct(channels, channels)
        self.p5_out_conv = ConvBNAct(channels, channels)

        self.down_p3 = ConvBNAct(channels, channels, stride=2)
        self.down_p4 = ConvBNAct(channels, channels, stride=2)

    @staticmethod
    def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape[-2:] == target.shape[-2:]:
            return source
        return F.interpolate(source, size=target.shape[-2:], mode="nearest")

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3_in, p4_in, p5_in = features

        p4_td = self.p4_td_conv(
            self.fuse_p4_td(
                p4_in,
                F.interpolate(p5_in, size=p4_in.shape[-2:], mode="nearest"),
            )
        )
        p3_out = self.p3_out_conv(
            self.fuse_p3_out(
                p3_in,
                F.interpolate(p4_td, size=p3_in.shape[-2:], mode="nearest"),
            )
        )

        p3_down = self._resize_like(self.down_p3(p3_out), p4_in)
        p4_out = self.p4_out_conv(
            self.fuse_p4_out(
                p4_in,
                p4_td,
                p3_down,
            )
        )

        p4_down = self._resize_like(self.down_p4(p4_out), p5_in)
        p5_out = self.p5_out_conv(
            self.fuse_p5_out(
                p5_in,
                p4_down,
            )
        )

        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    """3-scale BiFPN neck with learnable weighted feature fusion."""

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (512, 1024, 2048),
        out_channels: int = 256,
        num_repeats: int = 2,
    ) -> None:
        super().__init__()
        c3_channels, c4_channels, c5_channels = in_channels

        self.lateral3 = nn.Conv2d(c3_channels, out_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4_channels, out_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(c5_channels, out_channels, kernel_size=1)

        self.blocks = nn.ModuleList(
            BiFPNBlock(out_channels) for _ in range(num_repeats)
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features

        pyramid = (
            self.lateral3(c3),
            self.lateral4(c4),
            self.lateral5(c5),
        )
        for block in self.blocks:
            pyramid = block(pyramid)
        return pyramid


SimpleFPN = BiFPN
