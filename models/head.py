from __future__ import annotations

import math

import torch
from torch import nn

from .neck import ConvBNAct


class DetectionHead(nn.Module):
    """Anchor-free YOLO-style prediction heads for multiple feature scales."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 256,
        num_scales: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 5 + num_classes
        self.heads = nn.ModuleList(
            self._make_head(in_channels) for _ in range(num_scales)
        )
        self._init_prediction_bias()

    def _make_head(self, in_channels: int) -> nn.Sequential:
        return nn.Sequential(
            ConvBNAct(in_channels, in_channels),
            nn.Conv2d(in_channels, self.num_outputs, kernel_size=1),
        )

    def _init_prediction_bias(self) -> None:
        for head in self.heads:
            pred = head[-1]
            if not isinstance(pred, nn.Conv2d) or pred.bias is None:
                continue
            bias = pred.bias.detach()
            bias[4] = math.log(0.01 / 0.99)
            bias[5:] = math.log(0.01 / 0.99)
            pred.bias.data.copy_(bias)

    def forward(self, features: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        return [head(feature) for feature, head in zip(features, self.heads)]
