from __future__ import annotations

import math

import torch
from torch import nn

from .neck import ConvBNAct


class DecoupledPredictionHead(nn.Module):
    """Separate box/objectness and class towers for one feature scale."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.box_tower = nn.Sequential(
            ConvBNAct(in_channels, in_channels),
            ConvBNAct(in_channels, in_channels),
        )
        self.class_tower = nn.Sequential(
            ConvBNAct(in_channels, in_channels),
            ConvBNAct(in_channels, in_channels),
        )
        self.box_pred = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.objectness_pred = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.class_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        box_feature = self.box_tower(feature)
        class_feature = self.class_tower(feature)
        return torch.cat(
            [
                self.box_pred(box_feature),
                self.objectness_pred(box_feature),
                self.class_pred(class_feature),
            ],
            dim=1,
        )


class DetectionHead(nn.Module):
    """Anchor-free decoupled YOLO-style prediction heads."""

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
            DecoupledPredictionHead(in_channels, num_classes)
            for _ in range(num_scales)
        )
        self._init_prediction_bias()

    def _init_prediction_bias(self) -> None:
        for head in self.heads:
            nn.init.constant_(head.objectness_pred.bias, math.log(0.01 / 0.99))
            nn.init.constant_(head.class_pred.bias, math.log(0.01 / 0.99))

    def forward(self, features: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        return [head(feature) for feature, head in zip(features, self.heads)]
