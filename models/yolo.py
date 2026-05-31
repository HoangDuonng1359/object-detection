from __future__ import annotations

import torch
from torch import nn

from .backbone import ResNet34Backbone
from .head import DetectionHead
from .neck import SimpleFPN


DEFAULT_CLASSES = ("person", "car", "dog", "cat", "chair")
DEFAULT_STRIDES = (4, 8, 16, 32)


class YoloLite(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        pretrained_backbone: bool = False,
        freeze_backbone_stem: bool = False,
        fpn_channels: int = 256,
        strides: tuple[int, ...] = DEFAULT_STRIDES,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 5 + num_classes
        self.strides = tuple(int(stride) for stride in strides)

        self.backbone = ResNet34Backbone(
            pretrained=pretrained_backbone,
            freeze_stem=freeze_backbone_stem,
        )
        self.neck = SimpleFPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_channels,
        )
        self.head = DetectionHead(
            num_classes=num_classes,
            in_channels=fpn_channels,
            num_scales=len(self.strides),
        )

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone(images)
        pyramid = self.neck(features)
        return self.head(pyramid)

    def feature_shapes(self, image_size: int = 416) -> list[tuple[int, int]]:
        return [(image_size // stride, image_size // stride) for stride in self.strides]


def build_model(
    num_classes: int = len(DEFAULT_CLASSES),
    pretrained_backbone: bool = False,
    freeze_backbone_stem: bool = False,
    image_size: int = 416,
) -> YoloLite:
    return YoloLite(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        freeze_backbone_stem=freeze_backbone_stem,
    )
