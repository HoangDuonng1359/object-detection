from __future__ import annotations

import torch
from torch import nn

from .backbone import ResNet18Backbone
from .head import DetectionHead
from .neck import SimpleFPN


DEFAULT_CLASSES = ("person", "car", "dog", "cat", "chair")

# Anchors are k-means estimates from the provided train annotations after
# letterbox scaling to 416x416, grouped from small to large feature scale.
DEFAULT_ANCHOR_BASE_SIZE = 416
DEFAULT_ANCHORS = (
    ((17.0, 23.0), (43.0, 33.0), (26.0, 59.0)),
    ((47.0, 106.0), (77.0, 67.0), (87.0, 171.0)),
    ((145.0, 119.0), (159.0, 230.0), (297.0, 275.0)),
)
DEFAULT_STRIDES = (8, 16, 32)


def scale_anchors(
    anchors: tuple[tuple[tuple[float, float], ...], ...],
    image_size: int,
    base_size: int = DEFAULT_ANCHOR_BASE_SIZE,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    scale = float(image_size) / float(base_size)
    return tuple(
        tuple((float(width) * scale, float(height) * scale) for width, height in scale_anchors)
        for scale_anchors in anchors
    )


class YoloLite(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        pretrained_backbone: bool = False,
        freeze_backbone_stem: bool = False,
        fpn_channels: int = 256,
        anchors: tuple[tuple[tuple[float, float], ...], ...] = DEFAULT_ANCHORS,
        strides: tuple[int, int, int] = DEFAULT_STRIDES,
    ) -> None:
        super().__init__()
        if len(anchors) != len(strides):
            raise ValueError("anchors and strides must have the same number of scales")

        num_anchors = len(anchors[0])
        if any(len(scale_anchors) != num_anchors for scale_anchors in anchors):
            raise ValueError("all scales must use the same number of anchors")

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = 5 + num_classes
        self.strides = strides

        self.backbone = ResNet18Backbone(
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
            num_anchors=num_anchors,
            num_scales=len(anchors),
        )

        for scale_index, scale_anchors in enumerate(anchors):
            self.register_buffer(
                f"anchors_{scale_index}",
                torch.tensor(scale_anchors, dtype=torch.float32),
                persistent=False,
            )

    @property
    def anchors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            getattr(self, f"anchors_{scale_index}")
            for scale_index in range(len(self.strides))
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
    image_size: int = DEFAULT_ANCHOR_BASE_SIZE,
    anchor_base_size: int = DEFAULT_ANCHOR_BASE_SIZE,
) -> YoloLite:
    return YoloLite(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        freeze_backbone_stem=freeze_backbone_stem,
        anchors=scale_anchors(
            DEFAULT_ANCHORS,
            image_size=image_size,
            base_size=anchor_base_size,
        ),
    )
