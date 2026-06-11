from __future__ import annotations

import torch
from torch import nn

from .backbone import ResNetBackbone, YoloV8Backbone
from .head import DetectionHead
from .neck import BiFPN, YoloV8PAN


DEFAULT_CLASSES = ("person", "car", "dog", "cat", "chair")
DEFAULT_STRIDES = (8, 16, 32)


class YoloLite(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        pretrained_backbone: bool = False,
        freeze_backbone_stem: bool = False,
        backbone_name: str = "resnet50",
        neck_name: str = "yolov8_pan",
        fpn_channels: int = 256,
        strides: tuple[int, ...] = DEFAULT_STRIDES,
        reg_max: int = 16,
        image_size: int = 640,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = int(reg_max)
        self.box_channels = 4 * (self.reg_max + 1)
        self.num_outputs = self.box_channels + num_classes
        self.strides = tuple(int(stride) for stride in strides)
        self.image_size = int(image_size)
        self.backbone_name = backbone_name.lower()
        self.neck_name = neck_name.lower()

        if self.backbone_name.startswith("yolov8"):
            self.backbone = YoloV8Backbone(
                name=self.backbone_name,
                pretrained=pretrained_backbone,
                freeze_stem=freeze_backbone_stem,
            )
        else:
            self.backbone = ResNetBackbone(
                name=self.backbone_name,
                pretrained=pretrained_backbone,
                freeze_stem=freeze_backbone_stem,
            )

        if self.neck_name == "yolov8_pan":
            self.neck = YoloV8PAN(
                in_channels=self.backbone.out_channels,
                out_channels=fpn_channels,
                num_blocks=3,
            )
        elif self.neck_name == "bifpn":
            self.neck = BiFPN(
                in_channels=self.backbone.out_channels,
                out_channels=fpn_channels,
            )
        else:
            raise ValueError(
                f"Unsupported neck: {self.neck_name}. Choose yolov8_pan or bifpn."
            )
        self.head = DetectionHead(
            num_classes=num_classes,
            in_channels=fpn_channels,
            num_scales=len(self.strides),
            reg_max=self.reg_max,
            strides=self.strides,
            image_size=self.image_size,
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
    backbone_name: str = "resnet50",
    neck_name: str = "yolov8_pan",
    image_size: int = 416,
    reg_max: int = 16,
) -> YoloLite:
    return YoloLite(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        freeze_backbone_stem=freeze_backbone_stem,
        backbone_name=backbone_name,
        neck_name=neck_name,
        reg_max=reg_max,
        image_size=image_size,
    )
