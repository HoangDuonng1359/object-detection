from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet101_Weights, ResNet50_Weights, resnet101, resnet50

from .neck import C2f, ConvBNAct, SPPF

SUPPORTED_BACKBONES = ("resnet50", "resnet101", "yolov8n", "yolov8s", "yolov8m")


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


def scale_depth(repeats: int, depth_multiplier: float) -> int:
    return max(1, int(round(repeats * depth_multiplier)))


YOLOV8_SCALES = {
    "yolov8n": (0.33, 0.25),
    "yolov8s": (0.33, 0.50),
    "yolov8m": (0.67, 0.75),
}


class ResNetBackbone(nn.Module):
    """ResNet feature extractor returning strides 8, 16, and 32."""

    out_channels = (512, 1024, 2048)
    strides = (8, 16, 32)

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = False,
        freeze_stem: bool = False,
    ) -> None:
        super().__init__()
        name = name.lower()
        if name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
        elif name == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            model = resnet101(weights=weights)
        else:
            choices = ", ".join(SUPPORTED_BACKBONES)
            raise ValueError(f"Unsupported backbone: {name}. Choose one of: {choices}.")

        self.name = name

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
        )
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if freeze_stem:
            for parameter in self.stem.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class YoloV8Backbone(nn.Module):
    """YOLOv8-style CSPDarknet backbone returning P3/P4/P5 features."""

    strides = (8, 16, 32)

    def __init__(
        self,
        name: str = "yolov8s",
        pretrained: bool = False,
        freeze_stem: bool = False,
    ) -> None:
        super().__init__()
        name = name.lower()
        if name not in YOLOV8_SCALES:
            choices = ", ".join(YOLOV8_SCALES)
            raise ValueError(f"Unsupported YOLOv8 backbone: {name}. Choose one of: {choices}.")
        if pretrained:
            raise ValueError(
                f"{name} is implemented from scratch in this project and has no pretrained weights. "
                "Disable --pretrained_backbone or use resnet50/resnet101."
            )

        depth_multiplier, width_multiplier = YOLOV8_SCALES[name]
        channels = [
            make_divisible(channel * width_multiplier)
            for channel in (64, 128, 256, 512, 1024)
        ]
        c1, c2, c3, c4, c5 = channels
        n2 = scale_depth(3, depth_multiplier)
        n3 = scale_depth(6, depth_multiplier)
        n4 = scale_depth(6, depth_multiplier)
        n5 = scale_depth(3, depth_multiplier)

        self.name = name
        self.out_channels = (c3, c4, c5)

        self.stem = ConvBNAct(3, c1, stride=2)
        self.stage2 = nn.Sequential(
            ConvBNAct(c1, c2, stride=2),
            C2f(c2, c2, num_blocks=n2, shortcut=True),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(c2, c3, stride=2),
            C2f(c3, c3, num_blocks=n3, shortcut=True),
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(c3, c4, stride=2),
            C2f(c4, c4, num_blocks=n4, shortcut=True),
        )
        self.stage5 = nn.Sequential(
            ConvBNAct(c4, c5, stride=2),
            C2f(c5, c5, num_blocks=n5, shortcut=True),
            SPPF(c5, c5),
        )

        if freeze_stem:
            for parameter in self.stem.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class ResNet50Backbone(ResNetBackbone):
    """Backward-compatible ResNet-50 backbone alias."""

    def __init__(self, pretrained: bool = False, freeze_stem: bool = False) -> None:
        super().__init__(
            name="resnet50",
            pretrained=pretrained,
            freeze_stem=freeze_stem,
        )
