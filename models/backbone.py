from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class ResNet34Backbone(nn.Module):
    """ResNet-34 feature extractor returning strides 8, 16, and 32."""

    out_channels = (128, 256, 512)
    strides = (8, 16, 32)

    def __init__(self, pretrained: bool = False, freeze_stem: bool = False) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5
