from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(
        (
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            (x2 - x1).clamp(min=1e-6),
            (y2 - y1).clamp(min=1e-6),
        ),
        dim=-1,
    )


def bbox_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=1e-6)

    center1_x = (boxes1[:, 0] + boxes1[:, 2]) * 0.5
    center1_y = (boxes1[:, 1] + boxes1[:, 3]) * 0.5
    center2_x = (boxes2[:, 0] + boxes2[:, 2]) * 0.5
    center2_y = (boxes2[:, 1] + boxes2[:, 3]) * 0.5
    center_distance = (center1_x - center2_x).pow(2) + (center1_y - center2_y).pow(2)

    enclose_x1 = torch.minimum(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.minimum(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.maximum(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.maximum(boxes1[:, 3], boxes2[:, 3])
    enclose_diag = (enclose_x2 - enclose_x1).pow(2) + (
        enclose_y2 - enclose_y1
    ).pow(2)

    width1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=1e-6)
    height1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=1e-6)
    width2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=1e-6)
    height2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=1e-6)
    v = (4.0 / math.pi**2) * (
        torch.atan(width2 / height2) - torch.atan(width1 / height1)
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + 1e-6)

    return iou - center_distance / enclose_diag.clamp(min=1e-6) - alpha * v


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - p_t).pow(gamma) * bce


@dataclass
class LossTargets:
    objectness: list[torch.Tensor]
    boxes: list[torch.Tensor]
    classes: list[torch.Tensor]
    positive_masks: list[torch.Tensor]


class YoloDetectionLoss(nn.Module):
    def __init__(
        self,
        strides: tuple[int, ...],
        num_classes: int = 5,
        box_weight: float = 5.0,
        objectness_weight: float = 1.0,
        class_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        center_radius: int = 1,
        small_object_max_side: float = 64.0,
        medium_object_max_side: float = 160.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = tuple(int(stride) for stride in strides)
        self.box_weight = box_weight
        self.objectness_weight = objectness_weight
        self.class_weight = class_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.center_radius = center_radius
        self.small_object_max_side = small_object_max_side
        self.medium_object_max_side = medium_object_max_side

    def choose_scale(self, width: torch.Tensor, height: torch.Tensor) -> int:
        max_side = float(torch.maximum(width, height).item())
        if max_side <= self.small_object_max_side:
            return 0
        if max_side <= self.medium_object_max_side:
            return min(1, len(self.strides) - 1)
        return len(self.strides) - 1

    def decode_boxes(self, prediction: torch.Tensor, stride: int) -> torch.Tensor:
        _, _, height, width = prediction.shape
        device = prediction.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        grid_x = grid_x.view(1, height, width).float()
        grid_y = grid_y.view(1, height, width).float()

        center_x = (torch.sigmoid(prediction[:, 0]) + grid_x) * stride
        center_y = (torch.sigmoid(prediction[:, 1]) + grid_y) * stride
        box_w = prediction[:, 2].clamp(min=-6.0, max=6.0).exp() * stride
        box_h = prediction[:, 3].clamp(min=-6.0, max=6.0).exp() * stride
        return torch.stack(
            (
                center_x - box_w * 0.5,
                center_y - box_h * 0.5,
                center_x + box_w * 0.5,
                center_y + box_h * 0.5,
            ),
            dim=1,
        )

    def build_targets(
        self,
        predictions: list[torch.Tensor],
        targets: list[dict[str, Any]],
    ) -> LossTargets:
        device = predictions[0].device
        objectness_targets: list[torch.Tensor] = []
        box_targets: list[torch.Tensor] = []
        class_targets: list[torch.Tensor] = []
        positive_masks: list[torch.Tensor] = []
        area_targets: list[torch.Tensor] = []

        for prediction in predictions:
            batch_size, _, height, width = prediction.shape
            objectness_targets.append(torch.zeros(batch_size, height, width, device=device))
            box_targets.append(torch.zeros(batch_size, 4, height, width, device=device))
            class_targets.append(
                torch.full(
                    (batch_size, height, width),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
            )
            positive_masks.append(
                torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)
            )
            area_targets.append(torch.full((batch_size, height, width), float("inf"), device=device))

        for batch_index, target in enumerate(targets):
            boxes = target["boxes"].to(device=device, dtype=torch.float32)
            labels = target["labels"].to(device=device, dtype=torch.long)
            if boxes.numel() == 0:
                continue

            cxcywh = xyxy_to_cxcywh(boxes)
            for box_index, (cx, cy, box_w, box_h) in enumerate(cxcywh):
                scale_index = self.choose_scale(box_w, box_h)
                stride = self.strides[scale_index]
                _, grid_h, grid_w = objectness_targets[scale_index].shape
                center_x = int(torch.clamp((cx / stride).floor(), 0, grid_w - 1).item())
                center_y = int(torch.clamp((cy / stride).floor(), 0, grid_h - 1).item())
                area = box_w * box_h

                for offset_y in range(-self.center_radius, self.center_radius + 1):
                    for offset_x in range(-self.center_radius, self.center_radius + 1):
                        grid_x = center_x + offset_x
                        grid_y = center_y + offset_y
                        if grid_x < 0 or grid_x >= grid_w or grid_y < 0 or grid_y >= grid_h:
                            continue

                        point_x = (grid_x + 0.5) * stride
                        point_y = (grid_y + 0.5) * stride
                        x1, y1, x2, y2 = boxes[box_index]
                        if point_x < x1 or point_x > x2 or point_y < y1 or point_y > y2:
                            continue

                        current_area = area_targets[scale_index][batch_index, grid_y, grid_x]
                        if current_area <= area:
                            continue

                        objectness_targets[scale_index][batch_index, grid_y, grid_x] = 1.0
                        box_targets[scale_index][batch_index, :, grid_y, grid_x] = boxes[box_index]
                        class_targets[scale_index][batch_index, grid_y, grid_x] = labels[box_index]
                        positive_masks[scale_index][batch_index, grid_y, grid_x] = True
                        area_targets[scale_index][batch_index, grid_y, grid_x] = area

        return LossTargets(
            objectness=objectness_targets,
            boxes=box_targets,
            classes=class_targets,
            positive_masks=positive_masks,
        )

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        loss_targets = self.build_targets(predictions, targets)
        device = predictions[0].device
        objectness_loss_sum = torch.zeros((), device=device)
        box_loss_sum = torch.zeros((), device=device)
        class_loss_sum = torch.zeros((), device=device)
        positive_count = torch.zeros((), device=device)

        for scale_index, prediction in enumerate(predictions):
            objectness_logits = prediction[:, 4]
            objectness_targets = loss_targets.objectness[scale_index]
            objectness_loss_sum = objectness_loss_sum + focal_bce_with_logits(
                objectness_logits,
                objectness_targets,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            ).sum()

            positive_mask = loss_targets.positive_masks[scale_index]
            num_positive = positive_mask.sum()
            positive_count = positive_count + num_positive
            if num_positive == 0:
                continue

            decoded_boxes = self.decode_boxes(prediction, self.strides[scale_index])
            predicted_boxes = decoded_boxes.permute(0, 2, 3, 1)[positive_mask]
            target_boxes = loss_targets.boxes[scale_index].permute(0, 2, 3, 1)[positive_mask]
            box_loss_sum = box_loss_sum + (1.0 - bbox_ciou(predicted_boxes, target_boxes)).sum()

            class_logits = prediction[:, 5:].permute(0, 2, 3, 1)[positive_mask]
            class_targets = loss_targets.classes[scale_index][positive_mask]
            class_loss_sum = class_loss_sum + F.cross_entropy(
                class_logits,
                class_targets,
                reduction="sum",
            )

        normalizer = positive_count.clamp(min=1.0)
        box_loss = box_loss_sum / normalizer
        objectness_loss = objectness_loss_sum / normalizer
        class_loss = class_loss_sum / normalizer
        total_loss = (
            self.box_weight * box_loss
            + self.objectness_weight * objectness_loss
            + self.class_weight * class_loss
        )
        return {
            "loss": total_loss,
            "box_loss": box_loss.detach(),
            "objectness_loss": objectness_loss.detach(),
            "class_loss": class_loss.detach(),
            "num_positive": positive_count.detach(),
        }
