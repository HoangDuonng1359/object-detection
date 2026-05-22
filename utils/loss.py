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


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    half_w = width * 0.5
    half_h = height * 0.5
    return torch.stack(
        (cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        dim=-1,
    )


def box_wh_iou(box_wh: torch.Tensor, anchor_wh: torch.Tensor) -> torch.Tensor:
    box_wh = box_wh[:, None, :]
    anchor_wh = anchor_wh[None, :, :]
    intersection = torch.minimum(box_wh, anchor_wh).prod(dim=-1)
    union = box_wh.prod(dim=-1) + anchor_wh.prod(dim=-1) - intersection
    return intersection / union.clamp(min=1e-6)


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
        anchors: tuple[torch.Tensor | tuple[tuple[float, float], ...], ...],
        strides: tuple[int, ...],
        num_classes: int = 5,
        box_weight: float = 5.0,
        objectness_weight: float = 1.0,
        class_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if len(anchors) != len(strides):
            raise ValueError("anchors and strides must have the same number of scales")

        self.num_classes = num_classes
        self.strides = tuple(int(stride) for stride in strides)
        self.box_weight = box_weight
        self.objectness_weight = objectness_weight
        self.class_weight = class_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        for index, scale_anchors in enumerate(anchors):
            anchor_tensor = torch.as_tensor(scale_anchors, dtype=torch.float32)
            if anchor_tensor.ndim != 2 or anchor_tensor.shape[1] != 2:
                raise ValueError("each anchor scale must have shape [A, 2]")
            self.register_buffer(f"anchors_{index}", anchor_tensor)

    @property
    def anchors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            getattr(self, f"anchors_{index}") for index in range(len(self.strides))
        )

    def decode_boxes(
        self,
        prediction: torch.Tensor,
        anchors: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        _, num_anchors, _, height, width = prediction.shape
        device = prediction.device

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        grid_x = grid_x.view(1, 1, height, width).float()
        grid_y = grid_y.view(1, 1, height, width).float()
        anchors = anchors.to(device).view(1, num_anchors, 2, 1, 1)

        center_x = (torch.sigmoid(prediction[:, :, 0]) + grid_x) * stride
        center_y = (torch.sigmoid(prediction[:, :, 1]) + grid_y) * stride
        box_w = prediction[:, :, 2].clamp(min=-10.0, max=10.0).exp() * anchors[:, :, 0]
        box_h = prediction[:, :, 3].clamp(min=-10.0, max=10.0).exp() * anchors[:, :, 1]

        return torch.stack(
            (
                center_x - box_w * 0.5,
                center_y - box_h * 0.5,
                center_x + box_w * 0.5,
                center_y + box_h * 0.5,
            ),
            dim=2,
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
            batch_size, num_anchors, _, height, width = prediction.shape
            objectness_targets.append(
                torch.zeros(batch_size, num_anchors, height, width, device=device)
            )
            box_targets.append(
                torch.zeros(batch_size, num_anchors, 4, height, width, device=device)
            )
            class_targets.append(
                torch.full(
                    (batch_size, num_anchors, height, width),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
            )
            positive_masks.append(
                torch.zeros(
                    batch_size,
                    num_anchors,
                    height,
                    width,
                    dtype=torch.bool,
                    device=device,
                )
            )
            area_targets.append(
                torch.zeros(batch_size, num_anchors, height, width, device=device)
            )

        anchors = tuple(anchor.to(device) for anchor in self.anchors)
        all_anchors = torch.cat(anchors, dim=0)
        num_anchors = anchors[0].shape[0]

        for batch_index, target in enumerate(targets):
            boxes = target["boxes"].to(device=device, dtype=torch.float32)
            labels = target["labels"].to(device=device, dtype=torch.long)
            if boxes.numel() == 0:
                continue

            cxcywh = xyxy_to_cxcywh(boxes)
            anchor_ious = box_wh_iou(cxcywh[:, 2:4], all_anchors)
            best_anchor_indices = anchor_ious.argmax(dim=1)

            for box_index, best_anchor_index in enumerate(best_anchor_indices):
                scale_index = int(best_anchor_index.item() // num_anchors)
                anchor_index = int(best_anchor_index.item() % num_anchors)
                stride = self.strides[scale_index]
                _, _, grid_h, grid_w = objectness_targets[scale_index].shape

                cx, cy, box_w, box_h = cxcywh[box_index]
                grid_x = int(torch.clamp((cx / stride).floor(), 0, grid_w - 1).item())
                grid_y = int(torch.clamp((cy / stride).floor(), 0, grid_h - 1).item())
                area = box_w * box_h

                current_area = area_targets[scale_index][
                    batch_index,
                    anchor_index,
                    grid_y,
                    grid_x,
                ]
                if positive_masks[scale_index][
                    batch_index,
                    anchor_index,
                    grid_y,
                    grid_x,
                ] and current_area >= area:
                    continue

                objectness_targets[scale_index][
                    batch_index,
                    anchor_index,
                    grid_y,
                    grid_x,
                ] = 1.0
                box_targets[scale_index][
                    batch_index,
                    anchor_index,
                    :,
                    grid_y,
                    grid_x,
                ] = boxes[box_index]
                class_targets[scale_index][
                    batch_index,
                    anchor_index,
                    grid_y,
                    grid_x,
                ] = labels[box_index]
                positive_masks[scale_index][
                    batch_index,
                    anchor_index,
                    grid_y,
                    grid_x,
                ] = True
                area_targets[scale_index][
                    batch_index,
                    anchor_index,
                    grid_y,
                    grid_x,
                ] = area

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
            objectness_logits = prediction[:, :, 4]
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

            decoded_boxes = self.decode_boxes(
                prediction,
                self.anchors[scale_index],
                self.strides[scale_index],
            )
            predicted_boxes = decoded_boxes.permute(0, 1, 3, 4, 2)[positive_mask]
            target_boxes = loss_targets.boxes[scale_index].permute(0, 1, 3, 4, 2)[
                positive_mask
            ]
            box_loss_sum = box_loss_sum + (1.0 - bbox_ciou(predicted_boxes, target_boxes)).sum()

            class_logits = prediction[:, :, 5:].permute(0, 1, 3, 4, 2)[positive_mask]
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
