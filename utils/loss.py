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


def bbox_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            device=boxes1.device,
            dtype=boxes1.dtype,
        )

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-6)


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


def distribution_focal_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    reg_max: int,
) -> torch.Tensor:
    """DFL for l/t/r/b distributions.

    pred_logits: [N, 4, reg_max + 1]
    targets: [N, 4], continuous distances in feature-grid units.
    """
    targets = targets.clamp(min=0.0, max=float(reg_max) - 1e-3)
    left = targets.floor().long()
    right = (left + 1).clamp(max=reg_max)
    weight_right = targets - left.float()
    weight_left = 1.0 - weight_right

    logits = pred_logits.reshape(-1, reg_max + 1)
    left = left.reshape(-1)
    right = right.reshape(-1)
    weight_left = weight_left.reshape(-1)
    weight_right = weight_right.reshape(-1)

    loss_left = F.cross_entropy(logits, left, reduction="none") * weight_left
    loss_right = F.cross_entropy(logits, right, reduction="none") * weight_right
    return (loss_left + loss_right).view(-1, 4).mean(dim=1)


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
        dfl_weight: float = 1.0,
        objectness_weight: float = 1.0,
        class_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        center_radius: int = 1,
        small_object_max_side: float = 96.0,
        medium_object_max_side: float = 224.0,
        tal_topk: int = 10,
        tal_alpha: float = 1.0,
        tal_beta: float = 6.0,
        reg_max: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = tuple(int(stride) for stride in strides)
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        self.objectness_weight = objectness_weight
        self.class_weight = class_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.center_radius = center_radius
        self.small_object_max_side = small_object_max_side
        self.medium_object_max_side = medium_object_max_side
        self.tal_topk = int(tal_topk)
        self.tal_alpha = float(tal_alpha)
        self.tal_beta = float(tal_beta)
        self.reg_max = int(reg_max)
        self.reg_bins = self.reg_max + 1
        self.box_channels = 4 * self.reg_bins
        self.objectness_index = self.box_channels
        self.class_start_index = self.objectness_index + 1
        self.register_buffer(
            "dfl_projection",
            torch.arange(self.reg_bins, dtype=torch.float32),
            persistent=False,
        )

    def choose_scale(self, width: torch.Tensor, height: torch.Tensor) -> int:
        max_side = float(torch.maximum(width, height).item())
        if len(self.strides) >= 4:
            if max_side <= self.small_object_max_side:
                return 0
            if max_side <= self.medium_object_max_side:
                return 1
            if max_side <= self.medium_object_max_side * 2.0:
                return 2
            return min(3, len(self.strides) - 1)
        if max_side <= self.small_object_max_side:
            return 0
        if max_side <= self.medium_object_max_side:
            return min(1, len(self.strides) - 1)
        return len(self.strides) - 1

    @staticmethod
    def make_points(
        height: int,
        width: int,
        stride: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        point_x = (grid_x.float() + 0.5) * float(stride)
        point_y = (grid_y.float() + 0.5) * float(stride)
        return point_x, point_y

    def decode_boxes(self, prediction: torch.Tensor, stride: int) -> torch.Tensor:
        batch_size, _, height, width = prediction.shape
        device = prediction.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        grid_x = grid_x.view(1, height, width).float()
        grid_y = grid_y.view(1, height, width).float()

        box_logits = prediction[:, : self.box_channels].view(
            batch_size,
            4,
            self.reg_bins,
            height,
            width,
        )
        distances = (
            box_logits.softmax(dim=2)
            * self.dfl_projection.view(1, 1, self.reg_bins, 1, 1)
        ).sum(dim=2) * stride

        left = distances[:, 0]
        top = distances[:, 1]
        right = distances[:, 2]
        bottom = distances[:, 3]
        point_x = (grid_x + 0.5) * stride
        point_y = (grid_y + 0.5) * stride
        return torch.stack(
            (
                point_x - left,
                point_y - top,
                point_x + right,
                point_y + bottom,
            ),
            dim=1,
        )

    def encode_ltrb_targets(
        self,
        target_boxes: torch.Tensor,
        positive_mask: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        batch_size, _, height, width = target_boxes.shape
        device = target_boxes.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        point_x = ((grid_x.float() + 0.5) * stride).view(1, height, width).expand(
            batch_size,
            height,
            width,
        )
        point_y = ((grid_y.float() + 0.5) * stride).view(1, height, width).expand(
            batch_size,
            height,
            width,
        )
        boxes = target_boxes.permute(0, 2, 3, 1)[positive_mask]
        points_x = point_x[positive_mask]
        points_y = point_y[positive_mask]
        distances = torch.stack(
            (
                points_x - boxes[:, 0],
                points_y - boxes[:, 1],
                boxes[:, 2] - points_x,
                boxes[:, 3] - points_y,
            ),
            dim=1,
        )
        return (distances / float(stride)).clamp(
            min=0.0,
            max=float(self.reg_max) - 1e-3,
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
        alignment_targets: list[torch.Tensor] = []

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
            alignment_targets.append(torch.zeros(batch_size, height, width, device=device))

        with torch.no_grad():
            decoded_by_scale = [
                self.decode_boxes(prediction, self.strides[scale_index])
                .permute(0, 2, 3, 1)
                .detach()
                for scale_index, prediction in enumerate(predictions)
            ]
            objectness_by_scale = [
                torch.sigmoid(prediction[:, self.objectness_index]).detach()
                for prediction in predictions
            ]
            class_probs_by_scale = [
                torch.softmax(
                    prediction[:, self.class_start_index :],
                    dim=1,
                )
                .permute(0, 2, 3, 1)
                .detach()
                for prediction in predictions
            ]
            points_by_scale = [
                self.make_points(
                    prediction.shape[2],
                    prediction.shape[3],
                    self.strides[scale_index],
                    device,
                )
                for scale_index, prediction in enumerate(predictions)
            ]

            for batch_index, target in enumerate(targets):
                boxes = target["boxes"].to(device=device, dtype=torch.float32)
                labels = target["labels"].to(device=device, dtype=torch.long)
                if boxes.numel() == 0:
                    continue

                cxcywh = xyxy_to_cxcywh(boxes)
                for box_index, (cx, cy, box_w, box_h) in enumerate(cxcywh):
                    box = boxes[box_index]
                    label = labels[box_index]
                    candidate_metrics: list[torch.Tensor] = []
                    candidate_ious: list[torch.Tensor] = []
                    candidate_scales: list[torch.Tensor] = []
                    candidate_ys: list[torch.Tensor] = []
                    candidate_xs: list[torch.Tensor] = []

                    for scale_index, prediction in enumerate(predictions):
                        point_x, point_y = points_by_scale[scale_index]
                        inside_box = (
                            (point_x >= box[0])
                            & (point_x <= box[2])
                            & (point_y >= box[1])
                            & (point_y <= box[3])
                        )
                        ys, xs = torch.where(inside_box)
                        if ys.numel() == 0:
                            continue

                        pred_boxes = decoded_by_scale[scale_index][batch_index, ys, xs]
                        ious = bbox_iou_matrix(pred_boxes, box.view(1, 4)).squeeze(1)
                        class_scores = class_probs_by_scale[scale_index][
                            batch_index,
                            ys,
                            xs,
                            label,
                        ]
                        objectness_scores = objectness_by_scale[scale_index][
                            batch_index,
                            ys,
                            xs,
                        ]
                        scores = (objectness_scores * class_scores).clamp(min=0.0)
                        metrics = scores.pow(self.tal_alpha) * ious.clamp(min=0.0).pow(
                            self.tal_beta
                        )
                        candidate_metrics.append(metrics)
                        candidate_ious.append(ious)
                        candidate_scales.append(
                            torch.full_like(ys, scale_index, dtype=torch.long)
                        )
                        candidate_ys.append(ys)
                        candidate_xs.append(xs)

                    if not candidate_metrics:
                        scale_index = self.choose_scale(box_w, box_h)
                        _, grid_h, grid_w = objectness_targets[scale_index].shape
                        center_x = int(
                            torch.clamp((cx / self.strides[scale_index]).floor(), 0, grid_w - 1)
                            .item()
                        )
                        center_y = int(
                            torch.clamp((cy / self.strides[scale_index]).floor(), 0, grid_h - 1)
                            .item()
                        )
                        candidate_metrics = [torch.ones(1, device=device)]
                        candidate_ious = [torch.full((1,), 0.05, device=device)]
                        candidate_scales = [
                            torch.tensor([scale_index], dtype=torch.long, device=device)
                        ]
                        candidate_ys = [
                            torch.tensor([center_y], dtype=torch.long, device=device)
                        ]
                        candidate_xs = [
                            torch.tensor([center_x], dtype=torch.long, device=device)
                        ]

                    metrics = torch.cat(candidate_metrics)
                    ious = torch.cat(candidate_ious).clamp(min=0.0, max=1.0)
                    scale_indices = torch.cat(candidate_scales)
                    ys = torch.cat(candidate_ys)
                    xs = torch.cat(candidate_xs)

                    if torch.all(metrics <= 0):
                        center_dist = []
                        for scale_index, point_pair in enumerate(points_by_scale):
                            point_x, point_y = point_pair
                            scale_mask = scale_indices == scale_index
                            if not scale_mask.any():
                                continue
                            point_dist = (
                                (point_x[ys[scale_mask], xs[scale_mask]] - cx).pow(2)
                                + (point_y[ys[scale_mask], xs[scale_mask]] - cy).pow(2)
                            ).sqrt()
                            center_dist.append((scale_mask, point_dist))
                        fallback_metrics = torch.zeros_like(metrics)
                        for scale_mask, point_dist in center_dist:
                            fallback_metrics[scale_mask] = 1.0 / (
                                1.0 + point_dist / float(self.strides[0])
                            )
                        metrics = fallback_metrics

                    topk = min(self.tal_topk, metrics.numel())
                    top_values, top_indices = torch.topk(metrics, k=topk, largest=True)
                    top_ious = ious[top_indices]
                    top_scales = scale_indices[top_indices]
                    top_ys = ys[top_indices]
                    top_xs = xs[top_indices]

                    max_metric = top_values.max().clamp(min=1e-6)
                    quality_targets = (top_values / max_metric * top_ious).clamp(
                        min=0.05,
                        max=1.0,
                    )

                    for item_index in range(top_indices.numel()):
                        scale_index = int(top_scales[item_index].item())
                        grid_y = int(top_ys[item_index].item())
                        grid_x = int(top_xs[item_index].item())
                        metric_value = top_values[item_index]
                        current_metric = alignment_targets[scale_index][
                            batch_index,
                            grid_y,
                            grid_x,
                        ]
                        if current_metric > metric_value:
                            continue

                        objectness_targets[scale_index][batch_index, grid_y, grid_x] = (
                            quality_targets[item_index]
                        )
                        box_targets[scale_index][batch_index, :, grid_y, grid_x] = box
                        class_targets[scale_index][batch_index, grid_y, grid_x] = label
                        positive_masks[scale_index][batch_index, grid_y, grid_x] = True
                        alignment_targets[scale_index][batch_index, grid_y, grid_x] = metric_value

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
        dfl_loss_sum = torch.zeros((), device=device)
        class_loss_sum = torch.zeros((), device=device)
        positive_count = torch.zeros((), device=device)

        for scale_index, prediction in enumerate(predictions):
            objectness_logits = prediction[:, self.objectness_index]
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

            box_logits = prediction[:, : self.box_channels].view(
                prediction.shape[0],
                4,
                self.reg_bins,
                prediction.shape[2],
                prediction.shape[3],
            )
            box_logits = box_logits.permute(0, 3, 4, 1, 2)[positive_mask]
            target_ltrb = self.encode_ltrb_targets(
                loss_targets.boxes[scale_index],
                positive_mask,
                self.strides[scale_index],
            )
            dfl_loss_sum = dfl_loss_sum + distribution_focal_loss(
                box_logits,
                target_ltrb,
                self.reg_max,
            ).sum()

            class_logits = prediction[:, self.class_start_index :].permute(0, 2, 3, 1)[positive_mask]
            class_targets = loss_targets.classes[scale_index][positive_mask]
            class_loss_sum = class_loss_sum + F.cross_entropy(
                class_logits,
                class_targets,
                reduction="sum",
            )

        normalizer = positive_count.clamp(min=1.0)
        box_loss = box_loss_sum / normalizer
        dfl_loss = dfl_loss_sum / normalizer
        objectness_loss = objectness_loss_sum / normalizer
        class_loss = class_loss_sum / normalizer
        total_loss = (
            self.box_weight * box_loss
            + self.dfl_weight * dfl_loss
            + self.objectness_weight * objectness_loss
            + self.class_weight * class_loss
        )
        return {
            "loss": total_loss,
            "box_loss": box_loss.detach(),
            "dfl_loss": dfl_loss.detach(),
            "objectness_loss": objectness_loss.detach(),
            "class_loss": class_loss.detach(),
            "num_positive": positive_count.detach(),
        }
