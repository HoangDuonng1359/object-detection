from __future__ import annotations

import argparse
import csv
import gc
import math
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from models.backbone import SUPPORTED_BACKBONES
from models.yolo import DEFAULT_CLASSES, build_model
from utils.dataset import make_dataloader
from utils.loss import YoloDetectionLoss
from utils.thresholds import make_class_threshold_tensor, parse_class_thresholds


SUPPORTED_NECKS = ("yolov8_pan", "bifpn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the YOLO-lite detector.")
    parser.add_argument("--train_data", required=True, type=Path)
    parser.add_argument("--val_data", required=True, type=Path)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--val_image_dir", required=True, type=Path)
    parser.add_argument("--checkpoint_dir", required=True, type=Path)

    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--reg_max", type=int, default=16)
    parser.add_argument(
        "--small_object_max_side",
        type=float,
        default=96.0,
        help="Max resized box side assigned to the finest detection scale.",
    )
    parser.add_argument(
        "--medium_object_max_side",
        type=float,
        default=224.0,
        help="Max resized box side assigned to the middle detection scale.",
    )
    parser.add_argument(
        "--tal_topk",
        type=int,
        default=5,
        help="Top-k candidates per ground-truth box for task-aligned assignment.",
    )
    parser.add_argument(
        "--tal_alpha",
        type=float,
        default=0.5,
        help="Classification-score exponent for task-aligned assignment.",
    )
    parser.add_argument(
        "--tal_beta",
        type=float,
        default=4.0,
        help="IoU exponent for task-aligned assignment.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--oversample_classes",
        nargs="*",
        default=[],
        help="Class names whose images should be sampled more often during training.",
    )
    parser.add_argument(
        "--oversample_factor",
        type=float,
        default=1.0,
        help="Sampling weight for images containing oversample_classes. Use 1.0 to disable.",
    )

    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--freeze_backbone_stem", action="store_true")
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=SUPPORTED_BACKBONES,
        help="Feature extractor backbone.",
    )
    parser.add_argument(
        "--neck",
        default="yolov8_pan",
        choices=SUPPORTED_NECKS,
        help="Feature pyramid neck.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable tqdm progress bars and print only one summary line per epoch.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Alias for --disable_tqdm.",
    )

    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument(
        "--class_thresholds",
        default="",
        help='Comma-separated per-class thresholds for validation, e.g. "chair=0.15,car=0.20".',
    )
    parser.add_argument("--nms_threshold", type=float, default=0.45)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=0,
        help="Use 0 for full validation; useful for quick smoke tests.",
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="Use 0 for full training; useful for quick smoke tests.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_targets_to_device(
    targets: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    moved = []
    for target in targets:
        item = {}
        for key, value in target.items():
            item[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        moved.append(item)
    return moved


def bbox_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_box = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    union = area_box + area_boxes - intersection
    return intersection / union.clamp(min=1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break
        ious = bbox_iou(boxes[index], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep)


@torch.no_grad()
def decode_batch_predictions(
    predictions: list[torch.Tensor],
    criterion: YoloDetectionLoss,
    image_size: int,
    conf_threshold: float,
    class_thresholds: torch.Tensor | None,
    nms_threshold: float,
    max_detections: int,
) -> list[dict[str, torch.Tensor]]:
    batch_size = predictions[0].shape[0]
    batch_boxes = [[] for _ in range(batch_size)]
    batch_scores = [[] for _ in range(batch_size)]
    batch_labels = [[] for _ in range(batch_size)]

    for scale_index, prediction in enumerate(predictions):
        decoded_boxes = criterion.decode_boxes(
            prediction,
            criterion.strides[scale_index],
        )
        boxes = decoded_boxes.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        class_probs = torch.sigmoid(prediction[:, criterion.class_start_index :])
        scores, class_labels = class_probs.permute(0, 2, 3, 1).reshape(
            batch_size,
            -1,
            criterion.num_classes,
        ).max(dim=-1)
        thresholds = (
            class_thresholds[class_labels]
            if class_thresholds is not None
            else conf_threshold
        )

        boxes[..., 0::2].clamp_(0, float(image_size))
        boxes[..., 1::2].clamp_(0, float(image_size))
        valid = (
            (scores >= thresholds)
            & (boxes[..., 2] > boxes[..., 0])
            & (boxes[..., 3] > boxes[..., 1])
        )

        for batch_index in range(batch_size):
            mask = valid[batch_index]
            if not mask.any():
                continue
            batch_boxes[batch_index].append(boxes[batch_index][mask])
            batch_scores[batch_index].append(scores[batch_index][mask])
            batch_labels[batch_index].append(class_labels[batch_index][mask])

    results = []
    for boxes_parts, scores_parts, labels_parts in zip(
        batch_boxes,
        batch_scores,
        batch_labels,
    ):
        if not boxes_parts:
            results.append(
                {
                    "boxes": torch.empty((0, 4), device=predictions[0].device),
                    "scores": torch.empty((0,), device=predictions[0].device),
                    "labels": torch.empty((0,), dtype=torch.long, device=predictions[0].device),
                }
            )
            continue

        boxes = torch.cat(boxes_parts, dim=0)
        scores = torch.cat(scores_parts, dim=0)
        labels = torch.cat(labels_parts, dim=0)
        keep_all = []
        for label in labels.unique():
            label_indices = torch.where(labels == label)[0]
            kept = nms(boxes[label_indices], scores[label_indices], nms_threshold)
            keep_all.append(label_indices[kept])

        keep = torch.cat(keep_all) if keep_all else torch.empty(0, dtype=torch.long)
        keep = keep[scores[keep].argsort(descending=True)[:max_detections]]
        results.append(
            {
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            }
        )
    return results


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def compute_map_50(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    aps = []
    for class_index in range(num_classes):
        gt_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        num_gt = 0
        for gt in ground_truths:
            boxes = gt["boxes"]
            labels = gt["labels"]
            class_mask = labels == class_index
            for box in boxes[class_mask]:
                gt_by_image[gt["image_id"]].append({"bbox": box, "matched": False})
                num_gt += 1

        if num_gt == 0:
            continue

        class_predictions = []
        for pred in predictions:
            labels = pred["labels"]
            class_mask = labels == class_index
            for box, score in zip(pred["boxes"][class_mask], pred["scores"][class_mask]):
                class_predictions.append(
                    {
                        "image_id": pred["image_id"],
                        "bbox": box,
                        "score": float(score),
                    }
                )
        class_predictions.sort(key=lambda item: item["score"], reverse=True)

        tp_flags = []
        fp_flags = []
        for pred in class_predictions:
            candidates = gt_by_image.get(pred["image_id"], [])
            best_iou = 0.0
            best_index = -1
            for index, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = float(bbox_iou(pred["bbox"], gt["bbox"].unsqueeze(0))[0])
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index >= 0 and best_iou >= iou_threshold:
                candidates[best_index]["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        tp_sum = 0
        fp_sum = 0
        recalls = []
        precisions = []
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp
            fp_sum += fp
            recalls.append(tp_sum / num_gt)
            precisions.append(tp_sum / max(tp_sum + fp_sum, 1))
        aps.append(compute_ap(recalls, precisions))

    return sum(aps) / len(aps) if aps else 0.0


def compute_detection_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    total_tp = 0
    total_fp = 0
    total_gt = 0
    class_f1_scores = []

    for class_index in range(num_classes):
        gt_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        num_gt = 0
        for gt in ground_truths:
            boxes = gt["boxes"]
            labels = gt["labels"]
            class_mask = labels == class_index
            for box in boxes[class_mask]:
                gt_by_image[gt["image_id"]].append({"bbox": box, "matched": False})
                num_gt += 1

        class_predictions = []
        for pred in predictions:
            labels = pred["labels"]
            class_mask = labels == class_index
            for box, score in zip(pred["boxes"][class_mask], pred["scores"][class_mask]):
                class_predictions.append(
                    {
                        "image_id": pred["image_id"],
                        "bbox": box,
                        "score": float(score),
                    }
                )
        class_predictions.sort(key=lambda item: item["score"], reverse=True)

        tp = 0
        fp = 0
        for pred in class_predictions:
            candidates = gt_by_image.get(pred["image_id"], [])
            best_iou = 0.0
            best_index = -1
            for index, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = float(bbox_iou(pred["bbox"], gt["bbox"].unsqueeze(0))[0])
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index >= 0 and best_iou >= iou_threshold:
                candidates[best_index]["matched"] = True
                tp += 1
            else:
                fp += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(num_gt, 1)
        f1 = (
            2.0 * precision * recall / max(precision + recall, 1e-12)
            if num_gt > 0
            else 0.0
        )
        if num_gt > 0:
            class_f1_scores.append(f1)

        total_tp += tp
        total_fp += fp
        total_gt += num_gt

    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_gt, 1)
    micro_f1 = (
        2.0 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)
        if total_gt > 0
        else 0.0
    )
    macro_f1 = sum(class_f1_scores) / len(class_f1_scores) if class_f1_scores else 0.0
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if epochs <= warmup_epochs:
            return 1.0
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def train_one_epoch(
    model: nn.Module,
    criterion: YoloDetectionLoss,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    max_train_batches: int,
    show_progress: bool,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    totals = defaultdict(float)
    steps = 0

    total_batches = len(loader)
    if max_train_batches > 0:
        total_batches = min(total_batches, max_train_batches)
    progress = (
        tqdm(loader, total=total_batches, desc="train", leave=False)
        if show_progress
        else loader
    )

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses["loss"]

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            totals[key] += float(value.detach().cpu())
        steps += 1
        if show_progress:
            progress.set_postfix(
                loss=f"{totals['loss'] / steps:.4f}",
                box=f"{totals['box_loss'] / steps:.4f}",
                dfl=f"{totals['dfl_loss'] / steps:.4f}",
                cls=f"{totals['class_loss'] / steps:.4f}",
                pos=f"{totals['num_positive'] / steps:.0f}",
            )
        if max_train_batches > 0 and steps >= max_train_batches:
            break

    return {key: value / max(steps, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: YoloDetectionLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    image_size: int,
    conf_threshold: float,
    class_thresholds: torch.Tensor | None,
    nms_threshold: float,
    max_detections: int,
    max_val_batches: int,
    show_progress: bool,
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    steps = 0
    all_predictions = []
    all_ground_truths = []

    total_batches = len(loader)
    if max_val_batches > 0:
        total_batches = min(total_batches, max_val_batches)
    progress = (
        tqdm(loader, total=total_batches, desc="val", leave=False)
        if show_progress
        else loader
    )

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        device_targets = move_targets_to_device(targets, device)
        outputs = model(images)
        losses = criterion(outputs, device_targets)

        for key, value in losses.items():
            totals[key] += float(value.detach().cpu())
        steps += 1
        if show_progress:
            progress.set_postfix(loss=f"{totals['loss'] / steps:.4f}")

        decoded = decode_batch_predictions(
            outputs,
            criterion,
            image_size=image_size,
            conf_threshold=conf_threshold,
            class_thresholds=class_thresholds,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
        )
        for pred, target in zip(decoded, device_targets):
            all_predictions.append(
                {
                    "image_id": target["image_id"],
                    "boxes": pred["boxes"].detach().cpu(),
                    "scores": pred["scores"].detach().cpu(),
                    "labels": pred["labels"].detach().cpu(),
                }
            )
            all_ground_truths.append(
                {
                    "image_id": target["image_id"],
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            )

        if max_val_batches > 0 and steps >= max_val_batches:
            break

    metrics = {key: value / max(steps, 1) for key, value in totals.items()}
    metrics["mAP@0.5"] = compute_map_50(
        all_predictions,
        all_ground_truths,
        num_classes=len(DEFAULT_CLASSES),
    )
    metrics.update(
        compute_detection_metrics(
            all_predictions,
            all_ground_truths,
            num_classes=len(DEFAULT_CLASSES),
        )
    )
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_name: str,
    metric_value: float,
    best_metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "classes": list(DEFAULT_CLASSES),
        "architecture": (
            f"anchor_free_decoupled_yolov8_style_"
            f"{model.backbone_name}_{model.neck_name}_dfl"
        ),
        "head_type": "objectness_free_quality_class",
        "backbone_name": model.backbone_name,
        "neck_name": model.neck_name,
        "strides": list(model.strides),
        "reg_max": model.reg_max,
        "image_size": args.image_size,
        "assignment": {
            "small_object_max_side": args.small_object_max_side,
            "medium_object_max_side": args.medium_object_max_side,
            "tal_topk": args.tal_topk,
            "tal_alpha": args.tal_alpha,
            "tal_beta": args.tal_beta,
        },
        "checkpoint_metric": metric_name,
        "checkpoint_metric_value": metric_value,
        "best_metrics": best_metrics,
        "args": serializable_args,
    }
    torch.save(checkpoint, path)


def append_history(
    path: Path,
    epoch: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_map: float,
    best_val_loss: float,
    best_f1: float,
    args: argparse.Namespace,
    run_started_at: str,
    train_seconds: float,
    val_seconds: float,
    epoch_seconds: float,
    reset_file: bool = False,
) -> None:
    fieldnames = [
        "run_started_at",
        "epoch",
        "total_epochs",
        "train_data",
        "val_data",
        "image_dir",
        "val_image_dir",
        "image_size",
        "batch_size",
        "num_workers",
        "initial_lr",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "reg_max",
        "small_object_max_side",
        "medium_object_max_side",
        "tal_topk",
        "tal_alpha",
        "tal_beta",
        "backbone",
        "neck",
        "oversample_classes",
        "oversample_factor",
        "pretrained_backbone",
        "freeze_backbone_stem",
        "amp",
        "disable_tqdm",
        "conf_threshold",
        "class_thresholds",
        "nms_threshold",
        "max_detections",
        "seed",
        "train_seconds",
        "val_seconds",
        "epoch_seconds",
        "train_loss",
        "train_box_loss",
        "train_dfl_loss",
        "train_objectness_loss",
        "train_class_loss",
        "train_num_positive",
        "val_loss",
        "val_box_loss",
        "val_dfl_loss",
        "val_objectness_loss",
        "val_class_loss",
        "val_num_positive",
        "val_map50",
        "val_micro_precision",
        "val_micro_recall",
        "val_micro_f1",
        "val_macro_f1",
        "best_map50",
        "best_val_loss",
        "best_micro_f1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = reset_file or not path.exists()
    mode = "w" if reset_file else "a"

    row = {
        "run_started_at": run_started_at,
        "epoch": epoch,
        "total_epochs": args.epochs,
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "image_dir": str(args.image_dir),
        "val_image_dir": str(args.val_image_dir),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "initial_lr": args.lr,
        "lr": lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "reg_max": args.reg_max,
        "small_object_max_side": args.small_object_max_side,
        "medium_object_max_side": args.medium_object_max_side,
        "tal_topk": args.tal_topk,
        "tal_alpha": args.tal_alpha,
        "tal_beta": args.tal_beta,
        "backbone": args.backbone,
        "neck": args.neck,
        "oversample_classes": " ".join(args.oversample_classes),
        "oversample_factor": args.oversample_factor,
        "pretrained_backbone": args.pretrained_backbone,
        "freeze_backbone_stem": args.freeze_backbone_stem,
        "amp": args.amp,
        "disable_tqdm": args.disable_tqdm or args.no_progress,
        "conf_threshold": args.conf_threshold,
        "class_thresholds": args.class_thresholds,
        "nms_threshold": args.nms_threshold,
        "max_detections": args.max_detections,
        "seed": args.seed,
        "train_seconds": f"{train_seconds:.2f}",
        "val_seconds": f"{val_seconds:.2f}",
        "epoch_seconds": f"{epoch_seconds:.2f}",
        "train_loss": train_metrics.get("loss", 0.0),
        "train_box_loss": train_metrics.get("box_loss", 0.0),
        "train_dfl_loss": train_metrics.get("dfl_loss", 0.0),
        "train_objectness_loss": train_metrics.get("objectness_loss", 0.0),
        "train_class_loss": train_metrics.get("class_loss", 0.0),
        "train_num_positive": train_metrics.get("num_positive", 0.0),
        "val_loss": val_metrics.get("loss", 0.0),
        "val_box_loss": val_metrics.get("box_loss", 0.0),
        "val_dfl_loss": val_metrics.get("dfl_loss", 0.0),
        "val_objectness_loss": val_metrics.get("objectness_loss", 0.0),
        "val_class_loss": val_metrics.get("class_loss", 0.0),
        "val_num_positive": val_metrics.get("num_positive", 0.0),
        "val_map50": val_metrics.get("mAP@0.5", 0.0),
        "val_micro_precision": val_metrics.get("micro_precision", 0.0),
        "val_micro_recall": val_metrics.get("micro_recall", 0.0),
        "val_micro_f1": val_metrics.get("micro_f1", 0.0),
        "val_macro_f1": val_metrics.get("macro_f1", 0.0),
        "best_map50": best_map,
        "best_val_loss": best_val_loss,
        "best_micro_f1": best_f1,
    }

    with path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.backbone.startswith("yolov8") and args.pretrained_backbone:
        print(
            f"{args.backbone} is implemented from scratch in this project; "
            "ignoring --pretrained_backbone."
        )
        args.pretrained_backbone = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"device={device} amp={use_amp}")

    train_loader = make_dataloader(
        annotation_file=args.train_data,
        image_dir=args.image_dir,
        image_size=args.image_size,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        oversample_classes=args.oversample_classes,
        oversample_factor=args.oversample_factor,
    )
    val_loader = make_dataloader(
        annotation_file=args.val_data,
        image_dir=args.val_image_dir,
        image_size=args.image_size,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = build_model(
        num_classes=len(DEFAULT_CLASSES),
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone_stem=args.freeze_backbone_stem,
        backbone_name=args.backbone,
        neck_name=args.neck,
        image_size=args.image_size,
        reg_max=args.reg_max,
    ).to(device)

    criterion = YoloDetectionLoss(
        strides=model.strides,
        num_classes=len(DEFAULT_CLASSES),
        reg_max=model.reg_max,
        small_object_max_side=args.small_object_max_side,
        medium_object_max_side=args.medium_object_max_side,
        tal_topk=args.tal_topk,
        tal_alpha=args.tal_alpha,
        tal_beta=args.tal_beta,
    ).to(device)
    class_thresholds = make_class_threshold_tensor(
        DEFAULT_CLASSES,
        args.conf_threshold,
        parse_class_thresholds(args.class_thresholds),
        device=device,
    )
    if args.class_thresholds:
        pairs = ", ".join(
            f"{class_name}={float(class_thresholds[index]):.3f}"
            for index, class_name in enumerate(DEFAULT_CLASSES)
        )
        print(f"class_thresholds: {pairs}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)

    best_map = -1.0
    best_val_loss = float("inf")
    best_f1 = -1.0
    best_map_path = args.checkpoint_dir / "best_map50.pth"
    best_val_loss_path = args.checkpoint_dir / "best_val_loss.pth"
    best_f1_path = args.checkpoint_dir / "best_f1.pth"
    best_path = args.checkpoint_dir / "best.pth"
    last_path = args.checkpoint_dir / "last.pth"
    started = datetime.now()
    run_started_at = started.strftime("%Y-%m-%d %H:%M:%S")
    history_stamp = started.strftime("%Y%m%d_%H%M%S")
    history_path = args.checkpoint_dir / f"train_history_{history_stamp}.csv"
    show_progress = not (args.disable_tqdm or args.no_progress)

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            max_train_batches=args.max_train_batches,
            show_progress=show_progress,
        )
        train_seconds = time.perf_counter() - train_started

        val_started = time.perf_counter()
        val_metrics = validate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            image_size=args.image_size,
            conf_threshold=args.conf_threshold,
            class_thresholds=class_thresholds,
            nms_threshold=args.nms_threshold,
            max_detections=args.max_detections,
            max_val_batches=args.max_val_batches,
            show_progress=show_progress,
        )
        val_seconds = time.perf_counter() - val_started
        scheduler.step()

        current_map = val_metrics["mAP@0.5"]
        current_val_loss = val_metrics["loss"]
        current_f1 = val_metrics["micro_f1"]
        is_best_map = current_map > best_map
        is_best_val_loss = current_val_loss < best_val_loss
        is_best_f1 = current_f1 > best_f1

        if is_best_map:
            best_map = current_map
        if is_best_val_loss:
            best_val_loss = current_val_loss
        if is_best_f1:
            best_f1 = current_f1

        best_metrics = {
            "best_map50": best_map,
            "best_val_loss": best_val_loss,
            "best_micro_f1": best_f1,
        }

        if is_best_map:
            save_checkpoint(
                best_map_path,
                model,
                optimizer,
                epoch + 1,
                "map50",
                current_map,
                best_metrics,
                args,
            )
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch + 1,
                "map50",
                current_map,
                best_metrics,
                args,
            )
        if is_best_val_loss:
            save_checkpoint(
                best_val_loss_path,
                model,
                optimizer,
                epoch + 1,
                "val_loss",
                current_val_loss,
                best_metrics,
                args,
            )
        if is_best_f1:
            save_checkpoint(
                best_f1_path,
                model,
                optimizer,
                epoch + 1,
                "micro_f1",
                current_f1,
                best_metrics,
                args,
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch + 1,
            "last",
            current_map,
            best_metrics,
            args,
        )
        epoch_seconds = time.perf_counter() - epoch_started

        lr = optimizer.param_groups[0]["lr"]
        append_history(
            history_path,
            epoch=epoch + 1,
            lr=lr,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_map=best_map,
            best_val_loss=best_val_loss,
            best_f1=best_f1,
            args=args,
            run_started_at=run_started_at,
            train_seconds=train_seconds,
            val_seconds=val_seconds,
            epoch_seconds=epoch_seconds,
            reset_file=epoch == 0,
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"epoch_time={format_duration(epoch_seconds)} "
            f"train_time={format_duration(train_seconds)} "
            f"val_time={format_duration(val_seconds)} "
            f"lr={lr:.6g} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dfl={train_metrics.get('dfl_loss', 0.0):.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dfl={val_metrics.get('dfl_loss', 0.0):.4f} "
            f"val_map50={current_map:.4f} "
            f"val_f1={current_f1:.4f} "
            f"best_map50={best_map:.4f} "
            f"best_val_loss={best_val_loss:.4f} "
            f"best_f1={best_f1:.4f}"
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"best_map50_checkpoint={best_map_path}")
    print(f"best_val_loss_checkpoint={best_val_loss_path}")
    print(f"best_f1_checkpoint={best_f1_path}")
    print(f"best_checkpoint={best_path}")
    print(f"history={history_path}")


if __name__ == "__main__":
    main()
