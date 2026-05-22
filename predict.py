from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from models.yolo import DEFAULT_ANCHORS, DEFAULT_CLASSES, DEFAULT_STRIDES, YoloLite
from utils.augmentation import image_to_normalized_tensor, letterbox_resize
from utils.loss import YoloDetectionLoss


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run object detection inference.")
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pth"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.45)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def to_anchor_tuple(value: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(tuple((float(w), float(h)) for w, h in scale) for scale in value)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[YoloLite, list[str], int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = list(checkpoint.get("classes", DEFAULT_CLASSES))
    anchors = to_anchor_tuple(checkpoint.get("anchors", DEFAULT_ANCHORS))
    strides = tuple(int(stride) for stride in checkpoint.get("strides", DEFAULT_STRIDES))
    image_size = int(checkpoint.get("image_size", 416))

    model = YoloLite(
        num_classes=len(classes),
        pretrained_backbone=False,
        anchors=anchors,
        strides=strides,
    ).to(device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, classes, image_size


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def preprocess_image(
    image_path: Path,
    image_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        resized, _, scale, pad = letterbox_resize(
            image,
            torch.empty((0, 4), dtype=torch.float32),
            image_size,
        )
    tensor = image_to_normalized_tensor(resized)
    meta = {
        "image_id": image_path.name,
        "orig_w": orig_w,
        "orig_h": orig_h,
        "scale": scale,
        "pad_x": pad[0],
        "pad_y": pad[1],
    }
    return tensor, meta


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
def decode_predictions(
    predictions: list[torch.Tensor],
    criterion: YoloDetectionLoss,
    image_size: int,
    conf_threshold: float,
) -> list[dict[str, torch.Tensor]]:
    batch_size = predictions[0].shape[0]
    batch_boxes = [[] for _ in range(batch_size)]
    batch_scores = [[] for _ in range(batch_size)]
    batch_labels = [[] for _ in range(batch_size)]

    for scale_index, prediction in enumerate(predictions):
        decoded = criterion.decode_boxes(
            prediction,
            criterion.anchors[scale_index],
            criterion.strides[scale_index],
        )
        boxes = decoded.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
        objectness = torch.sigmoid(prediction[:, :, 4]).reshape(batch_size, -1)
        class_probs = torch.softmax(prediction[:, :, 5:], dim=2)
        class_scores, class_labels = class_probs.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            -1,
            criterion.num_classes,
        ).max(dim=-1)
        scores = objectness * class_scores

        boxes[..., 0::2].clamp_(0, float(image_size))
        boxes[..., 1::2].clamp_(0, float(image_size))
        valid = (
            (scores >= conf_threshold)
            & (boxes[..., 2] > boxes[..., 0])
            & (boxes[..., 3] > boxes[..., 1])
        )

        for batch_index in range(batch_size):
            mask = valid[batch_index]
            if mask.any():
                batch_boxes[batch_index].append(boxes[batch_index][mask])
                batch_scores[batch_index].append(scores[batch_index][mask])
                batch_labels[batch_index].append(class_labels[batch_index][mask])

    decoded_batch = []
    for boxes_parts, scores_parts, labels_parts in zip(
        batch_boxes,
        batch_scores,
        batch_labels,
    ):
        if not boxes_parts:
            device = predictions[0].device
            decoded_batch.append(
                {
                    "boxes": torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                }
            )
        else:
            decoded_batch.append(
                {
                    "boxes": torch.cat(boxes_parts, dim=0),
                    "scores": torch.cat(scores_parts, dim=0),
                    "labels": torch.cat(labels_parts, dim=0),
                }
            )
    return decoded_batch


def postprocess_single(
    decoded: dict[str, torch.Tensor],
    meta: dict[str, Any],
    classes: list[str],
    nms_threshold: float,
    max_detections: int,
) -> dict[str, Any]:
    boxes = decoded["boxes"]
    scores = decoded["scores"]
    labels = decoded["labels"]

    if boxes.numel() == 0:
        return {"image_id": meta["image_id"], "boxes": []}

    keep_parts = []
    for label in labels.unique():
        label_indices = torch.where(labels == label)[0]
        kept = nms(boxes[label_indices], scores[label_indices], nms_threshold)
        keep_parts.append(label_indices[kept])
    keep = torch.cat(keep_parts) if keep_parts else torch.empty(0, dtype=torch.long)
    keep = keep[scores[keep].argsort(descending=True)[:max_detections]]

    boxes = boxes[keep].clone()
    scores = scores[keep]
    labels = labels[keep]

    boxes[:, 0::2] = (boxes[:, 0::2] - float(meta["pad_x"])) / float(meta["scale"])
    boxes[:, 1::2] = (boxes[:, 1::2] - float(meta["pad_y"])) / float(meta["scale"])
    boxes[:, 0::2].clamp_(0, float(meta["orig_w"]))
    boxes[:, 1::2].clamp_(0, float(meta["orig_h"]))
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

    output_boxes = []
    for box, score, label in zip(boxes[valid], scores[valid], labels[valid]):
        output_boxes.append(
            {
                "class": classes[int(label.item())],
                "confidence": round(float(score.item()), 6),
                "bbox": [round(float(value), 2) for value in box.tolist()],
            }
        )

    return {"image_id": meta["image_id"], "boxes": output_boxes}


@torch.no_grad()
def predict(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = choose_device(args.device)
    model, classes, image_size = load_model(args.checkpoint, device)
    criterion = YoloDetectionLoss(
        anchors=model.anchors,
        strides=model.strides,
        num_classes=len(classes),
    ).to(device)

    image_paths = list_images(args.image_dir)
    results: list[dict[str, Any]] = []
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        samples = [preprocess_image(path, image_size) for path in batch_paths]
        images = torch.stack([sample[0] for sample in samples]).to(device)
        metas = [sample[1] for sample in samples]

        outputs = model(images)
        decoded_batch = decode_predictions(
            outputs,
            criterion,
            image_size=image_size,
            conf_threshold=args.conf_threshold,
        )
        for decoded, meta in zip(decoded_batch, metas):
            results.append(
                postprocess_single(
                    decoded,
                    meta,
                    classes=classes,
                    nms_threshold=args.nms_threshold,
                    max_detections=args.max_detections,
                )
            )

    return results


def main() -> None:
    args = parse_args()
    results = predict(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
