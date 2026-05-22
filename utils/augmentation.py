from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sanitize_boxes(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    width: int,
    height: int,
    min_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip boxes to image bounds and remove degenerate boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).float(), labels.reshape(0).long()

    boxes = boxes.float().clone()
    boxes[:, 0::2].clamp_(0, float(width))
    boxes[:, 1::2].clamp_(0, float(height))

    keep = (boxes[:, 2] - boxes[:, 0] >= min_size) & (
        boxes[:, 3] - boxes[:, 1] >= min_size
    )
    return boxes[keep], labels[keep]


def horizontal_flip(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    width, _ = image.size
    image = ImageOps.mirror(image)
    if boxes.numel() > 0:
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        boxes = flipped
    return image, boxes, labels


def random_color_jitter(
    image: Image.Image,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> Image.Image:
    transforms = [
        (ImageEnhance.Brightness, brightness),
        (ImageEnhance.Contrast, contrast),
        (ImageEnhance.Color, saturation),
    ]
    random.shuffle(transforms)
    for enhancer_cls, amount in transforms:
        if amount <= 0:
            continue
        factor = random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)
        image = enhancer_cls(image).enhance(factor)
    return image


def random_crop(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    min_scale: float = 0.65,
    max_trials: int = 10,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Random crop that keeps boxes whose centers remain inside the crop."""
    width, height = image.size
    if width <= 1 or height <= 1:
        return image, boxes, labels

    for _ in range(max_trials):
        crop_w = random.randint(max(1, int(width * min_scale)), width)
        crop_h = random.randint(max(1, int(height * min_scale)), height)
        if crop_w == width and crop_h == height:
            continue

        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        right = left + crop_w
        bottom = top + crop_h

        if boxes.numel() == 0:
            return image.crop((left, top, right, bottom)), boxes, labels

        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        keep = (
            (centers_x >= left)
            & (centers_x <= right)
            & (centers_y >= top)
            & (centers_y <= bottom)
        )
        if not keep.any():
            continue

        cropped_boxes = boxes[keep].clone()
        cropped_labels = labels[keep]
        cropped_boxes[:, 0::2] -= float(left)
        cropped_boxes[:, 1::2] -= float(top)
        cropped_boxes, cropped_labels = sanitize_boxes(
            cropped_boxes, cropped_labels, crop_w, crop_h
        )
        if cropped_boxes.numel() == 0:
            continue

        return image.crop((left, top, right, bottom)), cropped_boxes, cropped_labels

    return image, boxes, labels


def letterbox_resize(
    image: Image.Image,
    boxes: torch.Tensor,
    size: int | tuple[int, int],
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, torch.Tensor, float, tuple[int, int]]:
    if isinstance(size, int):
        out_w = out_h = size
    else:
        out_w, out_h = size

    width, height = image.size
    scale = min(out_w / width, out_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), fill)
    canvas.paste(resized, (pad_x, pad_y))

    if boxes.numel() > 0:
        boxes = boxes.float().clone()
        boxes[:, 0::2] = boxes[:, 0::2] * scale + pad_x
        boxes[:, 1::2] = boxes[:, 1::2] * scale + pad_y

    return canvas, boxes, scale, (pad_x, pad_y)


def image_to_normalized_tensor(
    image: Image.Image,
    mean: Iterable[float] = IMAGENET_MEAN,
    std: Iterable[float] = IMAGENET_STD,
) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean_tensor = torch.tensor(tuple(mean), dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(tuple(std), dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


@dataclass
class DetectionTransform:
    image_size: int = 416
    train: bool = True
    hflip_prob: float = 0.5
    crop_prob: float = 0.25
    color_jitter_prob: float = 0.8
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __call__(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        image = image.convert("RGB")
        boxes, labels = sanitize_boxes(boxes, labels, *image.size)

        if self.train:
            if random.random() < self.crop_prob:
                image, boxes, labels = random_crop(image, boxes, labels)
            if random.random() < self.hflip_prob:
                image, boxes, labels = horizontal_flip(image, boxes, labels)
            if random.random() < self.color_jitter_prob:
                image = random_color_jitter(image)

        image, boxes, scale, pad = letterbox_resize(image, boxes, self.image_size)
        boxes, labels = sanitize_boxes(boxes, labels, self.image_size, self.image_size)
        tensor = image_to_normalized_tensor(image, self.mean, self.std)

        meta = {
            "input_size": (self.image_size, self.image_size),
            "scale": scale,
            "pad": pad,
        }
        return tensor, boxes, labels, meta
