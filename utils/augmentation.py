from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


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


def random_scale_translate(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    min_scale: float = 0.85,
    max_scale: float = 1.20,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    width, height = image.size
    if width <= 1 or height <= 1:
        return image, boxes, labels

    scale = random.uniform(min_scale, max_scale)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    if new_w <= width:
        left = random.randint(0, width - new_w)
    else:
        left = random.randint(width - new_w, 0)
    if new_h <= height:
        top = random.randint(0, height - new_h)
    else:
        top = random.randint(height - new_h, 0)

    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(resized, (left, top))

    if boxes.numel() > 0:
        boxes = boxes.float().clone()
        boxes[:, 0::2] = boxes[:, 0::2] * scale + float(left)
        boxes[:, 1::2] = boxes[:, 1::2] * scale + float(top)
        boxes, labels = sanitize_boxes(boxes, labels, width, height)
    return canvas, boxes, labels


def random_grayscale(image: Image.Image) -> Image.Image:
    return ImageOps.grayscale(image).convert("RGB")


def random_blur(
    image: Image.Image,
    min_radius: float = 0.1,
    max_radius: float = 1.2,
) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(random.uniform(min_radius, max_radius)))


def random_cutout(
    image: Image.Image,
    max_holes: int = 2,
    max_size_ratio: float = 0.16,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> Image.Image:
    width, height = image.size
    if width <= 1 or height <= 1:
        return image

    draw = ImageDraw.Draw(image)
    holes = random.randint(1, max_holes)
    max_cut_w = max(1, int(width * max_size_ratio))
    max_cut_h = max(1, int(height * max_size_ratio))
    for _ in range(holes):
        cut_w = random.randint(1, max_cut_w)
        cut_h = random.randint(1, max_cut_h)
        left = random.randint(0, max(0, width - cut_w))
        top = random.randint(0, max(0, height - cut_h))
        draw.rectangle((left, top, left + cut_w, top + cut_h), fill=fill)
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
    scale_translate_prob: float = 0.35
    grayscale_prob: float = 0.08
    blur_prob: float = 0.10
    cutout_prob: float = 0.15
    focus_label_indices: tuple[int, ...] = ()
    focus_aug_boost: float = 1.5
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def _boosted_prob(self, probability: float, labels: torch.Tensor) -> float:
        if not self.focus_label_indices or labels.numel() == 0:
            return probability
        focus_labels = torch.tensor(
            self.focus_label_indices,
            dtype=labels.dtype,
            device=labels.device,
        )
        has_focus_label = torch.isin(labels, focus_labels).any().item()
        if not has_focus_label:
            return probability
        return min(1.0, probability * self.focus_aug_boost)

    def __call__(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        image = image.convert("RGB")
        boxes, labels = sanitize_boxes(boxes, labels, *image.size)

        if self.train:
            if random.random() < self._boosted_prob(self.scale_translate_prob, labels):
                image, boxes, labels = random_scale_translate(image, boxes, labels)
            if random.random() < self._boosted_prob(self.crop_prob, labels):
                image, boxes, labels = random_crop(image, boxes, labels)
            if random.random() < self.hflip_prob:
                image, boxes, labels = horizontal_flip(image, boxes, labels)
            if random.random() < self._boosted_prob(self.color_jitter_prob, labels):
                image = random_color_jitter(image)
            if random.random() < self._boosted_prob(self.grayscale_prob, labels):
                image = random_grayscale(image)
            if random.random() < self._boosted_prob(self.blur_prob, labels):
                image = random_blur(image)
            if random.random() < self._boosted_prob(self.cutout_prob, labels):
                image = random_cutout(image)

        image, boxes, scale, pad = letterbox_resize(image, boxes, self.image_size)
        boxes, labels = sanitize_boxes(boxes, labels, self.image_size, self.image_size)
        tensor = image_to_normalized_tensor(image, self.mean, self.std)

        meta = {
            "input_size": (self.image_size, self.image_size),
            "scale": scale,
            "pad": pad,
        }
        return tensor, boxes, labels, meta
