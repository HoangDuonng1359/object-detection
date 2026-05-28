from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augmentation import DetectionTransform


DEFAULT_CLASSES = ("person", "car", "dog", "cat", "chair")


class ObjectDetectionDataset(Dataset):
    def __init__(
        self,
        annotation_file: str | Path,
        image_dir: str | Path,
        image_size: int = 416,
        train: bool = True,
        classes: list[str] | tuple[str, ...] | None = None,
        transform: DetectionTransform | None = None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        self.train = train

        with self.annotation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.classes = list(classes or data.get("classes") or DEFAULT_CLASSES)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.idx_to_class = {index: name for name, index in self.class_to_idx.items()}

        self.images = data["images"]
        self.annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ann in data["annotations"]:
            class_name = ann["class"]
            if class_name not in self.class_to_idx:
                raise ValueError(f"Unknown class in annotation: {class_name}")
            self.annotations_by_image[ann["image_id"]].append(ann)

        self.transform = transform or DetectionTransform(
            image_size=image_size,
            train=train,
            crop_prob=0.25 if train else 0.0,
            hflip_prob=0.5 if train else 0.0,
            color_jitter_prob=0.8 if train else 0.0,
        )

    def __len__(self) -> int:
        return len(self.images)

    @property
    def image_size(self) -> int:
        return self.transform.image_size

    def set_image_size(self, image_size: int) -> None:
        """Allows epoch-level multi-scale training without rebuilding the dataset."""
        self.transform.image_size = int(image_size)

    def _resolve_image_path(self, image_info: dict[str, Any]) -> Path:
        filename = Path(image_info.get("file_name", image_info["id"])).name
        return self.image_dir / filename

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_info = self.images[index]
        image_id = image_info["id"]
        image_path = self._resolve_image_path(image_info)

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        anns = self.annotations_by_image.get(image_id, [])
        boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
        labels = torch.tensor(
            [self.class_to_idx[ann["class"]] for ann in anns],
            dtype=torch.long,
        )

        orig_size = (int(image_info["height"]), int(image_info["width"]))
        image_tensor, boxes, labels, meta = self.transform(image, boxes, labels)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "orig_size": torch.tensor(orig_size, dtype=torch.long),
            "size": torch.tensor(image_tensor.shape[-2:], dtype=torch.long),
            "scale": torch.tensor(float(meta["scale"]), dtype=torch.float32),
            "pad": torch.tensor(meta["pad"], dtype=torch.float32),
        }
        return image_tensor, target


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)


def make_class_oversampling_sampler(
    dataset: ObjectDetectionDataset,
    oversample_classes: list[str] | tuple[str, ...] | None = None,
    oversample_factor: float = 1.0,
) -> WeightedRandomSampler | None:
    if not oversample_classes or oversample_factor <= 1.0:
        return None

    unknown_classes = sorted(set(oversample_classes) - set(dataset.classes))
    if unknown_classes:
        raise ValueError(f"Unknown oversample classes: {unknown_classes}")

    target_classes = set(oversample_classes)
    weights = []
    for image_info in dataset.images:
        image_id = image_info["id"]
        annotations = dataset.annotations_by_image.get(image_id, [])
        has_target_class = any(ann["class"] in target_classes for ann in annotations)
        weights.append(float(oversample_factor if has_target_class else 1.0))

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def make_dataloader(
    annotation_file: str | Path,
    image_dir: str | Path,
    image_size: int = 416,
    train: bool = True,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool | None = None,
    oversample_classes: list[str] | tuple[str, ...] | None = None,
    oversample_factor: float = 1.0,
) -> DataLoader:
    dataset = ObjectDetectionDataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        image_size=image_size,
        train=train,
    )
    sampler = None
    if train:
        sampler = make_class_oversampling_sampler(
            dataset,
            oversample_classes=oversample_classes,
            oversample_factor=oversample_factor,
        )
    should_shuffle = train if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else should_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )
