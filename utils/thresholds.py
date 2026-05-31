from __future__ import annotations

import torch


def parse_class_thresholds(value: str | None) -> dict[str, float]:
    if value is None or not value.strip():
        return {}

    thresholds: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "Class thresholds must use name=value pairs, "
                f"got {item!r}."
            )
        class_name, threshold_text = item.split("=", 1)
        class_name = class_name.strip()
        if not class_name:
            raise ValueError(f"Missing class name in threshold item {item!r}.")
        threshold = float(threshold_text.strip())
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                f"Threshold for class {class_name!r} must be in [0, 1], "
                f"got {threshold}."
            )
        thresholds[class_name] = threshold
    return thresholds


def make_class_threshold_tensor(
    classes: list[str] | tuple[str, ...],
    default_threshold: float,
    class_thresholds: dict[str, float],
    device: torch.device | None = None,
) -> torch.Tensor:
    unknown_classes = sorted(set(class_thresholds) - set(classes))
    if unknown_classes:
        raise ValueError(f"Unknown classes in class_thresholds: {unknown_classes}")

    values = [float(default_threshold)] * len(classes)
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    for class_name, threshold in class_thresholds.items():
        values[class_to_idx[class_name]] = float(threshold)
    return torch.tensor(values, dtype=torch.float32, device=device)
