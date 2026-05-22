from __future__ import annotations

import argparse

from .dataset import ObjectDetectionDataset, make_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check the detection dataset.")
    parser.add_argument("--annotations", default="public/annotations/train.json")
    parser.add_argument("--image_dir", default="public/train/images")
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ObjectDetectionDataset(
        args.annotations,
        args.image_dir,
        image_size=args.image_size,
        train=args.train,
    )
    image, target = dataset[0]
    print(f"dataset_size={len(dataset)}")
    print(f"classes={dataset.classes}")
    print(f"sample_image_shape={tuple(image.shape)}")
    print(f"sample_image_id={target['image_id']}")
    print(f"sample_boxes={tuple(target['boxes'].shape)}")

    boxes = target["boxes"]
    if boxes.numel() > 0:
        assert boxes[:, 0].min() >= 0
        assert boxes[:, 1].min() >= 0
        assert boxes[:, 2].max() <= args.image_size
        assert boxes[:, 3].max() <= args.image_size
        assert ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all()

    loader = make_dataloader(
        args.annotations,
        args.image_dir,
        image_size=args.image_size,
        train=args.train,
        batch_size=args.batch_size,
        num_workers=0,
    )
    images, targets = next(iter(loader))
    print(f"batch_image_shape={tuple(images.shape)}")
    print(f"batch_box_counts={[len(item['boxes']) for item in targets]}")


if __name__ == "__main__":
    main()
