# Object Detection YOLO-Lite

This project implements an object detector from scratch with PyTorch. The model uses a pretrained ResNet34 feature extractor, a custom FPN/PAN neck, a decoupled anchor-free detection head, a custom detection loss, and per-class NMS during inference.

## Setup

Use Python 3.11.

```bash
pip install -r requirements.txt
```

For CUDA builds of PyTorch, install the matching PyTorch wheel for your machine if needed.

## Data

Place the provided dataset at:

```text
public/
  train/images/
  val/images/
  annotations/train.json
  annotations/val.json
  tools/evaluate_predictions.py
```

## Train

The required training command is supported:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Useful optional settings:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --image_size 512 \
  --epochs 70 \
  --batch_size 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --class_thresholds "chair=0.15,car=0.20" \
  --amp \
  --pretrained_backbone \
  --oversample_classes chair \
  --oversample_factor 2.0
```

The best model is saved to:

```text
models/best.pth
```

## Predict

The required inference command is supported:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

By default, prediction loads:

```text
models/best.pth
```

Optional inference settings:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint models/best.pth \
  --batch_size 8 \
  --conf_threshold 0.25 \
  --class_thresholds "chair=0.15,car=0.20" \
  --nms_threshold 0.35 \
  --max_detections 100
```

Classes not listed in `--class_thresholds` use `--conf_threshold`.

## Evaluate Validation Predictions

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions predictions.json \
  --output score.json
```

## Model Weights

Put trained weights at:

```text
models/best.pth
```

The checkpoint stores the class list, image size, strides, architecture name, model weights, optimizer state, and best validation mAP.
