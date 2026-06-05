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

## Model Architecture

The detector is implemented as a small anchor-free YOLO-style model. It does not use any complete object detection framework such as YOLOv5/v8, Detectron2, MMDetection, Faster R-CNN, or SSD. The only pretrained component is the ResNet34 classification backbone from `torchvision`.

### Overview

```text
Input image
  -> Letterbox resize + ImageNet normalization
  -> ResNet34 backbone
  -> 3-scale weighted FPN + PAN neck
  -> Decoupled anchor-free detection heads
  -> Decode boxes + confidence filtering + per-class NMS
  -> predictions.json
```

The model predicts 5 classes:

```text
person, car, dog, cat, chair
```

### Backbone

File: `models/backbone.py`

The backbone is `ResNet34Backbone`, built from `torchvision.models.resnet34`.

When `--pretrained_backbone` is enabled, ImageNet pretrained ResNet34 weights are used. The detector removes the classification head and returns intermediate feature maps:

```text
C3: stride 8,  channels 128
C4: stride 16, channels 256
C5: stride 32, channels 512
```

The model uses three detection scales to reduce memory use while still covering small, medium, and large objects.

### Neck

File: `models/neck.py`

The neck is a custom weighted FPN + PAN module named `SimpleFPN`.

Feature fusion points use learnable non-negative weights instead of plain addition:

```text
fused = (w1 * feature_a + w2 * feature_b) / (w1 + w2 + eps)
```

The weights are initialized equally, then learned during training. This lets the model decide how much to trust each top-down or bottom-up feature path.

First, a top-down FPN path fuses semantic information from deep layers into shallow layers:

```text
C5 -> P5
weighted(C4, upsample(P5)) -> P4
weighted(C3, upsample(P4)) -> P3
```

Then, a bottom-up PAN path sends localization-rich low-level information back to deeper scales:

```text
N3 = P3
weighted(P4, downsample(N3)) -> N4
weighted(P5, downsample(N4)) -> N5
```

Every neck output has 256 channels:

```text
N3: stride 8
N4: stride 16
N5: stride 32
```

For `IMAGE_SIZE = 512`, the prediction feature shapes are:

```text
stride 8:   64 x 64
stride 16:  32 x 32
stride 32:  16 x 16
```

### Detection Head

File: `models/head.py`

The head is anchor-free and decoupled. Each scale has two separate towers:

```text
feature
  -> box tower   -> bbox + objectness
  -> class tower -> class logits
```

Each scale outputs:

```text
[tx, ty, tw, th, objectness, class_logits...]
```

For 5 classes, each output tensor has:

```text
5 + num_classes = 10 channels
```

The full model output is a list of three tensors, one per stride:

```text
[
  B x 10 x H/8  x W/8,
  B x 10 x H/16 x W/16,
  B x 10 x H/32 x W/32,
]
```

### Box Parameterization

File: `utils/loss.py`

Predicted boxes are decoded per grid cell:

```text
center_x = (sigmoid(tx) + grid_x) * stride
center_y = (sigmoid(ty) + grid_y) * stride
box_w    = exp(tw) * stride
box_h    = exp(th) * stride
```

The decoded box is converted to:

```text
[xmin, ymin, xmax, ymax]
```

### Target Assignment

The loss assigns each ground-truth box to one feature scale based on object size:

```text
small objects  -> stride 8
medium objects -> stride 16
large objects  -> stride 32
```

Positive grid cells are chosen around the object center using a small center radius. If multiple objects compete for the same cell, the smaller-area object is kept for that cell.

### Loss Function

File: `utils/loss.py`

The training loss has three parts:

```text
total_loss =
  5.0 * box_loss
  + 1.0 * objectness_loss
  + 0.5 * class_loss
```

Where:

```text
box_loss        = 1 - CIoU
objectness_loss = focal BCE with logits
class_loss      = cross entropy
```

### Inference

File: `predict.py`

Inference follows these steps:

```text
1. Letterbox resize image to checkpoint image_size.
2. Normalize with ImageNet mean/std.
3. Run the model.
4. Decode boxes from all three feature scales.
5. Compute confidence = sigmoid(objectness) * max_class_probability.
6. Apply global or per-class confidence thresholds.
7. Run NMS independently for each class.
8. Map boxes back to original image coordinates.
9. Write predictions.json.
```

Per-class thresholds are supported:

```bash
--conf_threshold 0.25 \
--class_thresholds "person=0.25,car=0.20,dog=0.25,cat=0.25,chair=0.10"
```

Classes not listed in `--class_thresholds` use `--conf_threshold`.

### Checkpoint Selection

During training, `models/best.pth` can be selected by either validation mAP or validation loss:

```bash
--best_metric map50
```

or:

```bash
--best_metric val_loss
```

By default, `best.pth` is selected using validation `mAP@0.5`.

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
