# Object Detection YOLOv8-Style

This project implements an object detector with PyTorch. The default model is a practical YOLOv8-style hybrid: ImageNet-pretrained ResNet backbone, YOLOv8-style PAN-FPN neck, objectness-free decoupled detection head, DFL box regression, task-aligned assignment, and per-class NMS during inference.

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

The detector is implemented as an anchor-free YOLOv8-style model. It does not import or wrap any complete object detection framework such as Ultralytics YOLO, Detectron2, MMDetection, Faster R-CNN, or SSD. The default uses a pretrained ResNet feature extractor to avoid training all low-level features from scratch.

### Overview

```text
Input image
  -> Letterbox resize + ImageNet normalization
  -> ResNet50/ResNet101 pretrained backbone
  -> YOLOv8-style PAN-FPN neck
  -> Objectness-free decoupled detection heads
  -> Decode boxes + confidence filtering + per-class NMS
  -> predictions.json
```

The model predicts 5 classes:

```text
person, car, dog, cat, chair
```

### Backbone

File: `models/backbone.py`

The default backbone is `ResNetBackbone`, built from `torchvision.models.resnet50` or `torchvision.models.resnet101`.

Default notebook setting:

```text
BACKBONE_NAME = "resnet50"
USE_PRETRAINED_BACKBONE = True
```

The backbone returns four feature maps:

```text
C2: stride 4
C3: stride 8
C4: stride 16
C5: stride 32
```

YOLOv8-style backbone variants `yolov8n`, `yolov8s`, and `yolov8m` are still available, but they are implemented from scratch here and do not use pretrained Ultralytics weights.

### Neck

File: `models/neck.py`

The default neck is `YoloV8PAN`, a PAN-FPN neck with C2f fusion blocks:

```text
Top-down:
concat(P4, upsample(P5)) -> C2f -> P4_td
concat(P3, upsample(P4_td)) -> C2f -> P3_out
concat(P2, upsample(P3_out)) -> C2f -> P2_out

Bottom-up:
concat(P3_out, downsample(P2_out)) -> C2f -> P3_out
concat(P4_td, downsample(P3_out)) -> C2f -> P4_out
concat(P5, downsample(P4_out)) -> C2f -> P5_out
```

Every neck output has 256 channels:

```text
P2: stride 4
P3: stride 8
P4: stride 16
P5: stride 32
```

For `IMAGE_SIZE = 512`, the prediction feature shapes are:

```text
stride 4:   128 x 128
stride 8:   64 x 64
stride 16:  32 x 32
stride 32:  16 x 16
```

### Detection Head

File: `models/head.py`

The head is anchor-free and decoupled. Each scale has two separate towers:

```text
feature
  -> box tower   -> ltrb distribution
  -> class tower -> quality-aware class logits
```

Each scale outputs:

```text
[left_distribution, top_distribution, right_distribution, bottom_distribution,
 class_logits...]
```

With `reg_max = 16` and 5 classes, each output tensor has:

```text
4 * (reg_max + 1) + num_classes = 73 channels
```

The full model output is a list of four tensors, one per stride:

```text
[
  B x 73 x H/4  x W/4,
  B x 73 x H/8  x W/8,
  B x 73 x H/16 x W/16,
  B x 73 x H/32 x W/32,
]
```

### Box Parameterization

File: `utils/loss.py`

Predicted boxes are decoded as YOLOv8-style distributional distances from each
grid point to the four box sides:

```text
distance = expectation(softmax(distribution_bins)) * stride
point_x  = (grid_x + 0.5) * stride
point_y  = (grid_y + 0.5) * stride

xmin = point_x - left
ymin = point_y - top
xmax = point_x + right
ymax = point_y + bottom
```

The decoded box is:

```text
[xmin, ymin, xmax, ymax]
```

### Target Assignment

The loss uses a simplified Task-Aligned Assignment strategy inspired by modern
anchor-free YOLO detectors:

```text
candidate points = points inside each ground-truth box across all feature scales
alignment metric = class_score^tal_alpha * IoU^tal_beta
positive points  = top tal_topk candidates per ground-truth box
```

By default, `tal_topk=5`, `tal_alpha=0.5`, and `tal_beta=4.0`. If multiple
objects compete for the same grid cell, the assignment with the higher alignment
metric is kept. The selected class target is quality-aware, using the selected
candidate's normalized alignment and IoU instead of a hard `1.0` target.

The size thresholds `small_object_max_side` and `medium_object_max_side` are kept
as a fallback for extremely small boxes that have no candidate point inside the
box.

### Loss Function

File: `utils/loss.py`

The training loss has three active parts:

```text
total_loss =
  7.5 * box_loss
  + 1.5 * dfl_loss
  + 0.5 * class_loss
```

Where:

```text
box_loss   = quality-weighted 1 - CIoU
dfl_loss   = quality-weighted distribution focal loss for l/t/r/b distances
class_loss = BCE with quality-aware class targets
```

### Inference

File: `predict.py`

Inference follows these steps:

```text
1. Letterbox resize image to checkpoint image_size.
2. Normalize with ImageNet mean/std.
3. Run the model.
4. Decode boxes from all feature scales.
5. Compute confidence = max sigmoid(class_logit).
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

Training saves multiple best checkpoints automatically:

```text
best_map50.pth     best validation mAP@0.5
best_val_loss.pth  lowest validation loss
best_f1.pth        best validation micro F1
best.pth           compatibility alias for best_map50.pth
last.pth           most recent epoch
```

No `--best_metric` argument is needed.

## Training Augmentation

Training uses box-aware augmentation in `utils/augmentation.py`:

```text
random scale/translate  p=0.15
random crop             p=0.08
horizontal flip         p=0.50
color jitter            p=0.45
small cutout            p=0.03
letterbox resize
ImageNet normalization
```

Grayscale and blur are disabled by default to keep training examples closer to the validation distribution. Cutout is kept low to add mild occlusion robustness without pushing car false positives too aggressively. Images containing `chair` get a mild `1.2x` probability boost for scale, crop, color, and cutout transforms. Validation and prediction only use letterbox resize plus normalization.

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
  --backbone resnet50 \
  --neck yolov8_pan \
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

The checkpoint stores the class list, image size, strides, architecture name, model weights, optimizer state, the checkpoint metric, its value, and the best mAP/loss/F1 summary.
