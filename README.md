# Object Detection YOLO-Lite

This project implements an object detector from scratch with PyTorch. The model uses a pretrained ResNet50 feature extractor, a custom BiFPN neck, a decoupled anchor-free detection head, a custom detection loss, and per-class NMS during inference.

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

The detector is implemented as a small anchor-free YOLO-style model. It does not use any complete object detection framework such as YOLOv5/v8, Detectron2, MMDetection, Faster R-CNN, or SSD. The only pretrained component is the ResNet50 classification backbone from `torchvision`.

### Overview

```text
Input image
  -> Letterbox resize + ImageNet normalization
  -> ResNet50 backbone
  -> 3-scale BiFPN neck
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

The backbone is `ResNet50Backbone`, built from `torchvision.models.resnet50`.

When `--pretrained_backbone` is enabled, ImageNet pretrained ResNet50 weights are used. The detector removes the classification head and returns intermediate feature maps:

```text
C3: stride 8,  channels 512
C4: stride 16, channels 1024
C5: stride 32, channels 2048
```

The model uses three detection scales to reduce memory use while still covering small, medium, and large objects.

### Neck

File: `models/neck.py`

The neck is a custom 3-scale BiFPN module named `BiFPN`.

Feature fusion points use learnable non-negative weights instead of plain addition:

```text
fused = (w1 * feature_a + ... + wn * feature_n) / (w1 + ... + wn + eps)
```

The weights are initialized equally, then learned during training. This lets the model decide how much to trust each input path at every BiFPN node.

After 1x1 lateral projection, the BiFPN repeats bidirectional fusion blocks:

```text
Top-down:
weighted(P4_in, upsample(P5_in)) -> P4_td
weighted(P3_in, upsample(P4_td)) -> P3_out

Bottom-up:
weighted(P4_in, P4_td, downsample(P3_out)) -> P4_out
weighted(P5_in, downsample(P4_out)) -> P5_out
```

Every neck output has 256 channels:

```text
P3: stride 8
P4: stride 16
P5: stride 32
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
  -> box tower   -> ltrb distribution + objectness
  -> class tower -> class logits
```

Each scale outputs:

```text
[left_distribution, top_distribution, right_distribution, bottom_distribution,
 objectness, class_logits...]
```

With `reg_max = 16` and 5 classes, each output tensor has:

```text
4 * (reg_max + 1) + 1 + num_classes = 74 channels
```

The full model output is a list of three tensors, one per stride:

```text
[
  B x 74 x H/8  x W/8,
  B x 74 x H/16 x W/16,
  B x 74 x H/32 x W/32,
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

The loss assigns each ground-truth box to one feature scale based on object size:

```text
small objects  -> stride 8
medium objects -> stride 16
large objects  -> stride 32
```

Positive grid cells are chosen around the object center using a small center radius. If multiple objects compete for the same cell, the smaller-area object is kept for that cell.

### Loss Function

File: `utils/loss.py`

The training loss has four parts:

```text
total_loss =
  5.0 * box_loss
  + 1.0 * dfl_loss
  + 1.0 * objectness_loss
  + 0.5 * class_loss
```

Where:

```text
box_loss        = 1 - CIoU
dfl_loss        = distribution focal loss for l/t/r/b distances
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
random scale/translate  p=0.20
random crop             p=0.10
horizontal flip         p=0.50
color jitter            p=0.50
small cutout            p=0.05
letterbox resize
ImageNet normalization
```

Grayscale and blur are disabled by default to keep training examples closer to the validation distribution. Images containing `chair` get a mild `1.2x` probability boost for scale, crop, color, and cutout transforms. Validation and prediction only use letterbox resize plus normalization.

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

The checkpoint stores the class list, image size, strides, architecture name, model weights, optimizer state, the checkpoint metric, its value, and the best mAP/loss/F1 summary.
