# Kế Hoạch Chi Tiết: Object Detection From Scratch

## Tổng Quan Kiến Trúc

Xây dựng một mô hình kiểu **YOLO-lite** (anchor-based, single-stage) vì:

* Cân bằng giữa độ phức tạp và hiệu năng
* Dễ implement từ đầu
* Có thể đạt mAP ≥ 0.60 với backbone pretrained---

## Kế Hoạch Chi Tiết Từng Phần

### Cấu trúc thư mục nộp bài

```
my_submission/
├── public/                    ← dataset gốc
├── models/
│   ├── backbone.py            ← ResNet-18 wrapper
│   ├── neck.py                ← FPN đơn giản
│   ├── head.py                ← detection head
│   └── yolo.py           ← ghép toàn bộ model
├── utils/
│   ├── dataset.py             ← Dataset + DataLoader
│   ├── augmentation.py        ← transforms tự viết
│   ├── loss.py                ← 3 thành phần loss
│   ├── nms.py                 ← NMS tự cài đặt
│   └── metrics.py             ← IoU, mAP
├── train.py
├── predict.py
├── README.md
└── requirements.txt
```

---

### Giai đoạn 1 — Dữ liệu (≈ 1–2 ngày)

**`utils/dataset.py`** — tự viết `torch.utils.data.Dataset`:

* Đọc `train.json` / `val.json`, build dict `image_id → annotations`
* Xử lý nhiều bbox trong 1 ảnh (padding về max objects hoặc dùng custom collate\_fn)
* Resize ảnh về **416×416**, normalize `mean=[0.485, 0.456, 0.406]`

**`utils/augmentation.py`** — không dùng albumentations, tự viết:

* Lật ngang (bắt buộc) — phải đổi bbox theo
* Random crop + letterbox
* ColorJitter (brightness, contrast, saturation)
* Multi-scale training (320, 416, 512)

---

### Giai đoạn 2 — Kiến trúc (≈ 2–3 ngày)

Lựa chọn **anchor-based, 3 scale** vì dễ kiểm soát:

**`models/backbone.py`** — dùng ResNet-18 pretrained, bỏ `avgpool` + `fc`, lấy output ở `layer2`, `layer3`, `layer4`.

**`models/neck.py`** — FPN nhỏ: upsample + concat, 2 conv 3×3 để trộn feature.

**`models/head.py`** — mỗi scale 1 lớp conv `1×1` ra `[B, A*(5+C), H, W]` với A=3 anchors, C=5 classes. Output: `tx, ty, tw, th, objectness, class_scores`.

**Anchor boxes** — tự tính bằng k-means trên tập train:

```
Scale P5: (116,90), (156,198), (373,326)
Scale P4: (30,61),  (62,45),  (59,119)
Scale P3: (10,13),  (16,30),  (33,23)
```

---

### Giai đoạn 3 — Hàm mất mát (≈ 1–2 ngày)

**`utils/loss.py`** — tự cài đặt 3 thành phần:


| Thành phần | Hàm         | Ghi chú                                         |
| ------------ | ------------ | ------------------------------------------------ |
| Box loss     | CIoU         | phạt cả overlap, center distance, aspect ratio |
| Objectness   | BCE + focal  | giảm mất cân bằng positive/negative          |
| Class        | CrossEntropy | trên positive samples                           |

Trọng số: `λ_box=5.0, λ_obj=1.0, λ_cls=0.5`

---

### Giai đoạn 4 — Huấn luyện (≈ 1 ngày setup, chạy nhiều lần)

**`train.py`** với các đặc điểm:

* **Optimizer**: AdamW, `lr=1e-3`, `weight_decay=1e-4`
* **Scheduler**: cosine annealing + linear warmup 3 epoch đầu
* **Epochs**: 50–80 epoch
* Lưu `best.pth` theo val mAP@0.5
* In log mỗi epoch: train loss, val loss, val mAP

---

### Giai đoạn 5 — Suy luận & NMS (≈ 1 ngày)

**`utils/nms.py`** — tự cài đặt từ đầu:

```
1. Lọc theo conf_threshold (ví dụ 0.25)
2. Nhân objectness × class_score
3. NMS per-class: sort by score, greedy suppress IoU > 0.45
4. Scale bbox về kích thước ảnh gốc
```

**`predict.py`** — nhận `--image_dir`, xuất `predictions.json` đúng format.

### Điểm Kỹ Thuật Quan Trọng Cần Lưu Ý

**Target assignment** — phần khó nhất: với mỗi GT box, tìm anchor có IoU cao nhất → assign positive; IoU < 0.5 → negative; giữa → ignore.

**Collate function** — vì mỗi ảnh có số bbox khác nhau, cần viết `collate_fn` riêng thay vì dùng default của DataLoader.

**Decode bbox đúng** — khi inference phải map từ `tx,ty,tw,th` (grid-relative) về pixel coordinates của ảnh gốc, kể cả letterbox padding.

**Không dùng YOLO/Detectron** — chỉ dùng `torch.nn`, `torchvision.models.resnet18(pretrained=True)` là được phép (backbone pretrained là ok).
