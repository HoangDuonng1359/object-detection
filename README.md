## 1. Cách chạy 

### 1.1. Cài đặt môi trường
Sử dụng **Python 3.11** hoặc tương đương. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 1.2. Huấn luyện
Sử dụng lệnh sau để bắt đầu quá trình huấn luyện. Nếu không truyền tham số, script sẽ tự động nạp cấu hình mặc định từ file `config.py`:
```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```
Mô hình tốt nhất sau khi huấn luyện sẽ được lưu tự động tại `./models/best.pth`.

### 1.3. Chạy Dự đoán
Sử dụng lệnh sau để suy luận trên tập ảnh kiểm tra. Tương tự, cấu hình dự đoán sẽ nạp tự động từ `PredictConfig` trong file `config.py`:
```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json
```
### 1.4. Đánh giá
Để kiểm tra điểm mAP@0.5 của dự đoán:
```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions predictions.json \
  --output score.json
```

---

## 2. Giải thích cấu hình
Toàn bộ các siêu tham số quan trọng đều được tập trung quản lý tại `config.py`.

### 2.1. Cấu hình Huấn Luyện (`TrainConfig`)
- `IMAGE_SIZE`: Kích thước ảnh đầu vào (ví dụ: 512, 640). Kích thước lớn giúp nhận diện vật nhỏ tốt hơn.
- `EPOCHS` & `BATCH_SIZE`: Số vòng lặp huấn luyện và số lượng ảnh trong mỗi lô.
- `NUM_WORKERS`: Số tiến trình đọc dữ liệu song song. Để `0` trên Windows để tránh lỗi crash do multiprocessing.
- `LR` & `WEIGHT_DECAY`: Tốc độ học (Learning Rate) và hệ số chống quá khớp.
- **TAL (Task-Aligned Learning) Parameters:**
  - `TAL_TOPK`: Số lượng anchor (ô lưới) tốt nhất được chọn làm "positive" cho mỗi vật thể thật.
  - `TAL_ALPHA`: Trọng số ưu tiên độ tự tin phân lớp (Classification) khi gán nhãn anchor.
  - `TAL_BETA`: Trọng số ưu tiên độ khít của khung (IoU) khi gán nhãn.
- `BACKBONE_NAME`: Tên mạng trích xuất đặc trưng (VD: `resnet50`, `convnext_small`).
- `NECK_NAME`: Tên kiến trúc phần neck của mô hình (VD: `yolov8_pan`).
- `OVERSAMPLE_CLASSES` & `OVERSAMPLE_FACTOR`: Các class bị mất cân bằng (VD: `chair`) sẽ được nhân bản lên nhiều lần trong tập train để mô hình học kỹ hơn.

### 2.2. Cấu hình Dự Đoán
- `CONF_THRESHOLD`: Ngưỡng tự tin mặc định chung cho toàn bộ dự đoán. Dưới mức này hộp bao sẽ bị bỏ qua.
- `CLASS_THRESHOLDS`: Cài đặt ngưỡng tự tin riêng cho từng class. (VD: Nâng cao ngưỡng cho `chair` để giảm thiểu các dự đoán sai - False Positives).
- `NMS_THRESHOLD`: Ngưỡng độ gộp (IoU) trong thuật toán Non-Maximum Suppression. Mức độ cho phép các khung dự đoán được đè lên nhau.
- `MAX_DETECTIONS`: Số lượng hộp bao dự đoán tối đa trả về cho mỗi ảnh.
- `HF_REPO_ID`: Tên định danh kho chứa model trên Hugging Face để tự động fallback nếu không tìm thấy file local.
