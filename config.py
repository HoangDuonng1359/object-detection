class TrainConfig:
    IMAGE_SIZE = 512
    EPOCHS = 30
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    REG_MAX = 16
    SMALL_OBJECT_MAX_SIDE = 96.0
    MEDIUM_OBJECT_MAX_SIDE = 224.0
    TAL_TOPK = 5
    TAL_ALPHA = 0.5
    TAL_BETA = 4.0
    USE_AMP = True
    DISABLE_TQDM = False
    BACKBONE_NAME = "convnext_small"  # "convnext_small" "resnet50", "resnet101", "yolov8n", "yolov8s", or "yolov8m"
    NECK_NAME = "yolov8_pan"  # "yolov8_pan" or "bifpn"
    USE_PRETRAINED_BACKBONE = True
    FREEZE_BACKBONE_STEM = False
    OVERSAMPLE_CLASSES = ["chair"]
    OVERSAMPLE_FACTOR = 2.0

    MAX_TRAIN_BATCHES = 0
    MAX_VAL_BATCHES = 0

    CONF_THRESHOLD = 0.25
    CLASS_THRESHOLDS = {
        "person": 0.35,
        "car": 0.25,
        "dog": 0.3,
        "cat": 0.3,
        "chair": 0.2,
    }
    NMS_THRESHOLD = 0.35
    MAX_DETECTIONS = 100

    RUN_THRESHOLD_SWEEP = True
    CLASS_THRESHOLD_CONFIGS = [
        {
            "name": "balanced",
            "default": 0.25,
            "thresholds": CLASS_THRESHOLDS,
        },
        {
            "name": "more_recall",
            "default": 0.20,
            "thresholds": {"person": 0.20, "car": 0.15, "dog": 0.20, "cat": 0.20, "chair": 0.08},
        },
        {
            "name": "clean",
            "default": 0.30,
            "thresholds": {"person": 0.30, "car": 0.25, "dog": 0.25, "cat": 0.30, "chair": 0.15},
        },
        {
            "name": "map_oriented",
            "default": 0.15,
            "thresholds": {"person": 0.15, "car": 0.15, "dog": 0.15, "cat": 0.15, "chair": 0.08},
        },
    ]
    NMS_VALUES = [0.35, 0.45, 0.50]


class PredictConfig:
    BATCH_SIZE = 32
    CONF_THRESHOLD = 0.2
    CLASS_THRESHOLDS = {
        "person": 0.1,
        "car": 0.1,
        "dog": 0.1,
        "cat": 0.1,
        "chair": 0.1,
    }
    NMS_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    DEVICE = "cuda"
    HF_REPO_ID = "duowng/yolov8_object_detection"
