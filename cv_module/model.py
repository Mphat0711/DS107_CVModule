from functools import lru_cache

from cv_module.config import (
    CLASS_LABELS_VI,
    CLASS_NAMES,
    IMG_SIZE,
    MODEL_NAME,
    MODEL_TASK,
    WEIGHT_PATH,
)


def get_model_info():
    return {
        "name": MODEL_NAME,
        "task": MODEL_TASK,
        "imgsz": IMG_SIZE,
        "weight_path": str(WEIGHT_PATH),
        "classes": CLASS_NAMES,
        "class_labels_vi": CLASS_LABELS_VI,
    }


@lru_cache(maxsize=1)
def get_model():
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"Model weight not found: {WEIGHT_PATH}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: ultralytics. Install it with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    return YOLO(str(WEIGHT_PATH))

