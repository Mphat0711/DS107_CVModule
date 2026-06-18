from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WEIGHT_PATH = BASE_DIR / "E04_yolov8s_img416_default_best.pt"

MODEL_NAME = "E04_yolov8s_img416_default"
MODEL_TASK = "detect"
IMG_SIZE = 416
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_DEVICE = "cpu"
MAX_DETECTIONS = 20

CLASS_NAMES = [
    "asiatic_rice_borer",
    "brown_plant_hopper",
    "paddy_stem_maggot",
    "rice_gall_midge",
    "rice_leaf_caterpillar",
    "rice_leaf_hopper",
    "rice_leaf_roller",
    "rice_water_weevil",
    "small_brown_plant_hopper",
    "yellow_rice_borer",
]


def get_model_info():
    return {
        "name": MODEL_NAME,
        "task": MODEL_TASK,
        "imgsz": IMG_SIZE,
        "weight_path": str(WEIGHT_PATH),
        "classes": CLASS_NAMES,
    }


@lru_cache(maxsize=1)
def get_model():
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"Model weight not found: {WEIGHT_PATH}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: ultralytics. Install it with `pip install ultralytics`."
        ) from exc

    return YOLO(str(WEIGHT_PATH))
