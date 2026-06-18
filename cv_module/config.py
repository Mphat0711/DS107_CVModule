from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
WEIGHT_PATH = PROJECT_ROOT / "E08_yolov11s_img416_default_best.pt"

MODEL_NAME = "E08_yolov11s_img416_default"
MODEL_TASK = "detect"
IMG_SIZE = 416
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_DEVICE = "cpu"
MAX_DETECTIONS = 20

DEFAULT_INPUT_DIR = PROJECT_ROOT / "input_images"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_UPLOAD_CACHE_DIR = DEFAULT_CACHE_DIR / "uploads"
DEFAULT_BBOX_CACHE_DIR = DEFAULT_CACHE_DIR / "bbox"
DEFAULT_RESULT_PATH = PROJECT_ROOT / "classification_results.json"

HIGH_CONFIDENCE = 0.75
MEDIUM_CONFIDENCE = 0.45
MIN_CONFIDENCE_MARGIN = 0.08

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

CLASS_LABELS_VI = {
    "normal": "lúa bình thường",
    "asiatic_rice_borer": "sâu đục thân lúa",
    "brown_plant_hopper": "rầy nâu",
    "paddy_stem_maggot": "ruồi đục nõn lúa",
    "rice_gall_midge": "muỗi hành",
    "rice_leaf_caterpillar": "sâu ăn lá lúa",
    "rice_leaf_hopper": "rầy xanh",
    "rice_leaf_roller": "sâu cuốn lá lúa",
    "rice_water_weevil": "mọt nước hại lúa",
    "small_brown_plant_hopper": "rầy nâu nhỏ",
    "yellow_rice_borer": "sâu đục thân vàng",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BBOX_COLORS = [
    "#e53935",
    "#43a047",
    "#1e88e5",
    "#fdd835",
    "#8e24aa",
    "#fb8c00",
    "#00acc1",
    "#6d4c41",
    "#3949ab",
    "#d81b60",
]

