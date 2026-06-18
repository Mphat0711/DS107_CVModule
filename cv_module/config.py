from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
WEIGHT_PATH = PROJECT_ROOT / "E04_yolov8s_img416_default_best.pt"

MODEL_NAME = "E04_yolov8s_img416_default"
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
    "normal": "lua binh thuong",
    "asiatic_rice_borer": "sau duc than lua",
    "brown_plant_hopper": "ray nau",
    "paddy_stem_maggot": "ruoi duc non lua",
    "rice_gall_midge": "muoi hanh",
    "rice_leaf_caterpillar": "sau an la lua",
    "rice_leaf_hopper": "ray xanh",
    "rice_leaf_roller": "sau cuon la lua",
    "rice_water_weevil": "mot nuoc hai lua",
    "small_brown_plant_hopper": "ray nau nho",
    "yellow_rice_borer": "sau duc than vang",
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

