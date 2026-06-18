import argparse
import json
from pathlib import Path

from cv_module.config import (
    DEFAULT_BBOX_CACHE_DIR,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_INPUT_DIR,
    DEFAULT_IOU,
    DEFAULT_RESULT_PATH,
)
from cv_module.inference import classify_folder, classify_image


def write_json(path, data):
    path = Path(path)
    if not path.is_absolute():
        path = DEFAULT_RESULT_PATH.parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8s rice pest classification and bbox cache."
    )
    parser.add_argument(
        "--image",
        default="",
        help="Single image path. If omitted, --input-dir is used.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing images.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULT_PATH),
        help="Path to save JSON result.",
    )
    parser.add_argument(
        "--bbox-cache-dir",
        default=str(DEFAULT_BBOX_CACHE_DIR),
        help="Folder to cache images with bounding boxes.",
    )
    parser.add_argument(
        "--cache-upload",
        action="store_true",
        help="Copy source image into cache before inference, useful for upload flow.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Optional DS107 session id to include in ds107_payload.",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    if args.image:
        data = classify_image(
            args.image,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            bbox_cache_dir=args.bbox_cache_dir,
            cache_upload=args.cache_upload,
            session_id=args.session_id or None,
        )
    else:
        data = classify_folder(
            args.input_dir,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            bbox_cache_dir=args.bbox_cache_dir,
            cache_upload=args.cache_upload,
            session_id=args.session_id or None,
        )

    output_path = write_json(args.output, data)
    count = len(data) if isinstance(data, list) else 1
    print(f"Processed {count} image(s).")
    print(f"Saved result to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
