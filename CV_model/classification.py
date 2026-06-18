import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from model import (
    BASE_DIR,
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_IOU,
    IMG_SIZE,
    MAX_DETECTIONS,
    get_model,
    get_model_info,
)


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


def load_image(image_path):
    with Image.open(image_path) as img:
        return img.convert("RGB")


def _class_name(result, class_id):
    names = getattr(result, "names", None)
    if isinstance(names, dict) and class_id in names:
        return names[class_id]
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return str(class_id)


def _extract_detections(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    boxes = boxes.cpu()
    detections = []

    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        confidence = float(boxes.conf[index].item())
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[index].tolist()]
        width = x2 - x1
        height = y2 - y1

        detections.append(
            {
                "label": _class_name(result, class_id),
                "class_id": class_id,
                "confidence": round(confidence, 6),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_xywh": [
                    round(x1 + width / 2, 2),
                    round(y1 + height / 2, 2),
                    round(width, 2),
                    round(height, 2),
                ],
                "mask": None,
            }
        )

    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def _resolve_cv_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def draw_bboxes(image_path, detections, output_dir):
    image_path = Path(image_path)
    output_dir = _resolve_cv_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        class_id = detection["class_id"]
        color = BBOX_COLORS[class_id % len(BBOX_COLORS)]
        label = f'{detection["label"]} {detection["confidence"]:.2f}'

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - text_height - 6)

        draw.rectangle(
            [x1, text_y, x1 + text_width + 8, text_y + text_height + 6],
            fill=color,
        )
        draw.text((x1 + 4, text_y + 3), label, fill="white", font=font)

    output_path = output_dir / f"{image_path.stem}_bbox.jpg"
    image.save(output_path, quality=95)
    return output_path


def classify_image(
    image_path,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    device=DEFAULT_DEVICE,
    output_dir=None,
):
    image_path = Path(image_path)
    image = load_image(image_path)
    model = get_model()

    results = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=conf,
        iou=iou,
        device=device,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )

    result = results[0]
    detections = _extract_detections(result)
    top_detection = detections[0] if detections else None
    annotated_image = draw_bboxes(image_path, detections, output_dir) if output_dir else None

    return {
        "source": str(image_path),
        "annotated_image": str(annotated_image) if annotated_image else None,
        "model": get_model_info(),
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "summary": {
            "has_detection": top_detection is not None,
            "predicted_label": top_detection["label"] if top_detection else None,
            "predicted_class_id": top_detection["class_id"] if top_detection else None,
            "predicted_confidence": top_detection["confidence"] if top_detection else None,
            "count": len(detections),
            "segmentation_available": False,
        },
        "detections": detections,
    }


def iter_image_files(input_dir):
    input_dir = _resolve_cv_path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def classify_folder(
    input_dir,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    device=DEFAULT_DEVICE,
    output_dir=None,
):
    return [
        classify_image(
            image_path=image_path,
            conf=conf,
            iou=iou,
            device=device,
            output_dir=output_dir,
        )
        for image_path in iter_image_files(input_dir)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Classify rice pest/disease images from a folder with YOLOv8s."
    )
    parser.add_argument(
        "--input-dir",
        default="input_images",
        help="Folder containing images to classify.",
    )
    parser.add_argument(
        "--output",
        default="classification_results.json",
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--output-dir",
        default="output_images",
        help="Folder to save images with bounding boxes.",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    results = classify_folder(
        input_dir=args.input_dir,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        output_dir=args.output_dir,
    )

    output_path = _resolve_cv_path(args.output)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Processed {len(results)} image(s).")
    print(f"Saved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
