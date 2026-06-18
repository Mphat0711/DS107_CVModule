from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from cv_module.confidence import build_confidence_ir
from cv_module.config import (
    CLASS_LABELS_VI,
    CLASS_NAMES,
    DEFAULT_BBOX_CACHE_DIR,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_INPUT_DIR,
    DEFAULT_IOU,
    DEFAULT_UPLOAD_CACHE_DIR,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    MAX_DETECTIONS,
)
from cv_module.ds107_bridge import build_ds107_payload
from cv_module.model import get_model, get_model_info
from cv_module.visualization import draw_bboxes, load_image


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (DEFAULT_INPUT_DIR.parent / path).resolve()


def cache_upload_image(image_path, cache_dir=DEFAULT_UPLOAD_CACHE_DIR):
    image_path = Path(image_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / f"{uuid4().hex}{image_path.suffix.lower()}"
    shutil.copy2(image_path, cached_path)
    return cached_path


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
        x1, y1, x2, y2 = [float(value) for value in boxes.xyxy[index].tolist()]
        width = x2 - x1
        height = y2 - y1
        label = _class_name(result, class_id)

        detections.append(
            {
                "label": label,
                "label_vi": CLASS_LABELS_VI.get(label, label),
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


def classify_image(
    image_path,
    *,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    device=DEFAULT_DEVICE,
    bbox_cache_dir=DEFAULT_BBOX_CACHE_DIR,
    cache_upload=False,
    upload_cache_dir=DEFAULT_UPLOAD_CACHE_DIR,
    include_ds107_payload=True,
    session_id=None,
):
    original_path = resolve_path(image_path)
    source_path = (
        cache_upload_image(original_path, upload_cache_dir)
        if cache_upload
        else original_path
    )
    image = load_image(source_path)
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

    detections = _extract_detections(results[0])
    top_detection = detections[0] if detections else None
    annotated_image = draw_bboxes(source_path, detections, bbox_cache_dir)
    confidence_ir = build_confidence_ir(detections)

    result = {
        "source": str(source_path),
        "original_source": str(original_path),
        "annotated_image": str(annotated_image),
        "model": get_model_info(),
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "summary": {
            "has_detection": top_detection is not None,
            "predicted_label": top_detection["label"] if top_detection else None,
            "predicted_label_vi": top_detection["label_vi"] if top_detection else None,
            "predicted_class_id": top_detection["class_id"] if top_detection else None,
            "predicted_confidence": top_detection["confidence"] if top_detection else None,
            "count": len(detections),
            "segmentation_available": False,
        },
        "confidence_ir": confidence_ir,
        "detections": detections,
    }

    if include_ds107_payload:
        result["ds107_payload"] = build_ds107_payload(result, session_id=session_id)

    return result


def iter_image_files(input_dir=DEFAULT_INPUT_DIR):
    input_dir = Path(input_dir)
    if not input_dir.is_absolute() and input_dir.exists():
        input_dir = input_dir.resolve()
    elif not input_dir.is_absolute():
        input_dir = DEFAULT_INPUT_DIR.parent / input_dir
    input_dir = input_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def classify_folder(
    input_dir=DEFAULT_INPUT_DIR,
    *,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    device=DEFAULT_DEVICE,
    bbox_cache_dir=DEFAULT_BBOX_CACHE_DIR,
    cache_upload=False,
    upload_cache_dir=DEFAULT_UPLOAD_CACHE_DIR,
    include_ds107_payload=True,
    session_id=None,
):
    return [
        classify_image(
            image_path,
            conf=conf,
            iou=iou,
            device=device,
            bbox_cache_dir=bbox_cache_dir,
            cache_upload=cache_upload,
            upload_cache_dir=upload_cache_dir,
            include_ds107_payload=include_ds107_payload,
            session_id=session_id,
        )
        for image_path in iter_image_files(input_dir)
    ]
