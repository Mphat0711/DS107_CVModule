from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from cv_module.config import BBOX_COLORS, DEFAULT_BBOX_CACHE_DIR


def load_image(image_path):
    with Image.open(image_path) as image:
        return image.convert("RGB")


def draw_bboxes(image_path, detections, output_dir=None):
    image_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir else DEFAULT_BBOX_CACHE_DIR
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

