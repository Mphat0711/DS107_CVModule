# CV Module - Rice Pest YOLO

Module này đóng gói phần thị giác máy tính cho project RiceDisease.

Nhiệm vụ:

```text
Ảnh đầu vào
-> YOLOv8s detect sâu hại
-> chọn nhãn có confidence cao nhất làm nhãn phân loại chính
-> vẽ bbox vào ảnh cache
-> tạo confidence IR
-> xuất payload nối trực tiếp sang DS107
```

## Cài Đặt

Chạy từ thư mục `CV_model`:

```powershell
python -m pip install -r requirements.txt
```

Yêu cầu file trọng số nằm tại:

```text
E04_yolov8s_img416_default_best.pt
```

## Chạy Một Ảnh

```powershell
python -m cv_module.cli --image input_images\sample.jpg --output classification_results.json --device cpu
```

Kết quả:

- JSON: `classification_results.json`
- Ảnh có bbox: `cache\bbox\*_bbox.jpg`

## Chạy Một Folder Ảnh

```powershell
python -m cv_module.cli --input-dir input_images --output classification_results.json --device cpu
```

## Output Chính

Một result có các phần quan trọng:

```json
{
  "source": "...",
  "annotated_image": "...",
  "summary": {
    "predicted_label": "brown_plant_hopper",
    "predicted_label_vi": "ray nau",
    "predicted_confidence": 0.91
  },
  "confidence_ir": {
    "level": "high",
    "needs_llm_confirmation": false,
    "llm_strategy": "direct_advice"
  },
  "detections": [],
  "ds107_payload": {}
}
```

`annotated_image` là ảnh cache đã vẽ bbox. Đây là ảnh dùng để trả về frontend sau khi user upload.

## Nối Với DS107

CV module sinh sẵn `ds107_payload`, gồm:

- `class_id`: nhãn YOLO, ví dụ `brown_plant_hopper`
- `disease`: tên tiếng Việt tương ứng
- `confidence`: confidence cao nhất
- `image`: ảnh nguồn hoặc ảnh upload cache
- `annotated_image`: ảnh đã vẽ bbox
- `needs_llm_confirmation`: có cần DS107 hỏi thêm vì chưa chắc chắn không
- `notes`: ghi chú confidence IR cho LLM

Tạo payload riêng:

```powershell
python -m cv_module.ds107_bridge --result classification_results.json --session-id ruong-a --output ds107_payload.json
```

Tạo session trong DS107:

```powershell
python -m cv_module.ds107_bridge --result classification_results.json --session-id ruong-a --start-session --ds107-root ..\DS107
```

Sau đó hỏi DS107:

```powershell
cd ..\DS107
python advisor\gemini_rag.py ask "Tôi nên xử lý thế nào?" --session ruong-a --show-sources
```

## Cấu Trúc

```text
CV_model/
├─ E04_yolov8s_img416_default_best.pt  Trọng số YOLO ở root repo
├─ cv_module/
│  ├─ config.py          Cấu hình, class labels, threshold confidence
│  ├─ model.py           Load YOLO một lần
│  ├─ inference.py       Nhận ảnh, predict, xuất result
│  ├─ confidence.py      Confidence IR trung gian
│  ├─ visualization.py   Vẽ bbox vào ảnh cache
│  ├─ cli.py             CLI chạy inference
│  └─ ds107_bridge.py    Payload/lệnh nối DS107
├─ input_images/         Ảnh test local, bị ignore trừ .gitkeep
├─ cache/                Runtime cache, bị ignore trừ .gitkeep
├─ requirements.txt
└─ ARCHITECTURE.md
```

## Ghi Chú

- Model hiện tại là YOLO detection, chưa phải segmentation.
- Nếu không có segmentation, bài toán được quy về classification bằng nhãn detection confidence cao nhất.
- Confidence thấp hoặc ambiguous không chặn DS107, nhưng `confidence_ir.needs_llm_confirmation=true` để LLM tư vấn thận trọng.
