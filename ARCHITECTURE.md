# Kiến Trúc CV Module Và Ghép Nối DS107

## 1. Vai Trò Của CV Module

CV module là lớp đầu vào cho hệ thống:

```text
Người dùng upload ảnh
-> CV module nhận ảnh
-> YOLOv8s dự đoán sâu hại
-> CV module trả nhãn + ảnh bbox + confidence IR
-> DS107 dùng nhãn đó để tư vấn bằng RAG + Gemini
```

CV module không tự sinh lời khuyên nông nghiệp. Phần lời khuyên thuộc DS107.

## 2. Luồng Dữ Liệu

```text
image_path / uploaded image
        |
        v
cv_module.inference.classify_image()
        |
        |-- load YOLO một lần qua cv_module.model.get_model()
        |-- predict với imgsz=416
        |-- trích boxes/conf/class
        |-- sort detection theo confidence
        |-- chọn detection đầu tiên làm predicted_label
        |-- vẽ bbox vào cache/bbox
        |-- tạo confidence_ir
        |-- tạo ds107_payload
        v
CVResult dict
```

## 3. Các Khối Python

### `cv_module/config.py`

Chứa cấu hình tập trung:

- đường dẫn weight
- input size
- device mặc định
- class labels
- mapping tiếng Việt đồng bộ với DS107
- threshold confidence
- thư mục cache

### `cv_module/model.py`

Chỉ phụ trách model:

- kiểm tra file `.pt`
- import Ultralytics
- load YOLO bằng `@lru_cache(maxsize=1)`

Điểm quan trọng: model chỉ load một lần trong process backend/CLI.

### `cv_module/inference.py`

Khối xử lý chính:

- nhận ảnh
- optional cache ảnh upload
- chạy YOLO
- tạo `detections`
- tạo `summary`
- gọi vẽ bbox
- gọi confidence IR
- gắn `ds107_payload`

### `cv_module/confidence.py`

Tạo IR trung gian để quyết định cách nói chuyện với LLM:

```json
{
  "level": "high | medium | ambiguous | low | no_detection",
  "top_confidence": 0.91,
  "second_confidence": 0.83,
  "confidence_margin": 0.08,
  "needs_llm_confirmation": true,
  "llm_strategy": "cautious_advice_with_confirmation_question",
  "reason": "..."
}
```

Ý nghĩa:

- `high`: có thể tư vấn trực tiếp qua DS107.
- `medium`: vẫn nối DS107, nhưng LLM nên nói thận trọng.
- `ambiguous`: top-1 và top-2 gần nhau, nên hỏi thêm triệu chứng/ảnh/giai đoạn lúa.
- `low`: không nên kết luận chắc.
- `no_detection`: không phát hiện sâu hại đủ ngưỡng.

### `cv_module/visualization.py`

Vẽ bbox từ `bbox_xyxy` lên ảnh và lưu vào:

```text
cache/bbox/
```

Ảnh này là runtime cache để frontend/backend trả về sau upload.

### `cv_module/ds107_bridge.py`

Chuyển output CV thành payload DS107:

```json
{
  "session_id": "ruong-a",
  "source": "cv_yolo",
  "class_id": "brown_plant_hopper",
  "disease": "ray nau",
  "confidence": 0.91,
  "image": "...",
  "annotated_image": "...",
  "needs_llm_confirmation": false,
  "llm_strategy": "direct_advice",
  "notes": "..."
}
```

Bridge cũng có thể gọi trực tiếp DS107 CLI để tạo session.

### `cv_module/cli.py`

CLI chạy inference cho một ảnh hoặc một folder ảnh:

```powershell
python -m cv_module.cli --image input_images\sample.jpg --output classification_results.json
```

## 4. Contract Output Cho Backend

Backend nên gọi:

```python
from cv_module.inference import classify_image

result = classify_image(
    image_path,
    cache_upload=True,
    session_id="ruong-a",
)
```

Backend trả về frontend:

- `summary.predicted_label`
- `summary.predicted_label_vi`
- `summary.predicted_confidence`
- `annotated_image`
- `confidence_ir`
- `detections`

Backend truyền sang DS107:

- `result["ds107_payload"]`

## 5. Ghép Với DS107

DS107 nhận class qua lệnh:

```powershell
python advisor\gemini_rag.py sessions start ruong-a --class-id brown_plant_hopper --confidence 0.91
```

CV module có sẵn bridge:

```powershell
python -m cv_module.ds107_bridge --result classification_results.json --session-id ruong-a --start-session --ds107-root ..\DS107
```

Sau đó:

```powershell
cd ..\DS107
python advisor\gemini_rag.py ask "Tôi nên xử lý thế nào?" --session ruong-a --show-sources
```

## 6. Cách LLM Dùng Confidence IR

Khi `needs_llm_confirmation=false`:

- DS107 có thể tư vấn trực tiếp dựa trên class YOLO.
- Vẫn nên citation bằng RAG sources.

Khi `needs_llm_confirmation=true`:

- DS107 vẫn nhận class dự đoán tốt nhất.
- Prompt/session notes báo LLM rằng kết quả chưa chắc.
- LLM nên hỏi thêm hoặc dùng câu chữ như "nếu đúng là..." thay vì khẳng định tuyệt đối.

## 7. Quy Ước Repo Khi Up GitHub

Nên commit:

- `cv_module/`
- `README.md`
- `ARCHITECTURE.md`
- `requirements.txt`
- `input_images/.gitkeep`
- `cache/.gitkeep`, `cache/uploads/.gitkeep`, `cache/bbox/.gitkeep`
- file weight `.pt` nếu nhóm chấp nhận lưu model 22 MB trong repo

Không nên commit:

- ảnh upload thật
- ảnh bbox cache
- `classification_results.json`
- `__pycache__/`
