# Plan CV module cho RiceDisease

## 1. Mục tiêu bài toán

Người dùng gửi một ảnh lá/cây lúa vào hệ thống. CV module cần:

- Nhận ảnh đầu vào từ backend.
- Phân loại bệnh/tác nhân chính xuất hiện trong ảnh.
- Trả về nhãn dự đoán, độ tin cậy và bằng chứng trực quan nếu có.
- Phân đoạn vùng liên quan nếu có dữ liệu/model segmentation phù hợp; nếu không có phân đoạn thì bài toán được chấp nhận ở mức phân loại.
- Xuất kết quả đủ rõ để frontend hiển thị: ảnh gốc, bounding box, mask/overlay nếu có, và danh sách kết quả.

Luồng mục tiêu:

```text
Frontend upload ảnh
-> Backend nhận file
-> CV module tiền xử lí ảnh
-> YOLO inference
-> Hậu xử lí classification/detection/segmentation
-> Backend trả JSON + ảnh overlay
-> Frontend hiển thị kết quả
```

## 2. Hiện trạng folder `CV_model`

Hiện có:

- `E04_yolov8s_img416_default_best.pt`: trọng số YOLOv8s, input size 416, dung lượng khoảng 22.5 MB.
- `YOLO_rice_desease.ipynb`: notebook train/evaluate YOLO, gồm các cell cài thư viện, tìm dataset YOLO, patch `data.yaml`, kiểm tra label, kiểm tra leakage, train ablation và predict thử.

Thông tin đọc được từ notebook:

- Dataset đang dùng có format YOLO detection, không phải segmentation.
- Task train trong notebook là `task="detect"`.
- Dataset có 10 lớp:
  - `asiatic_rice_borer`
  - `brown_plant_hopper`
  - `paddy_stem_maggot`
  - `rice_gall_midge`
  - `rice_leaf_caterpillar`
  - `rice_leaf_hopper`
  - `rice_leaf_roller`
  - `rice_water_weevil`
  - `small_brown_plant_hopper`
  - `yellow_rice_borer`
- Dataset summary trong notebook:
  - train: 6630 ảnh, 21539 bbox
  - valid: 631 ảnh, 802 bbox
  - test: 315 ảnh, 420 bbox
- Model hiện tại phù hợp nhất với bài toán phát hiện sâu/rầy bằng bounding box.

Nhận xét quan trọng:

- Tên project là RiceDisease nhưng model/notebook hiện thiên về `rice pest detection`, tức phát hiện sâu hại/rầy, chưa phải nhận diện bệnh lá theo nghĩa bệnh học như đạo ôn, cháy bìa lá, đốm nâu.
- File `.pt` hiện tại là detection model, chưa có output mask. Vì vậy yêu cầu "phân đoạn" chưa thể làm bằng model hiện có nếu hiểu là segmentation mask thật.
- Nếu không triển khai phân đoạn, bài toán sẽ được quy về phân loại ảnh: chọn nhãn có confidence cao nhất trong các detection làm nhãn chính của ảnh.
- Bounding box được dùng như bằng chứng trực quan cho kết quả phân loại, không bắt buộc là output chính.

## 3. Định nghĩa phạm vi cho CV module

### MVP có thể làm ngay

Sử dụng `E04_yolov8s_img416_default_best.pt` để inference detection:

- Input: ảnh `.jpg`, `.jpeg`, `.png`.
- Resize/letterbox theo chuẩn Ultralytics YOLO với `imgsz=416`.
- Predict với ngưỡng mặc định:
  - `conf=0.25`
  - `iou=0.7`
  - `max_det=20`
- Output:
  - nhãn phân loại chính của ảnh, lấy từ detection có confidence cao nhất
  - danh sách detection
  - nhãn
  - confidence
  - bounding box theo pixel ảnh gốc
  - ảnh overlay có vẽ bbox
  - cờ `segmentation_available=false`

### Bản mở rộng có segmentation

Cần chọn một trong các hướng sau:

1. Train YOLO segmentation model:
   - Cần dataset có polygon/mask annotation.
   - Dùng model dạng `yolov8s-seg.pt` hoặc model seg tương đương.
   - Output có mask thật cho từng object.

2. Kết hợp detection + SAM/SAM2:
   - YOLO detect bbox.
   - SAM dùng bbox làm prompt để sinh mask.
   - Phù hợp demo nhanh nếu không có mask label, nhưng inference nặng hơn.

3. Pseudo-segmentation đơn giản:
   - Crop vùng bbox.
   - Dùng threshold/GrabCut để tách vùng nghi bệnh/tác nhân.
   - Chỉ nên dùng như minh họa, không nên báo cáo là segmentation chuẩn nếu chưa đánh giá.

Khuyến nghị cho môn học:

- Giai đoạn 1: hoàn thiện classification end-to-end từ YOLO detection, với bbox làm bằng chứng.
- Giai đoạn 2: nếu muốn nâng cấp từ phân loại sang phát hiện chi tiết, hiển thị đầy đủ nhiều bbox/object.
- Giai đoạn 3: nếu yêu cầu bắt buộc có "phân đoạn", dùng detection + SAM hoặc train YOLO-seg nếu có dataset mask.

## 4. Kiến trúc đề xuất trong repo

Giai đoạn hiện tại chia CV module thành 2 file Python chính:

```text
CV_model/
├─ E04_yolov8s_img416_default_best.pt
├─ model.py
├─ classification.py
├─ input_images/
│  └─ .gitkeep
├─ requirements.txt
├─ YOLO_rice_desease.ipynb
└─ plan.md
```

Vai trò:

- `model.py`: định nghĩa cấu hình mô hình, danh sách nhãn, đường dẫn trọng số, thông số inference mặc định và hàm `get_model()` để load YOLO một lần bằng cache.
- `classification.py`: nhận ảnh từ folder phụ, tiền xử lí ảnh, import model từ `model.py`, chạy inference, chọn detection confidence cao nhất làm nhãn phân loại chính và xuất JSON.
- `input_images/`: nơi đặt ảnh test local trước khi tích hợp backend.

Sau khi module ổn, có thể tách tiếp thành package `src/` nếu cần mở rộng backend/production.

## 5. API nội bộ của CV module

Hàm chính trong `classification.py`:

```python
classify_image(
    image_path: str | Path,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "cpu",
) -> dict
```

Hàm chạy theo folder:

```python
classify_folder(
    input_dir: str | Path,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "cpu",
) -> list[dict]
```

Schema đề xuất:

```json
{
  "model": {
    "name": "E04_yolov8s_img416_default",
    "task": "detect",
    "imgsz": 416
  },
  "image": {
    "width": 1280,
    "height": 720
  },
  "summary": {
    "has_detection": true,
    "predicted_label": "rice_leaf_roller",
    "predicted_confidence": 0.91,
    "top_label": "rice_leaf_roller",
    "top_confidence": 0.91,
    "count": 2,
    "segmentation_available": false
  },
  "detections": [
    {
      "label": "rice_leaf_roller",
      "class_id": 6,
      "confidence": 0.91,
      "bbox_xyxy": [120, 80, 420, 360],
      "bbox_xywh": [270, 220, 300, 280],
      "mask": null
    }
  ],
  "overlay_image_base64": "..."
}
```

Khi không phát hiện:

```json
{
  "summary": {
    "has_detection": false,
    "predicted_label": null,
    "predicted_confidence": null,
    "top_label": null,
    "top_confidence": null,
    "count": 0,
    "segmentation_available": false
  },
  "detections": []
}
```

## 6. Kết nối với backend

Backend FastAPI hiện có route `/predict` nhưng đang trả hard-code. Cần đổi route này để gọi CV module.

Endpoint đề xuất:

```text
POST /predict
Content-Type: multipart/form-data
field: file
```

Pseudo-flow:

```python
@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    # Giai đoạn đầu có thể lưu tạm ảnh rồi gọi classification.classify_image(path).
    # Giai đoạn sau nên bổ sung hàm nhận image_bytes trực tiếp để tránh ghi file tạm.
    result = classify_uploaded_image(image_bytes)
    return result
```

Các việc cần làm ở backend:

- Tạo `requirements.txt` riêng cho backend/CV:
  - `fastapi`
  - `uvicorn`
  - `python-multipart`
  - `ultralytics`
  - `torch`
  - `pillow`
  - `opencv-python`
  - `numpy`
- Load model một lần khi app start, không load lại mỗi request.
- Validate file upload:
  - đúng định dạng ảnh
  - giới hạn dung lượng, ví dụ 10 MB
  - xử lý lỗi ảnh hỏng
- Chuẩn hóa response lỗi:
  - `400`: file không hợp lệ
  - `500`: lỗi inference

## 7. Tiền xử lí ảnh

Cần đảm bảo:

- Đọc ảnh bằng Pillow/OpenCV.
- Convert về RGB.
- Giữ ảnh gốc để trả bbox theo kích thước gốc.
- Không tự resize thủ công nếu dùng Ultralytics vì YOLO đã xử lý letterbox.
- Chặn ảnh quá lớn nếu cần để tránh quá tải RAM.

Quy tắc input:

- Cho phép: `.jpg`, `.jpeg`, `.png`.
- Không cho phép: file không phải ảnh, ảnh rỗng, ảnh quá lớn.
- Nếu ảnh không có lá/cây lúa, model vẫn có thể dự đoán sai; cần hiển thị confidence và có ngưỡng `no_detection`.

## 8. Hậu xử lí kết quả

Classification:

- Nếu có ít nhất một detection, chọn detection có `confidence` cao nhất làm nhãn phân loại chính của ảnh.
- Trả `predicted_label`, `predicted_confidence` và `predicted_class_id` trong phần `summary`.
- Nếu không có detection vượt ngưỡng, trả `predicted_label=null` và trạng thái không phát hiện.
- Có thể ánh xạ nhãn tiếng Anh sang tên tiếng Việt để hiển thị trong frontend/báo cáo.

Detection:

- Lấy `boxes.xyxy`, `boxes.conf`, `boxes.cls`.
- Map `class_id` sang tên lớp.
- Sort theo confidence giảm dần.
- Có thể gom summary theo lớp:
  - lớp xuất hiện nhiều nhất
  - lớp confidence cao nhất
  - số lượng object theo từng lớp

Segmentation:

- Nếu model hiện tại không có `result.masks`, trả:
  - `segmentation_available=false`
  - `mask=null`
- Nếu sau này có YOLO-seg/SAM:
  - Trả polygon hoặc mask RLE/base64.
  - Vẽ overlay bán trong suốt trên ảnh.
  - Thêm metric diện tích mask, ví dụ `% diện tích ảnh`.

## 9. Đánh giá model

Metric detection nên giữ từ notebook:

- Precision
- Recall
- F1
- mAP50
- mAP50-95
- Per-class metric

Vì app nông nghiệp nên ưu tiên:

1. Accuracy/F1 ở mức phân loại ảnh nếu báo cáo theo bài toán classification.
2. Recall đủ cao để giảm bỏ sót sâu/bệnh.
3. F1 tốt để cân bằng false positive và false negative.
4. mAP50-95 tốt nếu vẫn đánh giá phần bbox/detection.

Ngưỡng inference đề xuất:

- Demo cân bằng: `conf=0.25`, `iou=0.7`.
- Muốn giảm bỏ sót: `conf=0.15`, `iou=0.7`.
- Muốn ít báo nhầm hơn: `conf=0.35` hoặc `0.4`.

Cần test riêng trên ảnh người dùng tự chụp vì dataset Roboflow có thể khác ánh sáng/góc chụp thực tế.

## 10. Kế hoạch triển khai

### Phase 1: Chuẩn hóa artifact

- Giữ weight hiện tại ở `CV_model/E04_yolov8s_img416_default_best.pt` trong giai đoạn đầu để code đơn giản.
- Tạo `requirements.txt`.
- Nếu cần hiển thị đẹp hơn, bổ sung mapping tên tiếng Việt cho 10 nhãn trong `model.py` hoặc file cấu hình riêng.
- Ghi rõ model card:
  - task: detect
  - imgsz: 416
  - dataset: rice pest YOLO
  - giới hạn: chưa có segmentation mask thật.

### Phase 2: Viết inference module

- Tạo `CV_model/model.py`.
- Tạo `CV_model/classification.py`.
- Load YOLO model một lần.
- Viết hàm `classify_image` và `classify_folder`.
- Trả JSON serializable.
- Chọn nhãn chính của ảnh từ detection có confidence cao nhất.
- Tạo ảnh overlay bbox bằng OpenCV/Pillow.
- Viết xử lý no-detection.

### Phase 3: Tích hợp backend

- Sửa `backend/main.py` để gọi predictor.
- Thêm route health check:
  - `GET /health`
  - `GET /model-info`
- Thêm CORS phù hợp frontend local.
- Kiểm thử bằng `curl` hoặc Postman:
  - ảnh hợp lệ
  - ảnh không có object
  - file không phải ảnh

### Phase 4: Chuẩn bị segmentation

- Xác nhận yêu cầu "phân đoạn" là:
  - phân đoạn object sâu/rầy
  - phân đoạn vùng bệnh trên lá
  - hay chỉ khoanh vùng bbox.
- Nếu không bắt buộc phân đoạn, giữ scope chính là classification và chỉ dùng bbox làm giải thích trực quan.
- Nếu cần mask thật:
  - tìm/chuẩn bị dataset segmentation.
  - train YOLOv8-seg input 416.
  - thay predictor để đọc `result.masks`.
- Nếu cần demo nhanh:
  - dùng YOLO bbox làm prompt cho SAM.
  - ghi rõ đây là mask suy ra từ bbox, không phải model bệnh được train trực tiếp.

### Phase 5: Tích hợp frontend sau

- Frontend upload ảnh qua `FormData`.
- Hiển thị:
  - ảnh preview
  - overlay bbox/mask
  - danh sách nhãn + confidence
  - khuyến nghị xử lý theo nhãn.
- Tách phần khuyến nghị nông nghiệp sang module LLM/rule-based riêng, không để CV tự sinh lời khuyên.

## 11. Rủi ro và điểm cần làm rõ

- Model hiện tại detect sâu hại, không detect các bệnh lá phổ biến như đạo ôn/đốm nâu nếu các lớp đó không có trong dataset.
- Notebook đặt tên `desease` nhưng nhãn dataset là pest; cần thống nhất thuật ngữ trong báo cáo.
- Chưa có segmentation annotation, nên nếu không làm phân đoạn thì cần mô tả bài toán là phân loại/nhận diện tác nhân chính trong ảnh.
- Weight `.pt` không đủ để biết metric cuối cùng nếu không lưu CSV/result; nên cần giữ thêm `ablation_summary.csv` hoặc model card.
- `backend/venv` hiện không nên commit/dùng lại vì môi trường ảo dễ hỏng khi chuyển máy.

## 12. Deliverable mong muốn

Sau khi hoàn thành CV module, repo nên có:

- Code inference chạy được local.
- API `/predict` trả kết quả phân loại từ model thật.
- Ảnh overlay bbox/mask nếu có.
- Tài liệu model card ngắn.
- Test tối thiểu cho input ảnh hợp lệ và không hợp lệ.
- Demo end-to-end: upload ảnh từ frontend và thấy kết quả thật từ YOLO.

## 13. Checklist nhanh

- [ ] Chuẩn hóa vị trí weight.
- [ ] Bổ sung tên tiếng Việt cho nhãn nếu cần.
- [x] Tạo `requirements.txt`.
- [x] Viết `model.py`.
- [x] Viết `classification.py`.
- [ ] Test inference bằng một ảnh local.
- [ ] Sửa backend `/predict`.
- [ ] Test API upload file.
- [ ] Quyết định chiến lược segmentation.
- [ ] Tích hợp frontend sau khi backend ổn.
