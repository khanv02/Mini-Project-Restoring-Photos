# Mini-Project-Restoring-Photos

Ứng dụng **xử lý và khôi phục ảnh cũ tự động** thuộc Học phần **Xử lý ảnh**.

Project tập trung vào việc mô phỏng các tình trạng ảnh bị hư hỏng, áp dụng các thuật toán khôi phục ảnh tiêu chuẩn, đánh giá chất lượng kết quả và hỗ trợ xử lý nhiều ảnh tự động thông qua **Batch Processing**.

---

## 📌 Mục tiêu dự án

Project được xây dựng nhằm minh họa quy trình cơ bản của một hệ thống khôi phục ảnh:

```text
Ảnh gốc
   │
   ▼
Giả lập ảnh bị hư hỏng
   │
   ▼
Ảnh degraded / damaged
   │
   ▼
Thuật toán khôi phục
   │
   ├── Median Filter
   ├── Gaussian Filter
   └── Inpainting
   │
   ▼
Ảnh sau khôi phục
   │
   ▼
Đánh giá chất lượng
   ├── PSNR
   └── SSIM
```

Ngoài việc xử lý từng ảnh, hệ thống còn hỗ trợ **Batch Processing** để tự động thực hiện pipeline trên nhiều ảnh.

---

## ✨ Chức năng chính

### 1. Giả lập dữ liệu ảnh bị hư hỏng

Module `generate_data.py` được sử dụng để tạo dữ liệu thử nghiệm từ ảnh ban đầu.

Các dạng hư hỏng có thể được mô phỏng tùy theo cách triển khai, chẳng hạn:

- Nhiễu ảnh.
- Nhiễu Salt-and-Pepper.
- Làm mờ ảnh.
- Vết xước hoặc vùng ảnh bị mất.
- Các dạng degradation khác phục vụ quá trình thử nghiệm.

Mục đích là tạo ra một cặp dữ liệu:

```text
Original Image
      │
      ▼
Damaged Image
```

để có thể kiểm tra khả năng khôi phục của từng thuật toán.

---

### 2. Khôi phục bằng Median Filter

`median_filter.py`

Sử dụng **Median Filter (bộ lọc trung vị)** để giảm nhiễu, đặc biệt phù hợp với các loại nhiễu dạng xung như **Salt-and-Pepper Noise**.

Ý tưởng cơ bản:

```text
Pixel hiện tại
      │
      ▼
Lấy các pixel trong vùng lân cận
      │
      ▼
Sắp xếp giá trị
      │
      ▼
Chọn giá trị trung vị
      │
      ▼
Pixel sau lọc
```

---

### 3. Khôi phục bằng Gaussian Filter

`gaussian_filter.py`

Sử dụng **Gaussian Filter (bộ lọc Gaussian)** để làm mượt ảnh và giảm nhiễu.

Gaussian Filter sử dụng kernel có phân bố Gaussian, trong đó các pixel gần tâm kernel có trọng số lớn hơn các pixel ở xa.

Thuật toán phù hợp cho:

- Giảm nhiễu Gaussian.
- Làm mượt ảnh.
- Tiền xử lý trước các bước khôi phục khác.

---

### 4. Khôi phục vết xước bằng Inpainting

`inpainting.py`

Sử dụng kỹ thuật **Image Inpainting** để phục hồi các vùng ảnh bị mất hoặc bị che phủ, ví dụ:

- Vết xước.
- Vết rách.
- Pixel bị mất.
- Các vùng nhỏ cần tái tạo.

Quy trình tổng quát:

```text
Ảnh bị hư hỏng
      │
      ├── Damaged Image
      │
      └── Damage Mask
              │
              ▼
          Inpainting
              │
              ▼
      Ảnh được khôi phục
```

---

## 📊 Đánh giá chất lượng

Module `metrics/evaluator.py` cung cấp các phương pháp đánh giá chất lượng ảnh sau khi khôi phục.

### PSNR

**PSNR (Peak Signal-to-Noise Ratio)** đo mức độ sai khác giữa ảnh gốc và ảnh được khôi phục.

Đơn vị:

```text
dB
```

Công thức:

$$
PSNR = 10 \log_{10}\left(\frac{MAX_I^2}{MSE}\right)
$$

Trong đó:

- `MAX_I`: Giá trị pixel lớn nhất.
- `MSE`: Mean Squared Error.

Thông thường, **PSNR càng cao thì ảnh khôi phục càng gần ảnh gốc**.

---

### SSIM

**SSIM (Structural Similarity Index Measure)** đánh giá mức độ tương đồng về cấu trúc giữa hai ảnh.

SSIM xem xét các yếu tố như:

- Độ sáng.
- Độ tương phản.
- Cấu trúc hình ảnh.

Giá trị SSIM thường nằm trong khoảng:

```text
-1 → 1
```

Trong đó giá trị càng gần `1` thể hiện hai ảnh càng tương đồng về cấu trúc.

---

## 📁 Cấu trúc thư mục

```text
Mini-Project-Restoring-Photos/
│
├── src/
│   └── project_1/
│       ├── __init__.py
│       ├── entrypoint.py
│       ├── generate_data.py
│       │
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── median_filter.py
│       │   ├── gaussian_filter.py
│       │   └── inpainting.py
│       │
│       ├── metrics/
│       │   ├── __init__.py
│       │   └── evaluator.py
│       │
│       └── utils/
│           ├── __init__.py
│           └── batch_processor.py
│
├── pyproject.toml
├── uv.lock
└── README.md
```

### Giải thích các module

| Module                     | Vai trò                                                  |
| -------------------------- | -------------------------------------------------------- |
| `entrypoint.py`            | Entrypoint chính, điều phối toàn bộ pipeline             |
| `generate_data.py`         | Tạo dữ liệu ảnh thử nghiệm và mô phỏng ảnh hư hỏng       |
| `algorithms/base.py`       | Định nghĩa interface/base class chung cho các thuật toán |
| `median_filter.py`         | Cài đặt Median Filter                                    |
| `gaussian_filter.py`       | Cài đặt Gaussian Filter                                  |
| `inpainting.py`            | Cài đặt thuật toán Inpainting                            |
| `metrics/evaluator.py`     | Tính toán PSNR và SSIM                                   |
| `utils/batch_processor.py` | Xử lý hàng loạt nhiều ảnh                                |
| `__init__.py`              | Khai báo Python package                                  |

---

## 🛠️ Yêu cầu môi trường

Project sử dụng:

- **Python >= 3.14**
- **uv** — quản lý môi trường và dependency.

Kiểm tra phiên bản:

```bash
python --version
uv --version
```

---

## ⚙️ Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd Mini-Project-Restoring-Photos
```

### 2. Tạo môi trường và cài dependency

Nếu project đã có `pyproject.toml` và `uv.lock`:

```bash
uv sync
```

`uv` sẽ tự động tạo virtual environment và cài đặt các dependency được khai báo trong project.

### 3. Kích hoạt virtual environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

---

## ▶️ Chạy chương trình

Entrypoint chính của project nằm tại:

```text
src/project_1/entrypoint.py
```

Có thể chạy project thông qua:

```bash
uv run python -m project_1.entrypoint
```

Hoặc nếu project khai báo script trong `pyproject.toml`, có thể sử dụng command tương ứng:

```bash
uv run <command>
```

> Command chạy chính xác phụ thuộc vào cấu hình `pyproject.toml`.

---

## 🔄 Pipeline xử lý

Hệ thống được thiết kế theo pipeline:

### Bước 1 — Input

Nhận ảnh đầu vào từ thư mục dữ liệu.

```text
Input Images
     │
     ▼
```

### Bước 2 — Generate Data

Tạo phiên bản ảnh bị hư hỏng để phục vụ thử nghiệm.

```text
Original
   │
   ▼
Degradation
   │
   ▼
Damaged Image
```

### Bước 3 — Restoration

Áp dụng một trong các thuật toán:

```text
Damaged Image
      │
      ├── Median Filter
      │
      ├── Gaussian Filter
      │
      └── Inpainting
```

### Bước 4 — Evaluation

So sánh ảnh sau khôi phục với ảnh gốc:

```text
Original Image ───────┐
                      ├──► PSNR
Restored Image ───────┤
                      └──► SSIM
```

### Bước 5 — Batch Processing

Pipeline có thể được áp dụng tự động cho nhiều ảnh:

```text
Image 1 ──► Restore ──► Evaluate
Image 2 ──► Restore ──► Evaluate
Image 3 ──► Restore ──► Evaluate
...
Image N ──► Restore ──► Evaluate
```

---

## 🧩 Kiến trúc thuật toán

Các thuật toán khôi phục được tách thành package riêng:

```text
algorithms/
│
├── base.py
│
├── median_filter.py
├── gaussian_filter.py
└── inpainting.py
```

`base.py` đóng vai trò định nghĩa chuẩn chung cho các thuật toán.

Điều này giúp hệ thống dễ dàng mở rộng thêm các phương pháp khác trong tương lai mà không cần thay đổi nhiều ở pipeline chính.

Ví dụ:

```text
BaseRestorationAlgorithm
          │
          ├── MedianFilter
          ├── GaussianFilter
          ├── Inpainting
          │
          └── Future Algorithm
```

---

## 📈 Kết quả đầu ra

Sau khi chạy pipeline, hệ thống hướng tới việc cung cấp:

- Ảnh gốc.
- Ảnh bị hư hỏng.
- Ảnh sau khôi phục.
- PSNR.
- SSIM.
- Kết quả theo từng thuật toán.
- Kết quả xử lý hàng loạt.

Ví dụ bảng đánh giá:

| Algorithm       | PSNR (dB) | SSIM |
| --------------- | --------: | ---: |
| Median Filter   |       ... |  ... |
| Gaussian Filter |       ... |  ... |
| Inpainting      |       ... |  ... |

Các chỉ số này có thể được sử dụng để **so sánh hiệu quả giữa các phương pháp khôi phục ảnh**.

---

## 🎯 Phạm vi project

Project tập trung vào việc minh họa các kỹ thuật xử lý ảnh kinh điển, bao gồm:

- Image Degradation.
- Image Denoising.
- Image Restoration.
- Image Inpainting.
- Image Quality Assessment.
- Batch Image Processing.

Project **không tập trung vào các mô hình Deep Learning hoặc Generative AI** mà ưu tiên các phương pháp xử lý ảnh truyền thống để phù hợp với mục tiêu của học phần **Xử lý ảnh**.

---

## 🚀 Khả năng mở rộng

Kiến trúc hiện tại cho phép bổ sung thêm:

- Các bộ lọc khác.
- Các thuật toán khử nhiễu khác.
- Các phương pháp Inpainting khác.
- Các metric đánh giá khác.
- GUI/Web interface.
- Visualization kết quả.
- So sánh tự động nhiều thuật toán.
- Batch Processing nâng cao.

Ví dụ:

```text
algorithms/
├── base.py
├── median_filter.py
├── gaussian_filter.py
├── inpainting.py
├── bilateral_filter.py       # Future
├── wiener_filter.py          # Future
└── ...
```

---

## 👨‍💻 Học phần

**Môn học:** Xử lý ảnh
**Project:** Mini-Project – Restoring Photos

Mục tiêu của project là áp dụng kiến thức xử lý ảnh vào một bài toán thực tế: **khôi phục và nâng cao chất lượng ảnh cũ bị hư hỏng**.
