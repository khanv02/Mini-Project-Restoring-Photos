# Mini-Project-Restoring-Photos

Ứng dụng **xử lý và khôi phục ảnh cũ tự động** thuộc Học phần **Xử lý ảnh**.

Project tập trung vào việc mô phỏng các tình trạng ảnh bị hư hỏng, áp dụng các thuật toán khôi phục ảnh tiêu chuẩn, đánh giá chất lượng kết quả bằng **PSNR & SSIM** và hỗ trợ xử lý ảnh theo nhiều chế độ thông qua giao diện **PySide6**, bao gồm **Single Image, Compare và Batch Processing**.

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

Ngoài việc xử lý từng ảnh, hệ thống còn hỗ trợ:

- **Single Image:** Khôi phục một ảnh.
- **Compare:** Chạy và so sánh kết quả của các thuật toán.
- **Batch Processing:** Tự động xử lý nhiều ảnh trong một thư mục.

---

## ✨ Chức năng chính

### 1. Giả lập dữ liệu ảnh bị hư hỏng

Module `generate_data.py` được sử dụng để tạo dữ liệu thử nghiệm từ ảnh ban đầu.

Project hỗ trợ mô phỏng **4 loại ảnh hỏng** nhằm phục vụ việc kiểm thử các thuật toán khôi phục.

Dữ liệu được tổ chức thành:

```text
dataset/
├── images_clean/        # Ảnh gốc
├── images_corrupted/    # Ảnh sau khi mô phỏng hư hỏng
└── images_masks/        # Mask vùng hư hỏng
```

Quy trình:

```text
Original Image
      │
      ▼
generate_data.py
      │
      ▼
Damaged / Corrupted Image
      │
      └── Damage Mask
```

Các ảnh được sinh ra phục vụ cho việc kiểm thử và đánh giá các thuật toán khôi phục.

---

### 2. Khôi phục bằng Median Filter

File:

```text
algorithms/median_filter.py
```

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

Median Filter giúp loại bỏ các pixel nhiễu bất thường đồng thời hạn chế làm mất các cạnh quan trọng của ảnh.

---

### 3. Khôi phục bằng Gaussian Filter

File:

```text
algorithms/gaussian_filter.py
```

Sử dụng **Gaussian Filter (bộ lọc Gaussian)** để làm mượt ảnh và giảm nhiễu.

Gaussian Filter sử dụng kernel có phân bố Gaussian, trong đó các pixel gần tâm kernel có trọng số lớn hơn các pixel ở xa.

Thuật toán phù hợp cho:

- Giảm nhiễu Gaussian.
- Làm mượt ảnh.
- Giảm các biến động nhỏ về giá trị pixel.
- Tiền xử lý trước các bước xử lý ảnh khác.

Quy trình:

```text
Input Image
     │
     ▼
Gaussian Kernel
     │
     ▼
Convolution
     │
     ▼
Smoothed / Restored Image
```

---

### 4. Khôi phục vết xước bằng Inpainting

File:

```text
algorithms/inpainting.py
```

Sử dụng kỹ thuật **Image Inpainting** để phục hồi các vùng ảnh bị mất hoặc bị che phủ dựa trên thông tin từ các vùng lân cận.

Có thể sử dụng để xử lý:

- Vết xước.
- Vết rách.
- Vùng ảnh bị mất.
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

## 🖥️ Giao diện người dùng

Project sử dụng **PySide6** để xây dựng giao diện đồ họa.

File giao diện chính:

```text
src/project_1/gui/main_window.py
```

Giao diện được tổ chức thành 3 chế độ chính:

```text
┌─────────────────────────────────────────────┐
│            RESTORING PHOTOS                 │
├──────────────┬──────────────┬───────────────┤
│    Single    │    Compare   │     Batch     │
├──────────────┴──────────────┴───────────────┤
│                                             │
│              Image Preview                  │
│                                             │
├─────────────────────────────────────────────┤
│ Algorithm: [ Median Filter ▼ ]              │
│                                             │
│ PSNR: XX.XX dB       SSIM: X.XXXX           │
│                                             │
│             [ Restore ] [ Save ]             │
└─────────────────────────────────────────────┘
```

### Single

Cho phép người dùng:

1. Chọn một ảnh.
2. Chọn thuật toán khôi phục.
3. Thực hiện khôi phục.
4. Xem ảnh kết quả.
5. Xem các chỉ số PSNR và SSIM.
6. Lưu ảnh kết quả.

---

### Compare

Cho phép chạy nhiều thuật toán trên cùng một ảnh để so sánh.

Ví dụ:

```text
                    Corrupted Image
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          Median        Gaussian     Inpainting
             │             │             │
             ▼             ▼             ▼
          Restored      Restored      Restored
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                      PSNR / SSIM
                           │
                           ▼
                    Compare Results
```

Các chỉ số PSNR và SSIM được sử dụng để đánh giá và so sánh hiệu quả của từng thuật toán.

---

### Batch Processing

Cho phép chọn một thư mục chứa nhiều ảnh và thực hiện khôi phục tự động.

```text
Input Folder
     │
     ├── image_01
     ├── image_02
     ├── image_03
     ├── ...
     └── image_N
            │
            ▼
     Batch Processor
            │
            ▼
    Restoration Pipeline
            │
            ▼
        outputs/
```

Batch Processing được thực hiện bằng `QThread` trong:

```text
gui/threads.py
```

Việc chạy tác vụ nền giúp giao diện không bị treo trong quá trình xử lý nhiều ảnh.

---

## 📊 Đánh giá chất lượng

Module:

```text
metrics/evaluator.py
```

cung cấp các phương pháp đánh giá chất lượng ảnh sau khi khôi phục.

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

Thông thường:

> **PSNR càng cao → ảnh khôi phục càng gần ảnh gốc.**

---

### SSIM

**SSIM (Structural Similarity Index Measure)** đánh giá mức độ tương đồng về cấu trúc giữa hai ảnh.

SSIM xem xét các yếu tố:

- Độ sáng.
- Độ tương phản.
- Cấu trúc hình ảnh.

Giá trị SSIM thường nằm trong khoảng:

```text
-1 → 1
```

Trong đó:

> **SSIM càng gần 1 → hai ảnh càng tương đồng về cấu trúc.**

---

## 📁 Cấu trúc thư mục

```text
Project-1/
│
├── .venv/                         # Môi trường ảo Python
├── .gitignore                    # Ignore dataset/, outputs/, __pycache__
├── .python-version               # Phiên bản Python
├── pyproject.toml                # Project & dependencies
├── README.md                     # Tài liệu dự án
│
├── dataset/                      # Dữ liệu ảnh test
│   ├── images_clean/             # Ảnh sạch gốc
│   ├── images_corrupted/         # Ảnh hỏng
│   └── images_masks/             # Mask vùng hư hỏng
│
├── outputs/                      # Kết quả xuất từ GUI
│
└── src/
    └── project_1/
        ├── __init__.py
        ├── main.py               # Launcher khởi chạy PySide6 GUI
        ├── entrypoint.py         # RestorationPipeline & Controller
        ├── generate_data.py      # Sinh 4 loại ảnh hỏng
        │
        ├── algorithms/           # Logic lõi khôi phục ảnh
        │   ├── __init__.py
        │   ├── base.py           # BaseRestorationAlgorithm
        │   ├── gaussian_filter.py# Gaussian Filter
        │   └── inpainting.py     # Image Inpainting
        │
        ├── metrics/              # Đánh giá chất lượng
        │   ├── __init__.py
        │   └── evaluator.py      # PSNR & SSIM
        │
        ├── gui/                  # Giao diện PySide6
        │   ├── __init__.py
        │   ├── main_window.py    # Cửa sổ chính
        │   └── threads.py        # QThread / Batch Processing
        │
        └── utils/                # Tiện ích bổ trợ
            ├── __init__.py
            └── batch_processor.py# Xử lý ảnh hàng loạt
```

---

## 🧩 Giải thích các module

| Module                     | Vai trò                                                             |
| -------------------------- | ------------------------------------------------------------------- |
| `.venv/`                   | Môi trường ảo Python được tạo bởi `uv`                              |
| `.gitignore`               | Cấu hình các file/thư mục không đưa lên Git                         |
| `.python-version`          | Xác định phiên bản Python sử dụng                                   |
| `pyproject.toml`           | Quản lý project và dependencies                                     |
| `test.ipynb`               | Notebook dùng để test thuật toán, phân tích và tạo hình cho báo cáo |
| `dataset/`                 | Chứa dữ liệu ảnh thử nghiệm                                         |
| `images_clean/`            | Chứa ảnh gốc sạch                                                   |
| `images_corrupted/`        | Chứa ảnh sau khi mô phỏng hư hỏng                                   |
| `images_masks/`            | Chứa mask cho các vùng hư hỏng                                      |
| `outputs/`                 | Chứa các kết quả được xuất từ GUI                                   |
| `main.py`                  | Launcher khởi chạy ứng dụng PySide6                                 |
| `entrypoint.py`            | Controller chính, điều phối Restoration Pipeline                    |
| `generate_data.py`         | Sinh dữ liệu ảnh hỏng                                               |
| `algorithms/base.py`       | Định nghĩa chuẩn chung cho các thuật toán khôi phục                 |
| `gaussian_filter.py`       | Cài đặt Gaussian Filter                                             |
| `inpainting.py`            | Cài đặt Image Inpainting                                            |
| `metrics/evaluator.py`     | Tính toán PSNR và SSIM                                              |
| `gui/main_window.py`       | Cửa sổ chính và các chức năng GUI                                   |
| `gui/threads.py`           | QThread xử lý tác vụ nền                                            |
| `utils/batch_processor.py` | Xử lý nhiều ảnh trong thư mục                                       |

---

## 🛠️ Yêu cầu môi trường

Project sử dụng:

- **Python 3.12**
- **uv** — quản lý môi trường và dependencies.
- **OpenCV** — xử lý ảnh.
- **PySide6** — xây dựng giao diện GUI.
- **scikit-image** — hỗ trợ các phép đo và xử lý ảnh.

Các dependencies được quản lý trong `pyproject.toml`.

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
cd Project-1
```

### 2. Cài đặt dependencies

Nếu project đã có `pyproject.toml` và `uv.lock`:

```bash
uv sync
```

`uv` sẽ tự động tạo virtual environment `.venv` và cài đặt các dependencies cần thiết.

### 3. Kích hoạt virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

---

## ▶️ Chạy chương trình

Ứng dụng GUI được khởi chạy từ:

```text
src/project_1/main.py
```

Sử dụng:

```bash
uv run python -m project_1.main
```

Nếu `pyproject.toml` đã khai báo script tương ứng, có thể chạy bằng command được định nghĩa trong project.

---

## 🧪 Tạo dữ liệu thử nghiệm

File:

```text
src/project_1/generate_data.py
```

được sử dụng để tạo dữ liệu ảnh hỏng.

Chạy:

```bash
uv run python -m project_1.generate_data
```

Dữ liệu sau khi tạo được tổ chức trong:

```text
dataset/
├── images_clean/
├── images_corrupted/
└── images_masks/
```

> Command cụ thể có thể thay đổi tùy theo cách triển khai của `generate_data.py`.

---

## 🔄 Pipeline xử lý

Hệ thống được thiết kế theo pipeline:

### Bước 1 — Input

Nhận ảnh đầu vào:

```text
Input Image
     │
     ▼
```

### Bước 2 — Generate Data

Tạo phiên bản ảnh bị hư hỏng:

```text
Original Image
     │
     ▼
Degradation
     │
     ▼
Damaged Image
```

### Bước 3 — Restoration

Áp dụng thuật toán khôi phục:

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

### Bước 5 — Output

Kết quả được xuất ra:

```text
outputs/
```

---

## 🧩 Kiến trúc thuật toán

Các thuật toán khôi phục được tách thành package riêng:

```text
algorithms/
│
├── base.py
├── median_filter.py
├── gaussian_filter.py
└── inpainting.py
```

`base.py` đóng vai trò định nghĩa chuẩn chung cho các thuật toán khôi phục.

Kiến trúc này giúp các thuật toán có cùng interface và dễ dàng mở rộng trong tương lai.

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

Sau khi chạy pipeline, hệ thống cung cấp:

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

Các chỉ số này được sử dụng để **so sánh hiệu quả giữa các phương pháp khôi phục ảnh**.

---

## 📝 Notebook thử nghiệm

File:

```text
test.ipynb
```

được sử dụng để:

- Test từng thuật toán.
- Thử nghiệm các tham số.
- So sánh kết quả.
- Tính PSNR / SSIM.
- Vẽ biểu đồ.
- Trực quan hóa ảnh trước và sau khôi phục.
- Chụp hình kết quả phục vụ báo cáo.

Notebook phục vụ mục đích thử nghiệm và phân tích, không phải thành phần bắt buộc để chạy ứng dụng GUI.

---

## 🎯 Phạm vi project

Project tập trung vào các kỹ thuật xử lý ảnh truyền thống:

- **Image Degradation**
- **Image Denoising**
- **Image Restoration**
- **Image Inpainting**
- **Image Quality Assessment**
- **Batch Image Processing**
- **Graphical User Interface với PySide6**

Project **không tập trung vào Deep Learning hoặc Generative AI**, mà ưu tiên các phương pháp xử lý ảnh truyền thống để phù hợp với nội dung của học phần **Xử lý ảnh**.

---

## 🚀 Khả năng mở rộng

Kiến trúc hiện tại cho phép bổ sung thêm:

- Các bộ lọc khử nhiễu khác.
- Các thuật toán Image Restoration khác.
- Các phương pháp Inpainting khác.
- Các metric đánh giá khác.
- Visualization nâng cao.
- So sánh tự động nhiều thuật toán.
- Batch Processing nâng cao.
- Các chức năng GUI mới.

Ví dụ:

```text
algorithms/
├── base.py
├── median_filter.py         # Future
├── gaussian_filter.py        
├── inpainting.py 
├── bilateral_filter.py      # Future
├── wiener_filter.py         # Future
└── ...
```

---

## 👨‍💻 Học phần

**Môn học:** Xử lý ảnh

**Project:** Mini-Project – Restoring Photos

Mục tiêu của project là áp dụng kiến thức xử lý ảnh vào một bài toán thực tế:

> **Khôi phục và nâng cao chất lượng ảnh cũ bị hư hỏng bằng các phương pháp xử lý ảnh truyền thống.**
