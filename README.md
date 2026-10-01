# Khôi phục ảnh cũ — Subject 2 / Project 1

Mini-project xử lý ảnh bằng Python và OpenCV, có giao diện desktop để giảm nhiễu, phục hồi vùng hư hỏng và so sánh chất lượng trước–sau. Phần mềm dùng các thuật toán xử lý ảnh truyền thống, không dùng mô hình AI.

## Mục lục

- [Giới thiệu và tính năng](#tong-quan)
- [Công nghệ](#cong-nghe)
- [Cấu trúc thư mục](#cau-truc)
- [Cài đặt và chạy nhanh](#cai-dat)
- [Chuẩn bị dữ liệu và chạy demo](#du-lieu)
- [Hướng dẫn sử dụng](#su-dung)
- [Thuật toán và thông số](#thuat-toan)
- [Đánh giá, kiểm thử và build](#kiem-thu)
- [API, lỗi thường gặp và giới hạn](#api)

<a id="tong-quan"></a>

## Giới thiệu và tính năng

Project tập trung vào **ảnh tĩnh** với ba chế độ: khôi phục đơn (Single), so sánh phương pháp (Compare) và xử lý hàng loạt (Batch).

- Gaussian Filter giảm nhiễu; Inpainting hỗ trợ Telea/Navier–Stokes; Combined kết hợp hai bước.
- Hiển thị ảnh trước–sau, lưu ảnh và xuất báo cáo CSV.
- Thanh kéo kèm ô nhập số để chỉnh thông số chính xác.
- Tăng nét tùy chọn sau phục hồi, mặc định tắt.
- Gợi ý mask vết xước sáng, xem lớp phủ và xác nhận trước khi sử dụng.
- Tính PSNR/SSIM khi có ảnh sạch tham chiếu tương ứng.
- Xử lý nền, tiến độ và hủy Batch sau file hiện tại.

Bộ dữ liệu dùng trong thực nghiệm hiện có 100 bộ clean–corrupted–mask. Đây là phạm vi phần mềm xử lý ảnh của Subject 2 / Project 1; chưa có xử lý video, báo cáo học phần hoặc video demo nộp bài.

<a id="cong-nghe"></a>

## Công nghệ

| Công nghệ | Yêu cầu khai báo | Vai trò trong project |
| --- | --- | --- |
| Python | ≥ 3.14 | Ngôn ngữ; `.python-version` chọn 3.14 |
| NumPy | ≥ 1.26.0 | Mảng pixel, tính toán số và tạo nhiễu |
| OpenCV | `opencv-python ≥ 4.8.0` | Đọc/lưu ảnh, Gaussian, Inpainting, morphology và tăng nét |
| PySide6 / Qt | ≥ 6.6.0 | GUI, thanh kéo, hiển thị ảnh và worker nền |
| scikit-image | ≥ 0.22.0 | PSNR và SSIM |
| uv / uv_build | Backend ≥ 0.12.10, < 0.13.0 | Đồng bộ môi trường/dependencies; build wheel và source distribution |
| unittest | Thư viện chuẩn Python | Kiểm thử core và GUI; không cần pytest |

`pyproject.toml` khai báo dependencies, `uv.lock` ghi phiên bản đã resolve. Matplotlib (≥ 3.8.0) và Pillow (≥ 10.0.0) cũng được khai báo, nhưng mã nguồn ứng dụng hiện không dùng trực tiếp hai thư viện này.

<a id="cau-truc"></a>

## Cấu trúc thư mục

```text
Project-1/
├── src/
│   └── project_1/
│       ├── __init__.py           # Console entrypoint project-1
│       ├── __main__.py           # Chạy bằng python -m project_1
│       ├── main.py               # Khởi tạo ứng dụng và đọc tham số CLI
│       ├── entrypoint.py         # Pipeline dùng chung cho GUI/API/Batch
│       ├── settings.py           # Registry thuật toán và mặc định
│       ├── generate_data.py      # Sinh ảnh hỏng và mask có seed
│       ├── algorithms/
│       │   ├── base.py           # Interface thuật toán phục hồi
│       │   ├── gaussian_filter.py
│       │   ├── inpainting.py
│       │   ├── unsharp_mask.py   # Tăng nét sau phục hồi
│       │   └── scratch_mask.py   # Detector, kiểm tra mask và overlay
│       ├── gui/
│       │   ├── main_window.py    # Các luồng Single/Compare/Batch
│       │   ├── widgets.py        # Viewer và control thanh kéo dùng chung
│       │   ├── mask_dialog.py    # Xem và xác nhận mask gợi ý
│       │   └── threads.py        # Worker Qt
│       ├── metrics/
│       │   ├── evaluator.py      # PSNR/SSIM ảnh phục hồi
│       │   └── mask_evaluator.py # Precision/recall/IoU của mask
│       └── utils/
│           ├── image_io.py       # I/O, kiểm tra ảnh, đường dẫn Unicode
│           ├── batch_processor.py
│           └── reports.py        # CSV chi tiết và tổng hợp
├── tests/                        # Kiểm thử unittest core, GUI và preview
├── tools/
│   ├── validate_dataset.py       # Kiểm chứng ba phương pháp trên dataset
│   ├── validate_enhancements.py  # Đánh giá tăng nét và mask holdout
│   └── generate_previews.py      # Chụp GUI, tạo manifest và ZIP preview
├── dataset/
│   ├── images_clean/             # Ảnh sạch nguồn/tham chiếu
│   ├── images_corrupted/         # Ảnh hỏng cục bộ, Git ignore
│   ├── images_masks/             # Mask cục bộ, Git ignore
│   └── generated/                # Các dataset sinh mới, Git ignore
├── output/                       # Ảnh phục hồi, CSV, preview, package; Git ignore
├── pyproject.toml                # Metadata, dependencies, entrypoint và build
├── uv.lock                       # Lockfile dependencies
├── .python-version
├── .gitignore
└── README.md
```

Các thư mục dữ liệu sinh và output có thể chưa tồn tại ở bản clone mới. Các file `__init__.py` của subpackage được lược bớt trong sơ đồ.

Luồng chính: **GUI → RestorationPipeline → thuật toán → ảnh kết quả → metric và báo cáo**. Detector mask là bước riêng trước Inpainting, không phải một phương pháp phục hồi thứ tư.

<a id="cai-dat"></a>

## Cài đặt và chạy nhanh

Yêu cầu: Python 3.14+ và uv đã được cài. Mở terminal tại thư mục gốc repository:

```powershell
uv sync
uv run project-1
```

Lệnh đầu đồng bộ dependencies; lệnh sau mở GUI. Hai cách chạy tương đương:

```powershell
uv run python -m project_1
uv run python -m project_1.main
```

Chọn bộ dữ liệu khác bằng `uv run project-1 --dataset-dir "duong-dan-den-dataset"`. Có thể mở ảnh riêng trong GUI mà không cần chuẩn bị cả dataset. Project đã được kiểm thử trên Windows; chưa xác nhận trên hệ điều hành khác.

<a id="du-lieu"></a>

## Chuẩn bị dữ liệu và chạy demo

Ảnh hỏng và mask trong `dataset/` bị Git ignore. Với bản clone mới, tạo dữ liệu từ ảnh sạch trước:

```powershell
uv run python -m project_1.generate_data --output dataset/generated/demo --seed 42
uv run project-1 --demo --dataset-dir dataset/generated/demo
```

`--demo` nạp bộ ảnh đầu tiên cùng reference/mask và chạy so sánh. Bộ sinh mặc định dùng 6 vết xước, 2 vết ố và nhiễu Gaussian variance 0,003.

Đích `dataset/generated/demo` phải **chưa tồn tại**. Khi sinh thêm lần khác, chọn tên mới; không xóa dữ liệu cũ chỉ để chạy lại. Bỏ `--output` sẽ tự tạo thư mục lượt chạy `dataset/generated/run_.../` và in đường dẫn.

Một dataset dùng trong GUI/validation có dạng:

```text
dataset/generated/demo/
├── images_clean/       # Ví dụ: a.png
├── images_corrupted/   # Cùng tên: a.png; thêm generation.json
└── images_masks/       # Cùng tên: a.png
```

Ảnh nguồn JPG/BMP được chuyển thành PNG; clean/corrupted/mask đầu ra có tên ghép tương ứng. `generation.json` lưu seed, cấu hình, phiên bản thư viện và danh sách nguồn. Cùng nguồn, seed, cấu hình và phiên bản thư viện cho cùng dữ liệu sinh.

Quy ước đầu vào:

- Hỗ trợ PNG, JPG/JPEG và BMP. Core nhận ảnh `uint8` xám hoặc BGR ba kênh; GUI đọc file ảnh thành BGR.
- Mask một kênh: trắng 255 là vùng cần phục hồi, đen 0 là vùng giữ nguyên khi Inpainting.
- Clean/mask phải tương ứng với ảnh hỏng và cùng kích thước; không tự resize để che ghép sai.
- Batch ghép theo **cùng tên file, gồm phần mở rộng**, chỉ duyệt file trực tiếp trong thư mục.

<a id="su-dung"></a>

## Hướng dẫn sử dụng

### Single và Compare

1. Mở ảnh hỏng trước; nạp ảnh sạch tham chiếu nếu muốn tính metric.
2. Chọn Gaussian, Inpainting hoặc Combined. Hai phương pháp sau cần mask.
3. Nạp mask có sẵn hoặc dùng hộp thoại gợi ý mask.
4. Chỉnh thông số, bấm Khôi phục hoặc So sánh, xem ảnh và lưu kết quả.

Thanh kéo và ô nhập số luôn đồng bộ. Nhập số rồi Enter/chuyển focus để xác nhận; kernel chẵn được đưa lên số lẻ kế tiếp. **Chỉnh thông số không tự chạy xử lý.** Compare dùng cùng cấu hình cho ba phương pháp; thiếu mask thì bỏ qua Inpainting/Combined. Chọn dòng kết quả để lưu ảnh hoặc xuất CSV.

Đổi ảnh hỏng xóa reference, mask và kết quả cũ. Không có reference vẫn phục hồi được, nhưng chỉ đánh giá trực quan.

### Tăng nét và gợi ý mask

Bật **Tăng nét sau phục hồi** để áp dụng cho Single/Compare/Batch; mặc định tắt. Mức 0 giữ nguyên kết quả trước tăng nét. Tăng nét có thể làm nổi nhiễu/tạo viền, không bảo đảm PSNR/SSIM tăng.

Trong Single/Compare, bấm **Gợi ý mask xước sáng… → Tạo / tạo lại mask**. Xem mask trắng/đen, overlay đỏ và thống kê, rồi bấm **Áp dụng mask**. Đổi thông số phải tạo lại; mask rỗng không được áp dụng; Hủy giữ mask cũ. Có thể lưu mask đã xác nhận thành PNG.

Detector chỉ gợi ý xước **sáng, mảnh**, không dùng ảnh sạch. Có thể nhận nhầm chi tiết sáng và bỏ sót xước; không tìm vết ố/xước tối. Diện tích vượt 10% sẽ cảnh báo, không phải kết luận mask đúng/sai.

### Batch

Chọn thư mục ảnh hỏng, clean tùy chọn, mask nếu cần và output; bấm **Bắt đầu Batch**. Batch không tự tạo mask. Mask PNG đã xuất cần ghép với ảnh đầu vào PNG cùng tên; không lưu mask dưới dạng JPEG.

Các tác vụ chạy nền và khóa điều khiển thay đầu vào/cấu hình. Hủy Batch dừng sau file hiện tại, giữ kết quả đã hoàn thành. File đọc lỗi hoặc thiếu mask được ghi nhận và không chặn ảnh khác; reference lỗi vẫn cho lưu kết quả nhưng không tính điểm. Đóng khi đang xử lý sẽ hỏi dừng và chờ worker.

Mỗi lượt Batch tạo thư mục riêng chứa PNG, `details.csv` và `summary.csv`; không ghi đè lượt cũ. Chọn dòng trong bảng để xem trước/sau.

<a id="thuat-toan"></a>

## Thuật toán và thông số

| Phương pháp | Cách xử lý | Cần mask |
| --- | --- | --- |
| Gaussian | `cv2.GaussianBlur`, giảm nhiễu toàn ảnh nhưng có thể làm mờ chi tiết | Không |
| Inpainting | `cv2.inpaint` với Telea hoặc Navier–Stokes | Có |
| Combined | Inpainting trước, Gaussian sau | Có |

Tăng nét dùng Unsharp Mask: `Y + amount × (Y − GaussianBlur(Y, sigma))`. Ảnh màu xử lý kênh độ sáng Y trong YCrCb; ảnh xám xử lý trực tiếp. Kết quả được làm tròn/giới hạn về `uint8`. Bước này chạy đúng một lần **sau** phương pháp phục hồi.

Detector dùng grayscale → white Top-hat → ngưỡng sáng/phản hồi → lọc thành phần liên thông mảnh → nới rộng mask nếu được chọn.

| Thông số GUI | Khoảng; bước | Mặc định |
| --- | --- | --- |
| Gaussian kernel | 3–31; chỉ số lẻ | 5 |
| Gaussian sigma | 0,1–10; 0,1 | 1,2 |
| Bán kính Inpainting | 1–20; 1 | 3 |
| Phương pháp Inpainting | Telea / Navier–Stokes | Telea |
| Tăng nét | Bật / tắt | Tắt |
| Mức tăng nét | 0–2; 0,05 | 0,5 |
| Sigma tăng nét | 0,3–3; 0,1 | 1 |
| Mask: ngưỡng sáng | 180–250; 1 | 220 |
| Mask: ngưỡng Top-hat | 5–100; 1 | 35 |
| Mask: kernel ellipse | 3–31; chỉ số lẻ | 9 |
| Mask: nới rộng | 0–3 pixel; 1 | 0 |

Các khoảng trên là giới hạn của GUI; không mặc định mọi tham số API có cùng giới hạn. Thuật toán, mặc định và validation thực tế nằm trong `algorithms/` và `settings.py`.

<a id="kiem-thu"></a>

## Đánh giá, kiểm thử và build

PSNR/SSIM cần ảnh sạch tương ứng, cùng kích thước và số kênh. Điểm sau tính trên **ảnh kết quả cuối**, bao gồm tăng nét nếu bật. PSNR dùng đơn vị dB; ảnh giống nhau có PSNR `inf`, SSIM bằng 1. SSIM cần cạnh nhỏ nhất từ 3 pixel. Delta là điểm sau trừ điểm trước; hai điểm vô hạn bằng nhau có delta 0.

Chỉ làm tròn khi hiển thị; CSV giữ độ chính xác đầy đủ, dùng UTF-8 BOM và lưu cấu hình dưới dạng JSON. Metric trống nghĩa là chưa được đánh giá. Thời gian đo xử lý thuật toán, gồm tăng nét nhưng không gồm đọc/lưu file hoặc tính metric.

`details.csv` ghi từng file, trạng thái `success/skipped/failed`, đường dẫn, metric trước/sau, delta và lỗi. `summary.csv` tổng hợp số lượng, điểm trung bình và số ảnh cải thiện; chỉ lấy các file có metric hợp lệ. Điểm tốt hơn không chứng minh nội dung mất đã được khôi phục chính xác.

Chạy test chức năng, không cần pytest:

```powershell
uv run python -m unittest discover -s tests -v
```

Đánh giá dữ liệu demo đã tạo ở trên và tạo preview:

```powershell
uv run python tools/validate_dataset.py --dataset dataset/generated/demo
uv run python tools/validate_enhancements.py --dataset dataset/generated/demo
uv run python tools/generate_previews.py --dataset dataset/generated/demo
```

- `validate_dataset.py`: chạy ba phương pháp, lưu PNG/CSV và tính lại PSNR/SSIM độc lập trên ảnh đã lưu.
- `validate_enhancements.py`: so sánh Combined bật/tắt tăng nét; đánh giá mask trên ảnh scratch-only, seed 43, loại 10 ảnh đầu đã dùng calibration. Mặc định cần hơn 10 ảnh sạch. Có precision/recall/IoU, không chấm detector xước bằng mask gộp xước/vết ố.
- `generate_previews.py`: chụp năm preview từ GUI offscreen, Batch mặc định 8 ảnh; tạo manifest/ZIP, sao lưu preview đích cũ trước khi cập nhật.

Validation tạo lượt mới trong `output/`; không sửa dataset. Dataset mới sinh có thể khác dữ liệu của các lượt thực nghiệm cũ, vì vậy không kỳ vọng số liệu luôn trùng nhau.

Trong lần kiểm chứng **01/10/2026**, 70/70 test đạt trên mã nguồn và cùng 70 test đạt khi import từ wheel; 680 ảnh phục hồi đã lưu được kiểm chứng metric. Đây là kết quả của lượt kiểm tra đó, không phải số test cố định hay bảo đảm trên mọi máy/dữ liệu.

Kiểm tra cú pháp và build package:

```powershell
uv run python -m compileall -q src/project_1 tests tools
uv build --out-dir output/package_check
```

Build tạo wheel và source distribution; không tạo executable độc lập. Không dùng `QT_QPA_PLATFORM=offscreen` khi muốn mở GUI tương tác; biến này dành cho kiểm thử/preview không có cửa sổ desktop.

<a id="api"></a>

## API, lỗi thường gặp và giới hạn

### Ví dụ API

Sau khi tạo dataset demo, chạy từ repository root trong môi trường project:

```python
from pathlib import Path

from project_1.entrypoint import RestorationPipeline
from project_1.utils.image_io import list_images, read_image

dataset = Path("dataset/generated/demo")
path = list_images(dataset / "images_corrupted")[0]
image = read_image(path)
clean = read_image(dataset / "images_clean" / path.name)
mask = read_image(dataset / "images_masks" / path.name, grayscale=True)

pipeline = RestorationPipeline()
result = pipeline.restore_result(
    "combined", image, clean_img=clean, mask=mask,
    sharpen_enabled=True, sharpen_amount=0.5, sharpen_sigma=1.0,
)
print(result["metrics"])  # {"PSNR": ..., "SSIM": ...}
```

`run_single(...)` trả `ndarray`; `restore_result(...)` bổ sung cấu hình, thời gian, baseline, metric, delta và cảnh báo. `compare_all(...)` trả trạng thái từng phương pháp; `run_batch(...)` trả `BatchResult`. Khi dùng kết quả, kiểm tra `status` và `metrics is not None`.

`detect_scratch_mask(image, ...)` trong `algorithms/scratch_mask.py` trả mask `uint8` 0/255 cùng kích thước, không sửa ảnh đầu vào. GUI yêu cầu xem và xác nhận mask; gọi hàm detector qua API không tự thay thế bước đánh giá của người dùng.

### Lỗi thường gặp

| Tình huống | Cách kiểm tra |
| --- | --- |
| Không nhận lệnh `uv` | Cài uv và bảo đảm lệnh có trong PATH |
| Sai phiên bản Python / thiếu dependency | Dùng Python 3.14+, chạy `uv sync` rồi chạy qua `uv run` |
| Demo thiếu ảnh hỏng/mask | Tạo dataset mới và truyền đúng `--dataset-dir` |
| Generator báo đích đã tồn tại | Chọn tên đích mới hoặc bỏ `--output`; không ghi đè dataset |
| Inpainting/Combined thiếu mask | Nạp hoặc xác nhận mask; Batch cần thư mục mask tương ứng |
| Không có PSNR/SSIM | Kiểm tra reference, tên file, kích thước và số kênh; xem cảnh báo metric |
| Không mở được cửa sổ GUI | Kiểm tra môi trường desktop và bỏ biến `QT_QPA_PLATFORM=offscreen` nếu đang đặt |

### Giới hạn và tài liệu bổ sung

Không xử lý video, không có AI/super-resolution hay cọ sửa mask; Batch không tự phát hiện mask. Reference cùng kích thước vẫn có thể sai nội dung; phần mềm không tự căn chỉnh hoặc xác minh cặp ảnh. Chưa kiểm chứng cài sạch trên máy khác, hệ điều hành khác hoặc bản executable.

Các file `PREVIEW.md`, `PROJECT_REVIEW.md`, `TEST_REPORT.md` và `output/` là tài liệu/artifact cục bộ đang bị Git ignore, có thể **không có trong bản clone**. Nếu có, dùng để xem screenshot, review và bằng chứng thực nghiệm; khi chia sẻ phải đính kèm riêng. README này không phụ thuộc vào những file đó.
