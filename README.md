# Khôi phục ảnh cũ — Subject 2 / Project 1

Ứng dụng desktop Python 3.14+, OpenCV, PySide6 và scikit-image. Có ba phương pháp: Gaussian Filter, Inpainting (Telea hoặc Navier–Stokes) và Combined (Inpainting rồi Gaussian).

Phạm vi triển khai là xử lý **ảnh**: Single, Compare và Batch; không có xử lý video. Đề Project_2026 trang 3 yêu cầu tối thiểu 50 ảnh hoặc video, PSNR/SSIM, Batch, so sánh phương pháp và GUI. Bộ dữ liệu hiện tại có 100 bộ ảnh clean–corrupted–mask. Phạm vi video cần xác nhận riêng với giảng viên.

[Preview giao diện mới](PREVIEW.md) · [Kết quả kiểm thử đầy đủ](TEST_REPORT.md) · [Review kỹ thuật](PROJECT_REVIEW.md)

## Cài đặt và khởi chạy

Cài dependencies bằng `uv`, từ thư mục gốc:

```powershell
uv sync
uv run project-1
```

Các cách chạy tương đương:

```powershell
uv run python -m project_1.main
uv run python -m project_1
```

Nạp ảnh mẫu, mask/reference tương ứng và chạy so sánh:

```powershell
uv run project-1 --demo
uv run project-1 --dataset-dir "D:\path\to\dataset"
```

Dependencies và Python thực tế được khai báo trong `pyproject.toml`, `.python-version` và `uv.lock`. Không cần notebook hoặc Median Filter để chạy ứng dụng.

## Cách sử dụng

Một vùng **Thuật toán & tham số chung** áp dụng cho cả ba tab. Mặc định: kernel 5, sigma 1,2, radius 3, Telea. Kernel chẵn được chuẩn hóa thành số lẻ kế tiếp. Gaussian kernel/sigma lớn có thể làm mờ chi tiết.

Mọi tham số số dùng thanh kéo ngang, kèm ô hiển thị/nhập số không có nút tăng/giảm. Kéo thanh cập nhật giá trị ngay; nhập số rồi Enter hoặc chuyển focus để xác nhận. Ô nhập đồng bộ với thanh kéo và làm tròn về bước tương ứng. Chỉnh tham số không tự chạy xử lý: bấm Khôi phục/So sánh/Bắt đầu Batch để áp dụng. Sidebar cuộn được khi cửa sổ thấp.

| Thông số | Khoảng; bước | Mặc định |
| --- | --- | --- |
| Gaussian kernel | 3–31; chỉ số lẻ | 5 |
| Gaussian sigma | 0,1–10; 0,1 | 1,2 |
| Bán kính Inpainting | 1–20; 1 | 3 |
| Mức tăng nét | 0–2; 0,05 | 0,5 |
| Sigma tăng nét | 0,3–3; 0,1 | 1 |

**Khôi phục đơn**

1. Nạp ảnh hỏng trước.
2. Nạp ảnh clean tham chiếu nếu có; nạp mask khi dùng Inpainting/Combined.
3. Chọn thuật toán, điều chỉnh tham số và chạy.
4. Xem trước/sau, PSNR/SSIM trước–sau và mức thay đổi; lưu ảnh kết quả.

Đổi ảnh hỏng sẽ xóa tham chiếu, mask và kết quả cũ. Nạp lại tham chiếu/mask cũng xóa kết quả trước đó. Không có reference vẫn phục hồi được; giao diện ghi rõ chỉ đánh giá trực quan.

**So sánh phương pháp**

Chạy Gaussian, Inpainting và Combined với tham số đang chọn. Tab Compare hiển thị toàn bộ điều khiển tham số, ảnh kết quả từng phương pháp và bảng chất lượng trước–sau, mức thay đổi, thời gian. Thiếu mask sẽ ghi trạng thái bỏ qua Inpainting/Combined. Chọn dòng để lưu ảnh phương pháp đó, hoặc xuất CSV cho toàn bộ bảng.

### Tăng nét tùy chọn

Bật **Tăng nét sau phục hồi** trong vùng tham số chung. Mặc định tắt để giữ nguyên kết quả cũ. Dùng Unsharp Mask: `Y + amount × (Y − GaussianBlur(Y, sigma))`, làm tròn và giới hạn về 0–255. Ảnh màu chỉ xử lý kênh độ sáng Y trong YCrCb; ảnh xám xử lý trực tiếp. Mức 0 giữ nguyên pixel, không chuyển đổi màu.

Áp dụng đúng một lần sau Gaussian/Inpainting; Combined chạy **Inpainting → Gaussian → tăng nét**. Cùng cấu hình dùng trong cả Single, Compare và Batch. Ảnh lưu và metric đều là kết quả cuối; CSV `parameters` chứa `sharpen_enabled`, `sharpen_amount`, `sharpen_sigma`. Thời gian bao gồm bước tăng nét nhưng không gồm metric/I/O.

Tăng nét có thể làm nổi nhiễu hoặc tạo viền; không bảo đảm PSNR/SSIM tốt hơn. Nên bắt đầu mức 0,5 và sigma 1, rồi so sánh bật/tắt trên cùng ảnh.

### Gợi ý và kiểm tra mask tự động

Trong Single/Compare, nạp ảnh hỏng rồi bấm **Gợi ý mask xước sáng…**. Không cần ảnh sạch; detector không sử dụng reference. Hộp thoại có các thanh kéo:

| Thông số | Khoảng; bước | Mặc định |
| --- | --- | --- |
| Ngưỡng sáng | 180–250; 1 | 220 |
| Ngưỡng Top-hat | 5–100; 1 | 35 |
| Kernel ellipse | 3–31; chỉ số lẻ | 9 |
| Nới rộng mask | 0–3 pixel; 1 | 0 |

1. Bấm **Tạo / tạo lại mask**. Xử lý chạy nền; chỉ khóa thông số liên quan.
2. Kiểm tra mask trắng/đen và lớp phủ đỏ, số vùng/pixel/tỷ lệ diện tích. Vượt 10% sẽ cảnh báo; đây không phải điểm chính xác của detector.
3. Đổi tham số phải tạo lại trước khi áp dụng. Mask rỗng không thể áp dụng.
4. Bấm **Áp dụng mask** để thay mask hiện tại; **Hủy** giữ nguyên mask và kết quả cũ. Đóng khi đang tạo sẽ chờ worker dừng an toàn.
5. Bấm **Lưu mask đã xác nhận (PNG)** ở cửa sổ chính để dùng lại.

Detector thử nghiệm chỉ tìm xước **sáng, mảnh**: grayscale → white Top-hat → ngưỡng sáng/phản hồi → lọc thành phần liên thông 8 hướng. Giữ vùng có diện tích ≥12, chiều rộng ước lượng ≤8 và chiều dài hiệu dụng ≥15 pixel. Chiều rộng = `max(1, 2 × percentile90(distanceTransform) − 1)`; chiều dài = diện tích/chiều rộng. Nới rộng bằng dilation ellipse nếu được chọn.

Có thể nhận nhầm chi tiết sáng và bỏ sót xước; không tìm vết ố/xước tối, không có cọ vẽ. Kiểm tra tự động chỉ xác nhận kích thước, kiểu `uint8`, giá trị nhị phân 0/255 và thống kê; không chứng minh mask đúng về nội dung.

Batch không tự tạo mask. Muốn dùng mask PNG đã xác nhận trong Batch, đặt trong thư mục mask với **cùng tên và phần mở rộng** ảnh đầu vào. Với đầu vào JPG/BMP, chuẩn bị một bản đầu vào PNG có tên tương ứng trước; không lưu mask thành JPEG vì mất tính nhị phân.

**Batch**

Chọn thư mục ảnh hỏng, mask khi cần, thư mục clean tùy chọn và thư mục output. Các file được ghép theo **cùng tên file, gồm phần mở rộng**. Chương trình chỉ duyệt file trực tiếp trong thư mục, không đệ quy.

Ảnh đọc lỗi được ghi nhận và không làm dừng các ảnh khác. Thiếu mask thì bỏ qua file cần Inpainting; thiếu hoặc sai reference vẫn lưu ảnh phục hồi nhưng không tính điểm, đồng thời ghi lý do. Gaussian không cần mask.

Mỗi lượt có thư mục riêng, chứa ảnh PNG không mất dữ liệu, `details.csv` và `summary.csv`. File PNG giữ tên `restored_<tên gốc>`; các định dạng khác thêm đuôi `.png` để giữ nguyên danh tính file và tránh trùng tên. Không ghi đè lượt chạy trước.

Nếu các tên nguồn khác nhau tạo cùng tên PNG đầu ra (ví dụ `a.jpg` và `a.jpg.png`), Batch và bộ sinh dữ liệu đánh số tên file trong lượt đó để tránh ghi đè.

Bảng Batch hiển thị trạng thái từng file; chọn dòng để xem trước/sau. Nút hủy dừng **sau file hiện tại**, giữ ảnh và báo cáo đã hoàn thành. Các tác vụ Single, Compare và Batch chạy nền; không cho bắt đầu tác vụ khác hoặc thay đầu vào/cấu hình khi đang chạy. Đóng khi đang chạy sẽ hỏi dừng và chờ worker kết thúc an toàn.

## PSNR, SSIM và CSV

Metric chỉ dùng ảnh tham chiếu tương ứng, cùng kích thước và số kênh. Core chấp nhận ảnh `uint8` xám hoặc BGR 3 kênh; đọc file trong GUI chuẩn hóa thành BGR và mask xám. Không tự resize ảnh hoặc mask để che lệch kích thước, không ép kiểu float âm thầm.

- PSNR: `10 log10(255² / MSE)`, đơn vị dB. Hai ảnh giống nhau có PSNR `inf`.
- SSIM: dùng `data_range=255`, đánh giá theo kênh màu hoặc ảnh xám; cửa sổ lẻ tối đa 7, thu nhỏ cho ảnh nhỏ. Cạnh nhỏ nhất phải từ 3 pixel.
- Độ thay đổi: điểm sau trừ điểm trước; `inf − inf` ở hai điểm bằng nhau được ghi là 0.
- Chỉ làm tròn khi hiển thị. CSV giữ giá trị đầy đủ; trường metric trống nghĩa là chưa đánh giá.
- Thời gian là thời gian xử lý thuật toán, không bao gồm đọc/lưu file hoặc tính metric.
- CSV UTF-8 có BOM để mở tiếng Việt thuận tiện. `parameters` chứa JSON cấu hình.

`details.csv` gồm tên file, thuật toán, trạng thái (`success/skipped/failed`), cấu hình, đường dẫn vào/ra, điểm trước/sau, mức thay đổi, thời gian, lỗi xử lý và lỗi metric. `summary.csv` gồm số lượng thành công/bỏ qua/lỗi/đánh giá, điểm trung bình, số ảnh cải thiện, cấu hình và trạng thái hủy. Nếu trung bình chứa cả hai dấu vô hạn, ghi `undefined`.

Điểm trung bình chỉ tính trên các file được đánh giá thành công. PSNR/SSIM tốt hơn không bảo đảm phục hồi chính xác nội dung bị mất; cần xem ảnh trước/sau. Gaussian, Inpainting và Combined có thể làm giảm điểm ở một số ảnh.

## Dataset và sinh dữ liệu

```text
dataset/
  images_clean/        # Ảnh tham chiếu
  images_corrupted/    # Ảnh hỏng cùng tên
  images_masks/        # Mask cùng tên: trắng 255 = vùng cần phục hồi
  generated/           # Dataset mới, được bỏ qua bởi Git
output/
  run_.../             # Một lượt Batch, PNG + CSV
  validation_.../      # Kiểm chứng toàn bộ dataset
```

Sinh nhiễu Gaussian, vết xước và vết ố với seed mặc định **42**:

```powershell
uv run python -m project_1.generate_data
uv run python -m project_1.generate_data --seed 42 --variance 0.003 --scratches 6 --stains 2 --output dataset/generated/experiment_01
```

Không có `--output` thì tạo thư mục lượt chạy mới. Có `--output` thì đích phải chưa tồn tại. Bộ sinh từ chối ghi đè dữ liệu và không sửa dataset hiện có. Mỗi lượt tạo `images_clean`, `images_corrupted`, `images_masks` dạng PNG cùng tên và `images_corrupted/generation.json` ghi seed, cấu hình, phiên bản NumPy/OpenCV và danh sách nguồn. Ảnh nguồn JPG/BMP được sao chép thành PNG tham chiếu cùng tên với ảnh sinh.

Dùng `--dataset-dir` khi khởi chạy GUI để chọn bộ dữ liệu mới. Tái tạo với cùng ảnh nguồn, seed, cấu hình và phiên bản thư viện cho cùng kết quả.

## Kiểm thử và kiểm chứng 100 ảnh

Không cần pytest:

```powershell
uv run python -m unittest discover -s tests -v
uv run python tools/validate_dataset.py
uv run python tools/validate_enhancements.py
uv run python tools/generate_previews.py
```

Bộ kiểm thử bao gồm metric đầy đủ độ chính xác, kích thước/kênh/kiểu dữ liệu sai, mask, tham số, Combined, Unicode I/O, CSV, lỗi từng file, hủy, không ghi đè dữ liệu, worker nền, GUI và entrypoint.

Hiện có 70 test, gồm 4 test cho việc render preview, xác nhận mask qua hộp thoại thật, đóng gói ZIP và sao lưu preview cũ. `generate_previews.py` cập nhật năm ảnh trong `output/previews/`, gồm Batch đã chạy thực tế (mặc định 8 ảnh). Có manifest cấu hình và `previews.zip`; bản cũ được giữ trong thư mục `refresh_.../previous/`. Xem [PREVIEW.md](PREVIEW.md) để biết nội dung và cách chia sẻ.

`validate_dataset.py` kiểm tra mọi bộ ảnh, chạy cả ba phương pháp bằng Batch, lưu ảnh + CSV và tính lại PSNR/SSIM độc lập trên **ảnh đã lưu** để đối chiếu. Xuất tổng hợp `all_results.csv` và `all_summary.csv` trong một thư mục `output/validation_...`. Có thể chỉ định `--dataset` và `--output`.

`validate_enhancements.py` so sánh Combined bật/tắt tăng nét trên toàn bộ dataset hiện có, lưu 200 ảnh nếu có 100 bộ đầu vào. Với detector, bỏ 10 ảnh sạch đầu tiên đã dùng chọn mặc định, dùng các ảnh còn lại làm holdout; sinh 6 xước, 0 vết ố, variance 0,003, seed 43. Không sửa dataset gốc.

Mỗi lượt tạo `output/enhancements_.../`: PNG, `sharpen_results.csv`, `sharpen_summary.csv`, `sharpen_differences.csv`, `mask_results.csv`, `mask_summary.csv`, `report.json`. Mask report có precision/recall/IoU trung bình theo ảnh, tỷ lệ vùng chọn và thời gian; so sánh Inpainting dùng mask dự đoán với mask ground truth ở cùng tham số. Kiểm chứng mask PNG giữ nguyên pixel và PSNR/SSIM độc lập trên ảnh phục hồi đã lưu. Có `--dataset`, `--output`, `--seed`, `--skip-calibration`.

Hai mask đều rỗng được chấm precision/recall/IoU = 1; dự đoán rỗng nhưng ground truth không rỗng được chấm 0; ground truth rỗng nhưng dự đoán không rỗng có precision/IoU = 0, recall = 1. Không dùng mask gộp xước và vết ố của dataset cũ để chấm detector chỉ tìm xước.

## Kiến trúc và API

- `algorithms/`: Gaussian và Inpainting triển khai `BaseRestorationAlgorithm.process`.
- `entrypoint.py`: pipeline dùng chung cho GUI, Batch và kiểm thử.
- `metrics/`: PSNR/SSIM với kiểm tra đầu vào nghiêm ngặt.
- `gui/`: cửa sổ, các control/viewer tái sử dụng và worker Qt.
- `utils/`: đọc/lưu ảnh, một vòng lặp Batch chung và xuất báo cáo.
- `settings.py`: registry thuật toán và giá trị mặc định.

`run_single(name, image, **parameters)` vẫn trả `ndarray`. `restore_result(..., clean_img=None, mask=None)` thêm ảnh, cấu hình, thời gian, baseline, metric, delta và cảnh báo metric.

```python
from project_1.entrypoint import RestorationPipeline
from project_1.algorithms.scratch_mask import detect_scratch_mask

mask = detect_scratch_mask(corrupted, brightness_threshold=220,
                           response_threshold=35, kernel_size=9, expand=0)
# Trong GUI, phải xem và xác nhận mask trước khi phục hồi.
restored = RestorationPipeline().run_single(
    "combined", corrupted, mask=mask,
    sharpen_enabled=True, sharpen_amount=0.5, sharpen_sigma=1.0,
)
```

Các tham số tăng nét có mặc định `False`, `0.5`, `1.0`. Mức phải hữu hạn trong 0–2, sigma trong 0,3–3; kiểm tra kể cả khi tắt. `detect_scratch_mask` trả mask `uint8` 0/255, cùng kích thước, không sửa ảnh đầu vào. `binary_mask_scores` trong `metrics/mask_evaluator.py` chỉ chấm khi có mask ground truth riêng.

`compare_all(clean_img, corrupted_img, mask=None, **parameters)` nhận tham số chung; trả entry cho cả ba phương pháp, kể cả phương pháp bị bỏ qua. Các caller phải kiểm tra `status` và `metrics is not None` trước khi dùng điểm.

`run_batch(..., clean_dir=None, result_callback=None, should_stop=None)` trả `BatchResult` gồm tổng số file, số thành công/bỏ qua/lỗi/đánh giá, trạng thái hủy, hàng báo cáo và thư mục lượt chạy. Callback kết quả nhận một dòng CSV; callback tiến độ tính cả file lỗi và bỏ qua.
