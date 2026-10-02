# Tutorial — khôi phục ảnh và đọc kết quả

Tài liệu này hướng dẫn nhanh cho người dùng và Dev kiểm tra luồng khôi phục ảnh trong Project 1.

## 1. Cài đặt và mở ứng dụng

Yêu cầu Python 3.14+ và `uv`.

```powershell
uv sync --locked
uv run python -m project_1
```

Có thể chạy bằng entrypoint tương đương:

```powershell
uv run project-1
```

Nếu chưa có dữ liệu demo, tạo từ ảnh sạch trong `dataset/images_clean/`:

```powershell
uv run python -m project_1.generate_data --output dataset/generated/demo --seed 42
uv run project-1 --demo --dataset-dir dataset/generated/demo
```

## 2. Khôi phục một ảnh

1. Vào tab **Single**, bấm **Mở ảnh hỏng…** hoặc chọn **Tùy chọn ảnh → Nạp bộ ảnh mẫu**.
2. Nạp ảnh sạch trong mục tham chiếu nếu muốn tính PSNR/SSIM.
3. Chọn **Gaussian**, **Inpainting** hoặc **Combined**. Inpainting và Combined cần mask.
4. Nạp mask có sẵn hoặc mở **Tùy chọn ảnh → Gợi ý mask xước sáng…** để tạo, xem overlay và bấm **Áp dụng mask**.
5. Chỉnh thông số rồi bấm **Khôi phục**. Kết quả xuất hiện ở panel bên phải.

Ảnh sạch chỉ dùng làm tham chiếu đánh giá, không tham gia phục hồi. Nếu thiếu ảnh sạch, ứng dụng vẫn khôi phục được nhưng không hiển thị PSNR/SSIM.

## 3. Preview nhanh khi kéo thanh chỉ số

Sau lần chạy đầu tiên, thay đổi thanh kéo trong Single hoặc Compare có hai giai đoạn:

- **Đang kéo:** ứng dụng thu nhỏ ảnh và mask tối đa còn 160 pixel ở cạnh dài, chạy pipeline trên ảnh nhỏ và hiển thị lại ảnh. PSNR/SSIM ở giai đoạn này được ghi rõ là **ước lượng**, giúp phản hồi nhanh.
- **Thả chuột:** ứng dụng chạy lại trên ảnh gốc ở độ phân giải đầy đủ và thay bằng PSNR/SSIM **chính xác**. Chỉ kết quả chính xác mới được lưu hoặc xuất CSV.

Preview dùng cơ chế latest-value-wins: nếu kéo tiếp khi một lượt preview đang chạy, kết quả cũ không được phép ghi đè giá trị mới nhất. Bộ nhớ đệm baseline trong Compare tránh tính lại metric của ảnh hỏng cho từng phương pháp.

Nếu cần số liệu tin cậy, luôn chờ trạng thái preview chính xác hoàn tất trước khi kết luận thông số nào tốt hơn.

![Preview nhanh khi kéo thanh](assets/tutorial/01_phuc_hoi.png)

## 4. Chọn thông số Gaussian

- **Kernel:** tăng để làm mượt nhiễu mạnh hơn nhưng dễ mất chi tiết. Giảm khi ảnh bị bệt hoặc cạnh bị nhòe.
- **Sigma:** tăng để giảm nhiễu mạnh hơn; giảm khi cần giữ texture và cạnh.
- Kernel phải là số lẻ. Ô nhập số và thanh kéo được đồng bộ; giá trị chẵn sẽ được đưa lên số lẻ kế tiếp.

Nếu có ảnh sạch, hãy chạy xong ảnh đầy đủ rồi đọc PSNR/SSIM. PSNR phản ánh sai khác pixel; SSIM phản ánh cấu trúc và độ tương đồng nhìn thấy. Không nên chọn một tham số chỉ vì một metric tăng khi metric còn lại giảm.

![So sánh kernel và sigma](assets/tutorial/02_kernel_sigma.png)

## 5. Inpainting, Combined và tăng nét

- **Bán kính Inpainting:** tăng khi vùng hỏng rộng hoặc cần lấy ngữ cảnh xa hơn; giảm khi chỉ cần sửa vết mảnh để tránh lan ảnh.
- **Telea:** thường phù hợp vết mảnh và nhanh.
- **Navier–Stokes:** là lựa chọn thay thế khi muốn so sánh cách lan truyền khác.
- **Combined:** inpainting trước, Gaussian sau; phù hợp khi vừa có vùng mất dữ liệu vừa có nhiễu.
- **Tăng nét sau phục hồi:** chỉ bật khi ảnh đầu ra bị mềm. Mức quá cao có thể làm nổi nhiễu hoặc tạo viền, vì vậy mặc định đang tắt.

![Mask cho Inpainting](assets/tutorial/03_mask.png)

## 6. Đọc và dùng gợi ý tham số

Trong nhóm **03 · Gợi ý tham số**, bấm **Phân tích & gợi ý** sau khi có ảnh và cấu hình hiện tại.

- Có ảnh sạch: hệ thống thử các giá trị lân cận hợp lệ của kernel, sigma, bán kính, phương pháp inpainting và tăng nét. Gợi ý chính chỉ xuất hiện khi PSNR và SSIM cùng tăng qua ngưỡng tối thiểu; nếu hai metric trái chiều, giao diện hiển thị đánh đổi để Dev tự quyết định.
- Không có ảnh sạch: hệ thống dùng heuristic từ mức nhiễu, độ mạnh cạnh và tỷ lệ mask. Kết quả có mức tin cậy thấp và chỉ mang tính định hướng.
- Hệ thống không tự thay đổi tham số. Gợi ý là một phép đánh giá có thể kiểm tra lại, không phải nhãn đúng tuyệt đối cho mọi ảnh.

Quy tắc đọc nhanh:

| Dấu hiệu | Hướng thử trước |
| --- | --- |
| Nhiễu hạt còn nhiều | Tăng kernel hoặc sigma từng bước nhỏ |
| Cạnh/texture bị nhòe | Giảm kernel hoặc sigma |
| Vết hỏng chưa được lấp | Kiểm tra mask, sau đó thử tăng bán kính 1 bước |
| Ảnh bị bệt sau Combined | Giảm Gaussian sau inpainting hoặc tắt tăng nét |
| Ảnh mềm nhưng sạch | Bật tăng nét ở mức thấp rồi kiểm tra halo |

## 7. Compare và Batch

Trong **Compare**, bấm **So sánh** để chạy ba phương pháp với cùng cấu hình. Chọn một dòng trong bảng để xem ảnh và lưu phương pháp tương ứng. Khi thiếu mask, Inpainting và Combined được bỏ qua theo trạng thái hiển thị.

Trong **Batch**, chọn thư mục ảnh hỏng, clean tùy chọn, mask nếu cần và thư mục output. Batch chạy nền, có tiến độ và có thể hủy sau file hiện tại. Kết quả gồm PNG, `details.csv` và `summary.csv`; các lượt chạy được đặt trong thư mục riêng.

## 8. Phóng to, lưu và xử lý lỗi

- Cuộn chuột để zoom tại con trỏ, kéo chuột trái để pan, nhấp đúp để vừa khung và chọn `1:1` để xem pixel.
- Khi preview còn là **ước lượng**, nút lưu/xuất bị khóa. Chờ preview đầy đủ.
- Ảnh và mask phải cùng kích thước. Mask một kênh dùng trắng `255` cho vùng cần phục hồi và đen `0` cho vùng giữ nguyên.
- Hỗ trợ PNG, JPG/JPEG và BMP; Batch ghép file theo đúng tên gồm phần mở rộng.
- Nếu ảnh bị tối/nhòe hoặc điểm thấp, kiểm tra lại mask trước khi tăng mạnh thông số. Detector hiện chỉ nhắm tới xước sáng, mảnh và luôn cần xem overlay trước khi áp dụng.

Để chạy toàn bộ kiểm thử và tạo lại ảnh preview:

```powershell
uv run python -m unittest discover -s tests -v
uv run python tools/generate_previews.py --dataset dataset --output output/previews
```
