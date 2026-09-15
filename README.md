# Mini-Project-Restoring-Photos


## Thuật toán xử lý bốn trường hợp ảnh bị corrupted như sau:
###  Vết xước & Ố vàng (Scratches & Stains): Dùng cv2.line và cv2.ellipse
### Nhiễu muốm tiêu (Salt & Pepper Noise): Hàm add_salt_and_pepper_noise
### Nhiễu hạt (Gaussian Noise): Hàm add_gaussian_noise
### Nhiễu mờ (Blur): Hàm cv2.GaussianBlur

## Cấu trúc thư mục
src/
└── project_1/
    ├── __init__.py
    ├── entrypoint.py           # Entrypoint quản lý chính (Pipeline / Controller)
    ├── generate_data.py        # Script tạo dữ liệu thử nghiệm
    │
    ├── algorithms/             # Tách riêng từng giải thuật khôi phục
    │   ├── __init__.py
    │   ├── base.py             # Base class định nghĩa chuẩn chung
    │   ├── median_filter.py    # Lọc Trung vị
    │   ├── gaussian_filter.py  # Lọc Gaussian
    │   └── inpainting.py       # Khôi phục vết xước
    │
    ├── metrics/                # Đánh giá chất lượng
    │   ├── __init__.py
    │   └── evaluator.py        # Tính PSNR & SSIM
    │
    └── utils/                  # Xử lý hàng loạt & Thao tác file
        ├── __init__.py
        └── batch_processor.py  # Batch Processing`