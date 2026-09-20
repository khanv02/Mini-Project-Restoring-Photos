import os
import cv2
from typing import Callable, Optional

def combined_restore(
    img, 
    mask: Optional[os.PathLike] = None, 
    inpaint_radius: int = 3, 
    inpaint_method: str = "Telea", 
    kernel_size: int = 5, 
    sigma: float = 1.2
):
    """
    Hàm kết hợp xử lý khôi phục ảnh 2 bước:
    1. Inpainting: Xóa vết xước/mảng ố dựa vào mask.
    2. Gaussian Filter: Khử lớp nhiễu hạt lấm chấm còn sót lại.
    """
    result = img.copy()

    # Bước 1: Chạy Inpainting nếu có mask
    if mask is not None:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        method = cv2.INPAINT_TELEA if inpaint_method == "Telea" else cv2.INPAINT_NS
        result = cv2.inpaint(result, mask, inpaintRadius=inpaint_radius, flags=method)

    # Bước 2: Chạy Gaussian Filter để làm mịn nhiễu hạt
    # Đảm bảo kernel_size luôn là số lẻ
    if kernel_size % 2 == 0:
        kernel_size += 1

    result = cv2.GaussianBlur(result, (kernel_size, kernel_size), sigmaX=sigma)

    return result


class BatchProcessor:
    @staticmethod
    def process_directory(
        input_dir: str, 
        output_dir: str, 
        process_fn: Optional[Callable] = None, 
        mask_dir: Optional[str] = None,
        progress_callback=None,
        # Thêm các tham số cấu hình trực tiếp cho Batch nếu không truyền process_fn
        inpaint_radius: int = 3,
        inpaint_method: str = "Telea",
        kernel_size: int = 5,
        sigma: float = 1.2
    ):
        """
        Duyệt thư mục và chạy hàm xử lý khôi phục hàng loạt.
        - Nếu process_fn=None, mặc định sử dụng combined_restore với các tham số truyền vào.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
        total = len(files)

        # Nếu người dùng không truyền process_fn từ ngoài, dùng mặc định combined_restore
        if process_fn is None:
            def default_pipeline(image, mask=None):
                return combined_restore(
                    img=image, 
                    mask=mask, 
                    inpaint_radius=inpaint_radius, 
                    inpaint_method=inpaint_method, 
                    kernel_size=kernel_size, 
                    sigma=sigma
                )
            process_fn = default_pipeline

        for idx, filename in enumerate(files):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Đọc mask tương ứng nếu có thư mục mask
            if mask_dir:
                mask_path = os.path.join(mask_dir, filename)
                mask = cv2.imread(mask_path) if os.path.exists(mask_path) else None
                restored = process_fn(img, mask=mask)
            else:
                restored = process_fn(img)

            if restored is not None:
                save_path = os.path.join(output_dir, f"restored_{filename}")
                cv2.imwrite(save_path, restored)

            if progress_callback:
                progress_callback(idx + 1, total)