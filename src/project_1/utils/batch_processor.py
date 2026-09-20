import os
import cv2
from typing import Callable, Optional

class BatchProcessor:
    @staticmethod
    def process_directory(
        input_dir: str, 
        output_dir: str, 
        process_fn: Callable, 
        mask_dir: Optional[str] = None,
        progress_callback=None
    ):
        """
        Duyệt thư mục và chạy hàm xử lý
        - mask_dir: Thư mục chứa các ảnh mask tương ứng (dùng cho Inpainting)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
        total = len(files)

        for idx, filename in enumerate(files):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Nếu có truyền thư mục mask (Inpainting)
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