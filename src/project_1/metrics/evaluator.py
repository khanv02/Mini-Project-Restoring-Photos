import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class QualityEvaluator:
    @staticmethod
    def evaluate(clean_img: np.ndarray, restored_img: np.ndarray) -> dict:
        """Tính chỉ số PSNR và SSIM an toàn"""
        if clean_img is None or restored_img is None:
            raise ValueError("Ảnh đầu vào để tính metrics không được là None.")

        # 1. Đảm bảo 2 ảnh cùng kích thước
        if clean_img.shape[:2] != restored_img.shape[:2]:
            restored_img = cv2.resize(restored_img, (clean_img.shape[1], clean_img.shape[0]))

        # 2. Đảm bảo kiểu dữ liệu là uint8
        if clean_img.dtype != np.uint8:
            clean_img = clean_img.astype(np.uint8)
        if restored_img.dtype != np.uint8:
            restored_img = restored_img.astype(np.uint8)

        # 3. Tính PSNR
        val_psnr = psnr(clean_img, restored_img, data_range=255)
        
        # 4. Tính SSIM
        if len(clean_img.shape) == 3:
            # Xử lý ảnh màu (RGB/BGR)
            val_ssim = ssim(clean_img, restored_img, channel_axis=2, data_range=255)
        else:
            # Xử lý ảnh xám
            val_ssim = ssim(clean_img, restored_img, data_range=255)

        return {
            "PSNR": round(float(val_psnr), 2),
            "SSIM": round(float(val_ssim), 4)
        }