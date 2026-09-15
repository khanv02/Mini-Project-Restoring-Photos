import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class QualityEvaluator:
    @staticmethod
    def evaluate(clean_img: np.ndarray, restored_img: np.ndarray) -> dict:
        """Tính chỉ số PSNR và SSIM"""
        if clean_img.shape != restored_img.shape:
            restored_img = cv2.resize(restored_img, (clean_img.shape[1], clean_img.shape[0]))

        val_psnr = psnr(clean_img, restored_img, data_range=255)
        
        if len(clean_img.shape) == 3:
            val_ssim = ssim(clean_img, restored_img, channel_axis=2, data_range=255)
        else:
            val_ssim = ssim(clean_img, restored_img, data_range=255)

        return {
            "PSNR": round(float(val_psnr), 2),
            "SSIM": round(float(val_ssim), 4)
        }