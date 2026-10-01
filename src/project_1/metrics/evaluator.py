"""Full-reference quality metrics for aligned uint8 images."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from project_1.utils.image_io import validate_image


class QualityEvaluator:
    @staticmethod
    def evaluate(clean_img: np.ndarray, restored_img: np.ndarray) -> dict[str, float]:
        validate_image(clean_img, "Ảnh gốc")
        validate_image(restored_img, "Ảnh phục hồi")
        if clean_img.shape != restored_img.shape:
            raise ValueError("Hai ảnh đánh giá phải cùng kích thước và số kênh; không tự resize ảnh để tính điểm.")
        min_side = min(clean_img.shape[:2])
        if min_side < 3:
            raise ValueError("Ảnh đánh giá SSIM cần kích thước tối thiểu 3×3.")
        win_size = min(7, min_side if min_side % 2 else min_side - 1)
        # Identical images have zero error, so PSNR is positive infinity.
        psnr = float("inf") if np.array_equal(clean_img, restored_img) else float(
            peak_signal_noise_ratio(clean_img, restored_img, data_range=255)
        )
        ssim = structural_similarity(
            clean_img, restored_img,
            channel_axis=2 if clean_img.ndim == 3 else None,
            data_range=255, win_size=win_size,
        )
        return {"PSNR": psnr, "SSIM": float(ssim)}
