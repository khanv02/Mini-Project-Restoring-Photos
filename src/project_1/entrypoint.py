import time
import numpy as np
from typing import Dict, Any

from project_1.algorithms.gaussian_filter import GaussianFilterAlgorithm
from project_1.algorithms.inpainting import InpaintingAlgorithm
from project_1.metrics.evaluator import QualityEvaluator
from project_1.utils.batch_processor import BatchProcessor

class RestorationPipeline:
    def __init__(self):
        # Khởi tạo tập hợp các thuật toán (Chỉ giữ Gaussian và Inpainting)
        self.algorithms = {
            "gaussian": GaussianFilterAlgorithm(),
            "inpainting": InpaintingAlgorithm()
        }
        self.evaluator = QualityEvaluator()

    def run_single(self, algorithm_name: str, image: np.ndarray, **kwargs) -> np.ndarray:
        """Chạy 1 thuật toán khôi phục"""
        algo = self.algorithms.get(algorithm_name)
        if not algo:
            raise ValueError(f"Thuật toán '{algorithm_name}' không tồn tại trong Pipeline.")
        return algo.process(image, **kwargs)

    def evaluate(self, clean_img: np.ndarray, restored_img: np.ndarray) -> Dict[str, float]:
        """Tính chỉ số PSNR và SSIM"""
        return self.evaluator.evaluate(clean_img, restored_img)

    def compare_all(self, clean_img: np.ndarray, corrupted_img: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """Chạy tất cả thuật toán hiện có và trả về kết quả so sánh"""
        results = {}

        # 1. Lọc Gaussian
        t0 = time.time()
        res_gauss = self.run_single("gaussian", corrupted_img, kernel_size=5, sigma=1.2)
        results["Gaussian Filter"] = {
            "image": res_gauss,
            "metrics": self.evaluate(clean_img, res_gauss),
            "time": round(time.time() - t0, 4)
        }

        # 2. Inpainting (Chạy nếu có cung cấp mask)
        if mask is not None:
            t0 = time.time()
            res_inpaint = self.run_single("inpainting", corrupted_img, mask=mask, radius=3, method='telea')
            results["Inpainting"] = {
                "image": res_inpaint,
                "metrics": self.evaluate(clean_img, res_inpaint),
                "time": round(time.time() - t0, 4)
            }

        return results

    def run_batch(self, input_dir: str, output_dir: str, algorithm_name: str, mask_dir: str = None, progress_callback=None, **kwargs):
        """Xử lý hàng loạt toàn bộ thư mục ảnh (Hỗ trợ truyền mask_dir cho Inpainting)"""
        def process_fn(img, mask=None):
            if mask is not None:
                kwargs["mask"] = mask
            return self.run_single(algorithm_name, img, **kwargs)

        BatchProcessor.process_directory(
            input_dir=input_dir, 
            output_dir=output_dir, 
            process_fn=process_fn, 
            mask_dir=mask_dir, 
            progress_callback=progress_callback
        )


if __name__ == "__main__":
    import cv2
    import os

    print("--- KIỂM THỬ RESTORATION PIPELINE ---")
    pipeline = RestorationPipeline()

    clean_path = "dataset/images_clean/001.png"
    corrupted_path = "dataset/images_corrupted/001.png"
    mask_path = "dataset/masks/001.png"

    if os.path.exists(clean_path) and os.path.exists(corrupted_path):
        clean = cv2.imread(clean_path)
        corrupted = cv2.imread(corrupted_path)
        mask = cv2.imread(mask_path) if os.path.exists(mask_path) else None

        # Test Gaussian
        restored = pipeline.run_single("gaussian", corrupted, kernel_size=5, sigma=1.2)
        metrics = pipeline.evaluate(clean, restored)
        print(f"✅ Gaussian Filter -> PSNR: {metrics['PSNR']} dB | SSIM: {metrics['SSIM']}")

        # Test So Sánh
        res_comp = pipeline.compare_all(clean, corrupted, mask=mask)
        print("✅ So sánh thành công các phương pháp:", list(res_comp.keys()))
    else:
        print("⚠️ Chưa tìm thấy file ảnh test trong thư mục dataset!")