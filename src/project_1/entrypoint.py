import time
import numpy as np
from typing import Dict, Any

from project_1.algorithms.gaussian_filter import GaussianFilterAlgorithm
from project_1.algorithms.inpainting import InpaintingAlgorithm
from project_1.metrics.evaluator import QualityEvaluator
from project_1.utils.batch_processor import BatchProcessor

class RestorationPipeline:
    def __init__(self):
        # Khởi tạo tập hợp các thuật toán cơ bản
        self.algorithms = {
            "gaussian": GaussianFilterAlgorithm(),
            "inpainting": InpaintingAlgorithm()
        }
        self.evaluator = QualityEvaluator()

    def run_single(self, algorithm_name: str, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Chạy thuật toán khôi phục. 
        Hỗ trợ: 'gaussian', 'inpainting', 'combined'
        """
        # Trường hợp 1: Kết hợp (Inpainting xóa xước -> Gaussian khử nhiễu)
        if algorithm_name == "combined":
            mask = kwargs.get("mask")
            if mask is None:
                raise ValueError("Thuật toán 'combined' yêu cầu truyền tham số 'mask'.")

            # Tách tham số riêng cho từng bước
            inpaint_kwargs = {
                "mask": mask,
                "radius": kwargs.get("radius", 3),
                "method": kwargs.get("method", "telea")
            }
            gauss_kwargs = {
                "kernel_size": kwargs.get("kernel_size", 5),
                "sigma": kwargs.get("sigma", 1.2)
            }

            # Bước 1: Inpainting
            inpainted = self.algorithms["inpainting"].process(image, **inpaint_kwargs)
            # Bước 2: Gaussian Filter
            restored = self.algorithms["gaussian"].process(inpainted, **gauss_kwargs)
            return restored

        # Trường hợp 2: Các thuật toán đơn lẻ ('gaussian', 'inpainting')
        algo = self.algorithms.get(algorithm_name)
        if not algo:
            raise ValueError(f"Thuật toán '{algorithm_name}' không tồn tại trong Pipeline.")
        return algo.process(image, **kwargs)

    def evaluate(self, clean_img: np.ndarray, restored_img: np.ndarray) -> Dict[str, float]:
        """Tính chỉ số PSNR và SSIM"""
        return self.evaluator.evaluate(clean_img, restored_img)

    def compare_all(self, clean_img: np.ndarray, corrupted_img: np.ndarray, mask: np.ndarray = None) -> Dict[str, Any]:
        """Chạy tất cả phương pháp hiện có và trả về kết quả so sánh"""
        results = {}

        # 1. Lọc Gaussian đơn lẻ
        t0 = time.time()
        res_gauss = self.run_single("gaussian", corrupted_img, kernel_size=5, sigma=1.2)
        results["Gaussian Filter"] = {
            "image": res_gauss,
            "metrics": self.evaluate(clean_img, res_gauss),
            "time": round(time.time() - t0, 4)
        }

        # 2. Inpainting đơn lẻ (Chạy nếu có mask)
        if mask is not None:
            t0 = time.time()
            res_inpaint = self.run_single("inpainting", corrupted_img, mask=mask, radius=3, method='telea')
            results["Inpainting"] = {
                "image": res_inpaint,
                "metrics": self.evaluate(clean_img, res_inpaint),
                "time": round(time.time() - t0, 4)
            }

            # 3. Combined (Kết hợp Inpainting + Gaussian)
            t0 = time.time()
            res_comb = self.run_single(
                "combined", 
                corrupted_img, 
                mask=mask, 
                radius=3, 
                method='telea', 
                kernel_size=5, 
                sigma=1.2
            )
            results["Combined (Inpaint + Gauss)"] = {
                "image": res_comb,
                "metrics": self.evaluate(clean_img, res_comb),
                "time": round(time.time() - t0, 4)
            }

        return results

    def run_batch(self, input_dir: str, output_dir: str, algorithm_name: str, mask_dir: str = None, progress_callback=None, **kwargs):
        """Xử lý hàng loạt toàn bộ thư mục ảnh (Tự động chuyển mask cho Inpainting/Combined)"""
        def process_fn(img, mask=None):
            task_kwargs = kwargs.copy()
            if mask is not None:
                task_kwargs["mask"] = mask
            return self.run_single(algorithm_name, img, **task_kwargs)

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
        restored_g = pipeline.run_single("gaussian", corrupted, kernel_size=5, sigma=1.2)
        m_g = pipeline.evaluate(clean, restored_g)
        print(f"✅ Gaussian Filter -> PSNR: {m_g['PSNR']} dB | SSIM: {m_g['SSIM']}")

        # Test Combined nếu có mask
        if mask is not None:
            restored_c = pipeline.run_single("combined", corrupted, mask=mask, radius=3, method='telea', kernel_size=5, sigma=1.2)
            m_c = pipeline.evaluate(clean, restored_c)
            print(f"✅ Combined Restoration -> PSNR: {m_c['PSNR']} dB | SSIM: {m_c['SSIM']}")

        # Test So Sánh
        res_comp = pipeline.compare_all(clean, corrupted, mask=mask)
        print("✅ So sánh thành công các phương pháp:", list(res_comp.keys()))
    else:
        print("⚠️ Chưa tìm thấy file ảnh test trong thư mục dataset!")