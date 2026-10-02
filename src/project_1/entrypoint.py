"""Application services shared by the desktop UI, scripts and tests."""

from time import perf_counter
from typing import Any

import numpy as np

from project_1.algorithms.gaussian_filter import GaussianFilterAlgorithm
from project_1.algorithms.inpainting import InpaintingAlgorithm
from project_1.algorithms.unsharp_mask import UnsharpMaskAlgorithm, validate_sharpen_parameters
from project_1.metrics.evaluator import QualityEvaluator
from project_1.settings import ALGORITHMS, DEFAULT_KERNEL_SIZE, DEFAULT_SIGMA, DEFAULT_RADIUS, DEFAULT_METHOD
from project_1.settings import DEFAULT_SHARPEN_AMOUNT, DEFAULT_SHARPEN_SIGMA
from project_1.utils.batch_processor import BatchProcessor, BatchResult
from project_1.utils.reports import metric_delta


class RestorationPipeline:
    def __init__(self):
        self.algorithms = {
            "gaussian": GaussianFilterAlgorithm(),
            "inpainting": InpaintingAlgorithm(),
        }
        self.evaluator = QualityEvaluator()
        self.sharpener = UnsharpMaskAlgorithm()

    def run_single(self, algorithm_name: str, image: np.ndarray, **kwargs) -> np.ndarray:
        enabled = kwargs.get("sharpen_enabled", False)
        amount = kwargs.get("sharpen_amount", DEFAULT_SHARPEN_AMOUNT)
        sigma = kwargs.get("sharpen_sigma", DEFAULT_SHARPEN_SIGMA)
        validate_sharpen_parameters(amount, sigma)
        if algorithm_name == "combined":
            inpainted = self.algorithms["inpainting"].process(image, **kwargs)
            restored = self.algorithms["gaussian"].process(inpainted, **kwargs)
        elif algorithm_name not in self.algorithms:
            raise ValueError(f"Thuật toán '{algorithm_name}' không tồn tại trong Pipeline.")
        else:
            restored = self.algorithms[algorithm_name].process(image, **kwargs)
        return self.sharpener.process(restored, amount=amount, sigma=sigma) if enabled else restored

    def evaluate(self, clean_img: np.ndarray, restored_img: np.ndarray) -> dict[str, float]:
        return self.evaluator.evaluate(clean_img, restored_img)

    @staticmethod
    def parameters(**kwargs) -> dict:
        kernel = max(3, int(kwargs.get("kernel_size", DEFAULT_KERNEL_SIZE)))
        return {"kernel_size": kernel + (kernel % 2 == 0),
                "sigma": kwargs.get("sigma", DEFAULT_SIGMA),
                "radius": max(1, int(kwargs.get("radius", DEFAULT_RADIUS))),
                "method": str(kwargs.get("method", DEFAULT_METHOD)).lower(),
                "sharpen_enabled": bool(kwargs.get("sharpen_enabled", False)),
                "sharpen_amount": kwargs.get("sharpen_amount", DEFAULT_SHARPEN_AMOUNT),
                "sharpen_sigma": kwargs.get("sharpen_sigma", DEFAULT_SHARPEN_SIGMA)}

    def restore_result(self, algorithm_name: str, image: np.ndarray,
                       clean_img: np.ndarray | None = None, mask=None,
                       baseline: dict[str, float] | None = None, **kwargs) -> dict:
        parameters = self.parameters(**kwargs)
        start = perf_counter()
        restored = self.run_single(algorithm_name, image, mask=mask, **parameters)
        result = {"image": restored, "time": perf_counter() - start,
                  "status": "success", "parameters": parameters,
                  "baseline": None, "metrics": None, "delta": None, "metrics_error": ""}
        if clean_img is not None:
            try:
                baseline = baseline if baseline is not None else self.evaluate(clean_img, image)
                metrics = self.evaluate(clean_img, restored)
            except ValueError as exc:
                result["metrics_error"] = str(exc)
            else:
                result.update(baseline=baseline, metrics=metrics, delta=metric_delta(baseline, metrics))
        return result

    def compare_all(
        self, clean_img: np.ndarray | None, corrupted_img: np.ndarray,
        mask: np.ndarray | None = None, **kwargs,
    ) -> dict[str, Any]:
        """Compare eligible methods using the same parameters; time processing only."""
        results = {}
        baseline = None
        if clean_img is not None:
            try:
                baseline = self.evaluate(clean_img, corrupted_img)
            except ValueError:
                # Keep the existing per-result validation/error behavior.
                baseline = None
        for spec in ALGORITHMS:
            if spec.needs_mask and mask is None:
                results[spec.label] = {"image": None, "status": "skipped", "metrics": None,
                                       "parameters": self.parameters(**kwargs),
                                       "error": "Thiếu mask: không chạy phương pháp này."}
                continue
            try:
                results[spec.label] = self.restore_result(
                    spec.key, corrupted_img, clean_img, mask=mask, baseline=baseline, **kwargs)
            except (ValueError, RuntimeError) as exc:
                results[spec.label] = {"image": None, "status": "failed", "metrics": None,
                                       "parameters": self.parameters(**kwargs), "error": str(exc)}
        return results

    def run_batch(
        self, input_dir: str, output_dir: str, algorithm_name: str,
        mask_dir: str | None = None, progress_callback=None,
        file_callback=None, error_callback=None, should_stop=None,
        clean_dir: str | None = None, result_callback=None, **kwargs,
    ) -> BatchResult:
        specs = {spec.key: spec for spec in ALGORITHMS}
        if algorithm_name not in specs:
            raise ValueError(f"Thuật toán '{algorithm_name}' không tồn tại trong Pipeline.")

        def process_fn(image, mask=None):
            task_kwargs = kwargs.copy()
            if mask is not None:
                task_kwargs["mask"] = mask
            return self.run_single(algorithm_name, image, **task_kwargs)

        return BatchProcessor.process_directory(
            input_dir, output_dir, process_fn, mask_dir=mask_dir,
            require_mask=specs[algorithm_name].needs_mask and kwargs.get("mask") is None,
            progress_callback=progress_callback, file_callback=file_callback,
            error_callback=error_callback, should_stop=should_stop,
            clean_dir=clean_dir, evaluate_fn=self.evaluate, algorithm_name=algorithm_name,
            parameters=self.parameters(**kwargs), result_callback=result_callback,
        )
