"""Parameter advice based on reference metrics or cautious image heuristics."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from project_1.settings import ALGORITHMS
from project_1.utils.image_io import validate_image


class ParameterAdvisor:
    """Suggest nearby parameter changes without changing the active configuration."""

    PSNR_EPSILON = 0.01
    SSIM_EPSILON = 0.0005

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def analyze(self, algorithm_name: str, image: np.ndarray, clean_img=None,
                mask=None, parameters: dict | None = None,
                current_result: dict | None = None) -> dict[str, Any]:
        validate_image(image)
        parameters = self.pipeline.parameters(**(parameters or {}))
        if clean_img is None:
            return {
                "mode": "heuristic",
                "suggestions": self._heuristic_suggestions(
                    algorithm_name, image, mask, parameters, current_result),
                "tradeoffs": [],
            }

        baseline = (current_result or {}).get("baseline")
        if baseline is None:
            baseline = self.pipeline.evaluate(clean_img, image)
        if current_result is None or current_result.get("metrics") is None:
            current_result = self.pipeline.restore_result(
                algorithm_name, image, clean_img, mask=mask,
                baseline=baseline, **parameters)
        current_metrics = current_result.get("metrics")
        if current_metrics is None:
            return {"mode": "reference", "suggestions": [], "tradeoffs": [],
                    "error": current_result.get("metrics_error", "Không tính được metric.")}

        suggestions, tradeoffs = [], []
        for candidate in self._candidates(algorithm_name, parameters):
            try:
                result = self.pipeline.restore_result(
                    algorithm_name, image, clean_img, mask=mask,
                    baseline=baseline, **candidate["parameters"])
            except (ValueError, RuntimeError):
                continue
            metrics = result.get("metrics")
            if metrics is None:
                continue
            delta_psnr = metrics["PSNR"] - current_metrics["PSNR"]
            delta_ssim = metrics["SSIM"] - current_metrics["SSIM"]
            item = {
                "parameter": candidate["parameter"],
                "label": candidate["label"],
                "direction": candidate["direction"],
                "value": candidate["parameters"].get(candidate["parameter"]),
                "metrics": metrics,
                "delta": {"PSNR": delta_psnr, "SSIM": delta_ssim},
            }
            if delta_psnr > self.PSNR_EPSILON and delta_ssim > self.SSIM_EPSILON:
                suggestions.append(item)
            elif ((delta_psnr > self.PSNR_EPSILON and delta_ssim < -self.SSIM_EPSILON) or
                  (delta_ssim > self.SSIM_EPSILON and delta_psnr < -self.PSNR_EPSILON)):
                tradeoffs.append(item)

        suggestions.sort(key=lambda item: (
            item["delta"]["PSNR"] / self.PSNR_EPSILON +
            item["delta"]["SSIM"] / self.SSIM_EPSILON), reverse=True)
        return {"mode": "reference", "suggestions": suggestions[:5],
                "tradeoffs": tradeoffs[:5], "current": current_metrics}

    def analyze_compare(self, image: np.ndarray, clean_img=None, mask=None,
                        parameters: dict | None = None,
                        results: dict | None = None) -> dict[str, dict[str, Any]]:
        results = results or {}
        output = {}
        for spec in ALGORITHMS:
            current = results.get(spec.label)
            if current and current.get("status") != "success":
                output[spec.label] = {"mode": "heuristic", "suggestions": [],
                                      "tradeoffs": [], "error": current.get("error", "")}
                continue
            if spec.needs_mask and mask is None:
                output[spec.label] = {"mode": "heuristic", "suggestions": [],
                                      "tradeoffs": [], "error": "Thiếu mask."}
                continue
            output[spec.label] = self.analyze(
                spec.key, image, clean_img, mask, parameters, current)
        return output

    @staticmethod
    def _candidates(algorithm_name: str, parameters: dict) -> list[dict]:
        candidates = []
        bounds = {
            "kernel_size": (3, 31), "sigma": (0.1, 10.0), "radius": (1, 20),
            "sharpen_amount": (0.0, 2.0), "sharpen_sigma": (0.3, 3.0),
        }

        def add(parameter, label, direction, value):
            if value == parameters.get(parameter):
                return
            if parameter in bounds and not bounds[parameter][0] <= value <= bounds[parameter][1]:
                return
            candidate = parameters.copy()
            candidate[parameter] = value
            candidates.append({"parameter": parameter, "label": label,
                               "direction": direction, "parameters": candidate})

        if algorithm_name in {"gaussian", "combined"}:
            kernel = parameters["kernel_size"]
            add("kernel_size", "Kernel", "tăng", kernel + 2)
            add("kernel_size", "Kernel", "giảm", kernel - 2)
            sigma = parameters["sigma"]
            add("sigma", "Sigma Gaussian", "tăng", round(sigma + 0.1, 1))
            add("sigma", "Sigma Gaussian", "giảm", round(sigma - 0.1, 1))
        if algorithm_name in {"inpainting", "combined"}:
            radius = parameters["radius"]
            add("radius", "Bán kính inpaint", "tăng", radius + 1)
            add("radius", "Bán kính inpaint", "giảm", radius - 1)
            add("method", "Inpaint method", "đổi", "navier-stokes"
                if parameters["method"] == "telea" else "telea")
        if parameters["sharpen_enabled"]:
            amount = parameters["sharpen_amount"]
            add("sharpen_amount", "Mức tăng nét", "tăng", round(amount + 0.05, 2))
            add("sharpen_amount", "Mức tăng nét", "giảm", round(amount - 0.05, 2))
            sigma = parameters["sharpen_sigma"]
            add("sharpen_sigma", "Sigma tăng nét", "tăng", round(sigma + 0.1, 1))
            add("sharpen_sigma", "Sigma tăng nét", "giảm", round(sigma - 0.1, 1))
        else:
            add("sharpen_enabled", "Tăng nét sau phục hồi", "bật", True)
        return candidates

    @classmethod
    def _heuristic_suggestions(cls, algorithm_name, image, mask, parameters, current_result):
        validate_image(image)
        output = (current_result or {}).get("image")
        if output is None:
            return []
        input_gray = cls._gray(image)
        output_gray = cls._gray(output)
        input_noise = cls._high_frequency_std(input_gray)
        output_noise = cls._high_frequency_std(output_gray)
        input_detail = float(cv2.Laplacian(input_gray, cv2.CV_32F).var())
        output_detail = float(cv2.Laplacian(output_gray, cv2.CV_32F).var())
        suggestions = []

        if algorithm_name in {"gaussian", "combined"}:
            if input_noise > 12 and output_detail > input_detail * 0.55:
                suggestions.append({"label": "Gaussian", "direction": "tăng",
                                     "reason": "Nhiễu cao và chi tiết biên vẫn còn khá tốt; thử tăng kernel hoặc sigma.",
                                     "confidence": "thấp"})
            elif output_detail < input_detail * 0.55:
                suggestions.append({"label": "Gaussian", "direction": "giảm",
                                     "reason": "Ảnh sau phục hồi có dấu hiệu mềm hơn nhiều; thử giảm kernel hoặc sigma.",
                                     "confidence": "thấp"})
        if algorithm_name in {"inpainting", "combined"} and mask is not None:
            coverage = float(np.count_nonzero(mask)) / mask.size
            if coverage > 0.03:
                suggestions.append({"label": "Bán kính inpaint", "direction": "tăng",
                                     "reason": "Vùng mask tương đối rộng; bán kính lớn hơn có thể nối vùng hỏng tốt hơn.",
                                     "confidence": "thấp"})
            elif coverage < 0.005:
                suggestions.append({"label": "Bán kính inpaint", "direction": "giảm",
                                     "reason": "Mask rất mảnh/nhỏ; bán kính lớn dễ làm lan chi tiết xung quanh.",
                                     "confidence": "thấp"})
        if output_detail < input_detail * 0.55 and not parameters["sharpen_enabled"]:
            suggestions.append({"label": "Tăng nét sau phục hồi", "direction": "bật",
                                 "reason": "Độ tương phản biên giảm rõ sau phục hồi; có thể thử bật tăng nét nhẹ.",
                                 "confidence": "thấp"})
        elif output_noise > max(input_noise * 1.15, input_noise + 2):
            suggestions.append({"label": "Mức tăng nét", "direction": "giảm",
                                 "reason": "Năng lượng chi tiết cao tần tăng; có thể đang làm nổi nhiễu hoặc halo.",
                                 "confidence": "thấp"})
        return suggestions[:5]

    @staticmethod
    def _gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    @staticmethod
    def _high_frequency_std(gray):
        smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        return float(np.std(gray.astype(np.float32) - smooth.astype(np.float32)))
