"""One batch loop for both programmatic use and Qt workers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from time import perf_counter
from datetime import datetime
from tempfile import mkdtemp

import cv2
import numpy as np

from project_1.settings import DEFAULT_KERNEL_SIZE, DEFAULT_METHOD, DEFAULT_RADIUS, DEFAULT_SIGMA
from project_1.utils.image_io import list_images, read_image, write_image
from project_1.utils.reports import report_row, metric_delta, write_csv, summarize


@dataclass
class BatchResult:
    total: int
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    evaluated: int = 0
    cancelled: bool = False
    errors: dict[str, str] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    output_dir: str = ""


def combined_restore(
    img: np.ndarray, mask: np.ndarray | None = None,
    inpaint_radius: int = DEFAULT_RADIUS, inpaint_method: str = DEFAULT_METHOD,
    kernel_size: int = DEFAULT_KERNEL_SIZE, sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    """Compatibility helper: inpaint if a mask is supplied, then blur."""
    from project_1.algorithms.gaussian_filter import GaussianFilterAlgorithm
    from project_1.algorithms.inpainting import InpaintingAlgorithm

    result = img
    if mask is not None:
        result = InpaintingAlgorithm().process(img, mask, inpaint_radius, inpaint_method)
    return GaussianFilterAlgorithm().process(result, kernel_size, sigma)


class BatchProcessor:
    @staticmethod
    def process_directory(
        input_dir: str, output_dir: str, process_fn: Callable | None = None,
        mask_dir: str | None = None, progress_callback=None,
        inpaint_radius: int = DEFAULT_RADIUS, inpaint_method: str = DEFAULT_METHOD,
        kernel_size: int = DEFAULT_KERNEL_SIZE, sigma: float = DEFAULT_SIGMA,
        *, require_mask: bool = False, file_callback=None,
        error_callback=None, should_stop=None, clean_dir: str | None = None,
        evaluate_fn=None, algorithm_name: str = "combined", parameters: dict | None = None,
        result_callback=None,
    ) -> BatchResult:
        files = list_images(input_dir)
        names = [f"restored_{p.name}" + ("" if p.suffix.lower() == ".png" else ".png") for p in files]
        if len({name.casefold() for name in names}) != len(names):
            names = [f"restored_{index:06d}_{p.name}.png" for index, p in enumerate(files, 1)]
        output = Path(output_dir)
        if output.resolve() == Path(input_dir).resolve():
            raise ValueError("Thư mục đầu ra phải khác thư mục đầu vào.")
        if mask_dir and not Path(mask_dir).is_dir():
            raise ValueError(f"Thư mục mask không tồn tại: {mask_dir}")
        if clean_dir and not Path(clean_dir).is_dir():
            raise ValueError(f"Thư mục clean không tồn tại: {clean_dir}")
        for source in (mask_dir, clean_dir):
            if source and output.resolve() == Path(source).resolve():
                raise ValueError("Thư mục đầu ra phải khác thư mục clean/mask.")
        if require_mask and not mask_dir:
            raise ValueError("Thuật toán này yêu cầu thư mục mask.")
        output.mkdir(parents=True, exist_ok=True)
        output = Path(mkdtemp(prefix=datetime.now().strftime("run_%Y%m%d_%H%M%S_"), dir=output))
        result = BatchResult(total=len(files), output_dir=str(output.resolve()))
        parameters = parameters or {"kernel_size": kernel_size, "sigma": sigma,
                                    "radius": inpaint_radius, "method": inpaint_method}
        use_mask = require_mask or process_fn is None
        if process_fn is None:
            def process_fn(image, mask=None):
                return combined_restore(image, mask, inpaint_radius, inpaint_method, kernel_size, sigma)

        for index, path in enumerate(files, start=1):
            if should_stop and should_stop():
                result.cancelled = True
                break
            data = {"status": "success", "parameters": parameters, "baseline": None,
                    "metrics": None, "delta": None, "metrics_error": ""}
            saved_path = ""
            try:
                image = read_image(path)
                mask = None
                if mask_dir and use_mask:
                    mask_path = Path(mask_dir) / path.name
                    if mask_path.is_file():
                        mask = read_image(mask_path, grayscale=True)
                    elif require_mask:
                        data.update(status="skipped", error=f"Thiếu mask tương ứng: {path.name}")
                if data["status"] == "skipped":
                    result.skipped += 1
                    result.errors[path.name] = data["error"]
                    if error_callback:
                        error_callback(path.name, data["error"])
                    continue
                start = perf_counter()
                restored = process_fn(image, mask=mask) if mask_dir else process_fn(image)
                data["time"] = perf_counter() - start
                if restored is None:
                    raise ValueError("Thuật toán không trả về ảnh kết quả.")
                name = names[index - 1]
                saved_path = str((output / name).resolve())
                write_image(saved_path, restored)
                if clean_dir:
                    try:
                        clean_path = Path(clean_dir) / path.name
                        if not clean_path.is_file():
                            raise ValueError(f"Thiếu ảnh tham chiếu: {path.name}")
                        clean = read_image(clean_path)
                        if evaluate_fn is None:
                            from project_1.metrics.evaluator import QualityEvaluator
                            evaluate_fn = QualityEvaluator.evaluate
                        baseline = evaluate_fn(clean, image)
                        metrics = evaluate_fn(clean, restored)
                    except (ValueError, OSError, cv2.error) as exc:
                        data["metrics_error"] = str(exc)
                    else:
                        data.update(baseline=baseline, metrics=metrics, delta=metric_delta(baseline, metrics))
                        result.evaluated += 1
            except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                result.failed += 1
                data.update(status="failed", error=str(exc))
                saved_path = ""
                result.errors[path.name] = str(exc)
                if error_callback:
                    error_callback(path.name, str(exc))
            else:
                result.processed += 1
                if file_callback:
                    file_callback(path.name)
            finally:
                row = report_row(path.name, algorithm_name, data, path.resolve(), saved_path)
                result.rows.append(row)
                if result_callback:
                    result_callback(row)
                if progress_callback:
                    progress_callback(index, result.total)
        write_csv(output / "details.csv", result.rows)
        summaries = summarize(result.rows)
        if not summaries:
            summaries = [{"algorithm": algorithm_name, "attempted": 0, "processed": 0,
                          "skipped": 0, "failed": 0, "evaluated": 0}]
        for summary in summaries:
            summary.update(total=result.total, cancelled=result.cancelled,
                           parameters=report_row("", algorithm_name, {"parameters": parameters})["parameters"])
        write_csv(output / "summary.csv", summaries, tuple(summaries[0]))
        return result
