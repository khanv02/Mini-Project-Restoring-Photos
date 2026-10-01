"""Evaluate held-out scratch masks and Combined sharpening on saved PNG outputs."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import cv2
import numpy as np
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from project_1.algorithms.scratch_mask import DEFAULT_MASK_PARAMETERS, detect_scratch_mask, mask_overlay, mask_summary
from project_1.entrypoint import RestorationPipeline
from project_1.generate_data import add_gaussian_noise, add_scratches_and_stains_with_mask
from project_1.metrics.mask_evaluator import binary_mask_scores
from project_1.utils.image_io import list_images, read_image, write_image
from project_1.utils.reports import summarize, write_csv


def independent_metrics(reference, restored):
    return {
        "PSNR": float(peak_signal_noise_ratio(reference, restored, data_range=255)),
        "SSIM": float(structural_similarity(reference, restored, channel_axis=2, data_range=255)),
    }


def verify_saved(reference, path, metrics):
    saved = read_image(path)
    independent = independent_metrics(reference, saved)
    np.testing.assert_allclose(list(metrics.values()), list(independent.values()), rtol=0, atol=1e-12)
    return saved


def validate_sharpening(dataset, root, pipeline):
    files = list_images(dataset / "images_corrupted")
    if not files:
        raise ValueError("Dataset is empty")
    rows, images = [], {}
    for enabled in (False, True):
        label = "on" if enabled else "off"
        def progress(current, total):
            if current % 25 == 0 or current == total:
                print(f"sharpen {label}: {current}/{total}", flush=True)
        batch = pipeline.run_batch(
            str(dataset / "images_corrupted"), str(root / ("sharpen_" + label)), "combined",
            mask_dir=str(dataset / "images_masks"), clean_dir=str(dataset / "images_clean"),
            sharpen_enabled=enabled, sharpen_amount=0.5, sharpen_sigma=1.0, progress_callback=progress)
        if batch.processed != len(files) or batch.evaluated != len(files):
            raise AssertionError(f"Batch failures: {batch.errors}")
        for row in batch.rows:
            reference = read_image(dataset / "images_clean" / row["filename"])
            verify_saved(reference, row["output_path"], {key: row[key] for key in ("PSNR", "SSIM")})
            row = row.copy()
            row["algorithm"] = "Combined / sharpen " + label
            rows.append(row)
        images[label] = {row["filename"]: row for row in batch.rows}
    write_csv(root / "sharpen_results.csv", rows)
    summaries = summarize(rows)
    write_csv(root / "sharpen_summary.csv", summaries, tuple(summaries[0]))
    differences = []
    for filename, after in images["on"].items():
        before = images["off"][filename]
        differences.append({"filename": filename,
            "off_PSNR": before["PSNR"], "on_PSNR": after["PSNR"],
            "delta_PSNR": after["PSNR"] - before["PSNR"],
            "off_SSIM": before["SSIM"], "on_SSIM": after["SSIM"],
            "delta_SSIM": after["SSIM"] - before["SSIM"]})
    write_csv(root / "sharpen_differences.csv", differences, tuple(differences[0]))
    comparison = {"images": len(differences)}
    for key in ("PSNR", "SSIM"):
        deltas = [row["delta_" + key] for row in differences]
        comparison.update({"improved_" + key: sum(v > 0 for v in deltas),
                           "decreased_" + key: sum(v < 0 for v in deltas),
                           "unchanged_" + key: sum(v == 0 for v in deltas)})
    return summaries, comparison


def validate_masks(dataset, root, pipeline, *, seed, skip):
    all_files = list_images(dataset / "images_clean")
    files = all_files[skip:]
    if not files:
        raise ValueError("No holdout images remain after --skip-calibration")
    directories = {name: root / "scratch_holdout" / name for name in
                   ("damaged", "truth", "predicted", "overlay", "restored_predicted", "restored_truth")}
    for directory in directories.values():
        directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rows = []
    for index, path in enumerate(files, 1):
        clean = read_image(path)
        image, truth = add_scratches_and_stains_with_mask(clean, 6, 0, rng=rng)
        image = add_gaussian_noise(image, var=0.003, rng=rng)
        start = perf_counter()
        predicted = detect_scratch_mask(image, **DEFAULT_MASK_PARAMETERS)
        seconds = perf_counter() - start
        scores, stats = binary_mask_scores(predicted, truth), mask_summary(predicted)
        predicted_result = pipeline.restore_result("inpainting", image, clean, mask=predicted)
        truth_result = pipeline.restore_result("inpainting", image, clean, mask=truth)
        name = f"{index:04d}_{path.name}.png"
        arrays = {"damaged": image, "truth": truth, "predicted": predicted,
                  "overlay": mask_overlay(image, predicted),
                  "restored_predicted": predicted_result["image"], "restored_truth": truth_result["image"]}
        for kind, array in arrays.items():
            write_image(directories[kind] / name, array)
        for kind, array in (("predicted", predicted), ("truth", truth)):
            np.testing.assert_array_equal(read_image(directories[kind] / name, grayscale=True), array)
        for kind, result in (("restored_predicted", predicted_result), ("restored_truth", truth_result)):
            if result["metrics"] is None:
                raise AssertionError(result["metrics_error"])
            verify_saved(clean, directories[kind] / name, result["metrics"])
        row = {"filename": path.name, "generated_filename": name,
               "clean_sha256": hashlib.sha256(clean.tobytes()).hexdigest(), **scores, **stats,
               "detection_seconds": seconds,
               "parameters": json.dumps(DEFAULT_MASK_PARAMETERS, sort_keys=True)}
        for prefix, metrics in (("damaged", predicted_result["baseline"]),
                                ("predicted", predicted_result["metrics"]), ("truth", truth_result["metrics"])):
            row.update({prefix + "_" + key: metrics[key] for key in ("PSNR", "SSIM")})
        rows.append(row)
        if index % 15 == 0 or index == len(files):
            print(f"scratch holdout: {index}/{len(files)}", flush=True)
    write_csv(root / "mask_results.csv", rows, tuple(rows[0]))
    summary = {"images": len(rows)}
    for key in ("precision", "recall", "IoU", "coverage", "detection_seconds",
                "damaged_PSNR", "damaged_SSIM", "predicted_PSNR", "predicted_SSIM", "truth_PSNR", "truth_SSIM"):
        summary["mean_" + key] = float(np.mean([row[key] for row in rows]))
    summary["empty_predictions"] = sum(row["pixels"] == 0 for row in rows)
    write_csv(root / "mask_summary.csv", [summary], tuple(summary))
    return summary, [path.name for path in all_files[:skip]], [path.name for path in files]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--skip-calibration", type=int, default=10)
    args = parser.parse_args()
    if args.skip_calibration < 0:
        parser.error("--skip-calibration must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix=datetime.now().strftime("enhancements_%Y%m%d_%H%M%S_"), dir=args.output))
    pipeline = RestorationPipeline()
    sharpening, comparison = validate_sharpening(args.dataset, root, pipeline)
    masks, calibration, holdout = validate_masks(args.dataset, root, pipeline, seed=args.seed, skip=args.skip_calibration)
    report = {"dataset": str(args.dataset.resolve()), "numpy": np.__version__, "opencv": cv2.__version__,
              "scikit_image": skimage.__version__, "seed": args.seed,
              "generation": {"scratches": 6, "stains": 0, "variance": 0.003},
              "detector": DEFAULT_MASK_PARAMETERS, "calibration_files_excluded": calibration,
              "holdout_files": holdout, "sharpening": sharpening,
              "sharpening_vs_off": comparison, "mask_holdout": masks}
    with (root / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"sharpening_vs_off": comparison, "mask_holdout": masks}, indent=2), flush=True)
    print(f"Saved PNGs and independently verified metrics: {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
