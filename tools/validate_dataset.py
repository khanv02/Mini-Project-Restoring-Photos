"""Run all methods, save CSV/PNG, and independently check every saved result."""

import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from project_1.entrypoint import RestorationPipeline
from project_1.settings import ALGORITHMS
from project_1.utils.image_io import list_images, read_image
from project_1.utils.reports import summarize, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    clean, damaged, masks = [args.dataset / name for name in
                             ("images_clean", "images_corrupted", "images_masks")]
    files = list_images(damaged)
    if not files:
        raise ValueError("Dataset is empty")
    hashes = set()
    for path in files:
        reference = read_image(clean / path.name)
        image = read_image(path)
        mask = read_image(masks / path.name, grayscale=True)
        if reference.shape != image.shape or image.shape[:2] != mask.shape:
            raise ValueError(f"Misaligned triple: {path.name}")
        if not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError(f"Non-binary mask: {path.name}")
        hashes.add(hashlib.sha256(reference.tobytes()).hexdigest())
    print(f"Validated {len(files)} triples, {len(hashes)} unique clean images", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix=datetime.now().strftime("validation_%Y%m%d_%H%M%S_"), dir=args.output))
    pipeline = RestorationPipeline()
    rows = []
    for spec in ALGORITHMS:
        def progress(current, total):
            if current % 25 == 0 or current == total:
                print(f"{spec.key}: {current}/{total}", flush=True)
        result = pipeline.run_batch(
            str(damaged), str(root / spec.key), spec.key,
            clean_dir=str(clean), mask_dir=str(masks) if spec.needs_mask else None,
            progress_callback=progress)
        if result.processed != len(files) or result.evaluated != len(files):
            raise AssertionError(f"Batch failures: {result.errors}")
        for row in result.rows:
            reference = read_image(clean / row["filename"])
            restored = read_image(row["output_path"])
            psnr = peak_signal_noise_ratio(reference, restored, data_range=255)
            ssim = structural_similarity(reference, restored, channel_axis=2, data_range=255)
            np.testing.assert_allclose([row["PSNR"], row["SSIM"]], [psnr, ssim], rtol=0, atol=1e-12)
        rows.extend(result.rows)
    summaries = summarize(rows)
    write_csv(root / "all_results.csv", rows)
    write_csv(root / "all_summary.csv", summaries, tuple(summaries[0]))
    for summary in summaries:
        print(f"{summary['algorithm']}: PSNR={summary['mean_PSNR']:.4f}, "
              f"SSIM={summary['mean_SSIM']:.6f}, verified={summary['evaluated']}", flush=True)
    print(f"Saved and independently verified {len(rows)} results: {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
