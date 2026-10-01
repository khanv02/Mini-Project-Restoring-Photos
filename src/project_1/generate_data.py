"""Reproducible synthetic damage; new destinations only, never overwrite data."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np

from project_1.utils.image_io import list_images, read_image, validate_image, write_image


def add_gaussian_noise(img: np.ndarray, mean: float = 0, var: float = 0.005,
                       *, rng: np.random.Generator | None = None) -> np.ndarray:
    validate_image(img)
    if not np.isfinite(var) or var < 0 or not np.isfinite(mean):
        raise ValueError("Mean/variance phải hữu hạn, variance không âm.")
    rng = rng if rng is not None else np.random.default_rng()
    noise = rng.normal(mean, var ** 0.5, img.shape)
    return (np.clip(img / 255.0 + noise, 0, 1) * 255).astype(np.uint8)


def add_scratches_and_stains_with_mask(img: np.ndarray, num_scratches: int = 5,
                                      num_stains: int = 2, *,
                                      rng: np.random.Generator | None = None):
    validate_image(img)
    rng = rng if rng is not None else np.random.default_rng()
    corrupted = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(num_scratches):
        pt1 = (int(rng.integers(w)), int(rng.integers(h)))
        pt2 = (int(rng.integers(w)), int(rng.integers(h)))
        thickness = int(rng.integers(1, 3))
        color = (255, 255, 255) if img.ndim == 3 else 255
        cv2.line(corrupted, pt1, pt2, color, thickness)
        cv2.line(mask, pt1, pt2, 255, thickness)
    for _ in range(num_stains):
        center = (int(rng.integers(w)), int(rng.integers(h)))
        axes = (int(rng.integers(10, 30)), int(rng.integers(10, 30)))
        angle = int(rng.integers(360))
        color = (120, 160, 180) if img.ndim == 3 else 160
        cv2.ellipse(corrupted, center, axes, angle, 0, 360, color, -1)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    return corrupted, mask


def create_corrupted_dataset(input_dir: str, output_dir: str, mask_dir: str, *,
                             seed: int = 42, clean_dir: str | None = None,
                             variance: float = 0.003, num_scratches: int = 6,
                             num_stains: int = 2) -> dict:
    files = list_images(input_dir)
    if not files:
        raise ValueError(f"Không tìm thấy ảnh trong {input_dir}")
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("Variance phải hữu hạn không âm.")
    if num_scratches < 0 or num_stains < 0:
        raise ValueError("Số vết xước / vết ố không được âm.")
    output, masks = Path(output_dir), Path(mask_dir)
    clean = Path(clean_dir) if clean_dir else output.parent / "images_clean_generated"
    destinations = [output.resolve(), masks.resolve(), clean.resolve()]
    if len(set(destinations)) != 3 or Path(input_dir).resolve() in destinations:
        raise ValueError("Các thư mục clean/corrupted/mask đầu ra phải riêng biệt và khác ảnh nguồn.")
    names = [p.name if p.suffix.lower() == ".png" else p.name + ".png" for p in files]
    if len({name.casefold() for name in names}) != len(names):
        names = [f"{index:06d}_{p.name}.png" for index, p in enumerate(files, 1)]
    targets = [directory / name for directory in (output, masks, clean) for name in names]
    config_path = output / "generation.json"
    if any(p.exists() for p in [*targets, config_path]):
        raise FileExistsError("Đích đã có dữ liệu. Hãy chọn thư mục mới; không ghi đè dataset.")
    rng = np.random.default_rng(seed)
    for directory in (output, masks, clean):
        directory.mkdir(parents=True, exist_ok=True)
    for path, name in zip(files, names):
        image = read_image(path)
        damaged, mask = add_scratches_and_stains_with_mask(
            image, num_scratches, num_stains, rng=rng)
        damaged = add_gaussian_noise(damaged, var=variance, rng=rng)
        write_image(clean / name, image)
        write_image(output / name, damaged)
        write_image(masks / name, mask)
    config = {"seed": seed, "variance": variance, "num_scratches": num_scratches,
              "num_stains": num_stains, "numpy_version": np.__version__,
              "opencv_version": cv2.__version__,
              "input_dir": str(Path(input_dir).resolve()),
              "clean_dir": str(clean.resolve()), "corrupted_dir": str(output.resolve()),
              "mask_dir": str(masks.resolve()),
              "files": [{"source": p.name, "generated": n} for p, n in zip(files, names)]}
    with config_path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    return config


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/images_clean")
    parser.add_argument("--output", help="New dataset root; omitted: unique dataset/generated/run_*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance", type=float, default=0.003)
    parser.add_argument("--scratches", type=int, default=6)
    parser.add_argument("--stains", type=int, default=2)
    args = parser.parse_args()
    if args.output:
        root = Path(args.output)
        if root.exists():
            parser.error("--output phải là thư mục mới, chưa tồn tại.")
    else:
        base = Path("dataset/generated")
        base.mkdir(parents=True, exist_ok=True)
        root = Path(mkdtemp(prefix=datetime.now().strftime("run_%Y%m%d_%H%M%S_"), dir=base))
    try:
        config = create_corrupted_dataset(
            args.input, str(root / "images_corrupted"), str(root / "images_masks"),
            clean_dir=str(root / "images_clean"), seed=args.seed, variance=args.variance,
            num_scratches=args.scratches, num_stains=args.stains)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {len(config['files'])} image triples; seed={args.seed}; output={root.resolve()}")


if __name__ == "__main__":
    main()
