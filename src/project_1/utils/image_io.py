"""Checked image I/O, including Unicode paths on Windows."""

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


def validate_image(image: np.ndarray, name: str = "Ảnh") -> None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError(f"{name} phải là mảng ảnh không rỗng.")
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError(f"{name} phải là ảnh xám hoặc ảnh BGR 3 kênh.")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} phải có kiểu uint8 (giá trị pixel 0–255).")


def read_image(path: str | Path, *, grayscale: bool = False) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, flags) if data.size else None
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    validate_image(image)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Định dạng ảnh không hỗ trợ: {path.suffix}")
    success, encoded = cv2.imencode(path.suffix.lower(), image)
    if not success:
        raise OSError(f"Không mã hóa được ảnh: {path}")
    encoded.tofile(path)


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Thư mục ảnh không tồn tại: {directory}")
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )
