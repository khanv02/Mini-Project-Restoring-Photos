"""Experimental bright, thin scratch detector; never uses a reference image."""

import cv2
import numpy as np

from project_1.utils.image_io import validate_image

DEFAULT_MASK_PARAMETERS = {
    "brightness_threshold": 220, "response_threshold": 35,
    "kernel_size": 9, "expand": 0,
}


def validate_binary_mask(mask: np.ndarray, shape: tuple) -> None:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError("Mask phải là ảnh xám uint8.")
    if mask.size == 0 or mask.shape != tuple(shape[:2]):
        raise ValueError("Mask phải cùng kích thước với ảnh hỏng.")
    if not np.all((mask == 0) | (mask == 255)):
        raise ValueError("Mask phải chỉ chứa giá trị 0 và 255.")


def mask_summary(mask: np.ndarray) -> dict:
    validate_binary_mask(mask, mask.shape)
    pixels = int(np.count_nonzero(mask))
    count, _ = cv2.connectedComponents(mask, connectivity=8)
    return {"pixels": pixels, "regions": count - 1, "coverage": pixels / mask.size}


def detect_scratch_mask(image: np.ndarray, brightness_threshold=220,
                        response_threshold=35, kernel_size=9, expand=0) -> np.ndarray:
    validate_image(image)
    values = ((brightness_threshold, 180, 250), (response_threshold, 5, 100),
              (kernel_size, 3, 31), (expand, 0, 3))
    if any(not np.isfinite(v) or int(v) != v or not low <= v <= high for v, low, high in values):
        raise ValueError("Tham số tạo mask phải là số nguyên trong khoảng cho phép.")
    brightness_threshold, response_threshold, kernel_size, expand = map(
        int, (brightness_threshold, response_threshold, kernel_size, expand))
    if kernel_size % 2 == 0:
        raise ValueError("Kernel tạo mask phải là số lẻ.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    response = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    candidate = ((gray >= brightness_threshold) & (response > response_threshold)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    distance = cv2.distanceTransform(candidate, cv2.DIST_L2, 5)
    mask = np.zeros(gray.shape, np.uint8)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 12:
            continue
        region = labels[y:y + h, x:x + w] == index
        width = max(1.0, 2 * float(np.percentile(distance[y:y + h, x:x + w][region], 90)) - 1)
        if width <= 8 and area / width >= 15:
            mask[y:y + h, x:x + w][region] = 255
    if expand:
        expansion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand + 1, 2 * expand + 1))
        mask = cv2.dilate(mask, expansion)
    return mask


def mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    validate_image(image)
    validate_binary_mask(mask, image.shape)
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    selected = mask > 0
    result[selected] = np.rint(0.55 * result[selected] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
    return result
