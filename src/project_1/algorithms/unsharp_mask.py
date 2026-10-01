"""Optional luminance-only sharpening after restoration."""

import cv2
import numpy as np

from project_1.algorithms.base import BaseRestorationAlgorithm
from project_1.settings import DEFAULT_SHARPEN_AMOUNT, DEFAULT_SHARPEN_SIGMA
from project_1.utils.image_io import validate_image


def validate_sharpen_parameters(amount, sigma):
    if not np.isfinite(amount) or not 0 <= amount <= 2:
        raise ValueError("Mức tăng nét phải hữu hạn trong khoảng 0–2.")
    if not np.isfinite(sigma) or not 0.3 <= sigma <= 3:
        raise ValueError("Sigma tăng nét phải hữu hạn trong khoảng 0,3–3.")


class UnsharpMaskAlgorithm(BaseRestorationAlgorithm):
    def process(self, image: np.ndarray, amount=DEFAULT_SHARPEN_AMOUNT,
                sigma=DEFAULT_SHARPEN_SIGMA, **kwargs) -> np.ndarray:
        validate_image(image)
        validate_sharpen_parameters(amount, sigma)
        if amount == 0:
            return image.copy()
        color = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) if image.ndim == 3 else None
        luminance = (color[:, :, 0] if color is not None else image).astype(np.float32)
        blurred = cv2.GaussianBlur(luminance, (0, 0), sigmaX=float(sigma))
        sharp = np.rint(np.clip(luminance + amount * (luminance - blurred), 0, 255)).astype(np.uint8)
        if color is None:
            return sharp
        color[:, :, 0] = sharp
        return cv2.cvtColor(color, cv2.COLOR_YCrCb2BGR)
