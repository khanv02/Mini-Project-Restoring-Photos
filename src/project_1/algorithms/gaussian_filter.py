import cv2
import numpy as np

from .base import BaseRestorationAlgorithm
from project_1.settings import DEFAULT_KERNEL_SIZE, DEFAULT_SIGMA
from project_1.utils.image_io import validate_image

class GaussianFilterAlgorithm(BaseRestorationAlgorithm):
    def process(self, image: np.ndarray, kernel_size: int = DEFAULT_KERNEL_SIZE, sigma: float = DEFAULT_SIGMA, **kwargs) -> np.ndarray:
        validate_image(image)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Sigma phải là số hữu hạn không âm.")
        # Ép kernel_size tối thiểu là 3
        kernel_size = max(3, int(kernel_size))
        
        # Nếu là số chẵn, cộng thêm 1 để biến thành số lẻ
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
