# gaussian.py
import cv2
import numpy as np
from .base import BaseRestorationAlgorithm

class GaussianFilterAlgorithm(BaseRestorationAlgorithm):
    def process(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, **kwargs) -> np.ndarray:
        # Ép kernel_size tối thiểu là 3
        kernel_size = max(3, int(kernel_size))
        
        # Nếu là số chẵn, cộng thêm 1 để biến thành số lẻ
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)