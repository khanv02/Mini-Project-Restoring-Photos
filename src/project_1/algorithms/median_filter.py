import cv2
import numpy as np
from .base import BaseRestorationAlgorithm

class MedianFilterAlgorithm(BaseRestorationAlgorithm):
    def process(self, image: np.ndarray, kernel_size: int = 5, **kwargs) -> np.ndarray:
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)