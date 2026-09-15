import cv2
import numpy as np
from .base import BaseRestorationAlgorithm

class InpaintingAlgorithm(BaseRestorationAlgorithm):
    def process(self, image: np.ndarray, mask: np.ndarray = None, radius: int = 3, method: str = 'telea', **kwargs) -> np.ndarray:
        if mask is None:
            raise ValueError("Inpainting yêu cầu truyền vào ảnh mask vết xước.")
            
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        flags = cv2.INPAINT_TELEA if method.lower() == 'telea' else cv2.INPAINT_NS
        return cv2.inpaint(image, mask_binary, inpaintRadius=radius, flags=flags)