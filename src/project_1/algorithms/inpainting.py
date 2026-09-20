# inpainting.py
import cv2
import numpy as np
from .base import BaseRestorationAlgorithm

class InpaintingAlgorithm(BaseRestorationAlgorithm):
    def process(self, image: np.ndarray, mask: np.ndarray = None, radius: int = 3, method: str = 'telea', **kwargs) -> np.ndarray:
        if mask is None:
            raise ValueError("Inpainting yêu cầu truyền vào ảnh mask vết xước.")
        
        # 1. Đảm bảo mask và image có cùng kích thước (Width x Height)
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 2. Chuyển mask về ảnh xám (Grayscale) 1 kênh nếu đang là 3 kênh
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        # 3. Chuyển mask thành dạng Binary chuẩn uint8 (0 và 255)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_binary = mask_binary.astype(np.uint8)

        # 4. Chọn phương pháp Inpaint
        flags = cv2.INPAINT_TELEA if str(method).lower() == 'telea' else cv2.INPAINT_NS
        
        # Ép kiểu image về uint8 nếu lỡ bị chuyển đổi dạng float trước đó
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return cv2.inpaint(image, mask_binary, inpaintRadius=max(1, int(radius)), flags=flags)