import cv2
import numpy as np

from .base import BaseRestorationAlgorithm
from project_1.settings import DEFAULT_METHOD, DEFAULT_RADIUS
from project_1.utils.image_io import validate_image

class InpaintingAlgorithm(BaseRestorationAlgorithm):
    def process(self, image: np.ndarray, mask: np.ndarray | None = None, radius: int = DEFAULT_RADIUS, method: str = DEFAULT_METHOD, **kwargs) -> np.ndarray:
        validate_image(image)
        if mask is None:
            raise ValueError("Inpainting yêu cầu truyền vào ảnh mask vết xước.")
        validate_image(mask, "Mask")
        methods = {"telea": cv2.INPAINT_TELEA, "navier-stokes": cv2.INPAINT_NS, "ns": cv2.INPAINT_NS}
        method = str(method).lower()
        if method not in methods:
            raise ValueError("Inpaint method phải là 'telea', 'navier-stokes' hoặc 'ns'.")
        
        # 1. Đảm bảo mask và image có cùng kích thước (Width x Height)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError("Mask phải cùng kích thước với ảnh; không tự resize mask.")

        # 2. Chuyển mask về ảnh xám (Grayscale) 1 kênh nếu đang là 3 kênh
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
        # 3. Chuyển mask thành dạng Binary chuẩn uint8 (0 và 255)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_binary = mask_binary.astype(np.uint8)

        # 4. Chọn phương pháp Inpaint
        flags = methods[method]

        return cv2.inpaint(image, mask_binary, inpaintRadius=max(1, int(radius)), flags=flags)
