# base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseRestorationAlgorithm(ABC):
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Mọi thuật toán đều nhận ảnh đầu vào (np.ndarray) 
        và trả về ảnh đã khôi phục (np.ndarray)
        """
        pass