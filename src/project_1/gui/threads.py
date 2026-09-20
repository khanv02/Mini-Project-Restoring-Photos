import os
import cv2
from PySide6.QtCore import QThread, Signal

class BatchWorkerThread(QThread):
    # Signals truyền dữ liệu về GUI
    progress_changed = Signal(int, int)  # (current, total)
    file_processed = Signal(str)          # filename
    finished_all = Signal()

    def __init__(self, pipeline, input_dir: str, output_dir: str, algo_name: str, kwargs: dict, mask_dir: str = None):
        super().__init__()
        self.pipeline = pipeline
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mask_dir = mask_dir
        self.algo_name = algo_name
        self.kwargs = kwargs
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(valid_exts)]
        total = len(files)

        for idx, filename in enumerate(files):
            if not self._is_running:
                break
            
            img_path = os.path.join(self.input_dir, filename)
            corrupted = cv2.imread(img_path)
            
            if corrupted is not None:
                task_kwargs = self.kwargs.copy()
                
                # Nếu chạy thuật toán Inpainting, tự tìm file mask trùng tên trong mask_dir
                if self.algo_name == "inpainting" and self.mask_dir:
                    mask_path = os.path.join(self.mask_dir, filename)
                    if os.path.exists(mask_path):
                        task_kwargs["mask"] = cv2.imread(mask_path)
                    else:
                        print(f"⚠️ Cảnh báo: Không tìm thấy mask tương ứng cho {filename}")
                        continue

                restored = self.pipeline.run_single(self.algo_name, corrupted, **task_kwargs)
                save_path = os.path.join(self.output_dir, f"restored_{filename}")
                cv2.imwrite(save_path, restored)

            self.progress_changed.emit(idx + 1, total)
            self.file_processed.emit(filename)

        self.finished_all.emit()