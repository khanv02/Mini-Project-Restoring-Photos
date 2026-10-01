"""Qt transport only: all image processing lives in the pipeline."""

from PySide6.QtCore import QThread, Signal


class BatchWorkerThread(QThread):
    progress_changed = Signal(int, int)
    file_processed = Signal(str)
    file_failed = Signal(str, str)
    result_ready = Signal(object)
    completed = Signal(object)
    failed = Signal(str)
    finished_all = Signal()  # Compatibility with previous callers.

    def __init__(self, pipeline, input_dir: str, output_dir: str,
                 algo_name: str, kwargs: dict, mask_dir: str | None = None, parent=None,
                 clean_dir: str | None = None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mask_dir = mask_dir
        self.clean_dir = clean_dir
        self.algo_name = algo_name
        self.kwargs = kwargs.copy()

    def stop(self):
        self.requestInterruption()

    def run(self):
        try:
            result = self.pipeline.run_batch(
                self.input_dir, self.output_dir, self.algo_name,
                mask_dir=self.mask_dir, progress_callback=self.progress_changed.emit,
                file_callback=self.file_processed.emit, error_callback=self.file_failed.emit,
                clean_dir=self.clean_dir, result_callback=self.result_ready.emit,
                should_stop=self.isInterruptionRequested, **self.kwargs,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(result)
        finally:
            self.finished_all.emit()


class TaskWorkerThread(QThread):
    """Run one Single/Compare task; results are delivered on the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, task, parent=None):
        super().__init__(parent)
        self.task = task

    def stop(self):
        self.requestInterruption()

    def run(self):
        try:
            result = self.task()
            if not self.isInterruptionRequested():
                self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
