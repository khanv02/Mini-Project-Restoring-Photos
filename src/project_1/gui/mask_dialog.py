"""Candidate mask preview; acceptance is always an explicit user action."""

from PySide6.QtWidgets import (
    QDialog, QFormLayout, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout,
)

from project_1.algorithms.scratch_mask import (
    DEFAULT_MASK_PARAMETERS, detect_scratch_mask, mask_overlay, mask_summary, validate_binary_mask,
)
from project_1.gui.threads import TaskWorkerThread
from project_1.gui.theme import apply_workspace_theme
from project_1.gui.widgets import ImagePanel, SliderControl


class MaskDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gợi ý mask — vết xước sáng (thử nghiệm)")
        self.resize(1000, 730)
        apply_workspace_theme(self)
        self.image = image.copy()
        self.candidate = None
        self.candidate_parameters = None
        self.thread = None
        self._closing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)
        hint = QLabel("Chỉ tìm vết xước sáng, mảnh; có thể nhầm chi tiết sáng hoặc bỏ sót xước. "
                      "Không phát hiện vết ố/xước tối. Kiểm tra lớp phủ đỏ trước khi áp dụng.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        form = QFormLayout()
        self.controls = {
            "brightness_threshold": SliderControl(180, 250, 1, DEFAULT_MASK_PARAMETERS["brightness_threshold"]),
            "response_threshold": SliderControl(5, 100, 1, DEFAULT_MASK_PARAMETERS["response_threshold"]),
            "kernel_size": SliderControl(3, 31, 2, DEFAULT_MASK_PARAMETERS["kernel_size"]),
            "expand": SliderControl(0, 3, 1, DEFAULT_MASK_PARAMETERS["expand"]),
        }
        for key, label in (("brightness_threshold", "Ngưỡng sáng"),
                           ("response_threshold", "Ngưỡng Top-hat"),
                           ("kernel_size", "Kernel (số lẻ)"), ("expand", "Nới rộng (pixel)")):
            form.addRow(label, self.controls[key])
            self.controls[key].valueChanged.connect(self._invalidate)
        layout.addLayout(form)
        views = QHBoxLayout()
        self.binary_panel = ImagePanel("Mask: trắng = vùng cần phục hồi")
        self.overlay_panel = ImagePanel("Lớp phủ đỏ: vùng được chọn")
        self.overlay_panel.set_image(self.image)
        views.addWidget(self.binary_panel)
        views.addWidget(self.overlay_panel)
        layout.addLayout(views, 1)
        self.status = QLabel("Bấm Tạo gợi ý để xem mask; mask đang dùng chưa bị thay đổi.")
        self.status.setObjectName("resultStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        self.generate_button = QPushButton("Tạo gợi ý")
        self.apply_button = QPushButton("Áp dụng mask")
        self.apply_button.setProperty("primary", True)
        self.apply_button.setEnabled(False)
        self.cancel_button = QPushButton("Hủy")
        self.generate_button.clicked.connect(self.generate)
        self.apply_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        actions.addWidget(self.generate_button)
        actions.addStretch()
        for button in (self.cancel_button, self.apply_button):
            actions.addWidget(button)
        layout.addLayout(actions)

    def parameters(self):
        return {key: control.value() for key, control in self.controls.items()}

    def _invalidate(self, *_):
        self.candidate_parameters = None
        self.apply_button.setEnabled(False)
        self.status.setText("Thông số đã thay đổi; hãy tạo lại mask trước khi áp dụng.")

    def generate(self):
        if self.thread is not None:
            return
        parameters = self.parameters()
        self.candidate = self.candidate_parameters = None
        self.binary_panel.set_image(None)
        self.overlay_panel.set_image(self.image)
        self.apply_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        for control in self.controls.values():
            control.setEnabled(False)
        self.status.setText("Đang tạo mask…")
        worker = TaskWorkerThread(lambda: (detect_scratch_mask(self.image, **parameters), parameters), self)
        self.thread = worker
        worker.completed.connect(self._completed)
        worker.failed.connect(self._failed)
        worker.finished.connect(self._finished)
        worker.start()

    def _completed(self, data):
        if self._closing:
            return
        mask, parameters = data
        try:
            validate_binary_mask(mask, self.image.shape)
            summary = mask_summary(mask)
            overlay = mask_overlay(self.image, mask)
        except ValueError as exc:
            self._failed(str(exc))
            return
        self.candidate, self.candidate_parameters = mask.copy(), parameters.copy()
        self.generate_button.setText("Tạo lại gợi ý")
        self.binary_panel.set_image(mask)
        self.overlay_panel.set_image(overlay)
        text = (f"{summary['regions']} vùng; {summary['pixels']} pixel; "
                f"chiếm {summary['coverage']:.2%} ảnh. Mask chưa được áp dụng.")
        if not summary["pixels"]:
            text += " Không tìm thấy vùng phù hợp; hãy chỉnh thông số hoặc nạp mask thủ công."
        elif summary["coverage"] > 0.1:
            text += " Cảnh báo: diện tích vượt 10%, có thể chọn nhầm nhiều chi tiết."
        self.status.setText(text)

    def _failed(self, message):
        if not self._closing:
            self.candidate = self.candidate_parameters = None
            self.apply_button.setEnabled(False)
            self.status.setText("Không tạo được mask: " + message)

    def _finished(self):
        worker = self.thread
        self.thread = None
        if worker is not None:
            worker.deleteLater()
        if self._closing:
            return
        for control in self.controls.values():
            control.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.apply_button.setEnabled(self._valid_candidate())

    def _valid_candidate(self):
        return (self.candidate is not None and self.candidate_parameters == self.parameters()
                and bool(self.candidate.any()))

    def accept(self):
        if self.thread is not None or not self._valid_candidate():
            return
        validate_binary_mask(self.candidate, self.image.shape)
        super().accept()

    def reject(self):
        self._closing = True
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait()
        super().reject()

    def closeEvent(self, event):
        self.reject()
        event.accept()
