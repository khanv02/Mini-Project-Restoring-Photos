"""Reusable parameter controls and image viewers."""

import math

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSignalBlocker, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpinBox,
    QVBoxLayout, QWidget,
)

from project_1.settings import (
    ALGORITHMS, DEFAULT_KERNEL_SIZE, DEFAULT_RADIUS, DEFAULT_SIGMA,
    DEFAULT_SHARPEN_AMOUNT, DEFAULT_SHARPEN_SIGMA,
)
from project_1.utils.image_io import validate_image


def cv_to_qpixmap(cv_img: np.ndarray | None, max_w=450, max_h=450) -> QPixmap:
    """Copy OpenCV buffers before displaying; support sliced/strided arrays."""
    if cv_img is None:
        return QPixmap()
    validate_image(cv_img)
    pixels = np.ascontiguousarray(
        cv_img if cv_img.ndim == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    )
    height, width = pixels.shape[:2]
    image_format = QImage.Format.Format_Grayscale8 if pixels.ndim == 2 else QImage.Format.Format_RGB888
    qimage = QImage(pixels.data, width, height, pixels.strides[0], image_format).copy()
    return QPixmap.fromImage(qimage).scaled(
        max(1, max_w), max(1, max_h),
        Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
    )


class ImagePanel(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        layout = QVBoxLayout(self)
        self.view = QLabel("Chưa nạp ảnh")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(120, 120)
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.view.setStyleSheet("background: #eef2f6; border-radius: 6px; color: #526274;")
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.caption)
        self.image = None

    def set_image(self, image: np.ndarray | None, caption: str = ""):
        self.image = image
        self.caption.setText(caption)
        self._refresh()

    def _refresh(self):
        if self.image is None:
            self.view.setPixmap(QPixmap())
            self.view.setText("Chưa nạp ảnh")
        else:
            self.view.setPixmap(cv_to_qpixmap(
                self.image, self.view.width() - 16, self.view.height() - 16,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class SliderControl(QWidget):
    """Discrete slider plus precise numeric entry; no increment/decrement buttons."""

    valueChanged = Signal(object)

    def __init__(self, minimum, maximum, step, default, *, decimals=0):
        super().__init__()
        self.minimum, self.maximum, self.step = minimum, maximum, step
        self.integer = decimals == 0
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, round((maximum - minimum) / step))
        self.slider.setMinimumWidth(60)
        self.slider.setPageStep(max(1, self.slider.maximum() // 10))
        self.editor = QSpinBox() if self.integer else QDoubleSpinBox()
        if not self.integer:
            self.editor.setDecimals(decimals)
        self.editor.setRange(minimum, maximum)
        self.editor.setSingleStep(step)
        self.editor.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.editor.setKeyboardTracking(False)
        self.editor.setFixedWidth(64)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.editor)
        self.slider.valueChanged.connect(self._slider_changed)
        self.editor.valueChanged.connect(self.setValue)
        self.setValue(default)

    def value(self):
        value = self.minimum + self.slider.value() * self.step
        return int(round(value)) if self.integer else round(value, self.editor.decimals())

    def setValue(self, value):
        if not math.isfinite(value):
            raise ValueError("Giá trị thanh kéo phải hữu hạn.")
        previous = self.value()
        value = min(self.maximum, max(self.minimum, value))
        # Odd kernels round upward, matching the existing pipeline's normalization.
        position = (value - self.minimum) / self.step
        index = math.ceil(position) if self.integer and self.step == 2 else math.floor(position + 0.5)
        with QSignalBlocker(self.slider), QSignalBlocker(self.editor):
            self.slider.setValue(index)
            self.editor.setValue(self.value())
        if self.value() != previous:
            self.valueChanged.emit(self.value())

    def _slider_changed(self, _):
        with QSignalBlocker(self.editor):
            self.editor.setValue(self.value())
        self.valueChanged.emit(self.value())


class AlgorithmControls(QGroupBox):
    def __init__(self, title="Thuật toán & tham số"):
        super().__init__(title)
        self.form = QFormLayout(self)
        self.algorithm = QComboBox()
        for spec in ALGORITHMS:
            self.algorithm.addItem("Combined" if spec.key == "combined" else spec.label, spec.key)
            self.algorithm.setItemData(self.algorithm.count() - 1, spec.label, Qt.ItemDataRole.ToolTipRole)
        self.kernel = SliderControl(3, 31, 2, DEFAULT_KERNEL_SIZE)
        self.sigma = SliderControl(0.1, 10, 0.1, DEFAULT_SIGMA, decimals=1)
        self.radius = SliderControl(1, 20, 1, DEFAULT_RADIUS)
        self.sharpen_enabled = QCheckBox("Tăng nét sau phục hồi")
        self.sharpen_amount = SliderControl(0, 2, 0.05, DEFAULT_SHARPEN_AMOUNT, decimals=2)
        self.sharpen_sigma = SliderControl(0.3, 3, 0.1, DEFAULT_SHARPEN_SIGMA, decimals=1)
        self.method = QComboBox()
        self.method.addItem("Telea", "telea")
        self.method.addItem("Navier–Stokes", "navier-stokes")
        self.form.addRow("Phương pháp", self.algorithm)
        self.form.addRow("Kernel (số lẻ)", self.kernel)
        self.form.addRow("Sigma", self.sigma)
        self.form.addRow("Bán kính inpaint", self.radius)
        self.form.addRow("Inpaint method", self.method)
        self.form.addRow(self.sharpen_enabled)
        self.form.addRow("Mức tăng nét", self.sharpen_amount)
        self.form.addRow("Sigma tăng nét", self.sharpen_sigma)
        self.sharpen_hint = QLabel("Tăng nét có thể làm nổi nhiễu/tạo viền và giảm PSNR/SSIM.")
        self.sharpen_hint.setWordWrap(True)
        self.form.addRow(self.sharpen_hint)
        self.sharpen_enabled.toggled.connect(self._update_sharpen)
        self._update_sharpen()
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.form.addRow(self.hint)
        self.algorithm.currentIndexChanged.connect(self._update_visibility)
        self._update_visibility()

    @property
    def algorithm_key(self) -> str:
        return self.algorithm.currentData()

    @property
    def needs_mask(self) -> bool:
        return ALGORITHMS[self.algorithm.currentIndex()].needs_mask

    def parameters(self) -> dict:
        return {
            "kernel_size": self.kernel.value(), "sigma": self.sigma.value(),
            "radius": self.radius.value(), "method": self.method.currentData(),
            "sharpen_enabled": self.sharpen_enabled.isChecked(),
            "sharpen_amount": self.sharpen_amount.value(), "sharpen_sigma": self.sharpen_sigma.value(),
        }

    def _update_sharpen(self, *_):
        for control in (self.sharpen_amount, self.sharpen_sigma):
            control.setEnabled(self.sharpen_enabled.isChecked())

    def _update_visibility(self):
        spec = ALGORITHMS[self.algorithm.currentIndex()]
        for control in (self.kernel, self.sigma):
            self.form.setRowVisible(control, spec.uses_gaussian)
        for control in (self.radius, self.method):
            self.form.setRowVisible(control, spec.needs_mask)
        self.hint.setText(
            "Mask trắng: vùng cần phục hồi. Mask đen: giữ nguyên. Combined chạy Inpainting trước, Gaussian sau."
            if spec.needs_mask else "Gaussian giảm nhiễu toàn ảnh; kernel/sigma lớn có thể làm mờ chi tiết."
        )
