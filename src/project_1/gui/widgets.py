"""Reusable parameter controls and image viewers."""

import math

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, QSignalBlocker, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListView, QMenu, QSizePolicy, QSlider, QSpinBox,
    QStyledItemDelegate, QToolButton, QVBoxLayout, QWidget,
)

from project_1.settings import (
    ALGORITHMS, DEFAULT_KERNEL_SIZE, DEFAULT_RADIUS, DEFAULT_SIGMA,
    DEFAULT_SHARPEN_AMOUNT, DEFAULT_SHARPEN_SIGMA,
)
from project_1.utils.image_io import validate_image


def cv_to_qimage(cv_img: np.ndarray | None) -> QImage:
    """Own a full-resolution Qt buffer; accept grayscale and strided BGR arrays."""
    if cv_img is None:
        return QImage()
    validate_image(cv_img)
    pixels = np.ascontiguousarray(
        cv_img if cv_img.ndim == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    )
    height, width = pixels.shape[:2]
    image_format = QImage.Format.Format_Grayscale8 if pixels.ndim == 2 else QImage.Format.Format_RGB888
    return QImage(pixels.data, width, height, pixels.strides[0], image_format).copy()


def cv_to_qpixmap(cv_img: np.ndarray | None, max_w=450, max_h=450) -> QPixmap:
    """Compatibility helper for callers that need a fitted, copied pixmap."""
    if cv_img is None:
        return QPixmap()
    return QPixmap.fromImage(cv_to_qimage(cv_img)).scaled(
        max(1, max_w), max(1, max_h),
        Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
    )


class InteractiveImageView(QWidget):
    """Full-resolution painting with cursor-anchored zoom and bounded mouse panning."""

    zoomChanged = Signal(float)
    viewportChanged = Signal()
    MAX_SCALE = 32.0
    MIN_SCALE = 0.01

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setToolTip("Cuộn chuột: zoom tại con trỏ. Giữ chuột trái: kéo ảnh. Nhấp đúp: vừa khung.")
        self._image = QImage()
        self.scale_factor = 1.0
        self.offset = QPointF()
        self.fit_mode = True
        self._drag_position = None

    @property
    def has_image(self):
        return not self._image.isNull()

    def set_image(self, image):
        # Validate/copy first so invalid input leaves the previous display intact.
        qimage = cv_to_qimage(image)
        self._image = qimage
        self._drag_position = None
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.has_image else Qt.CursorShape.ArrowCursor)
        self.fit_to_window(notify_view=False)

    def _fit_scale(self):
        return min(self.MAX_SCALE, max(1e-6, min(
            (self.width() - 16) / self._image.width(),
            (self.height() - 16) / self._image.height(),
        )))

    def _clamp_offset(self):
        def clamp(value, viewport, image_size):
            if image_size <= viewport:
                return (viewport - image_size) / 2
            return min(0.0, max(viewport - image_size, value))
        self.offset = QPointF(
            clamp(self.offset.x(), self.width(), self._image.width() * self.scale_factor),
            clamp(self.offset.y(), self.height(), self._image.height() * self.scale_factor),
        )

    def _notify(self, *, viewport_changed=False):
        self.zoomChanged.emit(self.scale_factor if self.has_image else 0.0)
        self.update()
        if viewport_changed:
            self.viewportChanged.emit()

    def fit_to_window(self, *, notify_view=True):
        self.fit_mode = True
        self._drag_position = None
        if self.has_image:
            self.scale_factor = self._fit_scale()
            self.offset = QPointF(
                (self.width() - self._image.width() * self.scale_factor) / 2,
                (self.height() - self._image.height() * self.scale_factor) / 2,
            )
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.scale_factor = 1.0
            self.offset = QPointF()
        self._notify(viewport_changed=notify_view)

    def viewport_state(self):
        """Return an image-space camera, independent of the viewport dimensions."""
        if not self.has_image:
            return None
        center = (QPointF(self.width() / 2, self.height() / 2) - self.offset) / self.scale_factor
        return self.scale_factor, center, self.fit_mode, self._image.size()

    def fill_to_window(self):
        """Cover the viewport without distortion, explicitly cropping outside edges."""
        if not self.has_image:
            return
        scale = max(self.width() / self._image.width(), self.height() / self._image.height())
        center = QPointF(self._image.width() / 2, self._image.height() / 2)
        self.apply_viewport_state((scale, center, False, self._image.size()))
        self.viewportChanged.emit()

    def apply_viewport_state(self, state):
        """Apply a matching camera without rebroadcasting a linked interaction."""
        if not self.has_image or state is None or self._image.size() != state[3]:
            return
        scale, center, fit_mode, _ = state
        if fit_mode:
            self.fit_to_window(notify_view=False)
            return
        self.scale_factor = min(self.MAX_SCALE, max(min(self.MIN_SCALE, self._fit_scale()), scale))
        self.offset = QPointF(self.width() / 2, self.height() / 2) - center * self.scale_factor
        self.fit_mode = False
        self._clamp_offset()
        self._notify()

    def set_zoom(self, scale, anchor=None):
        if not self.has_image:
            return
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Tỷ lệ zoom phải hữu hạn và lớn hơn 0.")
        anchor = QPointF(self.width() / 2, self.height() / 2) if anchor is None else QPointF(anchor)
        image_point = (anchor - self.offset) / self.scale_factor
        self.scale_factor = min(self.MAX_SCALE, max(min(self.MIN_SCALE, self._fit_scale()), scale))
        self.offset = anchor - image_point * self.scale_factor
        self.fit_mode = False
        self._clamp_offset()
        self._notify(viewport_changed=True)

    def zoom_by(self, factor, anchor=None):
        self.set_zoom(self.scale_factor * factor, anchor)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not self.has_image or not delta:
            event.ignore()
            return
        steps = max(-10.0, min(10.0, delta / 120.0))
        self.zoom_by(1.2 ** steps, event.position())
        event.accept()

    def mousePressEvent(self, event):
        if self.has_image and event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_position is not None:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.offset += event.position() - self._drag_position
                self._drag_position = event.position()
                self._clamp_offset()
                self._notify(viewport_changed=True)
                event.accept()
                return
            self._drag_position = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_position is not None:
            self._drag_position = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.has_image and event.button() == Qt.MouseButton.LeftButton:
            self.fit_to_window()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.has_image:
            return
        if self.fit_mode:
            self.fit_to_window(notify_view=False)
        else:
            self.offset += QPointF(
                (event.size().width() - event.oldSize().width()) / 2,
                (event.size().height() - event.oldSize().height()) / 2,
            )
            self._clamp_offset()
            self._notify()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eef2f6"))
        if not self.has_image:
            painter.setPen(QColor("#526274"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Chưa nạp ảnh")
            return
        # Downsampling is smooth; magnification exposes original pixels, not invented detail.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.scale_factor < 1)
        painter.translate(self.offset)
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawImage(QPointF(), self._image)


class LinkedImageViews(QObject):
    """Link only user camera changes; loading/resizing never replaces another camera."""

    def __init__(self, views, parent=None):
        super().__init__(parent)
        self.views = tuple(views)
        self.enabled = True
        for view in self.views:
            view.viewportChanged.connect(lambda source=view: self.sync_from(source))

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if self.enabled:
            source = next((view for view in self.views if view.isVisible() and view.has_image), None)
            if source is not None:
                self.sync_from(source)

    def sync_from(self, source):
        if self.enabled:
            state = source.viewport_state()
            for target in self.views:
                if target is not source:
                    target.apply_viewport_state(state)


class ElidedCaption(QLabel):
    """One-line captions keep before/after viewport heights equal; tooltip has full text."""

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        text = self.fontMetrics().elidedText(self.text().replace("\n", " "), Qt.TextElideMode.ElideRight, self.width())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)


class ImagePanel(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        layout = QVBoxLayout(self)
        self.view = InteractiveImageView()
        self.zoom_button = QToolButton()
        self.zoom_button.setObjectName("zoomMenu")
        self.zoom_button.setMinimumWidth(80)
        self.zoom_button.setToolTip(self.view.toolTip() + " Mở menu tỷ lệ để chọn 1:1 / Vừa khung.")
        self.zoom_button.setAccessibleName("Tỷ lệ zoom và chế độ xem ảnh")
        self.zoom_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.zoom_button)
        self.fit_action = QAction("Vừa khung", self)
        self.fill_action = QAction("Lấp khung (cắt viền)", self)
        self.fill_action.setToolTip("Giữ đúng tỷ lệ, phóng lớn để phủ khung; phần ngoài khung không hiển thị.")
        self.actual_size_action = QAction("1:1 — 100%", self)
        self.actual_size_action.setToolTip("100%: một pixel ảnh ứng với một đơn vị hiển thị Qt.")
        self.actual_size_action.triggered.connect(lambda: self.view.set_zoom(1.0))
        self.fit_action.triggered.connect(self.view.fit_to_window)
        self.fill_action.triggered.connect(self.view.fill_to_window)
        menu.addActions([self.fit_action, self.fill_action, self.actual_size_action])
        self.zoom_button.setMenu(menu)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.view.addActions([self.fit_action, self.fill_action, self.actual_size_action])
        toolbar = QHBoxLayout()
        self.caption = ElidedCaption()
        self.caption.setObjectName("viewerCaption")
        toolbar.addWidget(self.caption, 1)
        toolbar.addWidget(self.zoom_button)
        layout.addWidget(self.view, 1)
        layout.addLayout(toolbar)
        self.image = None
        self.view.zoomChanged.connect(self._update_zoom)
        self._update_zoom(0.0)

    def set_image(self, image: np.ndarray | None, caption: str = ""):
        self.view.set_image(image)
        self.image = image
        self.caption.setText(caption)
        self.caption.setToolTip(caption)

    def _update_zoom(self, scale):
        text = f"{scale * 100:.2f}%" if 0 < scale < 0.01 else f"{scale * 100:.0f}%"
        self.zoom_button.setText(text if scale else "—")
        self.zoom_button.setEnabled(bool(scale))
        self.actual_size_action.setEnabled(bool(scale))
        self.fit_action.setEnabled(bool(scale))
        self.fill_action.setEnabled(bool(scale))


class SliderControl(QWidget):
    """Discrete slider plus precise numeric entry; no increment/decrement buttons."""

    valueChanged = Signal(object)
    dragStarted = Signal()
    dragFinished = Signal()

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
        self.slider.sliderPressed.connect(self.dragStarted)
        self.slider.sliderReleased.connect(self.dragFinished)
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


class ContrastItemDelegate(QStyledItemDelegate):
    """Explicit selected-text colors, including inactive Windows popup palettes."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            option.palette.setColor(group, QPalette.ColorRole.Highlight, QColor("#2563eb"))
            option.palette.setColor(group, QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
            option.palette.setColor(group, QPalette.ColorRole.Text, QColor("#243248"))


def configure_combo_contrast(combo):
    """Use a styled list popup rather than platform-dependent native selection colors."""
    view = QListView(combo)
    combo.setView(view)
    view.setItemDelegate(ContrastItemDelegate(view))
    palette = view.palette()
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        palette.setColor(group, QPalette.ColorRole.Highlight, QColor("#2563eb"))
        palette.setColor(group, QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    view.setPalette(palette)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)


class AlgorithmControls(QGroupBox):
    def __init__(self, title="Thuật toán & tham số"):
        super().__init__(title)
        self.form = QFormLayout(self)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.form.setVerticalSpacing(10)
        self.algorithm = QComboBox()
        configure_combo_contrast(self.algorithm)
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
        configure_combo_contrast(self.method)
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
            self.form.setRowVisible(control, self.sharpen_enabled.isChecked())
        self.form.setRowVisible(self.sharpen_hint, self.sharpen_enabled.isChecked())

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
