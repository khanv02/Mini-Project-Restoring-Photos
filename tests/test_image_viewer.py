"""Mouse interaction and full-resolution rendering without changing image data."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from project_1.gui.main_window import MainWindow
from project_1.gui.mask_dialog import MaskDialog
from project_1.gui.widgets import ImagePanel, InteractiveImageView, cv_to_qimage, cv_to_qpixmap
from project_1.utils.image_io import read_image, write_image


class ImageViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.style().objectName().lower() != "fusion":
            cls.app.setStyle("Fusion")

    def setUp(self):
        self.view = InteractiveImageView()
        self.view.resize(320, 240)
        self.view.show()
        self.image = np.full((1000, 1200, 3), [20, 80, 150], np.uint8)
        self.addCleanup(self.close_widget, self.view)
        self.app.processEvents()

    def close_widget(self, widget):
        widget.close()
        widget.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def wheel(self, angle=120, *, pixel=0, point=QPointF(95, 75)):
        event = QWheelEvent(point, QPointF(self.view.mapToGlobal(point.toPoint())),
                            QPoint(0, pixel), QPoint(0, angle), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        self.app.sendEvent(self.view, event)
        return event

    def drag(self, start=QPoint(140, 110), end=QPoint(180, 135)):
        QTest.mousePress(self.view, Qt.MouseButton.LeftButton, pos=start)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(end),
                            QPointF(self.view.mapToGlobal(end)), Qt.MouseButton.NoButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(self.view, event)
        QTest.mouseRelease(self.view, Qt.MouseButton.LeftButton, pos=end)

    def assert_point_equal(self, a, b):
        self.assertAlmostEqual(a.x(), b.x(), places=7)
        self.assertAlmostEqual(a.y(), b.y(), places=7)

    def test_initial_fit_and_empty_controls(self):
        self.assertFalse(self.view.has_image)
        event = self.wheel()
        self.assertFalse(event.isAccepted())
        self.view.set_image(self.image)
        self.assertTrue(self.view.fit_mode)
        self.assertEqual(self.view._image.width(), 1200)
        self.assertEqual(self.view._image.height(), 1000)
        self.assertLessEqual(self.view._image.width() * self.view.scale_factor, self.view.width())
        self.assertLessEqual(self.view._image.height() * self.view.scale_factor, self.view.height())

    def test_wheel_zoom_keeps_image_point_under_cursor(self):
        self.view.set_image(self.image)
        self.view.set_zoom(1)
        anchor = QPointF(95, 75)
        before = (anchor - self.view.offset) / self.view.scale_factor
        self.assertTrue(self.wheel(point=anchor).isAccepted())
        self.assertAlmostEqual(self.view.scale_factor, 1.2)
        self.assertFalse(self.view.fit_mode)
        self.assert_point_equal(before, (anchor - self.view.offset) / self.view.scale_factor)
        self.wheel(angle=-120, point=anchor)
        self.assertAlmostEqual(self.view.scale_factor, 1)

    def test_trackpad_and_zoom_limits(self):
        self.view.set_image(self.image)
        self.view.set_zoom(1)
        self.wheel(angle=0, pixel=60)
        self.assertAlmostEqual(self.view.scale_factor, 1.2 ** 0.5)
        self.view.set_zoom(1e10)
        self.assertEqual(self.view.scale_factor, self.view.MAX_SCALE)
        self.view.set_zoom(1e-10)
        self.assertEqual(self.view.scale_factor, self.view.MIN_SCALE)
        for scale in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.view.set_zoom(scale)

    def test_drag_moves_enlarged_image_and_restores_cursor(self):
        self.view.set_image(self.image)
        self.view.set_zoom(2)
        before = QPointF(self.view.offset)
        self.drag()
        self.assert_point_equal(self.view.offset, before + QPointF(40, 25))
        self.assertIsNone(self.view._drag_position)
        self.assertEqual(self.view.cursor().shape(), Qt.CursorShape.OpenHandCursor)

    def test_pan_bounds_and_small_image_centering(self):
        self.view.set_image(self.image)
        self.view.set_zoom(1)
        self.view.offset = QPointF(10000, -10000)
        self.view._clamp_offset()
        self.assert_point_equal(self.view.offset, QPointF(0, self.view.height() - 1000))
        self.view.fit_to_window()
        before = QPointF(self.view.offset)
        self.drag()
        self.assert_point_equal(self.view.offset, before)

    def test_double_click_and_toolbar_reset_zoom(self):
        panel = ImagePanel("Test")
        self.addCleanup(self.close_widget, panel)
        panel.resize(500, 400)
        panel.show()
        self.app.processEvents()
        self.assertFalse(panel.fit_action.isEnabled())
        panel.set_image(self.image)
        self.assertTrue(panel.fit_action.isEnabled())
        panel.actual_size_action.trigger()
        self.assertEqual(panel.view.scale_factor, 1)
        self.assertEqual(panel.zoom_button.text(), "100%")
        panel.view.set_zoom(2)
        QTest.mouseDClick(panel.view, Qt.MouseButton.LeftButton)
        self.assertTrue(panel.view.fit_mode)
        panel.view.set_zoom(3)
        panel.fit_action.trigger()
        self.assertTrue(panel.view.fit_mode)
        panel.set_image(None)
        self.assertEqual(panel.zoom_button.text(), "—")
        self.assertFalse(panel.actual_size_action.isEnabled())

    def test_resize_refits_or_preserves_manual_zoom_and_center(self):
        self.view.set_image(self.image)
        before = self.view.scale_factor
        self.view.resize(400, 320)
        self.app.processEvents()
        self.assertGreater(self.view.scale_factor, before)
        self.view.set_zoom(2)
        center_before = (QPointF(self.view.width() / 2, self.view.height() / 2) - self.view.offset) / 2
        self.view.resize(460, 360)
        self.app.processEvents()
        self.assertEqual(self.view.scale_factor, 2)
        center_after = (QPointF(self.view.width() / 2, self.view.height() / 2) - self.view.offset) / 2
        self.assert_point_equal(center_before, center_after)

    def test_new_image_resets_transform_and_invalid_input_preserves_previous(self):
        self.view.set_image(self.image)
        self.view.set_zoom(2)
        self.drag()
        second = np.zeros((80, 96), np.uint8)
        self.view.set_image(second)
        self.assertTrue(self.view.fit_mode)
        scale, offset = self.view.scale_factor, QPointF(self.view.offset)
        with self.assertRaises(ValueError):
            self.view.set_image(second.astype(float))
        self.assertEqual(self.view.scale_factor, scale)
        self.assert_point_equal(self.view.offset, offset)
        self.assertEqual(self.view._image.width(), 96)
        self.view.set_image(None)
        self.assertFalse(self.view.has_image)
        self.assertEqual(self.view.scale_factor, 1)
        self.assertEqual(self.view.cursor().shape(), Qt.CursorShape.ArrowCursor)

    def test_grayscale_strided_and_full_resolution_buffers_are_owned(self):
        for pixels in (self.image[:, :, 0], self.image[::2, ::2, ::-1]):
            qimage = cv_to_qimage(pixels)
            self.assertEqual((qimage.height(), qimage.width()), pixels.shape[:2])
            before = qimage.pixelColor(0, 0)
            pixels[0, 0] = 255
            self.assertEqual(qimage.pixelColor(0, 0), before)
        self.assertTrue(cv_to_qimage(None).isNull())
        self.assertTrue(cv_to_qpixmap(None).isNull())
        pixmap = cv_to_qpixmap(self.image, 120, 100)
        self.assertLessEqual(pixmap.width(), 120)
        self.assertLessEqual(pixmap.height(), 100)

    def test_tiny_and_very_wide_images_have_finite_fit_and_valid_limits(self):
        for pixels in (np.zeros((1, 1), np.uint8), np.zeros((1, 50000), np.uint8)):
            self.view.set_image(pixels)
            self.assertGreater(self.view.scale_factor, 0)
            self.assertLessEqual(self.view.scale_factor, self.view.MAX_SCALE)
            self.wheel(angle=120000)
            self.wheel(angle=-120000)
            self.assertGreater(self.view.scale_factor, 0)

    def test_mouse_interaction_does_not_change_saved_pixels_or_masks(self):
        source = self.image.copy()
        self.view.set_image(source)
        self.view.set_zoom(2)
        self.drag()
        np.testing.assert_array_equal(source, self.image)
        mask = np.zeros((1000, 1200), np.uint8)
        mask[100:400, 200] = 255
        before = mask.copy()
        self.view.set_image(mask)
        self.view.set_zoom(3)
        self.drag()
        np.testing.assert_array_equal(mask, before)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mask.png"
            write_image(target, mask)
            np.testing.assert_array_equal(read_image(target, grayscale=True), before)

    def test_all_gui_panels_have_independent_interactive_viewers(self):
        with tempfile.TemporaryDirectory() as directory:
            window = MainWindow(dataset_dir=directory)
            try:
                panels = [window.before_panel, window.after_panel, window.compare_before,
                          *window.compare_panels.values(), window.batch_before, window.batch_after]
                dialog = MaskDialog(self.image, window)
                panels.extend([dialog.binary_panel, dialog.overlay_panel])
                self.assertTrue(all(isinstance(panel.view, InteractiveImageView) for panel in panels))
                panels[0].set_image(self.image)
                panels[1].set_image(self.image)
                window.single_link_checkbox.setChecked(False)
                panels[0].view.set_zoom(2)
                self.assertTrue(panels[1].view.fit_mode)
                self.assertFalse(panels[0].view.fit_mode)
                dialog.reject()
            finally:
                self.close_widget(window)


if __name__ == "__main__":
    unittest.main()
