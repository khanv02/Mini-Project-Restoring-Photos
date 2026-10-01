"""Large comparison workspaces, shared cameras, batch inspection and popup contrast."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QFontDatabase, QMouseEvent, QPalette
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from project_1.gui.main_window import MainWindow
from project_1.gui.widgets import InteractiveImageView, LinkedImageViews
from project_1.settings import ALGORITHMS
from project_1.utils.batch_processor import BatchResult
from project_1.utils.image_io import read_image, write_image
from project_1.utils.reports import report_row


class ComparisonGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.style().objectName().lower() != "fusion":
            cls.app.setStyle("Fusion")
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = np.full((512, 320, 3), [30, 80, 110], np.uint8)
        self.clean = np.full_like(self.image, 140)
        self.path = self.root / "input.png"
        write_image(self.path, self.image)
        self.window = MainWindow(dataset_dir=self.root)
        self.window.resize(1440, 940)
        self.window.show()
        self.window.load_input(self.path)
        self.addCleanup(self.close_widget, self.window)
        # Synthetic result payloads isolate UI selection/camera tests from algorithm quality.
        self.results = {spec.label: self.result(np.full_like(self.image, 50 + index * 40))
                        for index, spec in enumerate(ALGORITHMS)}
        self.app.processEvents()

    @staticmethod
    def result(image):
        return {"image": image, "status": "success", "parameters": {}, "error": "",
                "metrics_error": "", "baseline": {"PSNR": 20., "SSIM": .6},
                "metrics": {"PSNR": 24., "SSIM": .8}, "delta": {"PSNR": 4., "SSIM": .2},
                "time": .01}

    def close_widget(self, widget):
        if isinstance(widget, MainWindow) and widget.thread is not None:
            widget.thread.stop()
            widget.thread.wait()
            self.app.processEvents()
        widget.close()
        widget.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def compare(self):
        self.window.tabs.setCurrentIndex(1)
        self.window._comparison_completed(self.results)
        self.app.processEvents()
        return self.window

    def wait_for_task(self):
        deadline = time.monotonic() + 10
        while self.window.thread is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window.thread)

    def assert_same_camera(self, first, second):
        a, b = first.viewport_state(), second.viewport_state()
        self.assertAlmostEqual(a[0], b[0])
        self.assertAlmostEqual(a[1].x(), b[1].x())
        self.assertAlmostEqual(a[1].y(), b[1].y())
        self.assertEqual(a[2], b[2])

    def test_default_pair_is_large_and_overview_reuses_panels(self):
        w = self.compare()
        panels = list(w.compare_panels.values())
        self.assertEqual(sum(panel.isVisible() for panel in panels), 1)
        self.assertGreater(w.compare_before.view.height(), 400)
        pair_height = w.compare_before.view.height()
        identities = [id(panel) for panel in panels]
        for _ in range(3):
            w.compare_overview_action.trigger()
            self.app.processEvents()
            self.assertEqual(sum(panel.isVisible() for panel in panels), 3)
            self.assertLess(w.compare_before.view.height(), pair_height / 1.4)
            self.assertFalse(w.compare_selectors.isVisible())
            self.assertIs(w.compare_before.image, w.corrupted_img)
            w.compare_pair_action.trigger()
            self.app.processEvents()
            self.assertEqual(sum(panel.isVisible() for panel in panels), 1)
            self.assertTrue(w.compare_selectors.isVisible())
        self.assertEqual(identities, [id(panel) for panel in w.compare_panels.values()])

    def test_choose_algorithms_or_reference_without_processing(self):
        w = self.compare()
        with patch.object(w.pipeline, "compare_all") as compute:
            w.compare_left_selector.setCurrentIndex(3)  # Inpainting.
            w.compare_right_selector.setCurrentIndex(2)  # Combined.
            self.assertIs(w.compare_before.image, self.results[ALGORITHMS[1].label]["image"])
            self.assertEqual(w.compare_before.title(), ALGORITHMS[1].label)
            self.assertIs(w.compare_result_stack.currentWidget(), w.compare_panels[ALGORITHMS[2].label])
            w.clean_img = self.clean
            w.compare_left_selector.setCurrentIndex(1)
            self.assertIs(w.compare_before.image, self.clean)
            self.assertEqual(w.compare_before.title(), "Tham chiếu")
            compute.assert_not_called()
        self.assertIsNone(w.thread)

    def test_table_and_selector_save_the_same_right_image(self):
        w = self.compare()
        w.compare_right_selector.setCurrentIndex(2)
        self.assertEqual(w.table_comp.currentRow(), 2)
        w.table_comp.selectRow(1)
        self.assertEqual(w.compare_right_selector.currentIndex(), 1)
        target = self.root / "selected.png"
        with patch("project_1.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(target), "")):
            w._save_comparison()
        np.testing.assert_array_equal(read_image(target), self.results[ALGORITHMS[1].label]["image"])

    def test_skipped_results_and_new_input_do_not_leave_stale_images(self):
        w = self.compare()
        skipped = self.results[ALGORITHMS[1].label].copy()
        skipped.update(image=None, status="skipped", error="Missing mask")
        self.results[ALGORITHMS[1].label] = skipped
        w._clear_results()
        w._comparison_completed(self.results)
        w.table_comp.selectRow(1)
        self.assertIsNone(w.compare_result_stack.currentWidget().image)
        self.assertIn("Missing mask", w.compare_result_stack.currentWidget().caption.toolTip())
        w.compare_left_selector.setCurrentIndex(3)
        self.assertIsNone(w.compare_before.image)
        w.load_input(self.path)
        self.assertEqual(w.compare_left_selector.currentIndex(), 0)
        self.assertIs(w.compare_before.image, w.corrupted_img)
        self.assertTrue(all(panel.image is None for panel in w.compare_panels.values()))

    def test_zoom_and_drag_share_image_space_camera(self):
        w = self.compare()
        first = w.compare_before.view
        second = w.compare_result_stack.currentWidget().view
        first.set_zoom(3)
        self.assert_same_camera(first, second)
        start, end = QPoint(150, 150), QPoint(185, 170)
        QTest.mousePress(first, Qt.MouseButton.LeftButton, pos=start)
        move = QMouseEvent(QEvent.Type.MouseMove, QPointF(end), QPointF(first.mapToGlobal(end)),
                           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(first, move)
        QTest.mouseRelease(first, Qt.MouseButton.LeftButton, pos=end)
        self.assert_same_camera(first, second)
        QTest.mouseDClick(second, Qt.MouseButton.LeftButton)
        self.assertTrue(first.fit_mode)
        self.assertTrue(second.fit_mode)

    def test_link_can_be_disabled_and_reenabled(self):
        w = self.compare()
        first = w.compare_before.view
        second = w.compare_result_stack.currentWidget().view
        w.compare_link_checkbox.setChecked(False)
        first.set_zoom(3)
        self.assertTrue(second.fit_mode)
        w.compare_link_checkbox.setChecked(True)
        self.assert_same_camera(first, second)

    def test_selection_preserves_zoom_and_links_new_result(self):
        w = self.compare()
        w.compare_before.view.set_zoom(3)
        w.compare_right_selector.setCurrentIndex(2)
        self.app.processEvents()
        self.assert_same_camera(w.compare_before.view, w.compare_result_stack.currentWidget().view)
        w.compare_left_selector.setCurrentIndex(2)
        self.assertEqual(w.compare_before.view.scale_factor, 3)
        self.assert_same_camera(w.compare_before.view, w.compare_result_stack.currentWidget().view)

    def test_single_completed_uses_existing_input_camera(self):
        w = self.window
        w.before_panel.view.set_zoom(3)
        w._single_completed(self.results[ALGORITHMS[0].label])
        self.assert_same_camera(w.before_panel.view, w.after_panel.view)

    def test_hiding_table_and_sidebar_gives_images_more_space(self):
        w = self.compare()
        height, width = w.compare_before.view.height(), w.compare_before.view.width()
        w.compare_table_action.setChecked(False)
        self.app.processEvents()
        self.assertGreater(w.compare_before.view.height(), height)
        w.sidebar_toggle.setChecked(True)
        self.app.processEvents()
        self.assertGreater(w.compare_before.view.width(), width)
        self.assertFalse(w.sidebar_scroll.isVisible())
        w.sidebar_toggle.setChecked(False)
        self.assertTrue(w.sidebar_scroll.isVisible())

    def batch_row(self, name):
        source, output = self.root / name, self.root / ("out_" + name)
        write_image(source, self.image)
        write_image(output, self.clean)
        return report_row(name, "Gaussian Filter", self.result(self.clean), source, output)

    def test_batch_has_large_pair_beside_compact_file_list(self):
        w = self.window
        w.tabs.setCurrentIndex(2)
        w.batch_settings_button.setChecked(False)
        self.app.processEvents()
        w._batch_row(self.batch_row("a.png"))
        self.app.processEvents()
        self.assertEqual(w.table_batch.currentRow(), 0)
        self.assertGreater(w.batch_before.view.height(), 400)
        self.assertGreater(w.batch_before.view.scale_factor, .75)
        self.assertLess(w.table_batch.width(), w.batch_splitter.width() * .4)
        self.assertIn("a.png", w.batch_details.text())
        self.assertTrue(w.table_batch.isColumnHidden(2))
        w.batch_compact_action.setChecked(False)
        self.assertFalse(w.table_batch.isColumnHidden(2))
        self.assertIn("PSNR", w.table_batch.item(0, 0).toolTip())

    def test_batch_completion_preserves_selected_file_and_new_selection_fits(self):
        w = self.window
        w.tabs.setCurrentIndex(2)
        w._batch_row(self.batch_row("a.png"))
        w._batch_row(self.batch_row("b.png"))
        w.table_batch.selectRow(1)
        w.batch_before.view.set_zoom(3)
        self.assert_same_camera(w.batch_before.view, w.batch_after.view)
        w._batch_row(self.batch_row("c.png"))
        w._batch_completed(BatchResult(total=3, processed=3, evaluated=3))
        self.assertEqual(w.table_batch.currentRow(), 1)
        self.assertEqual(w.batch_before.view.scale_factor, 3)
        w.table_batch.selectRow(0)
        self.assertTrue(w.batch_before.view.fit_mode)
        self.assertTrue(w.batch_after.view.fit_mode)

    def test_batch_clears_after_on_skipped_or_unreadable_output(self):
        w = self.window
        w.tabs.setCurrentIndex(2)
        w._batch_row(self.batch_row("a.png"))
        row = self.batch_row("bad.png")
        row.update(status="skipped", output_path="", error="Missing mask")
        w._batch_row(row)
        w.table_batch.selectRow(1)
        self.assertIsNone(w.batch_after.image)
        self.assertIn("Missing mask", w.batch_details.text())
        row["output_path"] = str(self.root / "missing.png")
        w._show_batch_preview()
        self.assertIsNone(w.batch_after.image)
        self.assertTrue(w.batch_after.caption.toolTip())

    def test_batch_start_collapses_settings_and_failure_reopens_them(self):
        w = self.window
        w.tabs.setCurrentIndex(2)
        incoming = self.root / "incoming"
        incoming.mkdir()
        write_image(incoming / "a.png", self.image)
        w.batch_input.setText(str(incoming))
        w.batch_output.setText(str(self.root / "output"))
        w._start_batch()
        self.assertFalse(w.batch_settings_button.isChecked())
        self.wait_for_task()
        self.assertIsNotNone(w.batch_after.image)
        w.batch_input.setText(str(self.root / "missing"))
        with patch("project_1.gui.main_window.QMessageBox.warning"):
            w._start_batch()
            self.wait_for_task()
        self.assertTrue(w.batch_settings_button.isChecked())
        self.assertTrue(w.batch_settings.isVisible())

    def test_fill_crops_without_distortion_and_fit_resets_both_views(self):
        w = self.compare()
        panel = w.compare_before
        before = self.image.copy()
        panel.fill_action.trigger()
        view = panel.view
        self.assertGreaterEqual(self.image.shape[1] * view.scale_factor, view.width() - .01)
        self.assertGreaterEqual(self.image.shape[0] * view.scale_factor, view.height() - .01)
        self.assertFalse(view.fit_mode)
        self.assert_same_camera(view, w.compare_result_stack.currentWidget().view)
        panel.fit_action.trigger()
        self.assertTrue(view.fit_mode)
        self.assertTrue(w.compare_result_stack.currentWidget().view.fit_mode)
        np.testing.assert_array_equal(self.image, before)

    def test_camera_signals_ignore_load_and_resize_and_skip_mismatched_images(self):
        first, second = InteractiveImageView(), InteractiveImageView()
        self.addCleanup(self.close_widget, first)
        self.addCleanup(self.close_widget, second)
        first.resize(300, 250)
        second.resize(300, 250)
        first.show()
        second.show()
        spy = QSignalSpy(first.viewportChanged)
        first.set_image(self.image)
        second.set_image(np.zeros((30, 40), np.uint8))
        first.resize(350, 300)
        self.app.processEvents()
        self.assertEqual(spy.count(), 0)
        link = LinkedImageViews([first, second], first)
        first.set_zoom(3)
        self.assertEqual(spy.count(), 1)
        self.assertTrue(second.fit_mode)
        second.set_image(self.clean)
        self.assertEqual(first.scale_factor, 3)
        link.sync_from(first)
        self.assert_same_camera(first, second)

    def test_combobox_selected_text_is_white_on_blue_in_both_palette_groups(self):
        for combo in (self.window.controls.algorithm, self.window.controls.method,
                      self.window.compare_left_selector, self.window.compare_right_selector):
            option = QStyleOptionViewItem()
            combo.view().itemDelegate().initStyleOption(option, combo.model().index(0, 0))
            for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
                self.assertEqual(option.palette.color(group, QPalette.ColorRole.HighlightedText).name(), "#ffffff")
                self.assertEqual(option.palette.color(group, QPalette.ColorRole.Highlight).name(), "#2563eb")
            self.assertIn("selection-color: white", self.window.styleSheet())


if __name__ == "__main__":
    unittest.main()
