"""Compact action menus, conditional controls and state-aware GUI affordances."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFontDatabase, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea, QToolButton

from project_1.gui.main_window import MainWindow
from project_1.gui.widgets import ImagePanel
from project_1.utils.image_io import read_image, write_image


class CompactLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.style().objectName().lower() != "fusion":
            cls.app.setStyle("Fusion")
        # Offscreen Windows does not discover fonts as the desktop platform does.
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = np.full((64, 80, 3), 90, np.uint8)
        self.mask = np.zeros((64, 80), np.uint8)
        self.mask[15:25, 20] = 255
        self.path = self.root / "input.png"
        write_image(self.path, self.image)
        self.window = MainWindow(dataset_dir=self.root)
        self.window.show()
        self.app.processEvents()
        self.addCleanup(self.close_window)

    def close_window(self):
        if self.window.thread is not None:
            self.window.thread.stop()
            self.window.thread.wait()
            self.app.processEvents()
        self.window.close()
        self.window.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def wait_for_worker(self):
        deadline = time.monotonic() + 10
        while self.window.thread is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window.thread)

    def test_sidebar_has_two_input_controls_and_all_secondary_actions(self):
        w = self.window
        self.assertEqual(w.sidebar.findChildren(QPushButton), [w.open_button])
        self.assertEqual(w.sidebar.findChildren(QToolButton), [w.input_options_button])
        actions = [action for action in w.input_menu.actions() if not action.isSeparator()]
        self.assertEqual(actions, [w.reference_action, w.load_mask_action, w.auto_mask_action,
                                  w.save_mask_action, w.clear_references_action, w.demo_action])
        self.assertEqual(w.open_button.shortcut(), QKeySequence(QKeySequence.StandardKey.Open))
        self.assertTrue(w.open_button.property("primary"))

    def test_empty_and_loaded_action_states(self):
        w = self.window
        self.assertFalse(w.run_button.isEnabled())
        self.assertFalse(w.compare_button.isEnabled())
        self.assertFalse(w.single_export_button.isEnabled())
        self.assertFalse(w.compare_export_button.isEnabled())
        self.assertFalse(w.reference_action.isEnabled())
        self.assertTrue(w.demo_action.isEnabled())
        with patch("project_1.gui.main_window.QFileDialog.getOpenFileName", return_value=(str(self.path), "")):
            w.open_button.click()
        self.assertTrue(w.run_button.isEnabled())
        self.assertTrue(w.compare_button.isEnabled())
        self.assertTrue(w.reference_action.isEnabled())
        self.assertTrue(w.auto_mask_action.isEnabled())
        self.assertFalse(w.save_mask_action.isEnabled())

    def test_reference_and_mask_menu_actions_load_clear_and_save(self):
        w = self.window
        w.load_input(self.path)
        mask_path = self.root / "mask.png"
        write_image(mask_path, self.mask)
        with patch("project_1.gui.main_window.QFileDialog.getOpenFileName", return_value=(str(self.path), "")):
            w.reference_action.trigger()
        with patch("project_1.gui.main_window.QFileDialog.getOpenFileName", return_value=(str(mask_path), "")):
            w.load_mask_action.trigger()
        np.testing.assert_array_equal(w.clean_img, self.image)
        np.testing.assert_array_equal(w.mask_img, self.mask)
        self.assertTrue(w.clear_references_action.isEnabled())
        target = self.root / "saved_mask.png"
        with patch("project_1.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(target), "")):
            w.save_mask_action.trigger()
        np.testing.assert_array_equal(read_image(target, grayscale=True), self.mask)
        w.clear_references_action.trigger()
        self.assertIsNone(w.clean_img)
        self.assertIsNone(w.mask_img)
        self.assertIsNotNone(w.corrupted_img)
        self.assertFalse(w.clear_references_action.isEnabled())

    def test_required_mask_and_batch_context_control_available_actions(self):
        w = self.window
        w.load_input(self.path)
        w.controls.algorithm.setCurrentIndex(1)
        self.assertFalse(w.run_button.isEnabled())
        self.assertTrue(w.compare_button.isEnabled())
        w.apply_auto_mask(self.mask, {"expand": 0})
        self.assertTrue(w.run_button.isEnabled())
        w.tabs.setCurrentIndex(2)
        self.assertFalse(w.auto_mask_action.isEnabled())
        self.assertFalse(w.reference_action.isEnabled())
        self.assertFalse(w.save_mask_action.isEnabled())
        w.tabs.setCurrentIndex(0)
        self.assertTrue(w.save_mask_action.isEnabled())

    def test_export_menus_save_and_invalidate_on_new_input(self):
        w = self.window
        w.load_input(self.path)
        w.run_button.click()
        self.wait_for_worker()
        self.assertTrue(w.single_export_button.isEnabled())
        target = self.root / "restored.png"
        with patch("project_1.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(target), "")):
            w.save_single_action.trigger()
        np.testing.assert_array_equal(read_image(target), w.restored_img)
        w.tabs.setCurrentIndex(1)
        w.compare_button.click()
        self.wait_for_worker()
        self.assertEqual(len(w.compare_export_button.menu().actions()), 2)
        self.assertTrue(w.export_csv_action.isEnabled())
        self.assertTrue(w.save_comparison_action.isEnabled())
        w.table_comp.selectRow(1)  # No mask: skipped method cannot be saved.
        self.assertFalse(w.save_comparison_action.isEnabled())
        self.assertTrue(w.export_csv_action.isEnabled())
        csv_path = self.root / "comparison.csv"
        with patch("project_1.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(csv_path), "")):
            w.export_csv_action.trigger()
        self.assertTrue(csv_path.is_file())
        w.load_input(self.path)
        self.assertFalse(w.single_export_button.isEnabled())
        self.assertFalse(w.compare_export_button.isEnabled())

    def test_busy_locks_menu_actions_and_recovers_valid_states(self):
        w = self.window
        w.load_input(self.path)
        original = w.pipeline.restore_result
        def slow_task(*args, **kwargs):
            time.sleep(0.1)
            return original(*args, **kwargs)
        with patch.object(w.pipeline, "restore_result", side_effect=slow_task):
            w.run_button.click()
            self.assertFalse(w.input_options_button.isEnabled())
            self.assertFalse(w.demo_action.isEnabled())
            self.assertFalse(w.reference_action.isEnabled())
            self.assertFalse(w.save_single_action.isEnabled())
            self.wait_for_worker()
        self.assertTrue(w.input_options_button.isEnabled())
        self.assertTrue(w.reference_action.isEnabled())
        self.assertFalse(w.save_mask_action.isEnabled())
        self.assertTrue(w.save_single_action.isEnabled())

    def test_sharpen_rows_are_only_visible_when_enabled(self):
        controls = self.window.controls
        for control in (controls.sharpen_amount, controls.sharpen_sigma, controls.sharpen_hint):
            self.assertFalse(controls.form.isRowVisible(control))
        controls.sharpen_enabled.setChecked(True)
        controls.sharpen_amount.setValue(0.75)
        self.assertTrue(controls.form.isRowVisible(controls.sharpen_amount))
        controls.sharpen_enabled.setChecked(False)
        self.assertFalse(controls.form.isRowVisible(controls.sharpen_amount))
        self.assertEqual(controls.sharpen_amount.value(), 0.75)

    def test_zoom_menu_is_compact_and_mouse_selection_works(self):
        panel = ImagePanel("Viewer")
        self.addCleanup(self.close_widget, panel)
        panel.resize(400, 350)
        panel.show()
        panel.set_image(self.image)
        self.app.processEvents()
        self.assertEqual(panel.findChildren(QPushButton), [])
        self.assertEqual(panel.findChildren(QToolButton), [panel.zoom_button])
        self.assertEqual(panel.view.actions(), [panel.fit_action, panel.fill_action, panel.actual_size_action])
        menu = panel.zoom_button.menu()
        menu.popup(panel.zoom_button.mapToGlobal(panel.zoom_button.rect().bottomLeft()))
        self.app.processEvents()
        QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(panel.actual_size_action).center())
        self.assertEqual(panel.view.scale_factor, 1)
        self.assertEqual(panel.zoom_button.text(), "100%")
        panel.set_image(None)
        self.assertFalse(panel.zoom_button.isEnabled())

    def close_widget(self, widget):
        widget.close()
        widget.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_small_window_sidebar_and_directory_pickers(self):
        w = self.window
        w.resize(1024, 720)
        self.app.processEvents()
        self.assertEqual(w.findChild(QScrollArea).horizontalScrollBar().maximum(), 0)
        w.tabs.setCurrentIndex(2)
        buttons = w.tabs.widget(2).findChildren(QToolButton, "directoryPicker")
        self.assertEqual(len(buttons), 4)
        self.assertTrue(all(button.accessibleName() for button in buttons))
        with patch("project_1.gui.main_window.QFileDialog.getExistingDirectory", return_value=str(self.root)):
            buttons[0].click()
        self.assertEqual(w.batch_input.text(), str(self.root))


if __name__ == "__main__":
    unittest.main()
