"""Offscreen tests for sliders, mask review and shared sharpening controls."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QDialog

from project_1.algorithms.scratch_mask import detect_scratch_mask
from project_1.gui.main_window import MainWindow
from project_1.gui.mask_dialog import MaskDialog
from project_1.gui.widgets import SliderControl
from project_1.utils.image_io import read_image, write_image


class EnhancementGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = np.full((80, 96, 3), 80, np.uint8)
        cv2.line(self.image, (10, 15), (80, 60), (255, 255, 255), 1)
        self.mask = detect_scratch_mask(self.image)
        self.window = MainWindow(dataset_dir=self.root)
        self.window.show()
        write_image(self.root / "a.png", self.image)
        self.window.load_input(self.root / "a.png")
        self.dialogs = []
        self.addCleanup(self.cleanup_widgets)

    def cleanup_widgets(self):
        for dialog in self.dialogs:
            dialog.reject()
            self.app.processEvents()
            dialog.deleteLater()
        if self.window.thread is not None:
            self.window.thread.stop()
            self.window.thread.wait()
            self.app.processEvents()
        self.window.close()
        self.app.processEvents()

    def wait_for_worker(self, owner):
        deadline = time.monotonic() + 10
        while owner.thread is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(owner.thread, "worker did not finish")

    def dialog(self, image=None):
        dialog = MaskDialog(self.image if image is None else image, self.window)
        self.dialogs.append(dialog)
        dialog.show()
        return dialog

    def test_slider_synchronizes_steps_bounds_and_direct_entry(self):
        for settings in ((3, 31, 2, 5, 0), (0.1, 10, 0.1, 1.2, 1),
                         (0, 2, 0.05, 0.5, 2), (1, 20, 1, 3, 0)):
            low, high, step, default, decimals = settings
            control = SliderControl(low, high, step, default, decimals=decimals)
            self.assertEqual(control.value(), default)
            self.assertEqual(control.editor.buttonSymbols(), QAbstractSpinBox.ButtonSymbols.NoButtons)
            control.slider.setValue(control.slider.maximum())
            self.assertEqual(control.value(), high)
            self.assertEqual(control.editor.value(), high)
            control.editor.setValue(low)
            self.assertEqual(control.slider.value(), 0)
            control.setValue(high + 10)
            self.assertEqual(control.value(), high)
            if step == 2:
                control.editor.setValue(4)
                self.assertEqual(control.value(), 5)
                self.assertEqual(control.editor.value(), 5)
            elif step == 0.05:
                control.editor.setValue(0.53)
                self.assertEqual(control.value(), 0.55)
            control.deleteLater()
        control = self.window.controls.kernel
        control.editor.setFocus()
        control.editor.selectAll()
        QTest.keyClicks(control.editor, "11")
        QTest.keyClick(control.editor, Qt.Key.Key_Return)
        self.assertEqual(control.value(), 11)
        QTest.keyClick(control.slider, Qt.Key.Key_Right)
        self.assertEqual(control.value(), 13)

    def test_sliders_do_not_start_processing_and_sharpen_default_is_off(self):
        controls = self.window.controls
        self.assertFalse(controls.sharpen_enabled.isChecked())
        self.assertFalse(controls.sharpen_amount.isEnabled())
        controls.kernel.slider.setValue(5)
        controls.sigma.slider.setValue(17)
        self.assertIsNone(self.window.thread)
        self.assertIsNone(self.window.restored_img)
        controls.sharpen_enabled.setChecked(True)
        self.assertTrue(controls.sharpen_amount.isEnabled())
        controls.sharpen_amount.setValue(0.75)
        self.assertEqual(controls.parameters()["sharpen_amount"], 0.75)

    def test_candidate_generation_is_async_and_parameter_changes_require_regeneration(self):
        dialog = self.dialog()
        dialog.generate()
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertFalse(dialog.generate_button.isEnabled())
        self.assertTrue(all(not c.isEnabled() for c in dialog.controls.values()))
        self.wait_for_worker(dialog)
        self.assertTrue(dialog.apply_button.isEnabled())
        np.testing.assert_array_equal(dialog.candidate, self.mask)
        self.assertIn("chưa được áp dụng", dialog.status.text())
        dialog.controls["expand"].setValue(1)
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.accept()
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
        dialog.generate()
        self.wait_for_worker(dialog)
        self.assertEqual(dialog.candidate_parameters["expand"], 1)
        self.assertTrue(dialog.apply_button.isEnabled())

    def test_empty_candidate_cannot_be_applied(self):
        dialog = self.dialog(np.full(self.image.shape, 80, np.uint8))
        dialog.generate()
        self.wait_for_worker(dialog)
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertIn("Không tìm thấy", dialog.status.text())
        dialog.accept()
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_large_candidate_warns_but_can_still_be_confirmed(self):
        dialog = self.dialog()
        with patch("project_1.gui.mask_dialog.detect_scratch_mask", return_value=np.full(self.mask.shape, 255, np.uint8)):
            dialog.generate()
            self.wait_for_worker(dialog)
        self.assertIn("vượt 10%", dialog.status.text())
        self.assertTrue(dialog.apply_button.isEnabled())

    def test_failed_generation_recovers_controls(self):
        dialog = self.dialog()
        with patch("project_1.gui.mask_dialog.detect_scratch_mask", side_effect=ValueError("test failure")):
            dialog.generate()
            self.wait_for_worker(dialog)
        self.assertIn("test failure", dialog.status.text())
        self.assertTrue(dialog.generate_button.isEnabled())
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertTrue(all(c.isEnabled() for c in dialog.controls.values()))

    def test_modal_cancel_keeps_existing_mask_and_confirm_replaces_it(self):
        w = self.window
        old_mask = np.zeros(self.mask.shape, np.uint8)
        old_mask[0, 0] = 255
        w.mask_img = old_mask.copy()
        w.restored_img = self.image.copy()
        def review(dialog):
            self.assertFalse(w.controls.isEnabled())
            dialog.generate()
            self.wait_for_worker(dialog)
            dialog.reject()
            return dialog.result()
        with patch.object(MaskDialog, "exec", review):
            w._suggest_mask()
        np.testing.assert_array_equal(w.mask_img, old_mask)
        self.assertIsNotNone(w.restored_img)
        self.assertTrue(w.controls.isEnabled())
        def accept_review(dialog):
            dialog.generate()
            self.wait_for_worker(dialog)
            dialog.accept()
            return dialog.result()
        with patch.object(MaskDialog, "exec", accept_review):
            w._suggest_mask()
        np.testing.assert_array_equal(w.mask_img, self.mask)
        self.assertIsNone(w.restored_img)
        self.assertIn("đã xác nhận", w.input_label.text())
        self.assertEqual(w.mask_parameters["expand"], 0)

    def test_accepted_mask_save_and_new_input_clear(self):
        w = self.window
        w.apply_auto_mask(self.mask, {"expand": 0})
        output = self.root / "accepted.png"
        with patch("project_1.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(output), "")):
            w._save_mask()
        np.testing.assert_array_equal(read_image(output, grayscale=True), self.mask)
        self.mask[0, 0] = 255
        self.assertEqual(w.mask_img[0, 0], 0)
        w.load_input(self.root / "a.png")
        self.assertIsNone(w.mask_img)
        self.assertIsNone(w.mask_parameters)
        self.assertEqual(w.mask_source, "")
        self.assertFalse(w.save_mask_button.isEnabled())

    def test_auto_mask_is_unavailable_in_batch(self):
        w = self.window
        self.assertTrue(w.auto_mask_button.isEnabled())
        w.tabs.setCurrentIndex(2)
        self.assertFalse(w.auto_mask_button.isEnabled())
        self.assertFalse(w.save_mask_button.isEnabled())
        w._suggest_mask()
        self.assertIsNone(w.mask_dialog)
        w.tabs.setCurrentIndex(1)
        self.assertTrue(w.auto_mask_button.isEnabled())

    def test_cancel_running_mask_worker_waits_safely(self):
        dialog = self.dialog()
        def slow_detector(*args, **kwargs):
            time.sleep(0.08)
            return detect_scratch_mask(*args, **kwargs)
        with patch("project_1.gui.mask_dialog.detect_scratch_mask", side_effect=slow_detector):
            dialog.generate()
            worker = dialog.thread
            dialog.reject()
            self.assertFalse(worker.isRunning())
            self.app.processEvents()
        self.assertIsNone(dialog.candidate)
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_single_and_batch_share_sharpening_configuration(self):
        w = self.window
        w.controls.algorithm.setCurrentIndex(2)
        w.mask_img = self.mask.copy()
        w.controls.sharpen_enabled.setChecked(True)
        w.controls.sharpen_amount.setValue(0.75)
        w._run_single_restoration()
        self.wait_for_worker(w)
        expected = w.pipeline.run_single("combined", self.image, mask=self.mask, **w.controls.parameters())
        np.testing.assert_array_equal(w.restored_img, expected)
        (self.root / "input").mkdir()
        (self.root / "masks").mkdir()
        write_image(self.root / "input/a.png", self.image)
        write_image(self.root / "masks/a.png", self.mask)
        w.batch_input.setText(str(self.root / "input"))
        w.batch_masks.setText(str(self.root / "masks"))
        w.batch_output.setText(str(self.root / "output"))
        w.tabs.setCurrentIndex(2)
        w._start_batch()
        self.wait_for_worker(w)
        np.testing.assert_array_equal(read_image(w.batch_rows[0]["output_path"]), expected)


if __name__ == "__main__":
    unittest.main()
