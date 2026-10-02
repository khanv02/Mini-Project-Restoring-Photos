"""Tests for fast preview, exact settling and parameter advice."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from project_1.entrypoint import RestorationPipeline
from project_1.gui.main_window import MainWindow
from project_1.metrics.advisor import ParameterAdvisor
from project_1.metrics.evaluator import QualityEvaluator
from project_1.utils.image_io import write_image


class LivePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.style().objectName().lower() != "fusion":
            cls.app.setStyle("Fusion")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = np.full((96, 80, 3), 100, np.uint8)
        cv2.line(self.image, (8, 10), (70, 80), (230, 230, 230), 2)
        self.path = self.root / "input.png"
        write_image(self.path, self.image)
        self.window = MainWindow(dataset_dir=self.root)
        self.window.show()
        self.window.load_input(self.path)
        self.window.clean_img = self.image.copy()
        self.addCleanup(self.close_window)

    def close_window(self):
        if self.window.preview_thread is not None:
            self.window.preview_thread.stop()
            self.window.preview_thread.wait()
        if self.window.thread is not None:
            self.window.thread.stop()
            self.window.thread.wait()
        self.window.close()
        self.window.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def wait_until_idle(self, timeout=20):
        deadline = time.monotonic() + timeout
        while ((self.window.thread is not None or self.window.preview_thread is not None or
                self.window._preview_pending) and time.monotonic() < deadline):
            QTest.qWait(10)
        self.assertIsNone(self.window.thread)
        self.assertIsNone(self.window.preview_thread)
        self.assertFalse(self.window._preview_pending)

    def test_slider_drag_shows_fast_preview_then_exact_result(self):
        self.window._run_single_restoration()
        self.wait_until_idle()

        control = self.window.controls.kernel
        control.slider.sliderPressed.emit()
        control.setValue(9)
        deadline = time.monotonic() + 5
        while not self.window._preview_approximate and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(self.window._preview_approximate)
        self.assertIn("Preview nhanh", self.window.lbl_metrics.text())
        self.assertFalse(self.window.save_single_action.isEnabled())

        control.slider.sliderReleased.emit()
        self.wait_until_idle()
        self.assertFalse(self.window._preview_approximate)
        self.assertEqual(self.window.single_result["parameters"]["kernel_size"], 9)
        self.assertIn("Xem trước thời gian thực", self.window.lbl_metrics.text())
        self.assertTrue(self.window.save_single_action.isEnabled())

    def test_compare_reuses_one_baseline_evaluation(self):
        pipeline = RestorationPipeline()
        mask = np.zeros(self.image.shape[:2], np.uint8)
        mask[10:14, 10:18] = 255
        calls = []
        original = pipeline.evaluate

        def evaluate(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        pipeline.evaluate = evaluate
        pipeline.compare_all(self.image, self.image, mask, kernel_size=5, sigma=1.2)
        self.assertEqual(len(calls), 4)

    def test_preview_metrics_are_approximate_but_exact_evaluator_is_unchanged(self):
        restored = cv2.GaussianBlur(self.image, (5, 5), 1.2)
        preview = QualityEvaluator.evaluate_preview(self.image, restored, max_side=32)
        exact = QualityEvaluator.evaluate(self.image, restored)
        self.assertNotEqual(preview, exact)
        self.assertEqual(QualityEvaluator.evaluate_preview(self.image, restored, max_side=320), exact)

    def test_compare_slider_uses_fast_then_exact_preview(self):
        self.window.tabs.setCurrentIndex(1)
        self.window._run_comparison()
        self.wait_until_idle()
        control = self.window.controls.kernel
        control.slider.sliderPressed.emit()
        control.setValue(9)
        deadline = time.monotonic() + 10
        while not self.window._preview_approximate and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(self.window._preview_approximate)
        self.assertIn("Preview nhanh", self.window.statusBar().currentMessage())
        control.slider.sliderReleased.emit()
        self.wait_until_idle()
        self.assertFalse(self.window._preview_approximate)
        self.assertEqual(
            self.window.comparison_results["Gaussian Filter"]["parameters"]["kernel_size"], 9)

    def test_advisor_candidate_values_respect_gui_bounds(self):
        params = {
            "kernel_size": 31, "sigma": 10.0, "radius": 20, "method": "telea",
            "sharpen_enabled": True, "sharpen_amount": 2.0, "sharpen_sigma": 3.0,
        }
        candidates = ParameterAdvisor._candidates("combined", params)
        for candidate in candidates:
            values = candidate["parameters"]
            self.assertLessEqual(values["kernel_size"], 31)
            self.assertLessEqual(values["sigma"], 10.0)
            self.assertLessEqual(values["radius"], 20)
            self.assertLessEqual(values["sharpen_amount"], 2.0)
            self.assertLessEqual(values["sharpen_sigma"], 3.0)

    def test_advisor_runs_as_separate_action(self):
        self.window._run_single_restoration()
        self.wait_until_idle()
        self.assertTrue(self.window.advisor_button.isEnabled())
        self.window._run_advisor()
        self.wait_until_idle()
        self.assertNotIn("Đang phân tích", self.window.advisor_status.text())


if __name__ == "__main__":
    unittest.main()
