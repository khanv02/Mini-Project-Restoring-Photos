"""Qt workflow checks without opening a desktop window or requiring pytest."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from project_1.gui.main_window import MainWindow
from project_1.gui.threads import TaskWorkerThread
from project_1.utils.image_io import read_image, write_image


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("images_clean", "images_corrupted", "images_masks"):
            (self.root / name).mkdir()
        self.image = np.random.default_rng(8).integers(0, 256, (24, 24, 3), dtype=np.uint8)
        self.mask = np.zeros((24, 24), np.uint8)
        self.mask[8:10, 8:12] = 255
        for name in ("a.png", "b.png"):
            write_image(self.root / "images_clean" / name, self.image)
            write_image(self.root / "images_corrupted" / name, self.image)
            write_image(self.root / "images_masks" / name, self.mask)
        self.window = MainWindow(dataset_dir=self.root)
        self.window.show()
        self.warnings = []
        self.dialog_patch = patch.object(QMessageBox, "warning", side_effect=lambda *args: self.warnings.append(args[2]))
        self.dialog_patch.start()
        self.addCleanup(self.dialog_patch.stop)
        self.addCleanup(self.cleanup_window)

    def cleanup_window(self):
        if self.window.thread is not None:
            self.window.thread.stop()
            self.window.thread.wait()
            self.app.processEvents()
        self.window.close()
        self.app.processEvents()

    def wait_for_task(self):
        deadline = time.monotonic() + 10
        while self.window.thread is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window.thread, "worker did not finish")
        self.assertTrue(self.window.run_button.isEnabled())

    def test_single_background_and_input_invalidates_results(self):
        w = self.window
        w.load_input(self.root / "images_corrupted" / "a.png")
        w.clean_img, w.mask_img = self.image, self.mask
        w._run_single_restoration()
        self.assertFalse(w.run_button.isEnabled())
        worker = w.thread
        w._run_single_restoration()
        self.assertIs(w.thread, worker)
        self.wait_for_task()
        self.assertIsNotNone(w.restored_img)
        self.assertIn("PSNR:", w.lbl_metrics.text())
        w.load_input(self.root / "images_corrupted" / "b.png")
        self.assertIsNone(w.clean_img)
        self.assertIsNone(w.mask_img)
        self.assertIsNone(w.restored_img)
        self.assertEqual(w.table_comp.rowCount(), 0)

    def test_comparison_without_reference_and_export_selected_image(self):
        w = self.window
        w.load_input(self.root / "images_corrupted" / "a.png")
        w.tabs.setCurrentIndex(1)
        w.controls.kernel.setValue(9)
        w.controls.sigma.setValue(2)
        w._run_comparison()
        self.wait_for_task()
        self.assertEqual(w.table_comp.rowCount(), 3)
        self.assertEqual(w.comparison_results["Inpainting"]["status"], "skipped")
        self.assertIsNone(w.comparison_results["Gaussian Filter"]["metrics"])
        self.assertEqual(w.comparison_results["Gaussian Filter"]["parameters"]["kernel_size"], 9)
        csv_path = self.root / "compare.csv"
        with patch("project_1.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(csv_path), "")):
            w._export_comparison()
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)
        png_path = self.root / "selected.png"
        with patch("project_1.gui.main_window.QFileDialog.getSaveFileName", return_value=(str(png_path), "")):
            w.table_comp.selectRow(0)
            w._save_comparison()
        np.testing.assert_array_equal(read_image(png_path), w.comparison_results["Gaussian Filter"]["image"])
        self.assertFalse(self.warnings)

    def test_demo_and_shared_batch_configuration(self):
        w = self.window
        w.load_demo()
        self.wait_for_task()
        self.assertEqual(len(w.comparison_results), 3)
        self.assertTrue(all(d["status"] == "success" for d in w.comparison_results.values()))
        w.controls.kernel.setValue(7)
        w.controls.sigma.setValue(2)
        w.batch_output.setText(str(self.root / "output"))
        w._start_batch()
        self.wait_for_task()
        self.assertEqual(len(w.batch_rows), 2)
        self.assertEqual(w.progress_bar.value(), 100)
        self.assertIn("2 cặp được đánh giá", w.lbl_batch_status.text())
        expected = w.pipeline.run_single("combined", self.image, mask=self.mask, kernel_size=7, sigma=2)
        np.testing.assert_array_equal(read_image(w.batch_rows[0]["output_path"]), expected)
        self.assertIsNotNone(w.batch_after.image)

    def test_gui_event_loop_remains_responsive(self):
        w = self.window
        w.load_input(self.root / "images_corrupted" / "a.png")
        original = w.pipeline.restore_result
        def slow_task(*args, **kwargs):
            time.sleep(0.15)
            return original(*args, **kwargs)
        beats = []
        timer = QTimer()
        timer.setInterval(10)
        timer.timeout.connect(lambda: beats.append(1))
        timer.start()
        with patch.object(w.pipeline, "restore_result", side_effect=slow_task):
            w._run_single_restoration()
            self.wait_for_task()
        timer.stop()
        self.assertGreater(len(beats), 3)

    def test_batch_cancel_and_invalid_directory_recover_controls(self):
        w = self.window
        w.batch_output.setText(str(self.root / "output"))
        original = w.pipeline.run_single
        def slow_file(*args, **kwargs):
            time.sleep(0.1)
            return original(*args, **kwargs)
        with patch.object(w.pipeline, "run_single", side_effect=slow_file):
            w._start_batch()
            w._cancel_batch()
            self.wait_for_task()
        self.assertIn("Đã hủy", w.lbl_batch_status.text())
        self.assertFalse(w.cancel_button.isEnabled())
        w.batch_input.setText(str(self.root / "missing"))
        w._start_batch()
        self.wait_for_task()
        self.assertTrue(self.warnings)
        self.assertIn("thất bại", w.lbl_batch_status.text())

    def test_close_running_worker_cancel_or_wait(self):
        w = self.window
        worker = TaskWorkerThread(lambda: time.sleep(0.1), w)
        w._start_worker(worker, lambda result: None)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            event = QCloseEvent()
            w.closeEvent(event)
            self.assertFalse(event.isAccepted())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            event = QCloseEvent()
            w.closeEvent(event)
            self.assertTrue(event.isAccepted())
        self.assertFalse(worker.isRunning())
        self.app.processEvents()

    def test_package_console_entrypoint_delegates_to_gui_launcher(self):
        from project_1 import main
        with patch("project_1.main.main", return_value=7) as launch:
            self.assertEqual(main(), 7)
            launch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
