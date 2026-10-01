import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from project_1.entrypoint import RestorationPipeline
from project_1.generate_data import create_corrupted_dataset
from project_1.metrics.evaluator import QualityEvaluator
from project_1.utils.batch_processor import BatchProcessor
from project_1.utils.image_io import read_image, write_image
from project_1.utils.reports import metric_delta, summarize


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(4).integers(0, 256, (16, 17, 3), dtype=np.uint8)

    def test_identity_and_infinite_delta(self):
        metric = QualityEvaluator.evaluate(self.image, self.image)
        self.assertEqual(metric, {"PSNR": math.inf, "SSIM": 1.0})
        self.assertEqual(metric_delta(metric, metric), {"PSNR": 0.0, "SSIM": 0.0})

    def test_full_precision_matches_independent_metrics(self):
        restored = cv2.GaussianBlur(self.image, (5, 5), 1.2)
        metric = QualityEvaluator.evaluate(self.image, restored)
        self.assertEqual(metric["PSNR"], peak_signal_noise_ratio(self.image, restored, data_range=255))
        self.assertEqual(metric["SSIM"], structural_similarity(
            self.image, restored, channel_axis=2, data_range=255))
        self.assertNotEqual(metric["SSIM"], round(metric["SSIM"], 4))

    def test_small_grayscale_and_color(self):
        for shape in ((3, 3), (4, 4, 3), (6, 5)):
            with self.subTest(shape=shape):
                image = np.zeros(shape, np.uint8)
                self.assertEqual(QualityEvaluator.evaluate(image, image)["SSIM"], 1)
        with self.assertRaises(ValueError):
            QualityEvaluator.evaluate(np.zeros((2, 3), np.uint8), np.zeros((2, 3), np.uint8))

    def test_invalid_inputs_are_not_resized_or_cast(self):
        for other in (None, np.zeros((8, 8, 3), np.uint8), np.zeros((16, 17), np.uint8),
                      self.image.astype(float), np.empty((0, 0), np.uint8)):
            with self.subTest(other=type(other)), self.assertRaises(ValueError):
                QualityEvaluator.evaluate(self.image, other)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = RestorationPipeline()
        self.image = np.random.default_rng(5).integers(0, 256, (24, 24, 3), dtype=np.uint8)
        self.mask = np.zeros((24, 24), np.uint8)
        self.mask[10:12, 10:15] = 255
        self.parameters = {"kernel_size": 7, "sigma": 2.3, "radius": 2, "method": "navier-stokes"}

    def test_combined_sequence_and_compare_parameters(self):
        expected = cv2.GaussianBlur(cv2.inpaint(self.image, self.mask, 2, cv2.INPAINT_NS), (7, 7), 2.3)
        result = self.pipeline.run_single("combined", self.image, mask=self.mask, **self.parameters)
        np.testing.assert_array_equal(result, expected)
        compared = self.pipeline.compare_all(self.image, self.image, self.mask, **self.parameters)
        for label, data in compared.items():
            self.assertEqual(data["status"], "success", label)
            self.assertEqual(data["parameters"], self.pipeline.parameters(**self.parameters))
            self.assertGreaterEqual(data["time"], 0)
            self.assertIsNotNone(data["baseline"])
        np.testing.assert_array_equal(list(compared.values())[2]["image"], expected)

    def test_without_reference_and_mask(self):
        results = list(self.pipeline.compare_all(None, self.image).values())
        self.assertEqual([r["status"] for r in results], ["success", "skipped", "skipped"])
        self.assertIsNone(results[0]["metrics"])
        self.assertIsNotNone(results[0]["image"])

    def test_bad_reference_does_not_discard_restoration(self):
        data = self.pipeline.restore_result("gaussian", self.image, np.zeros((8, 8, 3), np.uint8))
        self.assertIsNotNone(data["image"])
        self.assertIsNone(data["metrics"])
        self.assertTrue(data["metrics_error"])

    def test_bad_mask_and_method_rejected(self):
        for kwargs in ({"mask": None}, {"mask": np.zeros((8, 8), np.uint8)},
                       {"mask": self.mask, "method": "wrong"}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.pipeline.run_single("inpainting", self.image, **kwargs)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "thử nghiệm ảnh"
        self.root.mkdir()
        self.inputs, self.masks, self.clean = [self.root / name for name in ("input", "masks", "clean")]
        for folder in (self.inputs, self.masks, self.clean):
            folder.mkdir()
        self.image = np.random.default_rng(3).integers(0, 256, (20, 20, 3), dtype=np.uint8)
        self.mask = np.zeros((20, 20), np.uint8)
        self.mask[7:9, 7:9] = 255
        self.pipeline = RestorationPipeline()

    def add_image(self, name="ảnh.png", *, mask=True, clean=True):
        write_image(self.inputs / name, self.image)
        if mask:
            write_image(self.masks / name, self.mask)
        if clean:
            write_image(self.clean / name, self.image)

    def test_unicode_io_and_write_errors(self):
        self.add_image()
        np.testing.assert_array_equal(read_image(self.inputs / "ảnh.png"), self.image)
        with self.assertRaises(ValueError):
            write_image(self.root / "invalid.xyz", self.image)
        with self.assertRaises(OSError):
            write_image(self.root / "missing" / "file.png", self.image)

    def test_batch_results_metrics_csv_and_unique_runs(self):
        self.add_image()
        kwargs = dict(mask_dir=str(self.masks), clean_dir=str(self.clean), kernel_size=7, sigma=2)
        result = self.pipeline.run_batch(str(self.inputs), str(self.root / "output"), "combined", **kwargs)
        again = self.pipeline.run_batch(str(self.inputs), str(self.root / "output"), "combined", **kwargs)
        self.assertNotEqual(result.output_dir, again.output_dir)
        self.assertEqual((result.total, result.processed, result.failed, result.evaluated), (1, 1, 0, 1))
        row = result.rows[0]
        restored = read_image(row["output_path"])
        expected = self.pipeline.run_single("combined", self.image, mask=self.mask, kernel_size=7, sigma=2)
        np.testing.assert_array_equal(restored, expected)
        self.assertEqual(row["PSNR"], peak_signal_noise_ratio(self.image, restored, data_range=255))
        self.assertEqual(json.loads(row["parameters"])["kernel_size"], 7)
        with (Path(result.output_dir) / "details.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["SSIM"]), row["SSIM"])
        self.assertTrue((Path(result.output_dir) / "summary.csv").is_file())

    def test_missing_mask_broken_file_and_progress(self):
        self.add_image("a.png", mask=False)
        self.add_image("b.png")
        (self.inputs / "c.png").touch()
        progress, rows = [], []
        result = self.pipeline.run_batch(
            str(self.inputs), str(self.root / "output"), "inpainting", mask_dir=str(self.masks),
            progress_callback=lambda *args: progress.append(args), result_callback=rows.append)
        self.assertEqual((result.processed, result.skipped, result.failed), (1, 1, 1))
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["status"], "skipped")

    def test_missing_invalid_reference_is_warning(self):
        self.add_image("a.png", clean=False)
        self.add_image("b.png")
        write_image(self.clean / "b.png", np.zeros((5, 5, 3), np.uint8))
        result = self.pipeline.run_batch(str(self.inputs), str(self.root / "output"), "gaussian",
                                         clean_dir=str(self.clean))
        self.assertEqual((result.processed, result.evaluated), (2, 0))
        self.assertTrue(all(r["metrics_error"] for r in result.rows))
        self.assertEqual(summarize(result.rows)[0]["evaluated"], 0)

    def test_cancel_empty_and_reject_input_output_overlap(self):
        for name in ("a.png", "b.png"):
            self.add_image(name)
        rows = []
        result = self.pipeline.run_batch(str(self.inputs), str(self.root / "output"), "gaussian",
                                         result_callback=rows.append, should_stop=lambda: bool(rows))
        self.assertTrue(result.cancelled)
        self.assertEqual(len(result.rows), 1)
        self.assertTrue((Path(result.output_dir) / "summary.csv").is_file())
        with self.assertRaises(ValueError):
            self.pipeline.run_batch(str(self.inputs), str(self.inputs), "gaussian")
        empty = self.root / "empty"
        empty.mkdir()
        result = self.pipeline.run_batch(str(empty), str(self.root / "output"), "gaussian")
        self.assertEqual(result.total, 0)
        self.assertTrue((Path(result.output_dir) / "summary.csv").is_file())

    def test_write_failure_is_counted_and_continues(self):
        self.add_image("a.png")
        self.add_image("b.png")
        with patch("project_1.utils.batch_processor.write_image", side_effect=OSError("disk error")):
            result = self.pipeline.run_batch(str(self.inputs), str(self.root / "output"), "gaussian")
        self.assertEqual(result.failed, 2)
        self.assertTrue(all(not r["output_path"] for r in result.rows))

    def test_converted_filenames_cannot_overwrite_each_other(self):
        self.add_image("a.jpg", mask=False, clean=False)
        self.add_image("a.jpg.png", mask=False, clean=False)
        result = self.pipeline.run_batch(str(self.inputs), str(self.root / "output"), "gaussian")
        self.assertEqual(result.processed, 2)
        self.assertEqual(len({row["output_path"] for row in result.rows}), 2)
        for row in result.rows:
            self.assertTrue(Path(row["output_path"]).is_file())
        root = self.root / "generated"
        config = create_corrupted_dataset(str(self.inputs), str(root / "damaged"), str(root / "masks"),
                                          clean_dir=str(root / "clean"))
        self.assertEqual(len({f["generated"] for f in config["files"]}), 2)

    def test_default_batch_helper_uses_optional_mask(self):
        self.add_image()
        result = BatchProcessor.process_directory(str(self.inputs), str(self.root / "output"),
                                                    mask_dir=str(self.masks))
        self.assertEqual(result.processed, 1)
        expected = self.pipeline.run_single("combined", self.image, mask=self.mask)
        np.testing.assert_array_equal(read_image(result.rows[0]["output_path"]), expected)

    def test_seed_reproducibility_and_no_overwrite(self):
        self.add_image()
        configs = []
        for folder in ("gen1", "gen2"):
            root = self.root / folder
            config = create_corrupted_dataset(str(self.inputs), str(root / "damaged"), str(root / "mask"),
                                               clean_dir=str(root / "clean"))
            configs.append(config)
        name = configs[0]["files"][0]["generated"]
        np.testing.assert_array_equal(read_image(Path(configs[0]["corrupted_dir"]) / name),
                                      read_image(Path(configs[1]["corrupted_dir"]) / name))
        self.assertEqual(configs[0]["seed"], 42)
        with self.assertRaises(FileExistsError):
            create_corrupted_dataset(str(self.inputs), configs[0]["corrupted_dir"], configs[0]["mask_dir"],
                                     clean_dir=configs[0]["clean_dir"])
        with self.assertRaises(ValueError):
            create_corrupted_dataset(str(self.inputs), str(self.inputs), str(self.masks))
        np.testing.assert_array_equal(read_image(self.inputs / "ảnh.png"), self.image)


if __name__ == "__main__":
    unittest.main()
