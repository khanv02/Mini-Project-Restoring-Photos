"""Behavioral regression checks; run with python -m unittest discover -s tests -v."""

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from project_1.entrypoint import RestorationPipeline
from project_1.generate_data import add_gaussian_noise, add_scratches_and_stains_with_mask
from project_1.metrics.evaluator import QualityEvaluator
from project_1.utils.batch_processor import combined_restore
from project_1.utils.image_io import list_images, read_image, write_image
from project_1.utils.reports import metric_delta, report_row, summarize, write_csv


class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = RestorationPipeline()
        self.image = np.random.default_rng(17).integers(0, 256, (32, 40, 3), dtype=np.uint8)
        self.mask = np.zeros((32, 40), dtype=np.uint8)
        self.mask[10:13, 5:20] = 255

    def test_gaussian_matches_opencv_and_normalizes_even_kernel(self):
        actual = self.pipeline.run_single("gaussian", self.image, kernel_size=4, sigma=1.2)
        expected = cv2.GaussianBlur(self.image, (5, 5), sigmaX=1.2)
        np.testing.assert_array_equal(actual, expected)

    def test_inpainting_both_methods_preserve_unmasked_pixels(self):
        for method, flag in (("Telea", cv2.INPAINT_TELEA), ("navier-stokes", cv2.INPAINT_NS)):
            actual = self.pipeline.run_single("inpainting", self.image, mask=self.mask, method=method)
            np.testing.assert_array_equal(actual, cv2.inpaint(self.image, self.mask, 3, flag))
            np.testing.assert_array_equal(actual[self.mask == 0], self.image[self.mask == 0])

    def test_color_mask_is_converted_and_misaligned_mask_is_rejected(self):
        small = np.zeros((16, 20, 3), dtype=np.uint8)
        small[3:5, 3:5] = 255
        with self.assertRaises(ValueError):
            self.pipeline.run_single("inpainting", self.image, mask=small)
        aligned = cv2.resize(small, (40, 32), interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        actual = self.pipeline.run_single("inpainting", self.image, mask=aligned)
        np.testing.assert_array_equal(actual, cv2.inpaint(self.image, mask, 3, cv2.INPAINT_TELEA))

    def test_combined_is_inpaint_then_gaussian_without_mutating_inputs(self):
        original, original_mask = self.image.copy(), self.mask.copy()
        expected = cv2.GaussianBlur(cv2.inpaint(self.image, self.mask, 3, cv2.INPAINT_TELEA), (5, 5), 1.2)
        actual = self.pipeline.run_single("combined", self.image, mask=self.mask)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(self.image, original)
        np.testing.assert_array_equal(self.mask, original_mask)
        np.testing.assert_array_equal(combined_restore(self.image, self.mask), expected)

    def test_invalid_algorithm_mask_image_and_parameters_are_rejected(self):
        cases = [
            ("missing", self.image, {}),
            ("combined", self.image, {}),
            ("inpainting", self.image, {"mask": self.mask, "method": "unknown"}),
            ("gaussian", None, {}),
            ("gaussian", self.image.astype(float), {}),
            ("gaussian", self.image, {"sigma": -1}),
            ("gaussian", self.image, {"sigma": float("nan")}),
        ]
        for algorithm, image, kwargs in cases:
            with self.subTest(algorithm=algorithm, kwargs=tuple(kwargs)):
                with self.assertRaises(ValueError):
                    self.pipeline.run_single(algorithm, image, **kwargs)

    def test_comparison_uses_supplied_parameters_and_baseline(self):
        results = self.pipeline.compare_all(self.image, self.image, self.mask, kernel_size=9, sigma=2)
        self.assertEqual(len(results), 3)
        expected = self.pipeline.run_single("gaussian", self.image, kernel_size=9, sigma=2)
        np.testing.assert_array_equal(results["Gaussian Filter"]["image"], expected)
        for result in results.values():
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["parameters"]["kernel_size"], 9)
            self.assertEqual(result["baseline"]["SSIM"], 1.0)
            self.assertGreaterEqual(result["time"], 0)

    def test_missing_mask_is_reported_as_skipped(self):
        results = self.pipeline.compare_all(self.image, self.image)
        self.assertEqual(results["Gaussian Filter"]["status"], "success")
        self.assertEqual(results["Inpainting"]["status"], "skipped")
        self.assertEqual(results["Combined (Inpainting + Gaussian)"]["status"], "skipped")

    def test_identical_color_and_small_grayscale_metrics(self):
        for image in (self.image, self.image[:3, :4, 0]):
            self.assertEqual(QualityEvaluator.evaluate(image, image), {"PSNR": float("inf"), "SSIM": 1.0})

    def test_metrics_reject_misaligned_shapes_channels_and_tiny_images(self):
        for other in (self.image[:20], self.image[:, :, 0]):
            with self.assertRaises(ValueError):
                QualityEvaluator.evaluate(self.image, other)
        with self.assertRaises(ValueError):
            QualityEvaluator.evaluate(np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8))

    def test_damage_generation_is_seeded_and_mask_is_binary(self):
        outputs = []
        for _ in range(2):
            rng = np.random.default_rng(42)
            damaged, mask = add_scratches_and_stains_with_mask(self.image, rng=rng)
            outputs.append((add_gaussian_noise(damaged, rng=rng), mask))
        for first, second in zip(outputs[0], outputs[1]):
            np.testing.assert_array_equal(first, second)
        self.assertTrue(set(np.unique(outputs[0][1])).issubset({0, 255}))
        with self.assertRaises(ValueError):
            add_gaussian_noise(self.image, var=-1)


class BatchAndIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "ảnh đầu vào"
        self.masks = self.root / "mặt nạ"
        self.clean = self.root / "ảnh gốc"
        for directory in (self.input, self.masks, self.clean):
            directory.mkdir()
        self.output = self.root / "kết quả"
        self.pipeline = RestorationPipeline()
        self.image = np.full((20, 24, 3), 120, np.uint8)
        self.mask = np.zeros((20, 24), np.uint8)

    def add_image(self, name, *, mask=True, clean=True):
        write_image(self.input / name, self.image)
        if mask:
            write_image(self.masks / name, self.mask)
        if clean:
            write_image(self.clean / name, self.image)

    def test_unicode_roundtrip_and_case_insensitive_sorted_files(self):
        self.add_image("ảnh B.PNG")
        self.add_image("ảnh A.png")
        np.testing.assert_array_equal(read_image(self.input / "ảnh B.PNG"), self.image)
        self.assertEqual([p.name for p in list_images(self.input)], ["ảnh A.png", "ảnh B.PNG"])
        np.testing.assert_array_equal(read_image(self.masks / "ảnh A.png", grayscale=True), self.mask)

    def test_unreadable_images_and_unsupported_output_formats_raise(self):
        with self.assertRaises(OSError):
            read_image(self.input / "missing.png")
        with self.assertRaises(ValueError):
            write_image(self.output / "file.txt", self.image)

    def test_batch_continues_after_bad_file_and_missing_mask_with_full_progress(self):
        self.add_image("a.png")
        self.add_image("b.png", mask=False)
        # A valid JPEG payload under a PNG name is readable by OpenCV; use an empty ndarray
        # to create an empty encoded input without relying on any image decoder.
        np.array([], dtype=np.uint8).tofile(self.input / "c.png")
        progress, errors = [], []
        result = self.pipeline.run_batch(
            str(self.input), str(self.output), "combined", mask_dir=str(self.masks),
            clean_dir=str(self.clean), progress_callback=lambda a, b: progress.append((a, b)),
            error_callback=lambda name, error: errors.append(name),
        )
        self.assertEqual((result.total, result.processed, result.skipped, result.failed), (3, 1, 1, 1))
        self.assertEqual(result.evaluated, 1)
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(errors, ["b.png", "c.png"])
        run_dir = Path(result.output_dir)
        self.assertTrue((run_dir / "restored_a.png").is_file())
        with (run_dir / "details.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["status"] for r in rows], ["success", "skipped", "failed"])
        self.assertTrue((run_dir / "summary.csv").is_file())

    def test_missing_clean_reference_does_not_discard_restored_image(self):
        self.add_image("a.png", clean=False)
        result = self.pipeline.run_batch(str(self.input), str(self.output), "gaussian", clean_dir=str(self.clean))
        self.assertEqual((result.processed, result.evaluated), (1, 0))
        self.assertTrue(result.rows[0]["metrics_error"])
        self.assertTrue(Path(result.rows[0]["output_path"]).is_file())

    def test_cancellation_keeps_completed_results_and_reports(self):
        for name in ("a.png", "b.png", "c.png"):
            self.add_image(name)
        progress = []
        result = self.pipeline.run_batch(
            str(self.input), str(self.output), "gaussian",
            progress_callback=lambda a, b: progress.append(a),
            should_stop=lambda: bool(progress),
        )
        self.assertTrue(result.cancelled)
        self.assertEqual((result.processed, result.total), (1, 3))
        self.assertTrue((Path(result.output_dir) / "details.csv").is_file())

    def test_input_output_alias_and_required_mask_are_rejected(self):
        self.add_image("a.png")
        with self.assertRaises(ValueError):
            self.pipeline.run_batch(str(self.input), str(self.input), "gaussian")
        with self.assertRaises(ValueError):
            self.pipeline.run_batch(str(self.input), str(self.output), "combined")

    def test_empty_batch_and_repeated_runs_have_separate_output(self):
        first = self.pipeline.run_batch(str(self.input), str(self.output), "gaussian")
        second = self.pipeline.run_batch(str(self.input), str(self.output), "gaussian")
        self.assertEqual(first.total, 0)
        self.assertNotEqual(first.output_dir, second.output_dir)
        self.assertTrue((Path(first.output_dir) / "summary.csv").is_file())

    def test_report_roundtrip_handles_infinite_metrics(self):
        identical = {"PSNR": float("inf"), "SSIM": 1.0}
        delta = metric_delta(identical, identical)
        self.assertEqual(delta["PSNR"], 0)
        row = report_row("ảnh.png", "gaussian", {
            "baseline": identical, "metrics": identical, "delta": delta,
            "parameters": {"sigma": 1.2}, "status": "success",
        })
        write_csv(self.root / "report.csv", [row])
        self.assertEqual(summarize([row])[0]["mean_PSNR"], float("inf"))
        with (self.root / "report.csv").open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))[0]
        self.assertEqual(saved["PSNR"], "inf")
        self.assertEqual(saved["filename"], "ảnh.png")


if __name__ == "__main__":
    unittest.main()
