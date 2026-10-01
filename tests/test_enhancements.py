"""Sharpening and experimental mask detection regressions, without GUI dependencies."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from project_1.algorithms.scratch_mask import (
    detect_scratch_mask, mask_overlay, mask_summary, validate_binary_mask,
)
from project_1.algorithms.unsharp_mask import UnsharpMaskAlgorithm
from project_1.entrypoint import RestorationPipeline
from project_1.metrics.mask_evaluator import binary_mask_scores
from project_1.utils.image_io import read_image, write_image


class SharpenTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(21).integers(0, 256, (32, 40, 3), dtype=np.uint8)
        self.mask = np.zeros(self.image.shape[:2], np.uint8)
        self.mask[12:14, 4:28] = 255
        self.pipeline = RestorationPipeline()
        self.sharpener = UnsharpMaskAlgorithm()

    def test_grayscale_formula_rounding_and_no_mutation(self):
        gray = self.image[:, :, 0].copy()
        original = gray.copy()
        values = gray.astype(np.float32)
        expected = np.rint(np.clip(values + 0.5 * (values - cv2.GaussianBlur(values, (0, 0), 1)), 0, 255)).astype(np.uint8)
        actual = self.sharpener.process(gray)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(gray, original)
        self.assertEqual(actual.dtype, np.uint8)

    def test_color_luminance_processing_and_no_mutation(self):
        original = self.image.copy()
        color = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
        color[:, :, 0] = self.sharpener.process(color[:, :, 0], amount=1, sigma=1.5)
        actual = self.sharpener.process(self.image, amount=1, sigma=1.5)
        np.testing.assert_array_equal(actual, cv2.cvtColor(color, cv2.COLOR_YCrCb2BGR))
        np.testing.assert_array_equal(self.image, original)
        self.assertEqual(actual.shape, original.shape)

    def test_disabled_and_zero_amount_are_exact_identity_after_restoration(self):
        for name in ("gaussian", "inpainting", "combined"):
            expected = self.pipeline.run_single(name, self.image, mask=self.mask)
            for kwargs in ({"sharpen_enabled": False, "sharpen_amount": 2},
                           {"sharpen_enabled": True, "sharpen_amount": 0}):
                actual = self.pipeline.run_single(name, self.image, mask=self.mask, **kwargs)
                np.testing.assert_array_equal(actual, expected)
        zero = self.sharpener.process(self.image, amount=0)
        np.testing.assert_array_equal(zero, self.image)
        self.assertFalse(np.shares_memory(zero, self.image))

    def test_all_methods_sharpen_once_after_base_restoration(self):
        for name in ("gaussian", "inpainting", "combined"):
            base = self.pipeline.run_single(name, self.image, mask=self.mask)
            with patch.object(self.pipeline.sharpener, "process", wraps=self.pipeline.sharpener.process) as process:
                actual = self.pipeline.run_single(name, self.image, mask=self.mask, sharpen_enabled=True)
                process.assert_called_once()
                np.testing.assert_array_equal(process.call_args.args[0], base)
            np.testing.assert_array_equal(actual, self.sharpener.process(base))

    def test_invalid_sharpening_values_and_images(self):
        for kwargs in ({"amount": -1}, {"amount": 2.1}, {"amount": float("nan")},
                       {"sigma": 0.2}, {"sigma": 3.1}, {"sigma": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sharpener.process(self.image, **kwargs)
        with self.assertRaises(ValueError):
            self.sharpener.process(self.image.astype(float))

    def test_comparison_metrics_measure_final_sharpened_image(self):
        results = self.pipeline.compare_all(self.image, self.image, self.mask, sharpen_enabled=True)
        for name, result in zip(("gaussian", "inpainting", "combined"), results.values()):
            expected = self.pipeline.run_single(name, self.image, mask=self.mask, sharpen_enabled=True)
            np.testing.assert_array_equal(result["image"], expected)
            self.assertEqual(result["metrics"], self.pipeline.evaluate(self.image, expected))
            self.assertTrue(result["parameters"]["sharpen_enabled"])

    def test_batch_saved_image_and_configuration_match_single(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("input", "mask", "clean"):
                (root / name).mkdir()
            write_image(root / "input/a.png", self.image)
            write_image(root / "clean/a.png", self.image)
            write_image(root / "mask/a.png", self.mask)
            result = self.pipeline.run_batch(str(root / "input"), str(root / "out"), "combined",
                mask_dir=str(root / "mask"), clean_dir=str(root / "clean"),
                sharpen_enabled=True, sharpen_amount=0.75, sharpen_sigma=1.5)
            self.assertEqual((result.processed, result.evaluated), (1, 1))
            row = result.rows[0]
            actual = read_image(row["output_path"])
            expected = self.pipeline.run_single("combined", self.image, mask=self.mask,
                sharpen_enabled=True, sharpen_amount=0.75, sharpen_sigma=1.5)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(row["PSNR"], self.pipeline.evaluate(self.image, actual)["PSNR"])
            self.assertEqual(json.loads(row["parameters"])["sharpen_amount"], 0.75)


class ScratchTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 96, 3), 80, np.uint8)
        self.truth = np.zeros((80, 96), np.uint8)
        cv2.line(self.image, (10, 15), (80, 60), (255, 255, 255), 1)
        cv2.line(self.truth, (10, 15), (80, 60), 255, 1)

    def test_bright_scratch_detection_shape_binary_and_unchanged_source(self):
        original = self.image.copy()
        mask = detect_scratch_mask(self.image)
        np.testing.assert_array_equal(mask, self.truth)
        np.testing.assert_array_equal(self.image, original)
        validate_binary_mask(mask, self.image.shape)
        self.assertEqual(binary_mask_scores(mask, self.truth), {"precision": 1, "recall": 1, "IoU": 1})

    def test_grayscale_and_expansion(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(detect_scratch_mask(gray), self.truth)
        expanded = detect_scratch_mask(gray, expand=1)
        expected = cv2.dilate(self.truth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        np.testing.assert_array_equal(expanded, expected)
        self.assertGreater(mask_summary(expanded)["pixels"], int(np.count_nonzero(self.truth)))

    def test_flat_bright_image_dark_scratches_and_small_specks_are_not_selected(self):
        for image in (np.full((80, 96), 255, np.uint8), np.full((80, 96), 80, np.uint8)):
            self.assertFalse(detect_scratch_mask(image).any())
        dark = np.full((80, 96), 150, np.uint8)
        cv2.line(dark, (10, 15), (80, 60), 0, 2)
        self.assertFalse(detect_scratch_mask(dark).any())
        specks = np.full((80, 96), 80, np.uint8)
        specks[::5, ::5] = 255
        self.assertFalse(detect_scratch_mask(specks).any())

    def test_invalid_parameters_and_mask_alignment(self):
        for kwargs in ({"kernel_size": 4}, {"kernel_size": 33}, {"expand": -1}, {"expand": 4},
                       {"brightness_threshold": 179}, {"response_threshold": 101},
                       {"response_threshold": float("nan")}, {"expand": 0.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detect_scratch_mask(self.image, **kwargs)
        for mask in (self.truth[:5], self.truth.astype(float), self.image, self.truth // 255):
            with self.assertRaises(ValueError):
                validate_binary_mask(mask, self.image.shape)

    def test_summary_and_overlay_only_change_mask_pixels(self):
        summary = mask_summary(self.truth)
        self.assertEqual(summary["regions"], 1)
        self.assertEqual(summary["pixels"], int((self.truth > 0).sum()))
        self.assertEqual(summary["coverage"], summary["pixels"] / self.truth.size)
        overlay = mask_overlay(self.image, self.truth)
        np.testing.assert_array_equal(overlay[self.truth == 0], self.image[self.truth == 0])
        self.assertTrue((overlay[self.truth > 0, 2] > overlay[self.truth > 0, 0]).all())
        self.assertEqual(mask_overlay(self.image[:, :, 0], self.truth).shape, self.image.shape)

    def test_empty_mask_metric_conventions_and_partial_overlap(self):
        empty = np.zeros(self.truth.shape, np.uint8)
        self.assertEqual(binary_mask_scores(empty, empty), {"precision": 1, "recall": 1, "IoU": 1})
        self.assertEqual(binary_mask_scores(empty, self.truth), {"precision": 0, "recall": 0, "IoU": 0})
        self.assertEqual(binary_mask_scores(self.truth, empty), {"precision": 0, "recall": 1, "IoU": 0})
        a, b = empty.copy(), empty.copy()
        a[0, :2], b[0, 1:3] = 255, 255
        self.assertEqual(binary_mask_scores(a, b), {"precision": 0.5, "recall": 0.5, "IoU": 1 / 3})


if __name__ == "__main__":
    unittest.main()
