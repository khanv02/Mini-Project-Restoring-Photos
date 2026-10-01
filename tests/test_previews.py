"""Preview publication safety and end-to-end rendering on a tiny image dataset."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from project_1.utils.image_io import read_image, write_image
from tools.generate_previews import PREVIEW_NAMES, generate, publish_previews


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_publishing_preserves_previous_and_unrelated_files(self):
        output, stage = self.root / "output", self.root / "stage"
        output.mkdir()
        stage.mkdir()
        old = np.full((10, 12), 30, np.uint8)
        new = np.full((10, 12), 200, np.uint8)
        write_image(output / "single.png", old)
        write_image(output / "unrelated.png", old)
        write_image(stage / "single.png", new)
        previous = publish_previews(stage, output, ("single.png",))
        self.assertEqual(previous, ["single.png"])
        np.testing.assert_array_equal(read_image(output / "single.png", grayscale=True), new)
        np.testing.assert_array_equal(read_image(stage / "previous/single.png", grayscale=True), old)
        np.testing.assert_array_equal(read_image(output / "unrelated.png", grayscale=True), old)

    def test_directory_target_is_rejected_before_overwriting(self):
        output, stage = self.root / "output", self.root / "stage"
        output.mkdir()
        stage.mkdir()
        (output / "single.png").mkdir()
        write_image(stage / "single.png", np.zeros((10, 12), np.uint8))
        with self.assertRaises(ValueError):
            publish_previews(stage, output, ("single.png",))
        self.assertTrue((output / "single.png").is_dir())
        self.assertFalse((stage / "previous").exists())

    def make_dataset(self):
        dataset = self.root / "dataset"
        for folder in ("images_clean", "images_corrupted", "images_masks"):
            (dataset / folder).mkdir(parents=True)
        clean = np.full((80, 96, 3), 80, np.uint8)
        damaged = clean.copy()
        mask = np.zeros(clean.shape[:2], np.uint8)
        cv2.line(damaged, (10, 15), (80, 60), (255, 255, 255), 1)
        cv2.line(mask, (10, 15), (80, 60), 255, 1)
        for folder, image in (("images_clean", clean), ("images_corrupted", damaged), ("images_masks", mask)):
            write_image(dataset / folder / "a.png", image)
        return dataset

    def test_invalid_batch_limit_does_not_create_previews(self):
        dataset = self.make_dataset()
        output = self.root / "previews"
        with self.assertRaises(ValueError):
            generate(dataset, output, batch_limit=0)
        self.assertFalse(output.exists())

    def test_all_workflows_render_and_zip_matches_canonical_files(self):
        dataset = self.make_dataset()
        before = {path: path.read_bytes() for path in dataset.rglob("*.png")}
        output = self.root / "previews"
        manifest = generate(dataset, output, batch_limit=1)
        self.assertEqual(manifest["batch_processed"], 1)
        self.assertTrue(manifest["parameters"]["sharpen_enabled"])
        self.assertEqual(set(manifest["screenshots"]), set(PREVIEW_NAMES))
        with ZipFile(output / "previews.zip") as archive:
            self.assertEqual(set(archive.namelist()), {*PREVIEW_NAMES, "preview_manifest.json"})
            for name in PREVIEW_NAMES:
                self.assertEqual(archive.read(name), (output / name).read_bytes())
                self.assertGreater(read_image(output / name).size, 0)
        metadata = json.loads((output / "preview_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata, manifest)
        self.assertEqual(manifest["screenshots"]["mask.png"], {"width": 1000, "height": 730})
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
