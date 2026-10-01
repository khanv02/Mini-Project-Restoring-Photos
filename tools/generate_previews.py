"""Render current GUI workflows, verify outputs and refresh canonical preview files."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp
from time import monotonic
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea

from project_1.gui.main_window import MainWindow
from project_1.utils.image_io import list_images, read_image, write_image

PREVIEW_NAMES = ("single.png", "compare.png", "batch.png", "mask.png", "auto_mask_single.png")


def configure_font(app):
    """The Windows offscreen platform may not discover system fonts automatically."""
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for path in (windows_fonts / "segoeui.ttf", windows_fonts / "arial.ttf",
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(path))
            if font_id >= 0:
                app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))
                break
    metrics = QFontMetrics(app.font())
    if not all(metrics.inFontUcs4(ord(char)) for char in "Ảộưế012"):
        raise RuntimeError("Preview font cannot display Vietnamese. Install a compatible system font.")
    return app.font().family()


def wait_for_worker(owner, timeout=120):
    deadline = monotonic() + timeout
    while owner.thread is not None:
        if monotonic() >= deadline:
            owner.thread.stop()
            owner.thread.wait()
            QApplication.processEvents()
            raise TimeoutError("GUI worker timed out")
        QTest.qWait(20)
    QApplication.processEvents()


def capture(widget, path):
    QTest.qWait(120)  # Settle layouts and queued worker/paint events, without blocking Qt.
    if not widget.grab().save(str(path), "PNG"):
        raise OSError(f"Cannot save preview: {path}")
    image = QImage(str(path))
    if image.isNull() or image.size() != widget.size():
        raise AssertionError(f"Invalid preview dimensions: {path}")
    return {"width": image.width(), "height": image.height()}


def publish_previews(stage, output, names):
    """Keep the previous canonical files before replacing only the explicit targets."""
    for name in names:
        target = output / name
        if target.exists() and not target.is_file():
            raise ValueError(f"Preview target is not a file: {target}")
    existing = [name for name in names if (output / name).is_file()]
    if existing:
        previous = stage / "previous"
        previous.mkdir()
        for name in existing:
            shutil.copy2(output / name, previous / name)
    for name in names:
        shutil.copy2(stage / name, output / name)
    return existing


def generate(dataset, output, batch_limit=8):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    files = list_images(dataset / "images_corrupted")
    if not files or batch_limit < 1:
        raise ValueError("Need image triples and a positive batch limit")
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix="refresh_", dir=output))
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    font = configure_font(app)
    window = MainWindow(dataset_dir=dataset)
    window.resize(1440, 940)
    window.show()
    warnings = []
    # Report unexpected errors instead of hanging on a modal warning offscreen.
    from unittest.mock import patch
    with patch.object(QMessageBox, "warning", side_effect=lambda *args: warnings.append(args[2])):
        try:
            window.load_demo()
            wait_for_worker(window)
            if len(window.comparison_results) != 3 or any(
                result["status"] != "success" for result in window.comparison_results.values()
            ):
                raise AssertionError("Demo comparison failed")
            window.controls.sharpen_enabled.setChecked(True)
            window.tabs.setCurrentIndex(0)
            window._run_single_restoration()
            wait_for_worker(window)
            if window.single_result is None or window.single_result["metrics"] is None:
                raise AssertionError("Single restoration failed")
            write_image(stage / "single_restored.png", window.restored_img)
            np.testing.assert_array_equal(read_image(stage / "single_restored.png"), window.restored_img)
            screenshots = {"single.png": capture(window, stage / "single.png")}
            single_metrics = window.single_result["metrics"].copy()
            parameters = window.controls.parameters()

            window.tabs.setCurrentIndex(1)
            window._run_comparison()
            wait_for_worker(window)
            if any(result["status"] != "success" for result in window.comparison_results.values()):
                raise AssertionError("Sharpened comparison failed")
            screenshots["compare.png"] = capture(window, stage / "compare.png")

            # A small, copied dataset exercises the actual Batch GUI without altering sources.
            batch = stage / "batch_sample"
            for folder in ("images_corrupted", "images_clean", "images_masks"):
                (batch / folder).mkdir(parents=True)
                for path in files[:batch_limit]:
                    shutil.copy2(dataset / folder / path.name, batch / folder / path.name)
            window.batch_input.setText(str(batch / "images_corrupted"))
            window.batch_clean.setText(str(batch / "images_clean"))
            window.batch_masks.setText(str(batch / "images_masks"))
            window.batch_output.setText(str(stage / "batch_results"))
            window.tabs.setCurrentIndex(2)
            window._start_batch()
            wait_for_worker(window)
            if len(window.batch_rows) != min(batch_limit, len(files)) or any(
                row["status"] != "success" or row["PSNR"] == "" for row in window.batch_rows
            ):
                raise AssertionError("Batch preview did not complete successfully")
            if window.progress_bar.value() != 100 or window.batch_after.image is None:
                raise AssertionError("Batch progress/selected-image preview missing")
            screenshots["batch.png"] = capture(window, stage / "batch.png")

            # Use the real modal review flow, including clicking the explicit Apply button.
            window.tabs.setCurrentIndex(0)
            review = {"started": False, "error": None}
            def review_mask():
                dialog = window.mask_dialog
                try:
                    if dialog is None:
                        raise AssertionError("Mask dialog did not open")
                    if not review["started"]:
                        review["started"] = True
                        dialog.generate()
                    if dialog.thread is not None:
                        QTimer.singleShot(20, review_mask)
                        return
                    if not dialog.apply_button.isEnabled():
                        raise AssertionError("No usable mask suggestion")
                    screenshots["mask.png"] = capture(dialog, stage / "mask.png")
                    review["parameters"] = dialog.candidate_parameters.copy()
                    write_image(stage / "accepted_mask.png", dialog.candidate)
                    dialog.apply_button.click()
                except Exception as exc:
                    review["error"] = exc
                    if dialog is not None:
                        dialog.reject()
            QTimer.singleShot(0, review_mask)
            window._suggest_mask()
            if review["error"] is not None:
                raise review["error"]
            if window.mask_source != "tự động — đã xác nhận":
                raise AssertionError("Mask was not explicitly accepted")
            window._run_single_restoration()
            wait_for_worker(window)
            if window.single_result is None:
                raise AssertionError("Restoration with accepted mask failed")
            screenshots["auto_mask_single.png"] = capture(window, stage / "auto_mask_single.png")
            scroll = window.findChild(QScrollArea)
            if scroll.horizontalScrollBar().maximum() != 0:
                raise AssertionError("Sidebar controls are horizontally clipped")
            if warnings:
                raise AssertionError(f"Unexpected GUI errors: {warnings}")
            manifest = {"dataset": str(dataset), "run_directory": str(stage), "font": font,
                        "sample": files[0].name, "parameters": parameters,
                        "single_manual_mask_metrics": single_metrics,
                        "single_auto_mask_metrics": window.single_result["metrics"],
                        "mask_parameters": review["parameters"], "batch_processed": len(window.batch_rows),
                        "screenshots": screenshots}
            with (stage / "preview_manifest.json").open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
            with ZipFile(stage / "previews.zip", "w", ZIP_DEFLATED) as archive:
                for name in (*PREVIEW_NAMES, "preview_manifest.json"):
                    archive.write(stage / name, arcname=name)
            previous = publish_previews(stage, output, (*PREVIEW_NAMES, "preview_manifest.json", "previews.zip"))
            print(f"Updated {len(PREVIEW_NAMES)} screenshots: {output}", flush=True)
            print(f"GUI Batch completed: {len(window.batch_rows)} images; no warnings", flush=True)
            if previous:
                print(f"Previous previews preserved: {stage / 'previous'}", flush=True)
            return manifest
        finally:
            if window.thread is not None:
                window.thread.stop()
                window.thread.wait()
                app.processEvents()
            window.close()
            window.deleteLater()
            # There is no persistent app.exec() loop here to deliver deferred deletes.
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("output/previews"))
    parser.add_argument("--batch-limit", type=int, default=8)
    args = parser.parse_args()
    generate(args.dataset, args.output, args.batch_limit)


if __name__ == "__main__":
    main()
