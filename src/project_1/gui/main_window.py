"""Desktop restoration workflows using one shared configuration and pipeline."""

from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
    QDialog, QScrollArea,
)

from project_1.entrypoint import RestorationPipeline
from project_1.gui.threads import BatchWorkerThread, TaskWorkerThread
from project_1.gui.widgets import AlgorithmControls, ImagePanel, cv_to_qpixmap
from project_1.gui.mask_dialog import MaskDialog
from project_1.algorithms.scratch_mask import validate_binary_mask
from project_1.settings import ALGORITHMS
from project_1.utils.image_io import list_images, read_image, write_image
from project_1.utils.reports import report_row, write_csv

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp)"
TABLE_HEADERS = ["Ảnh / phương pháp", "Trạng thái", "PSNR trước", "SSIM trước",
                 "PSNR sau", "SSIM sau", "Δ PSNR", "Δ SSIM", "Giây", "Ghi chú"]


def number(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


class MainWindow(QMainWindow):
    def __init__(self, dataset_dir: str | Path | None = None):
        super().__init__()
        self.setWindowTitle("Khôi phục ảnh cũ — Subject 2 / Project 1")
        self.resize(1320, 850)
        self.pipeline = RestorationPipeline()
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else Path.cwd() / "dataset"
        self.clean_img = self.corrupted_img = self.mask_img = self.restored_img = None
        self.input_path = ""
        self.single_result = None
        self.comparison_results = {}
        self.batch_rows = []
        self.thread = None
        self.mask_dialog = None
        self.mask_source = ""
        self.mask_parameters = None
        self._lockable = []
        self._init_ui()

    def _button(self, text, handler, layout, *, lock=True):
        button = QPushButton(text)
        button.clicked.connect(handler)
        layout.addWidget(button)
        if lock:
            self._lockable.append(button)
        return button

    def _init_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        sidebar = QWidget()
        left = QVBoxLayout(sidebar)
        self._button("1. Mở ảnh hỏng", self._load_corrupted, left)
        self._button("2. Mở ảnh tham chiếu (tùy chọn)", self._load_clean, left)
        self._button("3. Mở mask", self._load_mask, left)
        self.auto_mask_button = self._button("Gợi ý mask xước sáng…", self._suggest_mask, left)
        self.save_mask_button = self._button("Lưu mask đã xác nhận (PNG)", self._save_mask, left)
        self._button("Bỏ ảnh tham chiếu / mask", self._clear_references, left)
        self._button("Nạp bộ ảnh mẫu và so sánh", self.load_demo, left)
        self.input_label = QLabel("Chưa nạp ảnh. Hãy chọn ảnh hỏng trước.")
        self.input_label.setWordWrap(True)
        left.addWidget(self.input_label)
        self.controls = AlgorithmControls("Thuật toán & tham số chung")
        left.addWidget(self.controls)
        self._lockable.append(self.controls)
        hint = QLabel("Ảnh tham chiếu phải tương ứng với ảnh hỏng và cùng kích thước. "
                      "Không có tham chiếu: chỉ đánh giá trực quan. Batch ghép file theo cùng tên.")
        hint.setWordWrap(True)
        left.addWidget(hint)
        left.addStretch()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._single_tab(), "Khôi phục đơn")
        self.tabs.addTab(self._comparison_tab(), "So sánh phương pháp")
        self.tabs.addTab(self._batch_tab(), "Xử lý hàng loạt")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.controls.algorithm.currentIndexChanged.connect(self._tab_changed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(335)
        scroll.setMaximumWidth(365)
        scroll.setWidget(sidebar)
        layout.addWidget(scroll)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Sẵn sàng")
        self._update_mask_actions()

    def _single_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        actions = QHBoxLayout()
        self.run_button = self._button("Khôi phục ảnh", self._run_single_restoration, actions)
        self._button("Lưu ảnh kết quả", self._save_restored, actions)
        layout.addLayout(actions)
        views = QHBoxLayout()
        self.before_panel = ImagePanel("Trước: ảnh hỏng")
        self.after_panel = ImagePanel("Sau: ảnh khôi phục")
        views.addWidget(self.before_panel)
        views.addWidget(self.after_panel)
        layout.addLayout(views, 1)
        self.lbl_metrics = QLabel("Không có kết quả")
        self.lbl_metrics.setWordWrap(True)
        layout.addWidget(self.lbl_metrics)
        return widget

    @staticmethod
    def _table():
        table = QTableWidget(0, len(TABLE_HEADERS))
        table.setHorizontalHeaderLabels(TABLE_HEADERS)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _comparison_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        actions = QHBoxLayout()
        self._button("So sánh cả 3 phương pháp", self._run_comparison, actions)
        self._button("Lưu ảnh phương pháp được chọn", self._save_comparison, actions)
        self._button("Xuất CSV", self._export_comparison, actions)
        layout.addLayout(actions)
        grid = QGridLayout()
        for index in (0, 1):
            grid.setRowStretch(index, 1)
            grid.setColumnStretch(index, 1)
        self.compare_before = ImagePanel("Ảnh hỏng")
        grid.addWidget(self.compare_before, 0, 0)
        self.compare_panels = {}
        for index, spec in enumerate(ALGORITHMS, 1):
            panel = ImagePanel(spec.label)
            self.compare_panels[spec.label] = panel
            grid.addWidget(panel, index // 2, index % 2)
        layout.addLayout(grid, 1)
        self.table_comp = self._table()
        self.table_comp.setMaximumHeight(180)
        layout.addWidget(self.table_comp)
        return widget

    def _directory_control(self, title, layout, default=""):
        row = QHBoxLayout()
        entry = QLineEdit(default)
        entry.setPlaceholderText(title)
        button = QPushButton(title)
        button.clicked.connect(lambda: self._select_directory(entry, title))
        row.addWidget(entry, 1)
        row.addWidget(button)
        self._lockable.extend((entry, button))
        layout.addLayout(row)
        return entry

    def _batch_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.batch_input = self._directory_control("Thư mục ảnh hỏng", layout)
        self.batch_masks = self._directory_control("Mask (Inpainting / Combined)", layout)
        self.batch_clean = self._directory_control("Clean (tùy chọn)", layout)
        self.batch_output = self._directory_control("Thư mục lưu kết quả", layout, "output")
        for entry, name in ((self.batch_input, "images_corrupted"),
                            (self.batch_masks, "images_masks"), (self.batch_clean, "images_clean")):
            directory = self.dataset_dir / name
            if directory.is_dir():
                entry.setText(str(directory))
        actions = QHBoxLayout()
        self._button("Bắt đầu Batch", self._start_batch, actions)
        self.cancel_button = self._button("Hủy Batch", self._cancel_batch, actions, lock=False)
        self.cancel_button.setEnabled(False)
        layout.addLayout(actions)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.lbl_batch_status = QLabel("Sẵn sàng")
        self.lbl_batch_status.setWordWrap(True)
        layout.addWidget(self.lbl_batch_status)
        self.table_batch = self._table()
        self.table_batch.itemSelectionChanged.connect(self._show_batch_preview)
        layout.addWidget(self.table_batch, 1)
        views = QHBoxLayout()
        self.batch_before = ImagePanel("Batch: trước")
        self.batch_after = ImagePanel("Batch: sau")
        views.addWidget(self.batch_before)
        views.addWidget(self.batch_after)
        layout.addLayout(views, 1)
        return widget

    def _tab_changed(self, *_):
        self.controls._update_visibility()
        if self.tabs.currentIndex() == 1:
            for control in (self.controls.kernel, self.controls.sigma,
                            self.controls.radius, self.controls.method):
                self.controls.form.setRowVisible(control, True)
            self.controls.hint.setText("Compare chạy mọi phương pháp đủ điều kiện với các tham số này.")
        self._update_mask_actions()

    def _update_mask_actions(self):
        allowed = not self._busy() and self.tabs.currentIndex() != 2
        self.auto_mask_button.setEnabled(allowed and self.corrupted_img is not None)
        self.save_mask_button.setEnabled(allowed and self.mask_img is not None)

    def _select_directory(self, entry, title):
        directory = QFileDialog.getExistingDirectory(self, title, entry.text())
        if directory:
            entry.setText(directory)

    def _clear_results(self):
        self.restored_img = self.single_result = None
        self.comparison_results = {}
        self.after_panel.set_image(None)
        self.lbl_metrics.setText("Không có kết quả")
        self.table_comp.setRowCount(0)
        for panel in self.compare_panels.values():
            panel.set_image(None)

    def _update_input_label(self):
        self.input_label.setText(
            f"Ảnh hỏng: {Path(self.input_path).name or 'chưa nạp'}\n"
            f"Tham chiếu: {'đã nạp' if self.clean_img is not None else 'không có'}\n"
            f"Mask: {self.mask_source or ('đã nạp' if self.mask_img is not None else 'không có')}")
        self._update_mask_actions()

    def load_input(self, path):
        """Load a new input transactionally, then invalidate references and results."""
        image = read_image(path)
        self.corrupted_img = image
        self.input_path = str(path)
        self.clean_img = self.mask_img = None
        self.mask_source, self.mask_parameters = "", None
        self._clear_results()
        self.before_panel.set_image(image, str(path))
        self.compare_before.set_image(image, str(path))
        self._update_input_label()

    def _load(self, kind):
        if kind != "input" and self.corrupted_img is None:
            self._error("Hãy nạp ảnh hỏng trước, sau đó nạp tham chiếu / mask tương ứng.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", IMAGE_FILTER)
        if not path:
            return
        try:
            if kind == "input":
                self.load_input(path)
            else:
                image = read_image(path, grayscale=kind == "mask")
                if image.shape[:2] != self.corrupted_img.shape[:2]:
                    raise ValueError("Ảnh tham chiếu / mask phải cùng kích thước với ảnh hỏng.")
                if kind == "clean":
                    self.clean_img = image
                else:
                    self.mask_img = image
                    self.mask_source, self.mask_parameters = "nạp từ file", None
                self._clear_results()
                self._update_input_label()
        except Exception as exc:
            self._error(str(exc))

    def _load_corrupted(self):
        self._load("input")

    def load_demo(self):
        if self._busy():
            return
        try:
            images = list_images(self.dataset_dir / "images_corrupted")
            if not images:
                raise ValueError("Dataset chưa có ảnh hỏng để chạy mẫu.")
            path = images[0]
            self.load_input(path)
            clean = self.dataset_dir / "images_clean" / path.name
            mask = self.dataset_dir / "images_masks" / path.name
            self.clean_img = read_image(clean) if clean.is_file() else None
            self.mask_img = read_image(mask, grayscale=True) if mask.is_file() else None
            self.mask_source = "mask mẫu" if self.mask_img is not None else ""
            self._update_input_label()
            self.tabs.setCurrentIndex(1)
            self.controls.algorithm.setCurrentIndex(2)
            self._run_comparison()
        except Exception as exc:
            self._error(str(exc))

    def _load_clean(self):
        self._load("clean")

    def _load_mask(self):
        self._load("mask")

    def apply_auto_mask(self, mask, parameters):
        if self.corrupted_img is None:
            raise ValueError("Hãy nạp ảnh hỏng trước.")
        validate_binary_mask(mask, self.corrupted_img.shape)
        if not mask.any():
            raise ValueError("Mask rỗng không thể áp dụng.")
        self.mask_img = mask.copy()
        self.mask_source, self.mask_parameters = "tự động — đã xác nhận", parameters.copy()
        self._clear_results()
        self._update_input_label()

    def _suggest_mask(self):
        if self._busy() or self.tabs.currentIndex() == 2:
            return
        if self.corrupted_img is None:
            self._error("Hãy nạp ảnh hỏng trước.")
            return
        dialog = MaskDialog(self.corrupted_img, self)
        self.mask_dialog = dialog
        self._set_busy(True)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.apply_auto_mask(dialog.candidate, dialog.candidate_parameters)
        finally:
            self.mask_dialog = None
            dialog.deleteLater()
            self._set_busy(False)

    def _save_mask(self):
        if self.mask_img is None or self._busy():
            return
        name = Path(self.input_path).stem + ".png"
        path, _ = QFileDialog.getSaveFileName(self, "Lưu mask PNG", name, "PNG (*.png)")
        if path:
            try:
                if not Path(path).suffix:
                    path += ".png"
                if Path(path).suffix.lower() != ".png":
                    raise ValueError("Mask phải lưu dưới dạng PNG để giữ nguyên pixel.")
                write_image(path, self.mask_img)
            except Exception as exc:
                self._error(str(exc))
            else:
                self.statusBar().showMessage("Đã lưu mask: " + path)

    def _clear_references(self):
        self.clean_img = self.mask_img = None
        self.mask_source, self.mask_parameters = "", None
        self._clear_results()
        self._update_input_label()

    def _busy(self):
        return self.thread is not None or self.mask_dialog is not None

    def _set_busy(self, busy):
        for control in self._lockable:
            control.setEnabled(not busy)
        self.cancel_button.setEnabled(busy and isinstance(self.thread, BatchWorkerThread))
        self._update_mask_actions()

    def _start_worker(self, worker, callback):
        if self._busy():
            return
        self.thread = worker
        worker.completed.connect(callback)
        worker.failed.connect(self._worker_failed)
        worker.finished.connect(self._worker_finished)
        self._set_busy(True)
        self.statusBar().showMessage("Đang xử lý…")
        worker.start()

    def _worker_failed(self, message):
        if isinstance(self.thread, BatchWorkerThread):
            self.lbl_batch_status.setText("Batch thất bại: " + message)
        self._error(message)

    def _worker_finished(self):
        worker = self.thread
        self.thread = None
        self._set_busy(False)
        self.statusBar().showMessage("Sẵn sàng")
        if worker is not None:
            worker.deleteLater()

    def _run_single_restoration(self):
        if self._busy():
            return
        if self.corrupted_img is None:
            self._error("Vui lòng mở ảnh hỏng trước.")
            return
        if self.controls.needs_mask and self.mask_img is None:
            self._error("Phương pháp này cần mask.")
            return
        key, parameters = self.controls.algorithm_key, self.controls.parameters()
        image, clean, mask = self.corrupted_img, self.clean_img, self.mask_img
        self.restored_img = self.single_result = None
        self.after_panel.set_image(None)
        self.lbl_metrics.setText("Đang khôi phục…")
        worker = TaskWorkerThread(
            lambda: self.pipeline.restore_result(key, image, clean, mask=mask, **parameters), self)
        self._start_worker(worker, self._single_completed)

    def _single_completed(self, data):
        self.single_result = data
        self.restored_img = data["image"]
        self.after_panel.set_image(data["image"])
        before, after = data["baseline"], data["metrics"]
        if after:
            self.lbl_metrics.setText(
                f"PSNR: {number(before['PSNR'], 2)} → {number(after['PSNR'], 2)} dB "
                f"(Δ {number(data['delta']['PSNR'], 2)})   |   "
                f"SSIM: {number(before['SSIM'])} → {number(after['SSIM'])} "
                f"(Δ {number(data['delta']['SSIM'])})   |   {data['time']:.4f} s")
        else:
            self.lbl_metrics.setText(data["metrics_error"] or "Không có ảnh tham chiếu; đánh giá trực quan.")

    def _save_image(self, image, name="restored.png"):
        if image is None:
            self._error("Chưa có ảnh kết quả để lưu.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", name, IMAGE_FILTER)
        if path:
            try:
                write_image(path, image)
            except Exception as exc:
                self._error(str(exc))
            else:
                self.statusBar().showMessage("Đã lưu: " + path)

    def _save_restored(self):
        self._save_image(self.restored_img)

    def _run_comparison(self):
        if self._busy():
            return
        if self.corrupted_img is None:
            self._error("Vui lòng mở ảnh hỏng trước.")
            return
        self.comparison_results = {}
        self.table_comp.setRowCount(0)
        for panel in self.compare_panels.values():
            panel.set_image(None)
        image, clean, mask = self.corrupted_img, self.clean_img, self.mask_img
        parameters = self.controls.parameters()
        worker = TaskWorkerThread(
            lambda: self.pipeline.compare_all(clean, image, mask, **parameters), self)
        self._start_worker(worker, self._comparison_completed)

    @staticmethod
    def _append_row(table, row, label):
        values = [label, row["status"], number(row["before_PSNR"], 2), number(row["before_SSIM"]),
                  number(row["PSNR"], 2), number(row["SSIM"]), number(row["delta_PSNR"], 2),
                  number(row["delta_SSIM"]), number(row["time"]),
                  row["error"] or row["metrics_error"] or
                  ("Không có ảnh tham chiếu" if row["PSNR"] == "" and row["status"] == "success" else "")]
        index = table.rowCount()
        table.insertRow(index)
        for column, value in enumerate(values):
            table.setItem(index, column, QTableWidgetItem(str(value)))

    def _comparison_completed(self, results):
        self.comparison_results = results
        for label, data in results.items():
            self.compare_panels[label].set_image(data.get("image"), data.get("error", ""))
            row = report_row(Path(self.input_path).name, label, data, self.input_path)
            self._append_row(self.table_comp, row, label)
        if self.table_comp.rowCount():
            self.table_comp.selectRow(0)

    def _save_comparison(self):
        index = self.table_comp.currentRow()
        if index < 0 or not self.comparison_results:
            self._error("Chọn một phương pháp trong bảng kết quả.")
            return
        label = list(self.comparison_results)[index]
        self._save_image(self.comparison_results[label].get("image"), f"{ALGORITHMS[index].key}.png")

    def _export_comparison(self):
        if not self.comparison_results:
            self._error("Chưa có kết quả so sánh.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Xuất so sánh", "comparison.csv", "CSV (*.csv)")
        if path:
            try:
                rows = [report_row(Path(self.input_path).name, label, data, self.input_path)
                        for label, data in self.comparison_results.items()]
                write_csv(path, rows)
            except Exception as exc:
                self._error(str(exc))
            else:
                self.statusBar().showMessage("Đã xuất: " + path)

    def _start_batch(self):
        if self._busy():
            return
        input_dir, output_dir = self.batch_input.text().strip(), self.batch_output.text().strip()
        masks, clean = self.batch_masks.text().strip() or None, self.batch_clean.text().strip() or None
        if not input_dir or not output_dir:
            self._error("Chọn thư mục ảnh đầu vào và đầu ra.")
            return
        if self.controls.needs_mask and not masks:
            self._error("Inpainting / Combined cần thư mục mask.")
            return
        self.batch_rows = []
        self.table_batch.setRowCount(0)
        self.batch_before.set_image(None)
        self.batch_after.set_image(None)
        self.progress_bar.setValue(0)
        self.lbl_batch_status.setText("Đang xử lý…")
        worker = BatchWorkerThread(self.pipeline, input_dir, output_dir, self.controls.algorithm_key,
                                   self.controls.parameters(), mask_dir=masks, parent=self, clean_dir=clean)
        worker.progress_changed.connect(self._batch_progress)
        worker.result_ready.connect(self._batch_row)
        self._start_worker(worker, self._batch_completed)

    def _batch_progress(self, current, total):
        self.progress_bar.setValue(round(current / total * 100) if total else 100)

    def _batch_row(self, row):
        self.batch_rows.append(row)
        self._append_row(self.table_batch, row, row["filename"])
        self.lbl_batch_status.setText(f"Đã xử lý: {row['filename']} — {row['status']}")

    def _batch_completed(self, result):
        state = "Đã hủy" if result.cancelled else "Hoàn thành"
        self.lbl_batch_status.setText(
            f"{state}: {result.processed} thành công, {result.skipped} bỏ qua, {result.failed} lỗi; "
            f"{result.evaluated} cặp được đánh giá / {result.total} file.\n"
            f"Ảnh + details.csv + summary.csv: {result.output_dir}")
        if not result.cancelled:
            self.progress_bar.setValue(100)
        if self.table_batch.rowCount():
            self.table_batch.selectRow(0)

    def _cancel_batch(self):
        if isinstance(self.thread, BatchWorkerThread):
            self.thread.stop()
            self.cancel_button.setEnabled(False)
            self.lbl_batch_status.setText("Đang hủy sau file hiện tại…")

    def _show_batch_preview(self):
        index = self.table_batch.currentRow()
        if index < 0 or index >= len(self.batch_rows):
            return
        row = self.batch_rows[index]
        self.batch_before.set_image(None)
        self.batch_after.set_image(None)
        try:
            self.batch_before.set_image(read_image(row["input_path"]), row["filename"])
            if row["output_path"]:
                self.batch_after.set_image(read_image(row["output_path"]), row["metrics_error"])
            else:
                self.batch_after.set_image(None, row["error"])
        except Exception as exc:
            self.batch_after.set_image(None, str(exc))

    def _error(self, message):
        QMessageBox.warning(self, "Không thể thực hiện", message)

    def closeEvent(self, event):
        worker = self.thread
        if worker is not None and worker.isRunning():
            answer = QMessageBox.question(
                self, "Đang xử lý", "Dừng tác vụ và đóng ứng dụng?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            worker.stop()
            worker.wait()
        event.accept()
