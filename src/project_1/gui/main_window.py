import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QGroupBox, QTableWidget, QTableWidgetItem,
    QProgressBar, QMessageBox, QHeaderView
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from project_1.entrypoint import RestorationPipeline
from project_1.gui.threads import BatchWorkerThread

def cv_to_qpixmap(cv_img: np.ndarray, max_w=450, max_h=450) -> QPixmap:
    """Chuyển đổi OpenCV image (BGR) sang QPixmap để hiển thị trên PySide6"""
    if cv_img is None:
        return QPixmap()
    
    if len(cv_img.shape) == 2:
        h, w = cv_img.shape
        qimg = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, ch = cv_img.shape
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Khôi Phục Ảnh Cũ - Subject 2 Project 1 (Gaussian & Inpainting)")
        self.resize(1100, 750)

        self.pipeline = RestorationPipeline()
        self.clean_img = None
        self.corrupted_img = None
        self.restored_img = None
        self.mask_img = None

        self._init_ui()

    def _init_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._create_single_tab(), "🖼️ Khôi Phục Đơn")
        tabs.addTab(self._create_comparison_tab(), "⚖️ So Sánh Phương Pháp")
        tabs.addTab(self._create_batch_tab(), "📂 Xử Lý Hàng Loạt")
        self.setCentralWidget(tabs)

    # ================= TAB 1: KHÔI PHỤC ĐƠN =================
    def _create_single_tab(self) -> QWidget:
        widget = QWidget()
        main_layout = QHBoxLayout(widget)

        # Bảng điều khiển bên trái
        left_box = QGroupBox("Cấu Hình & Tham Số")
        left_layout = QVBoxLayout(left_box)

        btn_load_clean = QPushButton("1. Mở Ảnh Gốc (Clean)")
        btn_load_clean.clicked.connect(self._load_clean)
        left_layout.addWidget(btn_load_clean)

        btn_load_corr = QPushButton("2. Mở Ảnh Lỗi (Corrupted)")
        btn_load_corr.clicked.connect(self._load_corrupted)
        left_layout.addWidget(btn_load_corr)

        btn_load_mask = QPushButton("3. Mở Ảnh Mask (Vết xước)")
        btn_load_mask.clicked.connect(self._load_mask)
        left_layout.addWidget(btn_load_mask)

        left_layout.addWidget(QLabel("Chọn Thuật Toán:"))
        self.combo_algo = QComboBox()
        # Thêm lựa chọn Combined vào ComboBox
        self.combo_algo.addItems(["Gaussian Filter", "Inpainting", "Combined (Inpainting + Gaussian)"])
        self.combo_algo.currentIndexChanged.connect(self._on_algo_change)
        left_layout.addWidget(self.combo_algo)

        # Widget điều khiển cho Gaussian
        self.lbl_kernel = QLabel("Kernel Size (Số lẻ):")
        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(3, 31)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(5)
        
        self.lbl_sigma = QLabel("Sigma Gaussian:")
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setValue(1.2)

        # Widget điều khiển cho Inpainting
        self.lbl_radius = QLabel("Inpaint Radius:")
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(1, 20)
        self.spin_radius.setValue(3)

        self.lbl_inpaint_method = QLabel("Inpaint Method:")
        self.combo_inpaint_method = QComboBox()
        self.combo_inpaint_method.addItems(["Telea", "Navier-Stokes"])

        left_layout.addWidget(self.lbl_kernel)
        left_layout.addWidget(self.spin_kernel)
        left_layout.addWidget(self.lbl_sigma)
        left_layout.addWidget(self.spin_sigma)
        left_layout.addWidget(self.lbl_radius)
        left_layout.addWidget(self.spin_radius)
        left_layout.addWidget(self.lbl_inpaint_method)
        left_layout.addWidget(self.combo_inpaint_method)

        # Cập nhật hiển thị tham số phù hợp với thuật toán ban đầu
        self._on_algo_change(0)

        btn_run = QPushButton("🚀 Thực Thi Khôi Phục")
        btn_run.setStyleSheet("font-weight: bold; padding: 8px; background-color: #2b5b84; color: white;")
        btn_run.clicked.connect(self._run_single_restoration)
        left_layout.addWidget(btn_run)

        # Hiển thị chỉ số
        self.lbl_metrics = QLabel("PSNR: -- dB | SSIM: --")
        self.lbl_metrics.setStyleSheet("font-size: 14px; font-weight: bold; color: #008000;")
        left_layout.addWidget(self.lbl_metrics)

        btn_save = QPushButton("💾 Lưu Ảnh Kết Quả")
        btn_save.clicked.connect(self._save_restored)
        left_layout.addWidget(btn_save)
        left_layout.addStretch()

        # Khu vực xem ảnh bên phải (Trước / Sau)
        right_box = QGroupBox("Hiển Thị Chi Tiết")
        right_layout = QHBoxLayout(right_box)

        self.lbl_view_before = QLabel("Ảnh Nhiễu (Before)")
        self.lbl_view_before.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_view_before.setStyleSheet("border: 1px dashed gray;")

        self.lbl_view_after = QLabel("Ảnh Khôi Phục (After)")
        self.lbl_view_after.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_view_after.setStyleSheet("border: 1px dashed gray;")

        right_layout.addWidget(self.lbl_view_before)
        right_layout.addWidget(self.lbl_view_after)

        main_layout.addWidget(left_box, 1)
        main_layout.addWidget(right_box, 2)
        return widget

    # ================= TAB 2: SO SÁNH =================
    def _create_comparison_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_compare = QPushButton("📊 Chạy So Sánh Tất Cả Thuật Toán")
        btn_compare.clicked.connect(self._run_comparison)
        layout.addWidget(btn_compare)

        self.table_comp = QTableWidget(0, 4)
        self.table_comp.setHorizontalHeaderLabels(["Thuật Toán", "PSNR (dB)", "SSIM", "Thời Gian (s)"])
        self.table_comp.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table_comp)
        return widget

    # ================= TAB 3: BATCH PROCESSING =================
    def _create_batch_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        box = QGroupBox("Xử Lý Hàng Loạt (Batch Processing)")
        box_layout = QVBoxLayout(box)

        btn_in_dir = QPushButton("1. Chọn Thư Mục Ảnh Đầu Vào (Corrupted)")
        btn_in_dir.clicked.connect(self._select_batch_in)
        self.lbl_in_dir = QLabel("Chưa chọn...")
        box_layout.addWidget(btn_in_dir)
        box_layout.addWidget(self.lbl_in_dir)

        btn_mask_dir = QPushButton("2. Chọn Thư Mục Mask (Bắt buộc với Inpainting / Combined)")
        btn_mask_dir.clicked.connect(self._select_batch_mask)
        self.lbl_mask_dir = QLabel("Chưa chọn (Không bắt buộc với Gaussian)...")
        box_layout.addWidget(btn_mask_dir)
        box_layout.addWidget(self.lbl_mask_dir)

        btn_out_dir = QPushButton("3. Chọn Thư Mục Lưu Output")
        btn_out_dir.clicked.connect(self._select_batch_out)
        self.lbl_out_dir = QLabel("Chưa chọn...")
        box_layout.addWidget(btn_out_dir)
        box_layout.addWidget(self.lbl_out_dir)

        box_layout.addWidget(QLabel("Thuật toán áp dụng:"))
        self.combo_batch_algo = QComboBox()
        self.combo_batch_algo.addItems(["Gaussian Filter", "Inpainting", "Combined (Inpainting + Gaussian)"])
        box_layout.addWidget(self.combo_batch_algo)

        btn_start_batch = QPushButton("▶ Bắt Đầu Xử Lý Batch")
        btn_start_batch.setStyleSheet("font-weight: bold; padding: 6px; background-color: #2b5b84; color: white;")
        btn_start_batch.clicked.connect(self._start_batch)
        box_layout.addWidget(btn_start_batch)

        self.progress_bar = QProgressBar()
        box_layout.addWidget(self.progress_bar)
        self.lbl_batch_status = QLabel("Trạng thái: Sẵn sàng")
        box_layout.addWidget(self.lbl_batch_status)

        layout.addWidget(box)
        layout.addStretch()
        return widget

    # ================= EVENT HANDLERS =================
    def _load_clean(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh gốc", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.clean_img = cv2.imread(path)

    def _load_corrupted(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh nhiễu", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.corrupted_img = cv2.imread(path)
            self.lbl_view_before.setPixmap(cv_to_qpixmap(self.corrupted_img))

    def _load_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh mask", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.mask_img = cv2.imread(path)

    def _on_algo_change(self, idx):
        # idx 0: Gaussian, idx 1: Inpainting, idx 2: Combined
        show_gauss = (idx == 0 or idx == 2)
        show_inpaint = (idx == 1 or idx == 2)

        self.lbl_kernel.setVisible(show_gauss)
        self.spin_kernel.setVisible(show_gauss)
        self.lbl_sigma.setVisible(show_gauss)
        self.spin_sigma.setVisible(show_gauss)

        self.lbl_radius.setVisible(show_inpaint)
        self.spin_radius.setVisible(show_inpaint)
        self.lbl_inpaint_method.setVisible(show_inpaint)
        self.combo_inpaint_method.setVisible(show_inpaint)

    def _run_single_restoration(self):
        if self.corrupted_img is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng mở ảnh lỗi trước!")
            return

        algo_map = {0: "gaussian", 1: "inpainting", 2: "combined"}
        selected_key = algo_map[self.combo_algo.currentIndex()]
        
        kwargs = {}
        # Chuẩn bị tham số Gaussian
        if selected_key in ["gaussian", "combined"]:
            ksize = self.spin_kernel.value()
            if ksize % 2 == 0:
                ksize += 1
            kwargs["kernel_size"] = ksize
            kwargs["sigma"] = self.spin_sigma.value()

        # Chuẩn bị tham số Inpainting
        if selected_key in ["inpainting", "combined"]:
            if self.mask_img is None:
                QMessageBox.warning(self, "Lỗi", f"Thuật toán {selected_key.title()} cần phải nạp Ảnh Mask!")
                return
            kwargs["mask"] = self.mask_img
            kwargs["radius"] = self.spin_radius.value()
            kwargs["method"] = self.combo_inpaint_method.currentText().lower()

        # Thực thi
        self.restored_img = self.pipeline.run_single(selected_key, self.corrupted_img, **kwargs)
        self.lbl_view_after.setPixmap(cv_to_qpixmap(self.restored_img))

        # Tính chỉ số và In ra Terminal nếu có ảnh gốc
        if self.clean_img is not None:
            m = self.pipeline.evaluate(self.clean_img, self.restored_img)
            self.lbl_metrics.setText(f"PSNR: {m['PSNR']} dB | SSIM: {m['SSIM']}")
            print(f"[EVALUATION - {selected_key.upper()}] PSNR: {m['PSNR']} dB | SSIM: {m['SSIM']}")

    def _save_restored(self):
        if self.restored_img is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu ảnh", "restored.png", "Images (*.png *.jpg)")
        if path:
            cv2.imwrite(path, self.restored_img)
            QMessageBox.information(self, "Thông báo", "Đã lưu ảnh thành công!")

    def _run_comparison(self):
        if self.clean_img is None or self.corrupted_img is None:
            QMessageBox.warning(self, "Lỗi", "Cần mở cả ảnh gốc lẫn ảnh lỗi để so sánh!")
            return

        res = self.pipeline.compare_all(self.clean_img, self.corrupted_img, self.mask_img)
        self.table_comp.setRowCount(0)

        print("\n--- BẢNG SO SÁNH THUẬT TOÁN (TERMINAL LOG) ---")
        for row, (method_name, data) in enumerate(res.items()):
            self.table_comp.insertRow(row)
            self.table_comp.setItem(row, 0, QTableWidgetItem(method_name))
            self.table_comp.setItem(row, 1, QTableWidgetItem(str(data["metrics"]["PSNR"])))
            self.table_comp.setItem(row, 2, QTableWidgetItem(str(data["metrics"]["SSIM"])))
            self.table_comp.setItem(row, 3, QTableWidgetItem(str(data["time"])))
            
            print(f"[{method_name}] PSNR: {data['metrics']['PSNR']} dB | SSIM: {data['metrics']['SSIM']} | Time: {data['time']}s")
        print("----------------------------------------------\n")

    # ================= BATCH PROCESSING =================
    def _select_batch_in(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục đầu vào")
        if d:
            self.lbl_in_dir.setText(d)

    def _select_batch_mask(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục mask")
        if d:
            self.lbl_mask_dir.setText(d)

    def _select_batch_out(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả")
        if d:
            self.lbl_out_dir.setText(d)

    def _start_batch(self):
        in_d = self.lbl_in_dir.text()
        out_d = self.lbl_out_dir.text()
        mask_d = self.lbl_mask_dir.text() if self.lbl_mask_dir.text() != "Chưa chọn (Không bắt buộc với Gaussian)..." else None

        if in_d == "Chưa chọn..." or out_d == "Chưa chọn...":
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn thư mục đầu vào và đầu ra!")
            return

        idx = self.combo_batch_algo.currentIndex()
        algo_map = {0: "gaussian", 1: "inpainting", 2: "combined"}
        algo_key = algo_map[idx]
        
        if algo_key in ["inpainting", "combined"] and (not mask_d or mask_d == "Chưa chọn..."):
            QMessageBox.warning(self, "Lỗi", f"Vui lòng chọn thư mục Mask để chạy Batch {algo_key.title()}!")
            return

        # Lấy trực tiếp tham số từ giao diện GUI thay vì hardcode
        ksize = self.spin_kernel.value()
        if ksize % 2 == 0:
            ksize += 1

        kwargs = {
            "kernel_size": ksize,
            "sigma": self.spin_sigma.value(),
            "radius": self.spin_radius.value(),
            "method": self.combo_inpaint_method.currentText().lower()
        }
        
        self.thread = BatchWorkerThread(
            self.pipeline, in_d, out_d, algo_key, kwargs, mask_dir=mask_d
        )
        self.thread.progress_changed.connect(lambda cur, tot: self.progress_bar.setValue(int(cur / tot * 100)))
        self.thread.file_processed.connect(lambda f: self.lbl_batch_status.setText(f"Đang xử lý: {f}"))
        self.thread.finished_all.connect(lambda: self.lbl_batch_status.setText("✅ Hoàn thành Batch Processing!"))
        self.thread.start()