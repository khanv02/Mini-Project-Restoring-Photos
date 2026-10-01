"""Desktop restoration workflows using one shared configuration and pipeline."""

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
    QCheckBox, QComboBox, QDialog, QFormLayout, QGroupBox, QMenu, QScrollArea,
    QSplitter, QStackedWidget, QToolButton,
)

from project_1.entrypoint import RestorationPipeline
from project_1.gui.threads import BatchWorkerThread, TaskWorkerThread
from project_1.gui.theme import apply_workspace_theme
from project_1.gui.widgets import AlgorithmControls, ImagePanel, LinkedImageViews, configure_combo_contrast
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
        apply_workspace_theme(self)

    def _button(self, text, handler, layout, *, lock=True, primary=False):
        button = QPushButton(text)
        button.setProperty("primary", primary)
        button.clicked.connect(handler)
        layout.addWidget(button)
        if lock:
            self._lockable.append(button)
        return button

    def _action(self, text, handler, menu):
        action = QAction(text, self)
        action.triggered.connect(handler)
        menu.addAction(action)
        self._lockable.append(action)
        return action

    def _menu_button(self, text, layout):
        button = QToolButton()
        button.setText(text)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(button)
        button.setMenu(menu)
        layout.addWidget(button)
        self._lockable.append(button)
        return button, menu

    def _init_ui(self):
        root = QWidget()
        root.setObjectName("workspace")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 10)
        outer.setSpacing(14)
        header = QHBoxLayout()
        heading = QVBoxLayout()
        eyebrow = QLabel("SUBJECT 2  /  PROJECT 1")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Khôi phục ảnh cũ")
        title.setObjectName("appTitle")
        heading.addWidget(eyebrow)
        heading.addWidget(title)
        header.addLayout(heading)
        header.addStretch()
        viewer_hint = QLabel("Cuộn chuột: zoom  ·  Kéo: di chuyển  ·  Nhấp đúp: vừa khung")
        viewer_hint.setObjectName("muted")
        viewer_hint.setWordWrap(True)
        header.addWidget(viewer_hint)
        self.sidebar_toggle = QToolButton()
        self.sidebar_toggle.setText("Ẩn cấu hình")
        self.sidebar_toggle.setCheckable(True)
        self.sidebar_toggle.setToolTip("Thu gọn sidebar để dành thêm không gian cho ảnh.")
        self.sidebar_toggle.toggled.connect(self._toggle_sidebar)
        header.addWidget(self.sidebar_toggle)
        outer.addLayout(header)
        layout = QHBoxLayout()
        layout.setSpacing(18)
        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        left = QVBoxLayout(self.sidebar)
        left.setContentsMargins(0, 0, 6, 0)
        left.setSpacing(14)
        source_group = QGroupBox("01  ·  Ảnh đầu vào")
        source_layout = QVBoxLayout(source_group)
        self.open_button = self._button("Mở ảnh hỏng…", self._load_corrupted, source_layout, primary=True)
        self.open_button.setShortcut(QKeySequence.StandardKey.Open)
        self.open_button.setToolTip("Mở ảnh cần phục hồi (Ctrl+O). Đổi ảnh sẽ xóa mask và kết quả cũ.")
        self.input_options_button, self.input_menu = self._menu_button("Tùy chọn ảnh", source_layout)
        self.input_options_button.setToolTip("Ảnh tham chiếu, mask phục hồi và bộ ảnh mẫu.")
        self.reference_action = self._action("Mở ảnh tham chiếu…", self._load_clean, self.input_menu)
        self.input_menu.addSeparator()
        self.load_mask_action = self._action("Mở mask từ file…", self._load_mask, self.input_menu)
        self.auto_mask_action = self._action("Gợi ý mask xước sáng…", self._suggest_mask, self.input_menu)
        self.save_mask_action = self._action("Lưu mask đã xác nhận (PNG)…", self._save_mask, self.input_menu)
        self.input_menu.addSeparator()
        self.clear_references_action = self._action("Bỏ tham chiếu và mask", self._clear_references, self.input_menu)
        self.demo_action = self._action("Nạp bộ ảnh mẫu và so sánh", self.load_demo, self.input_menu)
        self.input_label = QLabel("Chưa nạp ảnh. Hãy chọn ảnh hỏng trước.")
        self.input_label.setObjectName("inputStatus")
        self.input_label.setWordWrap(True)
        source_layout.addWidget(self.input_label)
        left.addWidget(source_group)
        self.controls = AlgorithmControls("02  ·  Thuật toán & tham số")
        left.addWidget(self.controls)
        self._lockable.append(self.controls)
        hint = QLabel("Tham chiếu chỉ cần khi đo PSNR/SSIM.\n"
                      "Inpainting / Combined cần mask.\n"
                      "Chỉnh thông số rồi bấm xử lý để xem kết quả.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        left.addWidget(hint)
        left.addStretch()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._single_tab(), "Khôi phục đơn")
        self.tabs.addTab(self._comparison_tab(), "So sánh phương pháp")
        self.tabs.addTab(self._batch_tab(), "Xử lý hàng loạt")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.controls.algorithm.currentIndexChanged.connect(self._tab_changed)
        scroll = self.sidebar_scroll = QScrollArea()
        scroll.setObjectName("sidebarScroll")
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(345)
        scroll.setWidget(self.sidebar)
        layout.addWidget(scroll)
        layout.addWidget(self.tabs, 1)
        outer.addLayout(layout, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Sẵn sàng")
        self._update_mask_actions()

    def _toggle_sidebar(self, hidden):
        self.sidebar_scroll.setVisible(not hidden)
        self.sidebar_toggle.setText("Hiện cấu hình" if hidden else "Ẩn cấu hình")

    def _link_views(self, panels, actions):
        link = LinkedImageViews([panel.view for panel in panels], self)
        checkbox = QCheckBox("Đồng bộ zoom/kéo")
        checkbox.setChecked(True)
        checkbox.setToolTip("Thao tác trên một ảnh áp dụng cùng tỷ lệ và tâm nhìn cho ảnh tương ứng.")
        checkbox.toggled.connect(link.set_enabled)
        actions.addWidget(checkbox)
        return link, checkbox

    def _single_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        actions = QHBoxLayout()
        self.run_button = self._button("Khôi phục ảnh", self._run_single_restoration, actions, primary=True)
        actions.addStretch()
        self.single_export_button, menu = self._menu_button("Lưu kết quả", actions)
        self.save_single_action = self._action("Lưu ảnh khôi phục…", self._save_restored, menu)
        layout.addLayout(actions)
        views = QHBoxLayout()
        self.before_panel = ImagePanel("Trước: ảnh hỏng")
        self.after_panel = ImagePanel("Sau: ảnh khôi phục")
        views.addWidget(self.before_panel)
        views.addWidget(self.after_panel)
        self.single_view_link, self.single_link_checkbox = self._link_views(
            [self.before_panel, self.after_panel], actions)
        layout.addLayout(views, 1)
        self.lbl_metrics = QLabel("Không có kết quả")
        self.lbl_metrics.setObjectName("resultStatus")
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
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _comparison_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        actions = QHBoxLayout()
        self.compare_button = self._button("So sánh 3 phương pháp", self._run_comparison, actions, primary=True)
        actions.addStretch()
        self.compare_view_button, view_menu = self._menu_button("Hiển thị", actions)
        modes = QActionGroup(self)
        self.compare_pair_action = view_menu.addAction("Đối chiếu 2 ảnh lớn")
        self.compare_overview_action = view_menu.addAction("Tổng quan 4 ảnh")
        for action in (self.compare_pair_action, self.compare_overview_action):
            action.setCheckable(True)
            modes.addAction(action)
        self.compare_pair_action.setChecked(True)
        self.compare_pair_action.triggered.connect(lambda: self._set_compare_overview(False))
        self.compare_overview_action.triggered.connect(lambda: self._set_compare_overview(True))
        view_menu.addSeparator()
        self.compare_table_action = view_menu.addAction("Hiện bảng chỉ số")
        self.compare_table_action.setCheckable(True)
        self.compare_table_action.setChecked(True)
        self.compare_export_button, menu = self._menu_button("Xuất kết quả", actions)
        self.save_comparison_action = self._action("Lưu ảnh phương pháp đã chọn…", self._save_comparison, menu)
        self.export_csv_action = self._action("Xuất bảng so sánh (CSV)…", self._export_comparison, menu)
        layout.addLayout(actions)
        self.compare_selectors = QWidget()
        selectors = QHBoxLayout(self.compare_selectors)
        selectors.setContentsMargins(0, 0, 0, 0)
        self.compare_left_selector = QComboBox()
        self.compare_right_selector = QComboBox()
        for combo in (self.compare_left_selector, self.compare_right_selector):
            configure_combo_contrast(combo)
        self.compare_left_selector.addItem("Ảnh hỏng", "__input__")
        self.compare_left_selector.addItem("Tham chiếu", "__clean__")
        for spec in ALGORITHMS:
            text = "Combined" if spec.key == "combined" else spec.label
            self.compare_left_selector.addItem(text, spec.label)
            self.compare_right_selector.addItem(text, spec.label)
        selectors.addWidget(QLabel("Trái"))
        selectors.addWidget(self.compare_left_selector, 1)
        selectors.addWidget(QLabel("Phải"))
        selectors.addWidget(self.compare_right_selector, 1)
        layout.addWidget(self.compare_selectors)

        self.compare_workspace = QStackedWidget()
        self.compare_pair_widget = QWidget()
        self.compare_pair_layout = QHBoxLayout(self.compare_pair_widget)
        self.compare_pair_layout.setContentsMargins(0, 0, 0, 0)
        self.compare_before = ImagePanel("Ảnh hỏng")
        self.compare_pair_layout.addWidget(self.compare_before, 1)
        self.compare_result_stack = QStackedWidget()
        self.compare_pair_layout.addWidget(self.compare_result_stack, 1)
        self.compare_panels = {}
        for spec in ALGORITHMS:
            panel = ImagePanel(spec.label)
            self.compare_panels[spec.label] = panel
            self.compare_result_stack.addWidget(panel)
        self.compare_overview_widget = QWidget()
        self.compare_grid = QGridLayout(self.compare_overview_widget)
        self.compare_grid.setContentsMargins(0, 0, 0, 0)
        for index in (0, 1):
            self.compare_grid.setRowStretch(index, 1)
            self.compare_grid.setColumnStretch(index, 1)
        self.compare_workspace.addWidget(self.compare_pair_widget)
        self.compare_workspace.addWidget(self.compare_overview_widget)
        self.compare_view_link, self.compare_link_checkbox = self._link_views(
            [self.compare_before, *self.compare_panels.values()], actions)
        self.table_comp = self._table()
        self.table_comp.itemSelectionChanged.connect(self._update_result_actions)
        self.table_comp.itemSelectionChanged.connect(self._comparison_selection_changed)
        self.table_comp.setMinimumHeight(90)
        self.table_comp.setMaximumHeight(230)
        self.compare_splitter = QSplitter(Qt.Orientation.Vertical)
        self.compare_splitter.setChildrenCollapsible(False)
        self.compare_splitter.addWidget(self.compare_workspace)
        self.compare_splitter.addWidget(self.table_comp)
        self.compare_splitter.setStretchFactor(0, 1)
        self.compare_splitter.setStretchFactor(1, 0)
        self.compare_splitter.setSizes([650, 130])
        self.compare_table_action.toggled.connect(self.table_comp.setVisible)
        layout.addWidget(self.compare_splitter, 1)
        self.compare_left_selector.currentIndexChanged.connect(self._show_comparison_pair)
        self.compare_right_selector.currentIndexChanged.connect(self._right_comparison_changed)
        return widget

    def _comparison_selection_changed(self):
        index = self.table_comp.currentRow()
        if index >= 0:
            with QSignalBlocker(self.compare_right_selector):
                self.compare_right_selector.setCurrentIndex(index)
            self._show_comparison_pair()

    def _right_comparison_changed(self, *_):
        index = self.compare_right_selector.currentIndex()
        if index < self.table_comp.rowCount():
            self.table_comp.selectRow(index)
        self._show_comparison_pair()

    def _show_comparison_pair(self, *_):
        """Choose images without recomputing results, retaining the current inspection camera."""
        overview = self.compare_workspace.currentWidget() is self.compare_overview_widget
        key = "__input__" if overview else self.compare_left_selector.currentData()
        if key == "__input__":
            image, title, caption = self.corrupted_img, "Ảnh hỏng", Path(self.input_path).name
        elif key == "__clean__":
            image, title, caption = self.clean_img, "Tham chiếu", "" if self.clean_img is not None else "Chưa nạp tham chiếu"
        else:
            data = self.comparison_results.get(key, {})
            image, title = data.get("image"), key
            caption = data.get("error") or ("Chưa có kết quả" if image is None else "")
        camera = self.compare_before.view.viewport_state()
        if self.compare_before.image is not image:
            self.compare_before.set_image(image, caption)
            self.compare_before.view.apply_viewport_state(camera)
        self.compare_before.setTitle(title)
        self.compare_before.caption.setText(caption)
        self.compare_before.caption.setToolTip(caption)
        label = self.compare_right_selector.currentData()
        if not overview:
            self.compare_result_stack.setCurrentWidget(self.compare_panels[label])
        self.compare_view_link.sync_from(self.compare_before.view)

    def _set_compare_overview(self, overview):
        if overview == (self.compare_workspace.currentWidget() is self.compare_overview_widget):
            return
        panels = [self.compare_before, *self.compare_panels.values()]
        if overview:
            self.compare_pair_layout.removeWidget(self.compare_before)
            for panel in self.compare_panels.values():
                self.compare_result_stack.removeWidget(panel)
            for index, panel in enumerate(panels):
                self.compare_grid.addWidget(panel, index // 2, index % 2)
                panel.show()
            self.compare_workspace.setCurrentWidget(self.compare_overview_widget)
        else:
            for panel in panels:
                self.compare_grid.removeWidget(panel)
            self.compare_pair_layout.insertWidget(0, self.compare_before, 1)
            self.compare_before.show()
            for panel in self.compare_panels.values():
                self.compare_result_stack.addWidget(panel)
            self.compare_workspace.setCurrentWidget(self.compare_pair_widget)
        self.compare_pair_action.setChecked(not overview)
        self.compare_overview_action.setChecked(overview)
        self.compare_selectors.setVisible(not overview)
        self._show_comparison_pair()

    def _directory_control(self, title, layout, default=""):
        row = QHBoxLayout()
        entry = QLineEdit(default)
        entry.setPlaceholderText("Chọn thư mục…")
        button = QToolButton()
        button.setObjectName("directoryPicker")
        button.setText("…")
        button.setToolTip("Chọn " + title.lower())
        button.setAccessibleName("Chọn " + title.lower())
        button.clicked.connect(lambda: self._select_directory(entry, title))
        row.addWidget(entry, 1)
        row.addWidget(button)
        self._lockable.extend((entry, button))
        layout.addRow(title, row)
        return entry

    def _batch_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        directories = self.batch_settings = QGroupBox("Thư mục xử lý")
        form = QFormLayout(directories)
        self.batch_input = self._directory_control("Ảnh hỏng", form)
        self.batch_masks = self._directory_control("Mask", form)
        self.batch_clean = self._directory_control("Tham chiếu (tùy chọn)", form)
        self.batch_output = self._directory_control("Lưu kết quả", form, "output")
        for entry, name in ((self.batch_input, "images_corrupted"),
                            (self.batch_masks, "images_masks"), (self.batch_clean, "images_clean")):
            directory = self.dataset_dir / name
            if directory.is_dir():
                entry.setText(str(directory))
        actions = QHBoxLayout()
        self._button("Bắt đầu Batch", self._start_batch, actions, primary=True)
        self.batch_settings_button = QToolButton()
        self.batch_settings_button.setText("Thư mục")
        self.batch_settings_button.setCheckable(True)
        self.batch_settings_button.setChecked(True)
        self.batch_settings_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.batch_settings_button.setArrowType(Qt.ArrowType.DownArrow)
        self.batch_settings_button.toggled.connect(self._toggle_batch_settings)
        actions.addWidget(self.batch_settings_button)
        actions.addStretch()
        self.batch_view_button, menu = self._menu_button("Hiển thị", actions)
        self.batch_compact_action = menu.addAction("Danh sách file gọn")
        self.batch_compact_action.setCheckable(True)
        self.batch_compact_action.setChecked(True)
        self.batch_compact_action.toggled.connect(self._toggle_batch_columns)
        self.cancel_button = self._button("Hủy Batch", self._cancel_batch, actions, lock=False)
        self.cancel_button.setEnabled(False)
        layout.addLayout(actions)
        layout.addWidget(directories)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.lbl_batch_status = QLabel("Sẵn sàng")
        self.lbl_batch_status.setWordWrap(True)
        self.lbl_batch_status.setMaximumHeight(48)
        layout.addWidget(self.lbl_batch_status)
        self.table_batch = self._table()
        self.table_batch.setMinimumWidth(180)
        self.table_batch.itemSelectionChanged.connect(self._show_batch_preview)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        views = QHBoxLayout()
        self.batch_before = ImagePanel("Batch: trước")
        self.batch_after = ImagePanel("Batch: sau")
        views.addWidget(self.batch_before)
        views.addWidget(self.batch_after)
        preview_layout.addLayout(views, 1)
        self.batch_details = QLabel("Chọn một file để đối chiếu trước / sau.")
        self.batch_details.setObjectName("resultStatus")
        self.batch_details.setWordWrap(True)
        self.batch_details.setMaximumHeight(58)
        preview_layout.addWidget(self.batch_details)
        self.batch_view_link, self.batch_link_checkbox = self._link_views(
            [self.batch_before, self.batch_after], actions)
        self.batch_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.batch_splitter.setChildrenCollapsible(False)
        self.batch_splitter.addWidget(self.table_batch)
        self.batch_splitter.addWidget(preview)
        self.batch_splitter.setStretchFactor(0, 0)
        self.batch_splitter.setStretchFactor(1, 1)
        self.batch_splitter.setSizes([230, 900])
        layout.addWidget(self.batch_splitter, 1)
        self._toggle_batch_columns(True)
        return widget

    def _toggle_batch_settings(self, visible):
        self.batch_settings.setVisible(visible)
        self.batch_settings_button.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

    def _toggle_batch_columns(self, compact):
        for column in range(2, self.table_batch.columnCount()):
            self.table_batch.setColumnHidden(column, compact)

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
        loaded = self.corrupted_img is not None
        for action in (self.reference_action, self.load_mask_action, self.auto_mask_action):
            action.setEnabled(allowed and loaded)
        self.save_mask_action.setEnabled(allowed and self.mask_img is not None)
        self.clear_references_action.setEnabled(allowed and (self.clean_img is not None or self.mask_img is not None))
        self.run_button.setEnabled(not self._busy() and loaded and (not self.controls.needs_mask or self.mask_img is not None))
        self.run_button.setToolTip("Phục hồi bằng cấu hình bên trái." if self.run_button.isEnabled()
                                   else "Nạp ảnh hỏng và mask nếu phương pháp yêu cầu; chờ xử lý hiện tại kết thúc.")
        self.compare_button.setEnabled(not self._busy() and loaded)
        self._update_result_actions()

    def _update_result_actions(self):
        available = not self._busy()
        self.save_single_action.setEnabled(available and self.restored_img is not None)
        self.single_export_button.setEnabled(self.save_single_action.isEnabled())
        index = self.table_comp.currentRow()
        results = list(self.comparison_results.values())
        selected_image = 0 <= index < len(results) and results[index].get("image") is not None
        self.save_comparison_action.setEnabled(available and selected_image)
        self.export_csv_action.setEnabled(available and bool(results))
        self.compare_export_button.setEnabled(self.export_csv_action.isEnabled())

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
        self._show_comparison_pair()
        self._update_result_actions()

    def _update_input_label(self):
        self.input_label.setText(
            f"Ảnh hỏng: {Path(self.input_path).name or 'chưa nạp'}\n"
            f"Tham chiếu: {'đã nạp' if self.clean_img is not None else 'không có'}\n"
            f"Mask: {self.mask_source or ('đã nạp' if self.mask_img is not None else 'không có')}")
        self.input_label.setToolTip(self.input_path)
        self._update_mask_actions()

    def load_input(self, path):
        """Load a new input transactionally, then invalidate references and results."""
        image = read_image(path)
        self.corrupted_img = image
        self.input_path = str(path)
        with QSignalBlocker(self.compare_left_selector):
            self.compare_left_selector.setCurrentIndex(0)
        self.clean_img = self.mask_img = None
        self.mask_source, self.mask_parameters = "", None
        self._clear_results()
        for panel in (self.before_panel, self.compare_before):
            panel.set_image(image, Path(path).name)
            panel.caption.setToolTip(str(path))
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
            self.batch_settings_button.setChecked(True)
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
        self.single_view_link.sync_from(self.before_panel.view)
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
        self._show_comparison_pair()
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
            item = QTableWidgetItem(str(value))
            item.setToolTip("\n".join(f"{title}: {text}" for title, text in zip(TABLE_HEADERS, values)))
            table.setItem(index, column, item)

    def _comparison_completed(self, results):
        self.comparison_results = results
        for label, data in results.items():
            self.compare_panels[label].set_image(data.get("image"), data.get("error", ""))
            row = report_row(Path(self.input_path).name, label, data, self.input_path)
            self._append_row(self.table_comp, row, label)
        if self.table_comp.rowCount():
            self.table_comp.selectRow(self.compare_right_selector.currentIndex())
        self._show_comparison_pair()

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
            self.batch_settings_button.setChecked(True)
            self._error("Chọn thư mục ảnh đầu vào và đầu ra.")
            return
        if self.controls.needs_mask and not masks:
            self.batch_settings_button.setChecked(True)
            self._error("Inpainting / Combined cần thư mục mask.")
            return
        self.batch_rows = []
        self.table_batch.setRowCount(0)
        self.batch_before.set_image(None)
        self.batch_after.set_image(None)
        self.batch_details.setText("Chọn một file để đối chiếu trước / sau.")
        self.batch_settings_button.setChecked(False)
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
        if self.table_batch.currentRow() < 0:
            self.table_batch.selectRow(0)
        self.lbl_batch_status.setText(f"Đã xử lý: {row['filename']} — {row['status']}")

    def _batch_completed(self, result):
        state = "Đã hủy" if result.cancelled else "Hoàn thành"
        self.lbl_batch_status.setText(
            f"{state}: {result.processed} thành công, {result.skipped} bỏ qua, {result.failed} lỗi; "
            f"{result.evaluated} cặp được đánh giá / {result.total} file.\n"
            f"Ảnh và CSV: {Path(result.output_dir).name}")
        if not result.cancelled:
            self.progress_bar.setValue(100)
        self.lbl_batch_status.setToolTip(self.lbl_batch_status.text() + "\nThư mục: " + result.output_dir)
        if self.table_batch.rowCount() and self.table_batch.currentRow() < 0:
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
        details = (f"{row['filename']}  ·  {row['status']}  |  "
                   f"PSNR: {number(row['before_PSNR'], 2)} → {number(row['PSNR'], 2)} dB  |  "
                   f"SSIM: {number(row['before_SSIM'])} → {number(row['SSIM'])}")
        if row['error'] or row['metrics_error']:
            details += "\n" + (row['error'] or row['metrics_error'])
        self.batch_details.setText(details)
        self.batch_details.setToolTip(details)
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
            details += "\nKhông thể xem ảnh: " + str(exc)
            self.batch_details.setText(details)
            self.batch_details.setToolTip(details)

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
