"""Shared light theme for the desktop workspace and mask review dialog."""

from PySide6.QtGui import QFont


WORKSPACE_STYLE = """
QWidget { color: #243248; }
QMainWindow, QDialog, QWidget#workspace { background: #f2f5fa; }
QLabel { background: transparent; }
QLabel#appTitle { font-size: 23px; font-weight: 700; color: #14233b; }
QLabel#eyebrow { color: #2563eb; font-size: 10px; font-weight: 700; }
QLabel#muted, QLabel#viewerCaption { color: #68778d; }
QLabel#inputStatus, QLabel#resultStatus {
    background: #edf3fd; border-radius: 8px; padding: 10px; color: #435774;
}
QScrollArea#sidebarScroll { border: none; background: transparent; }
QWidget#sidebar { background: transparent; }
QGroupBox {
    background: white; border: 1px solid #dce4ef; border-radius: 10px;
    margin-top: 12px; padding: 14px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 14px; padding: 0 5px; color: #425570; font-weight: 600;
}
QPushButton, QToolButton {
    background: white; border: 1px solid #d4dfed; border-radius: 7px;
    padding: 7px 12px; font-weight: 600;
}
QPushButton:hover, QToolButton:hover { background: #edf3fd; border-color: #9eb8df; }
QPushButton:pressed, QToolButton:pressed { background: #dfeafb; }
QPushButton[primary="true"] { background: #2563eb; border-color: #2563eb; color: white; }
QPushButton[primary="true"]:hover { background: #1d4ed8; border-color: #1d4ed8; }
QPushButton[primary="true"]:pressed { background: #1e40af; }
QPushButton:disabled, QToolButton:disabled {
    background: #edf1f6; border-color: #e0e6ef; color: #97a4b7;
}
QPushButton:focus, QToolButton:focus, QLineEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #2563eb; }
QToolButton::menu-indicator { subcontrol-origin: padding; subcontrol-position: right center; right: 6px; }
QToolButton#zoomMenu { padding: 3px 18px 3px 8px; font-weight: 500; }
QToolButton#directoryPicker { padding: 5px 9px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: white; border: 1px solid #d7e1ee; border-radius: 5px;
    padding: 5px; selection-background-color: #2563eb; selection-color: white;
}
QComboBox { padding-right: 20px; }
QComboBox QAbstractItemView {
    background: white; color: #243248; border: 1px solid #d4dfed;
    selection-background-color: #2563eb; selection-color: white; outline: none;
}
QComboBox QAbstractItemView::item { padding: 7px 10px; min-height: 22px; }
QComboBox QAbstractItemView::item:selected { background: #2563eb; color: white; }
QSplitter::handle { background: #dce4ef; border-radius: 2px; }
QSplitter::handle:hover { background: #76a1f5; }
QSlider::groove:horizontal { background: #dbe4ef; height: 4px; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #76a1f5; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #2563eb; border: 2px solid white; width: 12px;
    margin: -6px 0; border-radius: 7px;
}
QSlider::handle:horizontal:disabled { background: #aebccc; }
QTabWidget::pane { border: none; background: transparent; }
QTabBar::tab {
    background: transparent; color: #6b7c94; padding: 10px 14px;
    border-bottom: 3px solid transparent; font-weight: 600;
}
QTabBar::tab:selected { color: #2563eb; border-bottom-color: #2563eb; }
QTabBar::tab:hover { background: #e8effa; }
QTableWidget {
    background: white; alternate-background-color: #f7f9fc;
    border: 1px solid #dce4ef; border-radius: 7px; gridline-color: #edf1f7;
    selection-background-color: #dfeafb; selection-color: #1e40af;
}
QHeaderView::section { background: #edf2f9; color: #50647f; border: none; padding: 7px; font-weight: 600; }
QProgressBar { border: none; background: #e4ebf5; border-radius: 5px; text-align: center; min-height: 16px; }
QProgressBar::chunk { background: #76a1f5; border-radius: 5px; }
QStatusBar { background: #e9eff8; color: #61738c; }
QMenu { background: white; border: 1px solid #d4dfed; padding: 5px; }
QMenu::item { padding: 8px 24px 8px 14px; border-radius: 4px; }
QMenu::item:selected { background: #edf3fd; color: #1d4ed8; }
QMenu::item:disabled { color: #97a4b7; }
QMenu::separator { height: 1px; background: #e5ebf4; margin: 5px 8px; }
QToolTip { background: #243248; color: white; border: none; padding: 6px; }
"""


def apply_workspace_theme(widget):
    widget.setFont(QFont("Segoe UI", 10))
    widget.setStyleSheet(WORKSPACE_STYLE)
