import sys
from pathlib import Path

# Thêm thư mục src vào Python Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from project_1.gui.main_window import MainWindow
from PySide6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()