"""Desktop launcher, also used by the project-1 console command."""

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # Windows redirected consoles may default to cp1252, which cannot print Vietnamese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Khôi phục ảnh cũ bằng Gaussian và Inpainting.")
    parser.add_argument("--demo", action="store_true", help="Nạp bộ ảnh mẫu và chạy cả ba phương pháp.")
    parser.add_argument("--dataset-dir", type=Path, default=Path.cwd() / "dataset")
    args = parser.parse_args(argv)

    from PySide6.QtWidgets import QApplication
    from project_1.gui.main_window import MainWindow

    app = QApplication([sys.argv[0]])
    app.setStyle("Fusion")
    window = MainWindow(dataset_dir=args.dataset_dir)
    window.show()
    if args.demo:
        window.load_demo()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
