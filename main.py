"""SOP 工序監控雛形 — 啟動入口。"""
from __future__ import annotations

import sys

try:
    import torch  # noqa: F401  先於 PySide6 載入，避免 Windows 上 DLL 載入順序衝突
except (ImportError, OSError) as exc:
    # Windows 應用程式控制可能封鎖模型 DLL；編輯與 OpenCV 定位仍可使用。
    # 模型載入時由背景管線顯示可讀的錯誤，避免整個介面啟動失敗。
    print(f"[WARNING] PyTorch unavailable; starting editor without AI inference: {exc}", file=sys.stderr)

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from sop_app.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SOP 工序監控")
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft JhengHei UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
