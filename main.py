"""SOP 工序監控雛形 — 啟動入口。"""
from __future__ import annotations

import sys

from sop_app.startup import StartupProgress


def main():
    progress = StartupProgress()
    progress.start()
    try:
        progress.report(10, "1 / 4 · 正在載入 AI 執行環境…")
        try:
            import torch  # noqa: F401  保留 Windows 的 torch → Qt 載入順序
        except (ImportError, OSError) as exc:
            print(f"[WARNING] PyTorch unavailable: {exc}", file=sys.stderr)
            progress.report(30, "AI 執行環境暫不可用，繼續開啟介面…")

        progress.report(40, "2 / 4 · 正在載入介面元件…")
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
        from sop_app.ui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("SOP 工序監控")
        app.setStyle("Fusion")
        app.setFont(QFont("Microsoft JhengHei UI", 10))
        progress.report(65, "3 / 4 · 正在建立工作視窗…")
        window = MainWindow(defer_initial_sop=True)
        progress.report(90, "4 / 4 · 介面就緒，準備讀取 SOP…")
        window.show()
        progress.report(100, "主介面已開啟")
        # 關閉前置視窗後才讀取 SOP，避免錯誤對話框被前置視窗遮住。
        QTimer.singleShot(0, window._open_initial_sop)
    finally:
        progress.close()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
