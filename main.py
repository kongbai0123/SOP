"""SOP 工序監控雛形 — 啟動入口。"""
from __future__ import annotations

import logging
import sys

from sop_app import __version__
from sop_app.inference_worker import InferenceWorker
from sop_app.startup import StartupProgress, initial_model_path


def main():
    progress = StartupProgress()
    progress.start()
    worker = None
    window = None
    try:
        progress.report(10, "1 / 4 · 正在背景準備 AI，並開啟介面…")
        # AI 程序先載入 torch；介面程序只載入 Qt，兩者不爭用 GIL 或 DLL loader。
        try:
            worker = InferenceWorker(initial_model_path()).start()
        except (OSError, RuntimeError):
            logging.getLogger("sop.launcher").exception("無法啟動 AI 程序，繼續開啟介面")

        progress.report(40, "2 / 4 · 正在載入介面元件…")
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
        from sop_app.ui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("SOP 工序監控")
        app.setApplicationVersion(__version__)
        app.setStyle("Fusion")
        app.setFont(QFont("Microsoft JhengHei UI", 10))
        progress.report(65, "3 / 4 · 正在建立工作視窗…")
        window = MainWindow(defer_initial_sop=True, inference_worker=worker)
        progress.report(90, "4 / 4 · 介面就緒，準備讀取 SOP…")
        window.show()
        progress.report(100, "主介面已開啟")
        # 關閉前置視窗後才讀取 SOP，避免錯誤對話框被前置視窗遮住。
        QTimer.singleShot(0, window._open_initial_sop)
        progress.close()
        return app.exec()
    finally:
        progress.close()
        if window is not None:
            if not window.pipeline.shutdown():
                window.pipeline.wait()
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    sys.exit(main())
