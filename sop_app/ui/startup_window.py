"""Independent splash process so torch can still load before Qt in the main process."""
import queue
import sys
import threading
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget

from ..startup import read_progress_messages


class StartupWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOP 工序監控 · 啟動中")
        self.setFixedSize(520, 270)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("QWidget { background: #f6f8fb; color: #172b42; }"
                           "QProgressBar { border: 0; border-radius: 5px; background: #dde5ee; height: 10px; }"
                           "QProgressBar::chunk { background: #2879d0; border-radius: 5px; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        title = QLabel("SOP 工序監控")
        title.setFont(QFont("Microsoft JhengHei UI", 20, QFont.Weight.Bold))
        layout.addWidget(title)
        layout.addWidget(QLabel("正在準備工作環境"))
        layout.addStretch()
        self.message = QLabel("正在啟動…")
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.elapsed = QLabel("已等待 0.0 秒 · 階段進度")
        layout.addWidget(self.elapsed)
        note = QLabel("主介面開啟後，AI 模型會繼續在背景載入。")
        note.setStyleSheet("color: #52677c;")
        layout.addWidget(note)
        self._started = time.monotonic()
        self._messages = queue.Queue()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)

    def read_messages(self):
        try:
            for message in read_progress_messages(sys.stdin.buffer):
                self._messages.put(message)
        finally:
            self._messages.put({"close": True})

    def poll(self):
        while not self._messages.empty():
            data = self._messages.get_nowait()
            if data.get("close"):
                self.close()
                return
            self.progress.setValue(data["progress"])
            self.message.setText(data["message"])
            self._started = time.monotonic() - data["elapsed"]
        self.elapsed.setText(f"已等待 {time.monotonic() - self._started:.1f} 秒 · 階段進度")


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft JhengHei UI", 10))
    window = StartupWindow()
    threading.Thread(target=window.read_messages, daemon=True).start()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
