"""Startup status transport. Keep this module free of Qt and torch imports."""
import json
import logging
from pathlib import Path
import subprocess
import sys
import time


def read_progress_messages(stream):
    """Decode the binary pipe explicitly; Windows stdin may default to CP950."""
    for line in stream:
        try:
            yield json.loads(line.decode("utf-8"))
        except (UnicodeError, ValueError):
            continue


class StartupProgress:
    def __init__(self):
        self.started = time.perf_counter()
        self.process = None

    def start(self):
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "sop_app.ui.startup_window"],
                cwd=Path(__file__).resolve().parent.parent,
                stdin=subprocess.PIPE, text=True, encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            logging.getLogger("sop.launcher").exception("無法顯示啟動進度視窗")

    def report(self, progress, message):
        elapsed = time.perf_counter() - self.started
        logging.getLogger("sop.launcher").info("啟動 %.2fs · %s", elapsed, message)
        self._send({"progress": progress, "message": message, "elapsed": elapsed})

    def _send(self, data):
        if self.process is None or self.process.stdin is None:
            return
        try:
            self.process.stdin.write(json.dumps(data, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError):
            pass  # 關閉前置視窗不影響主程式啟動。

    def close(self):
        self._send({"close": True})
        if self.process is not None and self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
