"""無命令列視窗的 Windows 啟動入口，啟動前先接好日誌。"""
from pathlib import Path
import ctypes
import logging
from logging.handlers import RotatingFileHandler
import os
import runpy
import sys


class LogStream:
    def __init__(self, logger, level):
        self.logger, self.level = logger, level
        self.encoding = "utf-8"

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.rstrip())
        return len(message)

    def flush(self):
        pass

    def isatty(self):
        return False


def start():
    root = Path(__file__).resolve().parent
    log_path = root / "logs" / "startup.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger = logging.getLogger("sop.launcher")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        sys.stdout = LogStream(logger, logging.INFO)
        sys.stderr = LogStream(logger, logging.ERROR)
        logger.info("啟動 SOP 工序監控")
        os.chdir(root)
        sys.path.insert(0, str(root))
        runpy.run_path(str(root / "main.py"), run_name="__main__")
    except Exception:
        logging.getLogger("sop.launcher").exception("啟動失敗")
        ctypes.windll.user32.MessageBoxW(
            None, f"SOP 啟動失敗，請查看日誌：\n{log_path}", "SOP 工序監控", 0x10)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(start())
