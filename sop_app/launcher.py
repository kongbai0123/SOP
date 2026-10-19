"""Shared source/frozen entry. Dispatch subprocesses before UI or log setup."""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
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


def _diagnose(stage, arguments=None):
    """Explicit opt-in evidence for frozen parent/child dispatch diagnostics."""
    destination = os.environ.get("SOP_LAUNCH_DIAGNOSTICS")
    if not destination:
        return
    try:
        record = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage,
                  "pid": os.getpid(), "parent_pid": os.getppid(),
                  "executable": sys.executable, "frozen": bool(getattr(sys, "frozen", False)),
                  "argv": list(sys.argv), "arguments": arguments,
                  "freeze_support_module": multiprocessing.freeze_support.__module__}
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, ValueError):
        pass


def start(argv=None):
    # PyInstaller dispatches worker/resource-tracker arguments here. This must
    # happen before opening a log or constructing another QApplication.
    _diagnose("before-freeze-support")
    multiprocessing.freeze_support()
    arguments = list(sys.argv[1:] if argv is None else argv)
    _diagnose("dispatch", arguments)
    if arguments and arguments[0] == "--sop-startup-window":
        if len(arguments) != 2:
            return 2
        from .ui.startup_window import main as splash_main
        return splash_main(Path(arguments[1]))
    if arguments and arguments[0] in ("--installer-smoke-test", "--installer-runtime-test"):
        if len(arguments) != 2:
            return 2
        from .installer_smoke import run
        if arguments[0] == "--installer-runtime-test":
            return run(Path(arguments[1]), check_ai_runtime=True)
        return run(Path(arguments[1]))
    if arguments:
        # A malformed smoke command or an undispatched worker must never open
        # a production GUI, write user settings, or start another AI worker.
        _diagnose("unsupported-arguments", arguments)
        return 2

    from .paths import LOGS_DIR, PROJECT_ROOT, initialize_runtime
    log_path = LOGS_DIR / "startup.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    handler = None
    logger = logging.getLogger("sop.launcher")
    try:
        initialize_runtime()
        handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        sys.stdout = LogStream(logger, logging.INFO)
        sys.stderr = LogStream(logger, logging.ERROR)
        logger.info("啟動 SOP 工序監控")
        _diagnose("normal-main", arguments)
        # Source launches historically use the checkout as cwd. Installed apps
        # use the per-user data root so relative library output is writable too.
        os.chdir(PROJECT_ROOT)
        from main import main
        return main()
    except Exception:
        logger.exception("啟動失敗")
        if hasattr(ctypes, "windll"):
            ctypes.windll.user32.MessageBoxW(
                None, f"SOP 啟動失敗，請查看日誌：\n{log_path}", "SOP 工序監控", 0x10)
        return 1
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
