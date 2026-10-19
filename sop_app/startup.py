"""Startup status transport. Keep this module free of Qt and torch imports."""
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .paths import APP_SETTINGS, SOPS_DIR, resolve


def initial_sop_path(settings, root=None) -> Path | None:
    """Choose the same initial SOP for early model preload and the main window."""
    last = settings.get("last_sop") if isinstance(settings, dict) else None
    if isinstance(last, str) and last.strip():
        try:
            path = (Path(last) if Path(last).is_absolute() else Path(root) / last) if root is not None else resolve(last)
            allowed = root is None or path.resolve().is_relative_to((Path(root) / "sops").resolve())
            if allowed and path.exists():
                return path
        except (OSError, ValueError):
            pass
    directory = Path(root) / "sops" if root is not None else SOPS_DIR
    example = directory / "example_sop.json"
    try:
        if example.exists():
            return example
        return next(iter(sorted(directory.glob("*.json"))), None) if root is not None else None
    except OSError:
        return None


def initial_model_path(root=None) -> Path | None:
    """Find an existing saved model without importing any UI or inference runtime.

    Invalid initial SOP content is left for the normal UI error handling; it
    must not cause an unrelated example model to preload instead.
    """
    try:
        settings_path = Path(root) / "app_settings.json" if root is not None else APP_SETTINGS
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    sop_path = initial_sop_path(settings, root=root) if root is not None else initial_sop_path(settings)
    if sop_path is None:
        return None
    try:
        sop = json.loads(sop_path.read_text(encoding="utf-8"))
        model = sop.get("model_path") if isinstance(sop, dict) else None
        if not isinstance(model, str) or not model.strip():
            return None
        path = (Path(model) if Path(model).is_absolute() else Path(root) / model) if root is not None else resolve(model)
        return path if path.exists() else None
    except (OSError, ValueError):
        return None


def read_progress_messages(stream):
    """Decode the binary pipe explicitly; Windows stdin may default to CP950."""
    for line in stream:
        try:
            yield json.loads(line.decode("utf-8"))
        except (UnicodeError, ValueError):
            continue


def read_progress_file(path, timeout=120.0):
    """Windowed frozen apps have no stdin; consume a private UTF-8 JSONL file.

    A timeout closes an orphaned splash if its parent exits without a packet.
    The reader owns cleanup, including after a normal close packet.
    """
    path = Path(path)
    try:
        with path.open("rb") as stream:
            last_packet = time.monotonic()
            while time.monotonic() - last_packet < timeout:
                line = stream.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                messages = list(read_progress_messages([line]))
                if messages:
                    last_packet = time.monotonic()
                    yield messages[0]
                    if messages[0].get("close"):
                        return
    except OSError:
        return
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


class StartupProgress:
    def __init__(self):
        self.started = time.perf_counter()
        self.process = None
        self.status_file = None
        self._closed = False

    def start(self):
        try:
            if getattr(sys, "frozen", False):
                with tempfile.NamedTemporaryFile(prefix="sop-startup-", suffix=".jsonl", delete=False) as stream:
                    self.status_file = Path(stream.name)
                self.process = subprocess.Popen(
                    [sys.executable, "--sop-startup-window", str(self.status_file)],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "sop_app.ui.startup_window"],
                    cwd=Path(__file__).resolve().parent.parent,
                    stdin=subprocess.PIPE, text=True, encoding="utf-8",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError:
            if self.status_file is not None:
                try:
                    self.status_file.unlink(missing_ok=True)
                except OSError:
                    pass
                self.status_file = None
            logging.getLogger("sop.launcher").exception("無法顯示啟動進度視窗")

    def report(self, progress, message):
        elapsed = time.perf_counter() - self.started
        logging.getLogger("sop.launcher").info("啟動 %.2fs · %s", elapsed, message)
        self._send({"progress": progress, "message": message, "elapsed": elapsed})

    def _send(self, data):
        if self._closed:
            return
        if self.status_file is not None:
            try:
                with self.status_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(data, ensure_ascii=False) + "\n")
            except OSError:
                pass
            return
        if self.process is None or self.process.stdin is None:
            return
        try:
            self.process.stdin.write(json.dumps(data, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError):
            pass  # 關閉前置視窗不影響主程式啟動。

    def close(self):
        if self._closed:
            return
        self._send({"close": True})
        self._closed = True
        if self.process is not None and self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
