"""Opt-in installer verification without camera, model, or production data.

Kept import-light because a multiprocessing child imports its target module.
The installed GUI executable can be checked with:
``SOP.exe --installer-smoke-test C:\\Temp\\sop-smoke.json``.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


def _smoke_worker(connection, cancelled):
    """Exercise the application's real spawn/pipe lifecycle without PyTorch."""
    try:
        connection.send({"kind": "result", "id": 0, "result": {
            "pid": os.getpid(), "frozen": bool(getattr(sys, "frozen", False)),
            "executable": sys.executable}})
        cancelled.wait(60)
    finally:
        connection.close()


def _smoke_ai_worker(connection, cancelled):
    """Check packaged native AI dependencies on CPU, without any model weights."""
    from .inference_worker import _error_record
    from .launcher import _diagnose

    try:
        # Ultralytics may initialize settings on import. Keep that incidental
        # output isolated just like the UI's settings and SQLite records.
        with tempfile.TemporaryDirectory(prefix="sop-runtime-smoke-") as temporary:
            os.environ["YOLO_CONFIG_DIR"] = temporary
            os.environ["YOLO_AUTOINSTALL"] = "false"
            _diagnose("ai-before-torch-import")
            import torch
            _diagnose("ai-after-torch-import")
            _diagnose("ai-before-torchvision-import")
            import torchvision
            _diagnose("ai-after-torchvision-import")
            _diagnose("ai-before-ultralytics-import")
            import ultralytics
            _diagnose("ai-after-ultralytics-import")

            _diagnose("ai-before-cpu-checks")
            tensor = torch.tensor([1.0, 2.0], device="cpu")
            boxes = torch.tensor([[0., 0., 2., 2.], [1., 1., 3., 3.]], device="cpu")
            keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
            _diagnose("ai-after-cpu-checks")
            connection.send({"kind": "result", "id": 0, "result": {
                "pid": os.getpid(), "frozen": bool(getattr(sys, "frozen", False)),
                "executable": sys.executable,
                "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                            "ultralytics": ultralytics.__version__},
                "checks": {"torch_cpu": tensor.sum().item() == 3.0,
                           "torchvision_nms": keep.tolist() == [0, 1],
                           "ultralytics": bool(ultralytics.__version__)}}})
            cancelled.wait(60)
    except Exception as exc:
        _diagnose("ai-runtime-error", [type(exc).__name__, str(exc)])
        connection.send({"kind": "fatal", "id": None, "error": _error_record(exc)})
    finally:
        connection.close()


@contextmanager
def _isolated_data_root(root):
    from . import paths

    names = ("DATA_ROOT", "PROJECT_ROOT", "SOPS_DIR", "RECORDS_DIR", "LOGS_DIR", "APP_SETTINGS")
    original = {name: getattr(paths, name) for name in names}
    try:
        paths.DATA_ROOT = paths.PROJECT_ROOT = root
        paths.SOPS_DIR = root / "sops"
        paths.RECORDS_DIR = root / "records"
        paths.LOGS_DIR = root / "logs"
        paths.APP_SETTINGS = root / "app_settings.json"
        for directory in (paths.SOPS_DIR, paths.RECORDS_DIR, paths.LOGS_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        yield
    finally:
        for name, value in original.items():
            setattr(paths, name, value)


def run(report_path: Path, *, check_ai_runtime=False) -> int:
    """Write machine-readable evidence and return a failing exit code on error."""
    from . import __version__, paths

    report_path = Path(report_path).absolute()
    report = {"ok": False, "application_version": __version__,
              "frozen": bool(getattr(sys, "frozen", False)),
              "ai_runtime_test": check_ai_runtime,
              "paths": {"install": str(paths.INSTALL_ROOT),
                        "resources": str(paths.RESOURCE_ROOT),
                        "user_data": str(paths.DATA_ROOT)},
              "checks": {}, "camera_opened": False, "model_loaded": False}
    worker, window = None, None
    try:
        readiness_timeout = 45.0
        if check_ai_runtime:
            readiness_timeout = float(os.environ.get("SOP_SMOKE_TIMEOUT", "180"))
            if not 45 <= readiness_timeout <= 600:
                raise ValueError("SOP_SMOKE_TIMEOUT must be between 45 and 600 seconds")
        report["worker_ready_timeout_sec"] = readiness_timeout
        with tempfile.TemporaryDirectory(prefix="sop-installer-smoke-") as temporary:
            root = Path(temporary)
            report["paths"]["smoke_data"] = str(root)
            # Imports below bind data locations only after isolation is active.
            with _isolated_data_root(root):
                from .inference_worker import InferenceWorker
                worker = InferenceWorker(worker_target=_smoke_ai_worker if check_ai_runtime else _smoke_worker).start()
                if not worker._connection.poll(readiness_timeout):
                    raise TimeoutError(f"Installer smoke worker did not report ready within {readiness_timeout:g} seconds")
                child = worker._receive(0)
                report["worker"] = child
                report["checks"]["worker_spawn"] = child["pid"] != os.getpid() and worker.is_alive
                report["checks"].update(child.get("checks", {}))
                if report["frozen"]:
                    report["checks"]["worker_frozen"] = child["frozen"]

                from .native_runtime import cv2
                import numpy as np
                native = cv2.require()
                image = native.cvtColor(np.zeros((8, 8, 3), dtype=np.uint8), native.COLOR_BGR2GRAY)
                report["checks"]["opencv"] = image.shape == (8, 8)
                report["opencv"] = {"version": native.__version__, "module": str(native.__file__)}

                from PySide6.QtCore import QLibraryInfo, QTimer, qVersion
                from PySide6.QtGui import QFont
                from PySide6.QtWidgets import QApplication
                from .ui.main_window import MainWindow

                app = QApplication.instance() or QApplication(["SOP installer verification"])
                app.setStyle("Fusion")
                app.setFont(QFont("Microsoft JhengHei UI", 10))
                window = MainWindow(defer_initial_sop=True, inference_worker=worker)
                window.show()
                report["qt"] = {"version": qVersion(), "platform": app.platformName(),
                                "plugins": QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)}

                def finish():
                    report["checks"]["main_window"] = window.isVisible() and window.tabs.count() == 3
                    report["checks"]["temporary_records"] = window.pipeline._records_dir == root / "records"
                    report["camera_opened"] = window.pipeline._source is not None
                    report["model_loaded"] = window.pipeline._detector is not None
                    report["checks"]["no_camera_or_model"] = not (report["camera_opened"] or report["model_loaded"])
                    report["checks"]["qt"] = bool(app.platformName())
                    window.close()
                    app.quit()

                QTimer.singleShot(500, finish)
                app.exec()
                if not window.pipeline.shutdown():
                    raise RuntimeError("Installer smoke pipeline did not stop")
                worker.close()
                report["checks"]["worker_stopped"] = not worker.is_alive
                report["ok"] = bool(report["checks"]) and all(report["checks"].values())
                window = None
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if window is not None:
            window.pipeline.shutdown()
        if worker is not None:
            worker.close()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1
