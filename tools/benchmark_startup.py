r"""Measure UI readiness, actual model warmup and inference without changing settings.

Run once after a reboot for a cold-start measurement:
  .venv\Scripts\python.exe tools\benchmark_startup.py --offscreen --output logs\startup-benchmark.json
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offscreen", action="store_true", help="Do not display the test window")
    parser.add_argument("--serial", action="store_true", help="Compare with the previous torch-first startup")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--source", type=Path, help="Optional image or video to verify real detections")
    parser.add_argument("--close-after", type=float, help="Test closing while the model is still loading")
    args = parser.parse_args()
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    started = time.perf_counter()
    samples = []

    def record(stage, **details):
        entry = {"stage": stage, "seconds": round(time.perf_counter() - started, 3), **details}
        samples.append(entry)
        print(json.dumps(entry, ensure_ascii=True), flush=True)

    from sop_app.inference_worker import InferenceWorker
    from sop_app.startup import initial_model_path

    model = initial_model_path()
    if args.serial:
        import torch
        record("runtime_loaded")

        class DirectWorker:
            is_alive = False

            def load_model(self, path, progress=None):
                from sop_app.detector import create_detector
                from sop_app.model_bundle import read_model_info
                return create_detector(read_model_info(path), progress=progress)

            def close(self):
                pass

        worker = DirectWorker()
    else:
        worker = InferenceWorker(model).start()
    record("worker_started", model=str(model) if model else None, serial=args.serial)
    try:
        from PySide6.QtWidgets import QApplication
        import sop_app.ui.main_window as main_window

        app = QApplication([])
        errors = []
        with tempfile.TemporaryDirectory() as folder, ExitStack() as cleanup:
            main_window.RECORDS_DIR = Path(folder)
            window = main_window.MainWindow(defer_initial_sop=True, inference_worker=worker)

            def stop_pipeline():
                if not window.pipeline.shutdown():
                    window.pipeline.wait()

            cleanup.callback(stop_pipeline)
            window._save_settings = lambda: None
            # Capture errors instead of displaying a modal dialog in unattended measurements.
            window.pipeline.error.disconnect()
            window.pipeline.error.connect(errors.append)
            window.pipeline.model_progress.connect(lambda stage: record(
                "model_progress", message=stage, displayed_status=window.model_status.text(),
                progress_percent=window.model_progress.value(),
                progress_label=window.model_progress.text()))
            loaded = []
            window.pipeline.model_loaded.connect(lambda info, device: loaded.append((info, device)))
            window.show()
            app.processEvents()
            record("window_shown", torch_in_ui="torch" in sys.modules)
            window._open_initial_sop()
            end = started + args.timeout
            while not loaded and time.perf_counter() < end:
                app.processEvents()
                if args.close_after is not None and time.perf_counter() - started >= args.close_after:
                    break
                time.sleep(.01)
            if loaded and loaded[-1][0] is not None:
                info, device = loaded[-1]
                record("model_ready", device=device, torch_in_ui="torch" in sys.modules)
                import numpy as np

                proxy = window.pipeline._detector
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                if args.source:
                    import cv2
                    image = cv2.imread(str(args.source)) if args.source.suffix.lower() in (
                        ".png", ".jpg", ".jpeg", ".bmp") else None
                    if image is None:
                        capture = cv2.VideoCapture(str(args.source))
                        try:
                            ok, image = capture.read()
                        finally:
                            capture.release()
                        if not ok:
                            raise ValueError(f"Cannot read source: {args.source}")
                    frame = cv2.resize(image, (1920, 1080))
                timings = []
                detections = []
                for _ in range(args.frames):
                    tick = time.perf_counter()
                    detections = proxy.detect(frame)
                    timings.append(round((time.perf_counter() - tick) * 1000, 3))
                record("inference", milliseconds=timings, detections=[{
                    "label": item.label, "score": round(item.score, 6), "box": list(item.box),
                    "mask_pixels": int(np.count_nonzero(item.mask)) if item.mask is not None else None,
                    "mask_shape": list(item.mask.shape) if item.mask is not None else None,
                } for item in detections])
            elif args.close_after is None:
                record("model_failed", errors=errors or ["Model missing or load timeout"])
            tick = time.perf_counter()
            window.close()
            app.processEvents()
            record("closed", shutdown_seconds=round(time.perf_counter() - tick, 3),
                   pipeline_running=window.pipeline.isRunning(), worker_alive=worker.is_alive)
    finally:
        worker.close()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    ready = bool(loaded and loaded[-1][0] is not None)
    return 1 if errors or (not ready and args.close_after is None) else 0


if __name__ == "__main__":
    sys.exit(main())
