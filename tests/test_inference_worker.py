"""Spawn real lightweight children; these tests never import PyTorch or Qt."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from sop_app.inference_worker import (InferenceWorker, InferenceWorkerClosed,
                                      InferenceWorkerError, _serve)


class FakeDetector:
    def __init__(self, path, loads):
        self.info = SimpleNamespace(source=path, model_version_id=path.name,
                                    load_count=loads, process_id=os.getpid())
        self.classes = ("part", "hand")
        self.device_name = "test CPU"

    def detect(self, frame):
        import numpy as np

        if frame[0, 0, 0] == 255:
            raise ValueError("invalid fake frame")
        height, width = frame.shape[:2]
        mask = frame[:, :, 0] % 2 == 1
        return [SimpleNamespace(label="part", score=float(frame[0, 0, 0]) / 100,
                                box=(0, 0, width, height), mask=mask),
                SimpleNamespace(label="hand", score=0.8, box=(1, 1, width, height), mask=None)]


def fake_worker(connection, cancelled):
    loads = 0

    def load(path, report):
        nonlocal loads
        loads += 1
        report(f"loading {path.name}")
        if path.name == "crash":
            os._exit(17)
        if path.name == "hang":
            while True:
                time.sleep(0.1)  # Simulate a library call that cannot be cancelled.
        if path.name == "error":
            raise ValueError("fake model rejected")
        if path.name == "blocked":
            error = OSError("Windows blocked fake DLL")
            error.winerror = 4551
            raise error
        report("warmup complete")
        return FakeDetector(path, loads)

    _serve(connection, cancelled, load)


def failed_bootstrap_worker(connection, cancelled):
    def bootstrap(report):
        report("importing runtime")
        error = OSError("runtime DLL denied")
        error.winerror = 4551
        raise RuntimeError("runtime import failed") from error

    _serve(connection, cancelled, lambda path, report: None, bootstrap=bootstrap)


class InferenceWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def worker(self, preload=None, target=fake_worker):
        worker = InferenceWorker(preload, worker_target=target)
        self.addCleanup(worker.close)
        return worker.start()

    def test_client_import_does_not_load_numeric_or_gui_runtime(self):
        script = ("import sys; import sop_app.inference_worker; "
                  "assert not any(name in sys.modules for name in "
                  "('torch', 'numpy', 'PySide6', 'sop_app.detector'))")
        subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                       check=True, capture_output=True, timeout=10)

    def test_preload_is_reused_for_normalized_path_and_reports_progress(self):
        path = self.root / "model"
        worker = self.worker(path)
        messages = []
        detector = worker.load_model(path.parent / "." / path.name, messages.append)
        self.assertEqual(detector.info.load_count, 1)
        self.assertEqual(detector.info.process_id, worker.pid)
        self.assertEqual(detector.classes, ("part", "hand"))
        self.assertEqual(detector.device_name, "test CPU")
        self.assertEqual(messages, ["loading model", "warmup complete", "載入完成"])
        self.assertTrue(all(message.mode == "startup" for message in messages))
        self.assertLessEqual(messages[0].measured_at, messages[-1].measured_at)
        self.assertIs(worker.start(), worker)

    def test_different_model_uses_same_child_and_does_not_return_preload(self):
        worker = self.worker(self.root / "initial")
        messages = []
        changed = worker.load_model(self.root / "changed", messages.append)
        self.assertEqual(changed.info.model_version_id, "changed")
        self.assertEqual(changed.info.load_count, 2)
        self.assertEqual(changed.info.process_id, worker.pid)
        self.assertEqual(messages, ["loading changed", "warmup complete", "載入完成"])
        self.assertTrue(all(message.mode == "reload" for message in messages))
        third = worker.load_model(self.root / "third")
        self.assertEqual(third.info.process_id, worker.pid)
        self.assertEqual(third.info.load_count, 3)

    def test_detection_roundtrip_preserves_noncontiguous_frame_and_masks(self):
        import numpy as np

        worker = self.worker()
        detector = worker.load_model(self.root / "model")
        frame = np.arange(3 * 10 * 3, dtype=np.uint8).reshape(3, 10, 3)[:, ::2]
        results = detector.detect(frame)
        self.assertEqual((results[0].label, results[0].box), ("part", (0, 0, 5, 3)))
        self.assertEqual(results[0].score, float(frame[0, 0, 0]) / 100)
        np.testing.assert_array_equal(results[0].mask, frame[:, :, 0] % 2 == 1)
        self.assertEqual(results[0].mask.dtype, np.bool_)
        self.assertIsNone(results[1].mask)
        # A larger next image checks buffer resizing and successive requests.
        larger = np.full((7, 9, 3), 11, dtype=np.uint8)
        np.testing.assert_array_equal(detector.detect(larger)[0].mask, np.ones((7, 9), dtype=bool))
        self.assertTrue(worker.is_alive)

    def test_old_proxy_cannot_silently_infer_with_replacement_model(self):
        import numpy as np

        worker = self.worker()
        old = worker.load_model(self.root / "old")
        current = worker.load_model(self.root / "current")
        frame = np.zeros((3, 5, 3), dtype=np.uint8)
        with self.assertRaisesRegex(InferenceWorkerError, "模型已變更"):
            old.detect(frame)
        self.assertEqual(len(current.detect(frame)), 2)

    def test_remote_model_and_detection_errors_are_recoverable(self):
        import numpy as np

        worker = self.worker()
        with self.assertRaises(InferenceWorkerError) as caught:
            worker.load_model(self.root / "error")
        error = caught.exception
        self.assertEqual(error.remote_type, "ValueError")
        self.assertIn("fake model rejected", str(error))
        self.assertIn("raise ValueError", error.remote_traceback)
        detector = worker.load_model(self.root / "valid")
        with self.assertRaises(InferenceWorkerError) as caught:
            detector.detect(np.full((3, 5, 3), 255, dtype=np.uint8))
        self.assertEqual(caught.exception.remote_type, "ValueError")
        self.assertTrue(worker.is_alive)
        self.assertEqual(len(detector.detect(np.zeros((3, 5, 3), dtype=np.uint8))), 2)

    def test_windows_error_4551_survives_rpc(self):
        worker = self.worker(self.root / "blocked")
        with self.assertRaises(InferenceWorkerError) as caught:
            worker.load_model(self.root / "blocked")
        self.assertIsInstance(caught.exception, OSError)
        self.assertEqual(caught.exception.winerror, 4551)
        self.assertEqual(caught.exception.remote_type, "OSError")
        self.assertIn("Windows blocked fake DLL", caught.exception.remote_traceback)

    def test_runtime_failure_preserves_nested_windows_error_and_full_traceback(self):
        worker = self.worker(target=failed_bootstrap_worker)
        messages = []
        with self.assertRaises(InferenceWorkerError) as caught:
            worker.load_model(self.root / "model", messages.append)
        self.assertEqual(messages, ["importing runtime"])
        self.assertEqual(caught.exception.winerror, 4551)
        self.assertEqual(caught.exception.remote_type, "RuntimeError")
        self.assertIn("runtime DLL denied", caught.exception.remote_traceback)
        self.assertIn("runtime import failed", caught.exception.remote_traceback)

    def test_shutdown_cancels_pending_load_and_reaps_unresponsive_child(self):
        worker = self.worker()
        started = threading.Event()
        errors = []

        def load():
            try:
                worker.load_model(self.root / "hang", lambda message: started.set())
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=load)
        thread.start()
        self.assertTrue(started.wait(10), "child did not start its fake load")
        before = time.monotonic()
        worker.close()
        elapsed = time.monotonic() - before
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertFalse(worker.is_alive)
        self.assertLess(elapsed, 2)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InferenceWorkerClosed)
        worker.close()
        with self.assertRaises(InferenceWorkerClosed):
            worker.load_model(self.root / "another")

    def test_crashed_child_does_not_leave_a_hanging_rpc(self):
        worker = self.worker()
        before = time.monotonic()
        with self.assertRaises(InferenceWorkerError) as caught:
            worker.load_model(self.root / "crash")
        self.assertLess(time.monotonic() - before, 10)
        self.assertEqual(caught.exception.remote_type, "WorkerExited")
        self.assertFalse(worker.is_alive)
        worker.close()

    def test_closed_worker_cannot_be_started_again(self):
        worker = InferenceWorker(worker_target=fake_worker)
        worker.close()
        worker.close()
        with self.assertRaises(InferenceWorkerClosed):
            worker.start()


if __name__ == "__main__":
    unittest.main()
