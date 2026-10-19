"""JPEG evidence reuse must preserve durable acceptance and replay semantics."""
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from sop_app.engine import EngineEvent
from sop_app.native_runtime import cv2
from sop_app.recorder import AsyncRecorder, Recorder, query


def evidence_events():
    return [EngineEvent("cycle_started", 0, 1),
            EngineEvent("alarm", 1, 1, 0, "A", "violation", {"alarm": "violation"}),
            EngineEvent("step_completed", 2, 1, 0, "A", data={"status": "done", "alarm": True}),
            EngineEvent("cycle_completed", 2, 1, data={"result": "NG"})]


class RecorderEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "records"
        self.frame = np.random.default_rng(29).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        self.jpeg = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def make_pending(self, image=None):
        """Produce the pre-optimization schema/payload, including no image hash."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload = dict(command="handle", session="legacy-session", at="2026-10-19 12:00:00",
                       events=[asdict(event) for event in evidence_events()], sop_name="test",
                       sop_version=1, model_version="m", context={}, sop_snapshot={"name": "test"})
        with closing(sqlite3.connect(self.root / "pending.db")) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "job_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, image BLOB)")
            db.execute("INSERT INTO jobs(job_key,payload,image) VALUES (?,?,?)",
                       ("legacy-job", json.dumps(payload), self.jpeg if image is None else image))
            db.commit()

    def writer(self, **kwargs):
        writer = AsyncRecorder(self.root, **kwargs)
        self.addCleanup(writer.close)
        return writer

    def assert_evidence(self, root=None):
        root = root or self.root
        steps = query(root / "records.db", "SELECT snapshot FROM step_records")
        alarms = query(root / "records.db", "SELECT snapshot FROM alarms")
        self.assertEqual(len(steps), 1)
        self.assertEqual(len(alarms), 1)
        for name, in steps + alarms:
            self.assertEqual((root / "snapshots" / name).read_bytes(), self.jpeg)
        self.assertEqual(query(root / "records.db", "SELECT result FROM cycles"), [("NG",)])

    def test_async_encodes_once_and_saves_identical_bytes_for_alarm_and_step(self):
        writer = self.writer()
        original_encode, original_decode = cv2.imencode, cv2.imdecode
        with patch("sop_app.recorder.cv2.imencode", wraps=original_encode) as encode, \
                patch("sop_app.recorder.cv2.imdecode", wraps=original_decode) as decode, \
                patch("sop_app.recorder.os.fsync", wraps=__import__("os").fsync) as fsync:
            writer.handle(evidence_events(), "test", 1, "m", frame=self.frame)
            self.assertTrue(writer.close())
            self.assertEqual(encode.call_count, 1)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(fsync.call_count, 2)
        self.assert_evidence()
        self.assertEqual(query(self.root / "pending.db", "SELECT COUNT(*) FROM jobs"), [(0,)])
        enqueue, background = writer.last_enqueue_metrics, writer.last_write_metrics
        self.assertTrue({"encode_ms", "pending_commit_ms", "total_ms"} <= enqueue.keys())
        self.assertTrue({"decode_ms", "formal_write_ms", "pending_delete_ms", "total_ms"} <= background.keys())
        self.assertTrue(all(value >= 0 for value in (*enqueue.values(), *background.values())))
        enqueue["encode_ms"] = -1
        self.assertGreaterEqual(writer.last_enqueue_metrics["encode_ms"], 0)

    def test_legacy_pending_replays_without_reencoding_and_remains_deduplicated(self):
        self.make_pending()
        with patch("sop_app.recorder.cv2.imencode", side_effect=AssertionError("must reuse JPEG")):
            writer = self.writer()
            self.assertTrue(writer.close())
        self.assert_evidence()
        # Simulate a crash after formal commit but before pending deletion.
        with closing(sqlite3.connect(self.root / "pending.db")) as db:
            db.execute("INSERT INTO jobs(job_key,payload,image) VALUES (?,?,?)", (
                "legacy-job", json.dumps(dict(command="handle", session="legacy-session",
                at="2026-10-19 12:00:00", events=[asdict(event) for event in evidence_events()],
                sop_name="test", sop_version=1, model_version="m", context={}, sop_snapshot={"name": "test"})), self.jpeg))
            db.commit()
        writer = self.writer()
        self.assertTrue(writer.close())
        self.assert_evidence()
        self.assertEqual(query(self.root / "records.db", "SELECT COUNT(*) FROM applied_jobs"), [(1,)])

    def test_corrupt_legacy_jpeg_remains_pending_without_formal_acceptance(self):
        self.make_pending(b"not a JPEG")
        errors = []
        writer = self.writer(on_error=errors.append)
        self.assertTrue(writer.close())
        self.assertTrue(any("截圖損毀" in message for message in errors))
        self.assertEqual(query(self.root / "pending.db", "SELECT image FROM jobs"), [(b"not a JPEG",)])
        self.assertEqual(query(self.root / "records.db", "SELECT COUNT(*) FROM applied_jobs"), [(0,)])
        self.assertEqual(query(self.root / "records.db", "SELECT COUNT(*) FROM cycles"), [(0,)])

    def test_failed_fsync_retains_pending_original_and_recovers_on_restart(self):
        failed = threading.Event()
        writer = self.writer(on_error=lambda _: failed.set())
        with patch("sop_app.recorder.os.fsync", side_effect=OSError("disk full")):
            writer.handle(evidence_events(), "test", 1, "m", frame=self.frame)
            self.assertTrue(failed.wait(5))
            self.assertTrue(writer.close())
        self.assertEqual(query(self.root / "pending.db", "SELECT image FROM jobs"), [(self.jpeg,)])
        self.assertEqual(query(self.root / "records.db", "SELECT COUNT(*) FROM applied_jobs"), [(0,)])
        writer = self.writer()
        self.assertTrue(writer.close())
        self.assert_evidence()
        self.assertEqual(query(self.root / "pending.db", "SELECT COUNT(*) FROM jobs"), [(0,)])

    def test_enqueue_returns_only_after_pending_full_commit_even_if_background_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        original = Recorder.handle
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release background writer")
            return original(*args, **kwargs)
        writer = self.writer()
        try:
            with patch.object(Recorder, "handle", new=blocked):
                writer.handle(evidence_events(), "test", 1, "m", frame=self.frame)
                self.assertTrue(entered.wait(5))
                # A separate connection sees the event and exact JPEG after the
                # caller returned, even though formal writing has not begun.
                self.assertEqual(query(self.root / "pending.db", "SELECT image FROM jobs"), [(self.jpeg,)])
                with writer._lock:
                    self.assertEqual(writer._journal.execute("PRAGMA synchronous").fetchone()[0], 2)
                release.set()
                self.assertTrue(writer.close())
        finally:
            release.set()
            writer.close()
        self.assert_evidence()

    def test_direct_recorder_frame_api_remains_supported(self):
        recorder = Recorder(self.root)
        try:
            with patch("sop_app.recorder.cv2.imencode", wraps=cv2.imencode) as encode:
                recorder.handle(evidence_events(), "test", 1, "m", frame=self.frame)
                self.assertEqual(encode.call_count, 2)
            self.assertEqual(recorder.db.execute("PRAGMA synchronous").fetchone()[0], 2)
        finally:
            recorder.close()
        self.assert_evidence()

    def test_process_exit_before_or_after_commit_replays_exact_evidence_once(self):
        script = '''
import os, sys, time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sop_app.engine import EngineEvent
from sop_app.recorder import AsyncRecorder, Recorder
root, stage = Path(sys.argv[1]), sys.argv[2]
writer = AsyncRecorder(root)
original = Recorder.handle
def crash(*args, **kwargs):
    if stage == "after":
        original(*args, **kwargs)
    os._exit(91)
frame = np.random.default_rng(29).integers(0, 256, (48, 64, 3), dtype=np.uint8)
events = [EngineEvent("cycle_started", 0, 1),
          EngineEvent("alarm", 1, 1, 0, "A", "violation", {"alarm":"violation"}),
          EngineEvent("step_completed", 2, 1, 0, "A", data={"status":"done","alarm":True}),
          EngineEvent("cycle_completed", 2, 1, data={"result":"NG"})]
with patch.object(Recorder, "handle", new=crash):
    writer.handle(events, "test", 1, "m", frame=frame)
    time.sleep(10)
'''
        for stage in ("before", "after"):
            with self.subTest(stage=stage):
                root = self.root.parent / stage / "records"
                completed = subprocess.run([sys.executable, "-c", script, str(root), stage],
                                           capture_output=True, timeout=15)
                self.assertEqual(completed.returncode, 91, completed.stderr)
                self.assertEqual(query(root / "pending.db", "SELECT image FROM jobs"), [(self.jpeg,)])
                writer = AsyncRecorder(root)
                self.addCleanup(writer.close)
                self.assertTrue(writer.close())
                self.assert_evidence(root)
                self.assertEqual(query(root / "records.db", "SELECT COUNT(*) FROM applied_jobs"), [(1,)])
                self.assertEqual(query(root / "pending.db", "SELECT COUNT(*) FROM jobs"), [(0,)])


if __name__ == "__main__":
    unittest.main()
