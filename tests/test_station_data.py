"""Station data durability and restore fault injection; uses temporary data only."""
import hashlib
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

from sop_app.engine import EngineEvent, SOPEngine, Phase
from sop_app.recorder import AsyncRecorder, Recorder, connect, query, read_sop_snapshot
from sop_app.sop_schema import SOPDefinition, Step, Condition, ROI, Workpiece, save_sop
from sop_app.station_backup import create_backup, inspect_backup, restore_backup, recover_restore, MARKER
from sop_app.pipeline import VideoPipeline
from sop_app.ui.records_page import RecordsPage


def canonical(value):
    return json.loads(json.dumps(value))


def events(cycle=1):
    return [EngineEvent("cycle_started", 1, cycle),
            EngineEvent("step_completed", 2, cycle, 0, "A", data={"status": "done", "manual": True, "duration": 1}),
            EngineEvent("cycle_completed", 2, cycle, data={"result": "OK", "duration": 1})]


def wait_for(predicate):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Background operation did not finish")


class StationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = self.root / "records"
        self.sop = SOPDefinition(name="工站", version=8, steps=[Step("A")],
                                 rois=[ROI("工作區", [(0, 0), (1, 0), (1, 1)])],
                                 workpiece=Workpiece(reference_png="reference-png"))

    def seed(self):
        recorder = Recorder(self.records)
        recorder.handle(events(), self.sop.name, self.sop.version, "model-1",
                        np.zeros((10, 20, 3), dtype=np.uint8), {"operator": "甲"}, self.sop.to_dict())
        recorder.close()
        save_sop(self.sop, self.root / "sops/test.json")
        (self.root / "app_settings.json").write_text('{"station":"A1"}', encoding="utf-8")

    def test_snapshot_binds_actual_content_even_when_version_is_unchanged(self):
        recorder = Recorder(self.records)
        self.addCleanup(recorder.close)
        original = self.sop.to_dict()
        recorder.handle(events(1), "工站", 8, "m", sop_snapshot=original)
        self.sop.steps[0].hold_sec = 9
        recorder.handle(events(2), "工站", 8, "m", sop_snapshot=self.sop.to_dict())
        first, first_hash = read_sop_snapshot(self.records / "records.db", 1)
        second, second_hash = read_sop_snapshot(self.records / "records.db", 2)
        self.assertEqual(first, canonical(original))
        self.assertEqual(second, canonical(self.sop.to_dict()))
        self.assertEqual(first["version"], second["version"])
        self.assertNotEqual(first_hash, second_hash)
        recorder.db.execute("UPDATE sop_snapshots SET content='{}' WHERE hash=?", (first_hash,))
        recorder.db.commit()
        with self.assertRaisesRegex(ValueError, "完整性"):
            read_sop_snapshot(self.records / "records.db", 1)

    def test_atomic_batch_and_retry_do_not_duplicate_records(self):
        recorder = Recorder(self.records)
        self.addCleanup(recorder.close)
        args = (events(), "test", 1, "m", np.zeros((5, 5, 3), np.uint8))
        with patch.object(recorder, "_save_snapshot", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                recorder.handle(*args, sop_snapshot=self.sop.to_dict(), job_key="job", session_id="run")
        self.assertEqual(query(self.records / "records.db", "SELECT COUNT(*) FROM cycles"), [(0,)])
        recorder.handle(*args, sop_snapshot=self.sop.to_dict(), job_key="job", session_id="run")
        recorder.handle(*args, sop_snapshot=self.sop.to_dict(), job_key="job", session_id="run")
        self.assertEqual(query(self.records / "records.db", "SELECT COUNT(*) FROM step_records"), [(1,)])
        self.assertEqual(query(self.records / "records.db", "SELECT result FROM cycles"), [("OK",)])

    def test_failed_jobs_survive_close_and_replay_on_next_start(self):
        errors = []
        with patch.object(Recorder, "handle", side_effect=sqlite3.OperationalError("temporarily locked")):
            writer = AsyncRecorder(self.records, on_error=errors.append)
            writer.handle(events(), "test", 8, "m", np.zeros((5, 5, 3), np.uint8), sop_snapshot=self.sop.to_dict())
            wait_for(lambda: bool(errors))
            self.assertEqual(writer.pending_count(), 1)
            self.assertTrue(writer.close())
        writer = AsyncRecorder(self.records)
        self.assertTrue(writer.close())
        self.assertEqual(query(self.records / "records.db", "SELECT result FROM cycles"), [("OK",)])
        self.assertEqual(read_sop_snapshot(self.records / "records.db", 1)[0], canonical(self.sop.to_dict()))
        with sqlite3.connect(self.records / "pending.db") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
        db.close()

    def test_sessions_separate_engine_restarts_and_unfinished_replay_is_interrupted(self):
        writer = AsyncRecorder(self.records)
        writer.handle(events(), "test", 1, "m", sop_snapshot=self.sop.to_dict())
        writer.handle(events(), "test", 1, "m", sop_snapshot=self.sop.to_dict())
        self.assertTrue(writer.close())
        self.assertEqual(query(self.records / "records.db", "SELECT result FROM cycles"), [("OK",), ("OK",)])
        with patch.object(Recorder, "handle", side_effect=OSError("unavailable")):
            writer = AsyncRecorder(self.records)
            writer.handle(events()[:1], "unfinished", 1, "m", sop_snapshot=self.sop.to_dict())
            self.assertTrue(writer.close())
        writer = AsyncRecorder(self.records)
        self.assertTrue(writer.close())
        self.assertEqual(query(self.records / "records.db", "SELECT result FROM cycles ORDER BY id DESC LIMIT 1"), [("INTERRUPTED",)])

    def test_one_writer_per_station_and_pending_failure_blocks_maintenance(self):
        writer = AsyncRecorder(self.records)
        self.addCleanup(writer.close)
        with self.assertRaisesRegex(RuntimeError, "另一個"):
            AsyncRecorder(self.records)
        callback, action, finished = Mock(), Mock(), threading.Event()
        with patch.object(Recorder, "handle", side_effect=OSError("unavailable")):
            writer.handle(events(), "test", 1, "m")
            writer.maintenance(action, lambda result, error: (callback(result, error), finished.set()))
            self.assertTrue(finished.wait(5))
        action.assert_not_called()
        self.assertIn("待寫入", callback.call_args.args[1])

    def test_backup_restore_roundtrip_preserves_evidence_and_safety_backup(self):
        self.seed()
        archive = self.root / "backup.zip"
        create_backup(self.root, archive)
        manifest = inspect_backup(archive)
        self.assertIn("sops/test.json", manifest["files"])
        self.assertTrue(any("snapshots/" in name for name in manifest["files"]))
        with connect(self.records / "records.db") as db:
            db.execute("UPDATE cycles SET operator='changed'")
        db.close()
        (self.root / "sops/extra.json").write_text("{}")
        safety = restore_backup(self.root, archive)
        self.assertTrue(Path(safety).exists())
        self.assertIn("sops/extra.json", inspect_backup(safety)["files"])
        self.assertFalse((self.root / "sops/extra.json").exists())
        self.assertEqual(query(self.records / "records.db", "SELECT operator FROM cycles"), [("甲",)])
        self.assertEqual(read_sop_snapshot(self.records / "records.db", 1)[0], canonical(self.sop.to_dict()))
        self.assertFalse((self.root / MARKER).exists())

    def test_backup_reads_committed_wal_and_excludes_pending_journal(self):
        self.seed()
        db = connect(self.records / "records.db")
        self.addCleanup(db.close)
        db.execute("UPDATE cycles SET operator='WAL'")
        db.commit()
        archive = self.root / "backup.zip"
        create_backup(self.root, archive)
        with zipfile.ZipFile(archive) as package:
            self.assertNotIn("records/records.db-wal", package.namelist())
            self.assertNotIn("records/pending.db", package.namelist())
            output = self.root / "copy.db"
            output.write_bytes(package.read("records/records.db"))
        self.assertEqual(query(output, "SELECT operator FROM cycles"), [("WAL",)])

    def test_tampering_and_path_traversal_are_rejected_before_replacement(self):
        self.seed()
        original = self.root / "backup.zip"
        create_backup(self.root, original)
        bad = self.root / "bad.zip"
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(bad, "w") as target:
            for name in source.namelist():
                target.writestr(name, b"bad" if name == "sops/test.json" else source.read(name))
        with self.assertRaises(ValueError):
            restore_backup(self.root, bad)
        self.assertEqual(query(self.records / "records.db", "SELECT operator FROM cycles"), [("甲",)])
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr("manifest.json", json.dumps(dict(format="sop-station-backup", version=1, files={
                "../outside": {"size": 1, "sha256": hashlib.sha256(b"x").hexdigest()},
                "records/records.db": {"size": 0, "sha256": ""}})))
            archive.writestr("../outside", b"x")
            archive.writestr("records/records.db", b"")
        with self.assertRaisesRegex(ValueError, "路徑"):
            inspect_backup(bad)

    def test_failed_replacement_rolls_back_and_interrupted_restore_recovers(self):
        self.seed()
        archive = self.root / "backup.zip"
        create_backup(self.root, archive)
        with connect(self.records / "records.db") as db:
            db.execute("UPDATE cycles SET operator='current'")
        db.close()
        original_replace = os.replace
        def fail_new_database(source, target):
            if "/new/records/records.db" in str(source).replace("\\", "/"):
                raise OSError("simulated replacement failure")
            return original_replace(source, target)
        with patch("sop_app.station_backup.os.replace", side_effect=fail_new_database):
            with self.assertRaises(OSError):
                restore_backup(self.root, archive)
        self.assertEqual(query(self.records / "records.db", "SELECT operator FROM cycles"), [("current",)])
        def interrupt_new_database(source, target):
            if "/new/records/records.db" in str(source).replace("\\", "/"):
                raise KeyboardInterrupt()
            return original_replace(source, target)
        with patch("sop_app.station_backup.os.replace", side_effect=interrupt_new_database):
            with self.assertRaises(KeyboardInterrupt):
                restore_backup(self.root, archive)
        self.assertTrue((self.root / MARKER).exists())
        recover_restore(self.root)
        self.assertEqual(query(self.records / "records.db", "SELECT operator FROM cycles"), [("current",)])

    def test_pipeline_stops_when_durable_enqueue_fails_and_passes_actual_snapshot(self):
        pipe = VideoPipeline(self.records)
        pipe._engine = SOPEngine(self.sop)
        recorder = Mock()
        pipe._emit_events(pipe._engine.start(1), recorder)
        self.assertEqual(recorder.handle.call_args.kwargs["sop_snapshot"], self.sop.to_dict())
        recorder.handle.side_effect = OSError("disk full")
        pipe._emit_events(events(), recorder)
        self.assertEqual(pipe._engine.phase, Phase.IDLE)

    def test_history_can_export_verified_snapshot_and_legacy_is_explicit(self):
        self.seed()
        page = RecordsPage(self.records)
        self.addCleanup(page.close)
        page.refresh()
        self.assertTrue(page.export_sop_button.isEnabled())
        output = self.root / "export.json"
        with patch("sop_app.ui.records_page.QFileDialog.getSaveFileName", return_value=(str(output), "")):
            page.export_sop_button.click()
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), canonical(self.sop.to_dict()))
        with connect(self.records / "records.db") as db:
            db.execute("UPDATE cycles SET sop_snapshot_hash=NULL")
        db.close()
        page.refresh()
        self.assertFalse(page.export_sop_button.isEnabled())
        self.assertIn("舊紀錄", page.cycle_summary.text())

    def test_abrupt_process_exit_replays_both_before_and_after_formal_commit(self):
        script = '''
import json, os, sys
from pathlib import Path
from unittest.mock import patch
from sop_app.recorder import Recorder, AsyncRecorder
from sop_app.engine import EngineEvent
from sop_app.sop_schema import SOPDefinition, Step
root, stage = Path(sys.argv[1]), sys.argv[2]
writer = AsyncRecorder(root)
original = Recorder.handle
def crash(*args, **kwargs):
    if stage == "after":
        original(*args, **kwargs)
    os._exit(91)
with patch.object(Recorder, "handle", new=crash):
    writer.handle([EngineEvent("cycle_started", 1, 1),
                   EngineEvent("step_completed", 2, 1, 0, "A", data={"status":"done"}),
                   EngineEvent("cycle_completed", 2, 1, data={"result":"OK"})],
                  "test", 1, "m", sop_snapshot=SOPDefinition(steps=[Step("A")]).to_dict())
    import time
    time.sleep(10)
'''
        for stage in ("before", "after"):
            with self.subTest(stage=stage):
                records = self.root / stage / "records"
                completed = subprocess.run([sys.executable, "-c", script, str(records), stage],
                                           capture_output=True, timeout=15)
                self.assertEqual(completed.returncode, 91, completed.stderr)
                writer = AsyncRecorder(records)
                self.assertTrue(writer.close())
                self.assertEqual(query(records / "records.db", "SELECT result FROM cycles"), [("OK",)])
                self.assertEqual(query(records / "records.db", "SELECT COUNT(*) FROM step_records"), [(1,)])
                self.assertIsNotNone(read_sop_snapshot(records / "records.db", 1))

    def test_async_maintenance_closes_live_database_and_resumes_after_restore(self):
        self.seed()
        writer = AsyncRecorder(self.records)
        self.addCleanup(writer.close)
        completed, results = threading.Event(), []
        archive = self.root / "live-backup.zip"
        def callback(result, error):
            results.append((result, error))
            completed.set()
        writer.maintenance(lambda: create_backup(self.root, archive), callback)
        self.assertTrue(completed.wait(5))
        self.assertFalse(results[-1][1])
        completed.clear()
        writer.maintenance(lambda: restore_backup(self.root, archive), callback)
        self.assertTrue(completed.wait(5))
        self.assertFalse(results[-1][1])
        writer.handle(events(), "new", 1, "m", sop_snapshot=self.sop.to_dict())
        self.assertTrue(writer.close())
        self.assertEqual(query(self.records / "records.db", "SELECT COUNT(*) FROM cycles"), [(2,)])

    def test_corrupt_current_database_can_restore_without_discarding_original_bytes(self):
        self.seed()
        archive = self.root / "good.zip"
        create_backup(self.root, archive)
        broken = b"damaged database for fault injection"
        (self.records / "records.db").write_bytes(broken)
        page = RecordsPage(self.records)
        self.addCleanup(page.close)
        page.refresh()
        self.assertIn("無法讀取", page.summary.text())
        writer = AsyncRecorder(self.records)
        self.addCleanup(writer.close)
        finished, results = threading.Event(), []
        writer.maintenance(lambda: restore_backup(self.root, archive),
                           lambda result, error: (results.append((result, error)), finished.set()),
                           allow_unavailable=True)
        self.assertTrue(finished.wait(5))
        self.assertFalse(results[0][1])
        self.assertEqual((Path(results[0][0]) / "records/records.db").read_bytes(), broken)
        self.assertEqual(query(self.records / "records.db", "SELECT result FROM cycles"), [("OK",)])
