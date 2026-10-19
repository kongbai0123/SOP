"""Station isolation, legacy migration and real Qt/pipeline station switching."""
from contextlib import closing
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

from sop_app.engine import EngineEvent
from sop_app.recorder import AsyncRecorder, Recorder, query, read_sop_snapshot
from sop_app.sop_schema import SOPDefinition, Step, save_sop, load_sop
from sop_app.stations import StationStore
from sop_app.station_backup import create_backup, restore_backup, inspect_backup
from sop_app.startup import initial_sop_path, initial_model_path
from sop_app.ui.main_window import MainWindow


class StationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = StationStore(self.root)

    def wait(self, predicate):
        until = time.monotonic() + 8
        while time.monotonic() < until:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        self.fail("工站操作未完成")

    def seed(self):
        sop = SOPDefinition(name="原裝配流程", model_path="models/old.zip", steps=[Step("原工序")])
        save_sop(sop, self.root / "sops/original.json")
        model = self.root / "models/old.zip"
        model.parent.mkdir()
        model.write_bytes(b"external model placeholder")
        (self.root / "app_settings.json").write_text(json.dumps({
            "station": "裝配站", "last_sop": str(self.root / "sops/original.json"),
            "run_context": {"operator": "甲", "work_order": "WO-1"}}, ensure_ascii=False), encoding="utf-8")
        recorder = Recorder(self.root / "records")
        recorder.handle([EngineEvent("cycle_started", 1, 1),
                         EngineEvent("step_completed", 2, 1, 0, "原工序", data={"status": "done"}),
                         EngineEvent("cycle_completed", 2, 1, data={"result": "OK"})],
                        sop.name, 1, "old", np.zeros((6, 8, 3), dtype=np.uint8),
                        {"station": "裝配站", "operator": "甲"}, sop.to_dict())
        recorder.close()
        return sop

    def window(self):
        window = MainWindow(station_store=self.store)
        def close():
            window.dirty = False
            self.wait(lambda: not window._maintenance_busy)
            window.close()
            self.app.processEvents()
        self.addCleanup(window.deleteLater)
        self.addCleanup(close)
        window.show()
        self.wait(lambda: window.pipeline.isRunning())
        return window

    def test_migration_preserves_originals_and_snapshots_and_is_idempotent(self):
        sop = self.seed()
        original_json = (self.root / "sops/original.json").read_bytes()
        original_settings = (self.root / "app_settings.json").read_bytes()
        station = self.store.initialize()
        self.assertEqual(station.id, "ST001")
        self.assertEqual(station.name, "裝配站")
        self.assertEqual(station.root, self.root / "data/stations/ST001")
        self.assertEqual(query(station.records / "records.db", "SELECT operator FROM cycles"), [("甲",)])
        self.assertEqual(read_sop_snapshot(station.records / "records.db", 1)[0]["model_path"], sop.model_path)
        migrated = load_sop(station.sops / "original.json")
        self.assertEqual(migrated.model_path, str(self.root / "models/old.zip"))
        self.assertEqual(json.loads(station.settings.read_text(encoding="utf-8"))["last_sop"], "sops/original.json")
        self.assertEqual((self.root / "sops/original.json").read_bytes(), original_json)
        self.assertEqual((self.root / "app_settings.json").read_bytes(), original_settings)
        self.assertEqual(len(list((station.records / "snapshots").glob("*.jpg"))), 1)
        (station.sops / "original.json").write_text("user edit", encoding="utf-8")
        self.store.initialize()
        self.assertEqual((station.sops / "original.json").read_text(encoding="utf-8"), "user edit")
        self.assertEqual(initial_model_path(root=station.root), None)  # malformed SOP is left for UI repair

    def test_migration_failure_preserves_source_and_can_retry(self):
        self.seed()
        with patch("sop_app.stations.shutil.copy2", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.initialize()
        self.assertTrue((self.root / "sops/original.json").exists())
        self.assertFalse((self.store.root / "ST001").exists())
        self.assertFalse(self.store.index.exists())
        self.assertEqual(self.store.initialize().id, "ST001")

    def test_legacy_writer_must_close_before_migration_and_pending_jobs_replay(self):
        writer = AsyncRecorder(self.root / "records")
        with self.assertRaises(RuntimeError):
            self.store.initialize()
        with patch.object(Recorder, "handle", side_effect=OSError("unavailable")):
            writer.handle([EngineEvent("cycle_started", 1, 1),
                           EngineEvent("cycle_completed", 2, 1, data={"result": "OK"})], "pending", 1, "m")
            self.assertTrue(writer.close())
        station = self.store.initialize()
        migrated = AsyncRecorder(station.records)
        self.assertTrue(migrated.close())
        self.assertEqual(query(station.records / "records.db", "SELECT sop_name,result FROM cycles"), [("pending", "OK")])
        with closing(sqlite3.connect(self.root / "records/pending.db")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)

    def test_station_creation_uses_fixed_ids_and_empty_independent_folders(self):
        first = self.store.initialize()
        second = self.store.create("包裝站")
        self.assertEqual(second.id, "ST002")
        self.assertEqual(self.store.active, first)
        self.assertFalse(list(second.sops.iterdir()))
        for directory in ("sops", "records/snapshots", "backups", "logs"):
            self.assertTrue((second.root / directory).is_dir())
        self.store.activate(second.id)
        self.assertEqual(StationStore(self.root).active.id, "ST002")
        for name in ("", "包裝站", "\n工站"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.create(name)

    def test_station_backup_rejects_other_station_before_touching_target(self):
        self.seed()
        first = self.store.initialize()
        second = self.store.create("包裝站")
        Recorder(second.records).close()
        archive = self.root / "first.zip"
        create_backup(first.root, archive)
        self.assertEqual(inspect_backup(archive)["station_id"], "ST001")
        before = (second.root / "station.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "不屬於"):
            restore_backup(second.root, archive)
        self.assertEqual((second.root / "station.json").read_bytes(), before)
        self.assertEqual(query(second.records / "records.db", "SELECT COUNT(*) FROM cycles"), [(0,)])

    def test_startup_only_selects_current_station_sops(self):
        first = self.store.initialize()
        second = self.store.create("包裝站")
        save_sop(SOPDefinition(name="A", steps=[Step("A")]), first.sops / "a.json")
        save_sop(SOPDefinition(name="B", steps=[Step("B")]), second.sops / "b.json")
        self.assertEqual(initial_sop_path({"last_sop": str(first.sops / "a.json")}, root=second.root), second.sops / "b.json")

    def test_new_station_button_and_switch_isolate_context_records_and_sops(self):
        first = self.store.initialize()
        save_sop(SOPDefinition(name="裝配", steps=[Step("A")]), first.sops / "assembly.json")
        window = self.window()
        self.assertEqual(window.editor.sop.name, "裝配")
        window.run_page.operator_edit.setText("甲")
        window.run_page.operator_edit.editingFinished.emit()
        window.pipeline.start_engine(window.editor.sop)
        self.wait(lambda: window._engine_running)
        self.assertFalse(window.station_combo.isEnabled())
        self.assertFalse(window.new_station_button.isEnabled())
        with patch("sop_app.ui.main_window.QMessageBox.information"):
            # A direct request is rejected too; disabling the UI is not the only guard.
            second = self.store.create("包裝站")
            window._switch_to_station(second)
        self.assertEqual(window.current_station.id, first.id)
        window.pipeline.confirm_step()
        window.pipeline.stop_engine()
        self.wait(lambda: not window._engine_running)
        window._switch_to_station(second)
        self.wait(lambda: not window._maintenance_busy)
        self.assertEqual(window.current_station.id, "ST002")
        self.assertEqual(window.station, "包裝站")
        self.assertEqual(window.records_page.db_path, second.records / "records.db")
        self.assertEqual(window.run_page.operator_edit.text(), "")
        self.assertEqual(window._source_label, "")
        self.assertEqual(window.records_page.cycles.rowCount(), 0)
        window.editor._add_step()
        self.assertTrue(window._write(second.sops / "packing.json", new_file=True))
        window.run_page.operator_edit.setText("乙")
        window.run_page.operator_edit.editingFinished.emit()
        window.pipeline.start_engine(window.editor.sop)
        self.wait(lambda: window._engine_running)
        window.pipeline.stop_engine()
        self.wait(lambda: not window._engine_running)
        window._switch_to_station(first)
        self.wait(lambda: not window._maintenance_busy)
        self.assertEqual(window.editor.sop.name, "裝配")
        self.assertEqual(window.run_page.operator_edit.text(), "甲")
        self.assertEqual(query(first.records / "records.db", "SELECT station,operator FROM cycles"), [(first.name, "甲")])
        self.assertEqual(query(second.records / "records.db", "SELECT station,operator FROM cycles"), [(second.name, "乙")])
        self.assertFalse((first.sops / "packing.json").exists())
        self.assertEqual(self.store.active.id, first.id)
        with patch("sop_app.ui.main_window.QInputDialog.getText", return_value=("檢查站", True)):
            window.new_station_button.click()
        self.wait(lambda: not window._maintenance_busy)
        self.assertEqual(window.current_station.id, "ST003")
        self.assertTrue((window.current_station.root / "logs/workstation.log").exists())

    def test_cancelled_switch_and_locked_target_keep_old_station_usable(self):
        first = self.store.initialize()
        second = self.store.create("包裝站")
        window = self.window()
        window.editor._add_step()
        with patch("sop_app.ui.main_window.QMessageBox.question", return_value=QMessageBox.StandardButton.Cancel):
            window._switch_to_station(second)
        self.assertEqual(window.current_station.id, first.id)
        self.assertTrue(window.dirty)
        window.dirty = False
        other = AsyncRecorder(second.records)
        try:
            with patch("sop_app.ui.main_window.QMessageBox.warning"):
                window._switch_to_station(second)
                self.wait(lambda: not window._maintenance_busy)
            self.assertEqual(window.current_station.id, first.id)
            self.assertEqual(self.store.active.id, first.id)
            self.assertEqual(window.pipeline._records_dir, first.records)
        finally:
            other.close()
        window._switch_to_station(second)
        self.wait(lambda: not window._maintenance_busy)
        self.assertEqual(window.current_station.id, second.id)

    def test_external_sop_is_imported_and_cross_station_save_is_rejected(self):
        first = self.store.initialize()
        second = self.store.create("包裝站")
        source = second.sops / "other.json"
        save_sop(SOPDefinition(name="其他站", steps=[Step("B")]), source)
        original = source.read_bytes()
        window = self.window()
        window._open_path(source)
        self.assertEqual(window.sop_path, first.sops / "other.json")
        window.editor._add_step()
        self.assertTrue(window._write(window.sop_path))
        self.assertEqual(source.read_bytes(), original)
        with patch("sop_app.ui.main_window.QMessageBox.warning"):
            self.assertFalse(window._write(source))

    def test_pending_write_failure_blocks_switch_without_changing_station(self):
        first = self.store.initialize()
        second = self.store.create("包裝站")
        window = self.window()
        with patch.object(Recorder, "handle", side_effect=OSError("database unavailable")), \
                patch("sop_app.ui.main_window.QMessageBox.warning"):
            window.pipeline.start_engine(window.editor.sop)
            self.wait(lambda: window._engine_running)
            window.pipeline.stop_engine()
            self.wait(lambda: not window._engine_running)
            window._switch_to_station(second)
            self.wait(lambda: not window._maintenance_busy)
            self.assertEqual(window.current_station.id, first.id)
            self.assertEqual(self.store.active.id, first.id)
            self.assertEqual(window.pipeline._records_dir, first.records)
        self.wait(lambda: query(first.records / "records.db", "SELECT COUNT(*) FROM cycles") == [(1,)])
        window._switch_to_station(second)
        self.wait(lambda: not window._maintenance_busy)
        self.assertEqual(window.current_station.id, second.id)

    def test_station_startup_recovers_interrupted_restore_before_loading_settings(self):
        self.seed()
        first = self.store.initialize()
        archive = self.root / "backup.zip"
        create_backup(first.root, archive)
        settings = json.loads(first.settings.read_text(encoding="utf-8"))
        settings["run_context"] = {"operator": "目前作業員"}
        first.settings.write_text(json.dumps(settings), encoding="utf-8")
        replace = os.replace
        def interrupted(source, target):
            if "/new/records/records.db" in str(source).replace("\\", "/"):
                raise KeyboardInterrupt()
            return replace(source, target)
        with patch("sop_app.station_backup.os.replace", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                restore_backup(first.root, archive)
        self.store.initialize()
        self.assertEqual(json.loads(first.settings.read_text(encoding="utf-8"))["run_context"]["operator"], "目前作業員")
