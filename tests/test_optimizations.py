"""Regression tests for the robustness, rule, record and operator-UI optimizations."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

from sop_app.conditions import combine_results, evaluate_condition, evaluate_condition_set
from sop_app.detection import Detection, FrameResult
from sop_app.engine import Phase, SOPEngine, StepStatus
from sop_app.pipeline import FramePacket, VideoPipeline
from sop_app.recorder import AsyncRecorder, Recorder, export_csv, query
from sop_app.sop_schema import ROI, Condition, CycleSettings, SOPDefinition, Step
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.run_page import RunPage

W, H = 200, 100
IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


def det(label, box=(10, 10, 30, 30), score=.9, mask=None):
    return Detection(label, score, box, mask)


def frame(t, *detections, transform=None, valid=True):
    return FrameResult(W, H, t, list(detections), transform, detection_valid=valid)


def run(engine, start, seconds, *detections, step=.1, transform=None):
    events, t = [], start
    while t <= start + seconds + 1e-9:
        events += engine.update(frame(t, *detections, transform=transform))
        t = round(t + step, 6)
    return events, t


def alarms(events):
    return [e.data.get("alarm") for e in events if e.kind == "alarm"]


class ConditionTests(unittest.TestCase):
    def test_box_sized_mask_matches_full_frame_mask(self):
        roi = ROI("左半", [(0, 0), (.5, 0), (.5, 1), (0, 1)])
        full = np.zeros((H, W), bool)
        full[20:40, 120:180] = True
        cropped = full[20:40, 20:180].copy()
        cond = Condition(label="a", roi="左半")
        for mask in (full, cropped):
            result = evaluate_condition(cond, frame(0, det("a", (20, 20, 180, 40), mask=mask)), {"左半": roi})
            self.assertFalse(result.met)

    def test_any_mode_is_not_blocked_by_a_position_out_of_frame(self):
        inside = ROI("左", [(.05, .1), (.2, .1), (.2, .4), (.05, .4)], "workpiece")
        outside = ROI("右", [(.9, .1), (.99, .1), (.99, .4), (.9, .4)], "workpiece")
        conditions = [Condition(label="grip", roi="左"), Condition(label="grip", roi="右")]
        shifted = IDENTITY.copy()
        shifted[0, 2] = .05            # 右側位置被推出畫面，左側仍在畫面內
        result = frame(0, det("grip", (14, 15, 36, 35)), transform=shifted)
        rois = {"左": inside, "右": outside}
        self.assertEqual(evaluate_condition_set(conditions, result, rois, "any")[:2], (True, False))
        self.assertEqual(evaluate_condition_set(conditions, result, rois, "all")[:2], (False, True))
        # 都沒成立且有位置無法確認：any 也要暫停累積，不能當成「未成立」。
        empty = frame(0, transform=shifted)
        self.assertEqual(evaluate_condition_set(conditions, empty, rois, "any")[:2], (False, True))

    def test_missing_region_configuration_still_blocks_any_mode(self):
        results = [evaluate_condition(Condition(label="a", roi="不存在"), frame(0, det("a")), {}),
                   evaluate_condition(Condition(label="a"), frame(0, det("a")), {})]
        self.assertEqual(combine_results(results, "any"), (False, True))


class EngineRuleTests(unittest.TestCase):
    def test_forbidden_any_mode_and_own_window(self):
        step = Step("A", conditions=[Condition(label="a")], forbidden=[Condition(label="x"), Condition(label="y")],
                    hold_sec=5, forbidden_mode="any", forbidden_hold_sec=.3, forbidden_ratio=1.0)
        engine = SOPEngine(SOPDefinition(steps=[step]))
        engine.start(0)
        events, _ = run(engine, 0, .5, det("x"))
        self.assertEqual(alarms(events), ["violation"])
        self.assertFalse(engine.cycle_ok)
        all_mode = SOPEngine(SOPDefinition(steps=[Step("A", conditions=[Condition(label="a")],
                                                        forbidden=step.forbidden, hold_sec=.3)]))
        all_mode.start(0)
        self.assertEqual(alarms(run(all_mode, 0, 1, det("x"))[0]), [])

    def test_too_fast_completion_alarms_and_fails_cycle(self):
        engine = SOPEngine(SOPDefinition(steps=[Step("鎖附", conditions=[Condition(label="a")], hold_sec=.2,
                                                     min_duration_sec=3)]))
        engine.start(0)
        events, _ = run(engine, 0, .5, det("a"))
        self.assertEqual(alarms(events), ["too_fast"])
        completed = [e for e in events if e.kind == "cycle_completed"]
        self.assertEqual(completed[0].data["result"], "NG")
        self.assertTrue(engine.runtimes[0].has_alarm)

    def test_sequence_detection_flags_later_step_satisfied_first(self):
        steps = [Step("A", conditions=[Condition(label="a")], hold_sec=.3),
                 Step("B", conditions=[Condition(label="b")], hold_sec=.3),
                 Step("C", conditions=[Condition("disappear", "c")], hold_sec=.3)]
        quiet = SOPEngine(SOPDefinition(steps=steps))
        quiet.start(0)
        self.assertEqual(alarms(run(quiet, 0, 1, det("b"))[0]), [])

        engine = SOPEngine(SOPDefinition(steps=steps, cycle=CycleSettings(detect_sequence=True)))
        engine.start(0)
        events, t = run(engine, 0, 1, det("b"))
        # 只對 B 警報一次；只有「消失」條件的 C 一開始就成立，不能當成跳步。
        self.assertEqual(alarms(events), ["sequence"])
        self.assertEqual(engine.index, 0)
        self.assertFalse(engine.cycle_ok)
        self.assertTrue(engine.runtimes[0].sequence_alarmed)
        events, _ = run(engine, t, 1, det("a"))
        self.assertEqual(alarms(events), [])

    def test_skip_is_explicit_and_confirm_never_skips_automatic_step(self):
        engine = SOPEngine(SOPDefinition(steps=[Step("自動", conditions=[Condition(label="a")]), Step("人工")]))
        engine.start(0)
        self.assertEqual(engine.confirm_manual_step(1), [])
        self.assertEqual(engine.index, 0)
        events = engine.skip_step(2)
        self.assertEqual(alarms(events), ["skipped"])
        self.assertEqual(engine.runtimes[0].status, StepStatus.SKIPPED)
        events = engine.confirm_manual_step(3)
        self.assertEqual(engine.runtimes[1].status, StepStatus.DONE)
        self.assertEqual([e.data["result"] for e in events if e.kind == "cycle_completed"], ["NG"])

    def test_reset_ratio_is_configurable(self):
        definition = SOPDefinition(steps=[Step("A", conditions=[Condition(label="a")], hold_sec=0)],
                                   cycle=CycleSettings(reset_conditions=[Condition("disappear", "a")],
                                                       reset_hold_sec=1.0, reset_ratio=.5))
        engine = SOPEngine(definition)
        engine.start(0)
        engine.update(frame(0, det("a")))
        self.assertEqual(engine.phase, Phase.CYCLE_DONE)
        self.assertEqual(engine._window.ratio, .5)

    def test_new_fields_roundtrip_and_validate_ranges(self):
        definition = SOPDefinition(steps=[Step("A", forbidden_mode="any", forbidden_hold_sec=.4, forbidden_ratio=.6,
                                               min_duration_sec=2)],
                                   cycle=CycleSettings(reset_ratio=.7, detect_sequence=True))
        restored = SOPDefinition.from_dict(definition.to_dict())
        self.assertEqual(restored, definition)
        self.assertEqual(SOPDefinition.from_dict({"steps": [{"name": "舊"}]}).steps[0].forbidden_hold_sec, None)
        bad = SOPDefinition(steps=[Step("A", hold_sec=-1, conditions=[Condition(label="a", min_score=2)]),
                                   Step("A", forbidden_mode="sometimes")],
                            cycle=CycleSettings(reset_ratio=0))
        errors, warnings = bad.validate()
        self.assertTrue(any("觀察時間" in e for e in errors))
        self.assertTrue(any("信心度" in e for e in errors))
        self.assertTrue(any("違規判定方式" in e for e in errors))
        self.assertTrue(any("成立比例" in e for e in errors))
        self.assertTrue(any("重複" in w for w in warnings))


class PipelineRobustnessTests(unittest.TestCase):
    def pipeline(self):
        pipe = VideoPipeline(Path("unused"))
        pipe._detector = Mock(classes=["a"], info=Mock(model_version_id="m"),
                              detect=Mock(return_value=[det("a")]))
        return pipe

    def test_gui_ack_between_checks_cannot_crash_or_mismatch_display(self):
        pipe = self.pipeline()
        packets = []
        pipe.packet_ready.connect(packets.append)
        image = np.zeros((H, W, 3), np.uint8)
        pipe._gui_ready = Mock()
        # 舊版在同一幀內讀兩次：第一次未就緒、第二次就緒會使用未定義變數而讓執行緒停止。
        pipe._gui_ready.is_set.side_effect = [False, True, True]
        pipe._process(image, 0, Mock())
        pipe._process(image, .1, Mock())
        self.assertEqual(len(packets), 1)
        self.assertIsNotNone(packets[0].display)
        self.assertEqual([r.timestamp for r in packets[0].results], [0, .1])
        self.assertEqual(packets[0].classes, ("a",))

    def test_frame_errors_are_throttled_and_release_gui(self):
        pipe = self.pipeline()
        errors = []
        pipe.error.connect(errors.append)
        pipe._gui_ready.clear()
        for _ in range(3):
            pipe._report_frame_error(RuntimeError("壞掉的幀"))
        self.assertEqual(len(errors), 1)
        self.assertTrue(pipe._gui_ready.is_set())

    def test_pending_results_are_bounded(self):
        pipe = self.pipeline()
        pipe._gui_ready.clear()
        image = np.zeros((H, W, 3), np.uint8)
        for index in range(pipe.PENDING_RESULTS + 20):
            pipe._process(image, index / 10, Mock())
        self.assertEqual(len(pipe._pending_results), pipe.PENDING_RESULTS)

    def test_stale_model_load_is_discarded(self):
        pipe = VideoPipeline(Path("unused"))
        loaded = []
        pipe.model_loaded.connect(lambda *args: loaded.append(args))
        pipe._model_generation = 2
        pipe._handle_command("model_ready", (1, Mock(device_name="CPU"), Mock(), "old.zip"), Mock())
        self.assertIsNone(pipe._detector)
        self.assertEqual(loaded, [])
        detector = Mock(device_name="CPU")
        pipe._handle_command("model_ready", (2, detector, Mock(model_version_id="v"), "new.zip"), Mock())
        self.assertIs(pipe._detector, detector)

    def test_model_command_loads_in_background_thread(self):
        pipe = VideoPipeline(Path("unused"))
        started = threading.Event()
        threads = []

        def build(path):
            threads.append(threading.current_thread().name)
            started.set()
            return Mock(device_name="CPU"), Mock(model_version_id="v")

        with patch.object(pipe, "_build_detector", side_effect=build):
            pipe._handle_command("model", "a.zip", Mock())
            self.assertTrue(started.wait(5))
        command, payload = pipe._commands.get(timeout=5)
        self.assertEqual(command, "model_ready")
        self.assertEqual(threads, ["sop-model-loader"])
        pipe._handle_command(command, payload, Mock())
        self.assertIsNotNone(pipe._detector)

    def test_confirm_command_only_completes_manual_steps(self):
        pipe = VideoPipeline(Path("unused"))
        pipe._handle_command("start", SOPDefinition(steps=[Step("自動", conditions=[Condition(label="a")])]), Mock())
        pipe._handle_command("confirm", None, Mock())
        self.assertEqual(pipe._engine.index, 0)
        self.assertEqual(pipe._engine.runtimes[0].status, StepStatus.ACTIVE)
        pipe._handle_command("skip", None, Mock())
        self.assertEqual(pipe._engine.runtimes[0].status, StepStatus.SKIPPED)


class RecorderTests(unittest.TestCase):
    def test_legacy_database_is_migrated_and_interrupted_cycles_recovered(self):
        with tempfile.TemporaryDirectory() as folder:
            db_path = Path(folder) / "records.db"
            with closing(sqlite3.connect(db_path)) as db:
                db.executescript("""
                    CREATE TABLE cycles (id INTEGER PRIMARY KEY AUTOINCREMENT, sop_name TEXT, sop_version INTEGER,
                        model_version TEXT, started_at TEXT, ended_at TEXT, duration_sec REAL, result TEXT);
                    CREATE TABLE step_records (id INTEGER PRIMARY KEY AUTOINCREMENT, cycle_id INTEGER,
                        step_index INTEGER, step_name TEXT, status TEXT, manual INTEGER, had_alarm INTEGER,
                        duration_sec REAL, finished_at TEXT, snapshot TEXT);
                    CREATE TABLE alarms (id INTEGER PRIMARY KEY AUTOINCREMENT, cycle_id INTEGER, step_index INTEGER,
                        step_name TEXT, kind TEXT, message TEXT, created_at TEXT);
                    INSERT INTO cycles (sop_name, result) VALUES ('舊', 'RUNNING');""")
            recorder = Recorder(Path(folder))
            recorder.close()
            self.assertEqual(query(db_path, "SELECT result FROM cycles"), [("INTERRUPTED",)])
            self.assertEqual(query(db_path, "PRAGMA user_version"), [(2,)])
            columns = {row[1] for row in query(db_path, "PRAGMA table_info(cycles)")}
            self.assertTrue({"operator", "serial_no", "station", "work_order"} <= columns)
            # positioning_events 在舊庫不存在時也會補建
            self.assertEqual(query(db_path, "SELECT COUNT(*) FROM positioning_events"), [(0,)])

    def test_context_alarm_snapshot_ack_and_csv_export(self):
        engine = SOPEngine(SOPDefinition(name="測試", steps=[
            Step("A", conditions=[Condition(label="a")], forbidden=[Condition(label="x")], hold_sec=.2)]))
        events = engine.start(0)
        more, _ = run(engine, 0, .3, det("x"))
        events += more + run(engine, .5, .3, det("a"))[0]
        image = np.zeros((H, W, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            written = threading.Event()
            recorder = AsyncRecorder(Path(folder), on_written=written.set)
            recorder.handle(events, "測試", 3, "model", image,
                            {"operator": "王小明", "serial_no": "SN-001", "station": "A1"})
            recorder.acknowledge_alarms("王小明")
            recorder.close()
            self.assertTrue(written.is_set())
            db_path = Path(folder) / "records.db"
            self.assertEqual(query(db_path, "SELECT operator, serial_no, station, result FROM cycles"),
                             [("王小明", "SN-001", "A1", "NG")])
            alarm = query(db_path, "SELECT kind, snapshot, acknowledged_by FROM alarms")
            self.assertEqual(alarm[0][0], "violation")
            self.assertTrue((Path(folder) / "snapshots" / alarm[0][1]).exists())
            self.assertEqual(alarm[0][2], "王小明")
            target = Path(folder) / "export.csv"
            self.assertEqual(export_csv(db_path, target, text="SN-001"), 1)
            self.assertTrue(target.read_bytes().startswith(b"\xef\xbb\xbf"))
            rows = list(csv.reader(target.open(encoding="utf-8-sig")))
            self.assertEqual(rows[1][4], "NG")
            self.assertIn("違規", rows[1][-1])
            self.assertEqual(export_csv(db_path, target, text="不存在"), 0)


class OperatorUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def running_page(self, definition):
        page = RunPage()
        self.addCleanup(page.close)
        page.set_sop(definition)
        page.set_running(True)
        engine = SOPEngine(definition)
        engine.start(0)
        image = np.zeros((H, W, 3), np.uint8)
        page.on_packet(FramePacket(image, image, frame(0), engine.snapshot(), 10, 5))
        return page

    def test_confirm_only_for_manual_step_and_skip_needs_confirmation(self):
        page = self.running_page(SOPDefinition(steps=[Step("自動", conditions=[Condition(label="a")])]))
        self.assertFalse(page.confirm_button.isEnabled())
        skipped = []
        page.skip_requested.connect(lambda: skipped.append(True))
        with patch("sop_app.ui.run_page.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
            page.skip_button.click()
        self.assertEqual(skipped, [])
        with patch("sop_app.ui.run_page.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            page.skip_button.click()
        self.assertEqual(skipped, [True])
        manual = self.running_page(SOPDefinition(steps=[Step("人工")]))
        self.assertTrue(manual.confirm_button.isEnabled())

    def test_stop_requires_confirmation_and_alarm_needs_acknowledgement(self):
        page = self.running_page(SOPDefinition(steps=[Step("人工")]))
        stopped, acked = [], []
        page.stop_requested.connect(lambda: stopped.append(True))
        page.alarm_ack_requested.connect(lambda: acked.append(True))
        with patch("sop_app.ui.run_page.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
            page.start_button.click()
        self.assertEqual(stopped, [])
        engine = SOPEngine(SOPDefinition(steps=[Step("A", timeout_sec=1)]))
        engine.start(0)
        with patch("sop_app.ui.run_page.QApplication.beep"):
            page.on_events(engine.update(frame(2)))
        self.assertFalse(page.alarm_panel.isHidden())
        self.assertIn("超時", page.alarm_text.text())
        page.alarm_ack_button.click()
        self.assertTrue(page.alarm_panel.isHidden())
        self.assertEqual(acked, [True])

    def test_context_is_emitted_before_start(self):
        page = RunPage()
        self.addCleanup(page.close)
        page.set_sop(SOPDefinition(steps=[Step("人工")]))
        contexts = []
        page.context_changed.connect(contexts.append)
        page.operator_edit.setText("陳")
        page.serial_edit.setText("SN-9")
        page.start_button.click()
        self.assertEqual(contexts[-1], {"operator": "陳", "work_order": "", "serial_no": "SN-9"})

    def test_editor_trial_consumes_every_background_frame(self):
        editor = EditorPage()
        self.addCleanup(editor.close)
        editor.set_sop(SOPDefinition(steps=[Step("前置檢查"),
            Step("安裝", conditions=[Condition(label="a")], hold_sec=.4)]))
        editor.set_classes(["a"])
        editor.set_model_state("available")
        image = np.zeros((H, W, 3), np.uint8)
        editor.on_packet(FramePacket(image, image, frame(0, det("a")), None, 10, 5))
        editor.step_editors[1].trial_button.click()
        self.assertTrue(editor._validation.running)
        self.assertEqual(editor.validation_step.currentIndex(), 1)
        self.assertEqual(editor.validation_mode.currentData(), "single")
        # 介面只收到一個封包，但背景處理了 6 幀；試跑應以全部 6 幀累積時間窗。
        results = tuple(frame(t / 10, det("a")) for t in range(1, 7))
        editor.on_packet(FramePacket(image, image, results[-1], None, 10, 5, results=results))
        self.assertTrue(editor._validation.completed)


if __name__ == "__main__":
    unittest.main()
