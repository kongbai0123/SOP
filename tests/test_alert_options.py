"""Alert preferences affect presentation, never decisions or recorded events."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import ExitStack
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sop_app.detection import TRACKING_CONFIRMED, TRACKING_RELOCATING, Detection, FrameResult
from sop_app.engine import Phase, SOPEngine
from sop_app.pipeline import FramePacket
from sop_app.recorder import connect
from sop_app.sop_schema import Condition, CycleSettings, ROI, SOPDefinition, Step, Workpiece
from sop_app.tracking import workpiece_key
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.editor_validation import EditorValidation
from sop_app.ui.main_window import MainWindow
from sop_app.ui.run_page import RunPage


IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])
POINTS = [(.6, .1), (.9, .1), (.9, .4), (.6, .4)]
BLANK_ALERT = dict(text="", regions=(), active=False)


def definition(*, following=False):
    return SOPDefinition(
        name="Alert preference acceptance",
        workpiece=Workpiece(reference_png="fixture", points=POINTS) if following else None,
        rois=[ROI("C1盒", [(0., 0.), (.4, 0.), (.4, 1.), (0., 1.)]),
              ROI("安裝位置", POINTS, "workpiece" if following else "fixed")],
        steps=[Step("安裝", conditions=[Condition(label="done", roi="安裝位置" if following else "")],
                    forbidden=[Condition(label="hand", roi="C1盒")],
                    hold_sec=0, forbidden_hold_sec=0)],
        cycle=CycleSettings(enabled=False))


def frame(at=.1, *, hand=False, done=False, matrix=None, valid=True, sop=None):
    detections = ([Detection("hand", .95, (10, 10, 30, 30))] if hand else [])
    if done:
        detections.append(Detection("done", .95, (130, 15, 165, 35)))
    return FrameResult(200, 100, at, detections, detection_valid=valid,
                       workpiece_transform=matrix,
                       tracking_state=TRACKING_CONFIRMED if matrix is not None else TRACKING_RELOCATING,
                       tracking_status="位置已確認" if matrix is not None else "重新定位中",
                       workpiece_key=workpiece_key(sop.workpiece) if sop else "")


def packet(result, snapshot=None):
    raw = np.zeros((result.height, result.width, 3), np.uint8)
    return FramePacket(raw, raw.copy(), result, snapshot, 10., 5., classes=("done", "hand"))


class AlertOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def dispose(widget):
        widget.close()
        for timer in widget.findChildren(QTimer):
            timer.stop()

    def page(self, kind, sop=None):
        page = kind()
        self.addCleanup(self.dispose, page)
        with ExitStack() as stack:
            if kind is RunPage:
                stack.enter_context(patch("sop_app.ui.run_page.WorkpieceLocator", return_value=SimpleNamespace()))
            else:
                stack.enter_context(patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()))
                stack.enter_context(patch("sop_app.ui.editor_page.decode_reference",
                                          return_value=np.zeros((100, 200, 3), np.uint8)))
            page.set_sop(sop or definition())
        return page

    def update_run(self, page, engine, result):
        events = engine.update(result)
        page.on_events(events)
        page.on_packet(packet(result, engine.snapshot()))
        return events

    def window(self, saved=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        connect(root / "records.db").close()
        if saved is not None:
            (root / "settings.json").write_text(json.dumps(saved), encoding="utf-8")
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, path in (("RECORDS_DIR", root), ("PROJECT_ROOT", root),
                           ("SOPS_DIR", root / "sops"), ("APP_SETTINGS", root / "settings.json")):
            stack.enter_context(patch(f"sop_app.ui.main_window.{name}", path))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.start"))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.shutdown", return_value=True))
        window = MainWindow(defer_initial_sop=True)
        self.addCleanup(self.dispose, window)
        return window, root

    def test_disabled_workflow_preserves_engine_alarm_completion_ng_and_history(self):
        sop = definition()
        original = deepcopy(sop.to_dict())
        page = self.page(RunPage, sop)
        engine = SOPEngine(sop)
        page.set_running(True)
        page.on_events(engine.start(0))
        page.set_alert_options(False, False)
        acknowledged = Mock()
        page.alarm_ack_requested.connect(acknowledged)
        result = frame(hand=True)
        with patch("sop_app.ui.run_page.QApplication.beep") as beep:
            events = engine.update(result)
            page.on_events(events)
            page.on_packet(packet(result, engine.snapshot()))
            self.assertTrue(any(event.kind == "alarm" for event in events))
            self.assertFalse(engine.cycle_ok)
            self.assertTrue(engine.snapshot().steps[0].active_violation)
            self.assertTrue(page.alarm_panel.isHidden())
            self.assertFalse(page._flash_timer.isActive())
            self.assertEqual(page.alarm_slot.minimumHeight(), 0)
            self.assertIn("違規", "\n".join(page.log.item(i).text() for i in range(page.log.count())))
            completed = frame(.2, done=True)
            completion_events = engine.update(completed)
            page.on_events(completion_events)
            page.on_packet(packet(completed, engine.snapshot()))
            self.assertEqual(engine.phase, Phase.CYCLE_DONE)
            self.assertFalse(engine.cycle_ok)
            self.assertEqual(page._stats["NG"], 1)
            self.assertIn("NG", page.step_title.text())
            self.assertTrue(any(event.kind == "cycle_completed" and event.data.get("result") == "NG"
                                for event in completion_events))
            beep.assert_not_called()
        acknowledged.assert_not_called()
        self.assertEqual(sop.to_dict(), original)

    def test_hiding_and_restoring_violation_never_acknowledges_or_replays_sound(self):
        sop = definition()
        page = self.page(RunPage, sop)
        engine = SOPEngine(sop)
        page.set_running(True)
        page.on_events(engine.start(0))
        acknowledged = Mock()
        page.alarm_ack_requested.connect(acknowledged)
        with patch("sop_app.ui.run_page.QApplication.beep") as beep:
            self.update_run(page, engine, frame(hand=True))
            beep.assert_called_once()
            self.assertFalse(page.alarm_panel.isHidden())
            self.assertTrue(page._flash_timer.isActive())
            message, count = page.alarm_text.text(), page.log.count()
            page.set_alert_options(True, False)
            self.assertTrue(page.alarm_panel.isHidden())
            self.assertFalse(page._flash_timer.isActive())
            self.assertFalse(page._flash_on)
            page.set_alert_options(False, True)
            self.assertFalse(page.alarm_panel.isHidden())
            self.assertTrue(page._flash_timer.isActive())
            self.assertEqual(page.alarm_text.text(), message)
            self.assertEqual(page.log.count(), count)
            beep.assert_called_once()
        acknowledged.assert_not_called()

    def test_violation_dismisses_on_confirmed_clear_without_acknowledging_or_undoing_ng(self):
        sop = definition(following=True)
        sop.rois[0].anchor = "workpiece"
        sop.steps[0].forbidden_hold_sec = .2
        page = self.page(RunPage, sop)
        engine = SOPEngine(sop)
        page.set_running(True)
        page.on_events(engine.start(0))
        acknowledged = Mock()
        page.alarm_ack_requested.connect(acknowledged)

        with patch("sop_app.ui.run_page.QApplication.beep"):
            self.update_run(page, engine, frame(.01, matrix=IDENTITY, sop=sop))
            normal_banner = page.video._banner_color.name()
            self.update_run(page, engine, frame(.05, hand=True, matrix=IDENTITY, sop=sop))
            events = self.update_run(page, engine, frame(.26, hand=True, matrix=IDENTITY, sop=sop))
            self.assertEqual([e.data.get("alarm") for e in events if e.kind == "alarm"], ["violation"])
            self.assertFalse(page.alarm_panel.isHidden())
            self.assertTrue(page.alarm_ack_button.isHidden())
            self.assertTrue(page._flash_timer.isActive())
            self.assertNotEqual(page.video._banner_color.name(), normal_banner)
            self.assertEqual(page.step_overview._card(0).kind, "ng")

            self.update_run(page, engine, frame(.31, sop=sop))
            self.assertFalse(page.alarm_panel.isHidden())
            self.assertEqual(page.step_overview._card(0).kind, "ng")
            self.update_run(page, engine, frame(.32, matrix=IDENTITY, sop=sop))
            self.assertTrue(page.alarm_panel.isHidden())
            self.assertFalse(page._flash_timer.isActive())
            self.assertEqual(page.video._banner_color.name(), normal_banner)
            self.assertEqual(page.step_overview._card(0).kind, "active")
            self.assertIn("有警報", page.step_overview._card(0).detail)
            self.assertTrue(engine.snapshot().steps[0].violation_alarmed)
            self.assertFalse(engine.cycle_ok)

            self.update_run(page, engine, frame(.37, hand=True, matrix=IDENTITY, sop=sop))
            self.assertTrue(page.alarm_panel.isHidden())
            events = self.update_run(page, engine, frame(.62, hand=True, matrix=IDENTITY, sop=sop))
            self.assertFalse(page.alarm_panel.isHidden())
            self.assertEqual(page.step_overview._card(0).kind, "ng")
            self.assertFalse(any(e.kind == "alarm" for e in events))
            violation_log = [page.log.item(i).text() for i in range(page.log.count())
                             if "違規：" in page.log.item(i).text()]
            self.assertEqual(len(violation_log), 1)
            events = self.update_run(page, engine, frame(.70, done=True, matrix=IDENTITY, sop=sop))
            self.assertTrue(page.alarm_panel.isHidden())
            self.assertEqual(page.step_overview._card(0).kind, "ng")
            self.assertTrue(any(e.kind == "cycle_completed" and e.data.get("result") == "NG" for e in events))
        acknowledged.assert_not_called()

    def test_reminder_clears_automatically_and_allows_ok_in_monitor(self):
        sop = definition()
        sop.steps[0].forbidden_action = "remind"
        page = self.page(RunPage, sop)
        engine = SOPEngine(sop)
        page.set_running(True)
        page.on_events(engine.start(0))
        acknowledged = Mock()
        page.alarm_ack_requested.connect(acknowledged)

        with patch("sop_app.ui.run_page.QApplication.beep") as beep:
            events = self.update_run(page, engine, frame(.1, hand=True, done=True))
            self.assertEqual([e.kind for e in events], ["reminder"])
            self.assertFalse(page.alarm_panel.isHidden())
            self.assertEqual(page.alarm_panel.property("reminder"), "true")
            self.assertTrue(page.alarm_ack_button.isHidden())
            self.assertFalse(page._flash_timer.isActive())
            self.assertEqual(page.step_overview._card(0).kind, "warn")
            self.assertEqual(page.progress.format(), "等待提醒解除")
            self.assertTrue(engine.cycle_ok)

            self.update_run(page, engine, frame(.2, valid=False))
            self.assertFalse(page.alarm_panel.isHidden())
            page.set_alert_options(True, False)
            self.assertTrue(page.alarm_panel.isHidden())
            page.set_alert_options(True, True)
            self.assertFalse(page.alarm_panel.isHidden())
            events = self.update_run(page, engine, frame(.3, done=True))
            self.assertTrue(any(e.kind == "cycle_completed" and e.data["result"] == "OK" for e in events))
            self.assertTrue(page.alarm_panel.isHidden())
            self.assertEqual(page.step_overview._card(0).kind, "ok")
            beep.assert_not_called()
        acknowledged.assert_not_called()

    def test_clearing_violation_keeps_timeout_available_for_manual_acknowledgement(self):
        sop = definition()
        sop.steps[0].timeout_sec = .25
        page = self.page(RunPage, sop)
        engine = SOPEngine(sop)
        page.set_running(True)
        page.on_events(engine.start(0))
        acknowledged = Mock()
        page.alarm_ack_requested.connect(acknowledged)
        with patch("sop_app.ui.run_page.QApplication.beep"):
            self.update_run(page, engine, frame(.10, hand=True))
            self.update_run(page, engine, frame(.30, hand=True))
            self.update_run(page, engine, frame(.40))
        self.assertFalse(page.alarm_panel.isHidden())
        self.assertFalse(page.alarm_ack_button.isHidden())
        self.assertIn("超時", page.alarm_text.text())
        self.assertNotIn("違規", page.alarm_text.text())
        acknowledged.assert_not_called()
        page.alarm_ack_button.click()
        self.assertTrue(page.alarm_panel.isHidden())
        acknowledged.assert_called_once()
        self.assertFalse(engine.cycle_ok)

    def test_tracking_off_shows_confirmed_regions_but_never_unlabelled_old_position(self):
        sop = definition(following=True)
        page = self.page(RunPage, sop)
        page.set_alert_options(False, True)
        page.on_packet(packet(frame(matrix=IDENTITY.copy(), sop=sop)))
        self.assertEqual({roi.name for roi in page.video._rois}, {"C1盒", "安裝位置"})
        self.assertTrue(page.tracking_label.isHidden())
        confirmed_points = deepcopy(page.video._rois[1].points)
        page.on_packet(packet(frame(.2, sop=sop)))
        self.assertEqual([roi.name for roi in page.video._rois], ["C1盒"])
        self.assertEqual(page.video._stale, set())
        self.assertIsNone(page._transform)
        np.testing.assert_array_equal(page._last_transform, IDENTITY)
        page.set_alert_options(True, True)
        self.assertFalse(page.tracking_label.isHidden())
        self.assertEqual(page.video._rois[1].points, confirmed_points)
        self.assertEqual(page.video._stale, {"安裝位置"})
        self.assertIn("僅供參考", page.tracking_label.text())
        self.assertIsNone(page._transform)

    def test_tracking_setting_preserves_required_configuration_error_and_start_guard(self):
        sop = definition(following=True)
        page = RunPage()
        self.addCleanup(self.dispose, page)
        with patch("sop_app.ui.run_page.WorkpieceLocator", side_effect=ValueError("參考特徵不足")):
            page.set_sop(sop)
        page.set_alert_options(False, False)
        self.assertTrue(page.tracking_label.isHidden())
        self.assertFalse(page.start_button.isEnabled())
        self.assertIn("參考特徵不足", page.start_button.toolTip())
        self.assertIn("重新設定參考工件", page.start_button.toolTip())

    def test_editor_workflow_off_retains_true_ng_and_trial_details_then_restores_overlay(self):
        page = self.page(EditorPage)
        page.set_alert_options(True, False)
        trial = page._validation
        trial.start(page.sop, "single", 0, 0)
        trial.update(frame(hand=True))
        page._update_validation_view()
        self.assertFalse(trial.engine.cycle_ok)
        self.assertTrue(trial.alert()["active"])
        self.assertTrue(page.validation_alert.isHidden())
        self.assertEqual(page.video._alert, "")
        self.assertEqual(page._active_alert_regions, set())
        self.assertIn("違規", page.validation_details.toPlainText())
        events = list(trial.events)
        page.set_alert_options(False, True)
        self.assertFalse(page.validation_alert.isHidden())
        self.assertIn("違規警示", page.video._alert)
        self.assertEqual(page.video._alert_regions, {"C1盒"})
        self.assertEqual(list(trial.events), events)
        self.assertFalse(trial.engine.cycle_ok)

    def test_editor_reminder_highlights_region_and_trial_completes_ok_after_correction(self):
        sop = definition()
        sop.steps[0].forbidden_action = "remind"
        page = self.page(EditorPage, sop)
        page.set_model_state("available")
        trial = page._validation
        trial.start(page.sop, "single", 0, 0)
        page.on_packet(packet(frame(.1, hand=True, done=True)))

        self.assertFalse(page.validation_alert.isHidden())
        self.assertEqual(page.validation_alert.property("state"), "reminder")
        self.assertTrue(page.video._alert_reminder)
        self.assertEqual(page.video._alert_regions, {"C1盒"})
        self.assertIn("暫停", page.validation_elapsed.text())
        self.assertIn("0 道工序完成條件成立", page.live_label.text())
        self.assertIn("1 道提醒條件成立", page.live_label.text())
        self.assertTrue(trial.engine.cycle_ok)

        page.on_packet(packet(frame(.2, done=True)))
        self.assertTrue(page.validation_alert.isHidden())
        self.assertFalse(page.video._alert_reminder)
        self.assertEqual(page.video._alert_regions, set())
        self.assertIn("單步驗證完成：OK", page.validation_status.text())
        self.assertIn("提醒：", page.validation_details.toPlainText())
        self.assertNotIn("本輪已NG", page.validation_details.toPlainText())

    def test_editor_tracking_only_overlay_obeys_tracking_not_workflow_switch(self):
        sop = definition(following=True)
        page = self.page(EditorPage, sop)
        trial = page._validation
        trial.start(sop, "single", 0, 0)
        trial.update(frame(sop=sop))
        original = deepcopy(page.sop.to_dict())
        events = list(trial.events)
        page.set_alert_options(False, True)
        self.assertTrue(page.validation_alert.isHidden())
        self.assertEqual(page.video._alert, "")
        self.assertTrue(trial.engine.cycle_ok)
        self.assertIn("無法確認", page.validation_status.text())
        page.set_alert_options(True, False)
        self.assertFalse(page.validation_alert.isHidden())
        self.assertIn("無法確認", page.video._alert)
        self.assertEqual(page.video._alert_regions, set())
        self.assertEqual(list(trial.events), events)
        self.assertEqual(page.sop.to_dict(), original)

    def test_alert_partition_keeps_model_and_configuration_failures_in_workflow_category(self):
        for source, observed in ((definition(following=True), frame(valid=False)),
                                 (definition(), frame())):
            with self.subTest(model_available=observed.detection_valid):
                if observed.detection_valid:
                    source.steps[0].conditions[0].roi = "不存在的區域"
                trial = EditorValidation()
                trial.start(source, "single", 0, 0)
                trial.update(observed)
                self.assertTrue(trial.alert(include_tracking=False, include_workflow=True)["text"])
                self.assertEqual(trial.alert(include_tracking=True, include_workflow=False), BLANK_ALERT)
                self.assertTrue(trial.engine.cycle_ok)

    def test_alert_partition_hides_real_workflow_history_without_changing_it(self):
        trial = EditorValidation()
        trial.start(definition(), "single", 0, 0)
        trial.update(frame(hand=True))
        trial.stop()
        before = list(trial.events)
        self.assertEqual(trial.alert(include_tracking=True, include_workflow=False), BLANK_ALERT)
        self.assertIn("本輪已NG", trial.alert(include_tracking=False, include_workflow=True)["text"])
        self.assertFalse(trial.engine.cycle_ok)
        self.assertEqual(list(trial.events), before)

    def test_toolbar_has_exactly_two_independent_choices_shared_with_both_pages(self):
        window, root = self.window()
        window._apply_sop(definition(), None)
        original = deepcopy(window.editor.sop.to_dict())
        modified = Mock()
        window.editor.modified.connect(modified)
        self.assertEqual(set(window.alert_actions), {"tracking", "workflow"})
        self.assertIs(window.alert_menu_button.menu(), window.alert_menu)
        self.assertFalse(window.alert_actions["tracking"].isChecked())
        self.assertTrue(window.alert_actions["workflow"].isChecked())
        for tracking, workflow in product((False, True), repeat=2):
            with self.subTest(tracking=tracking, workflow=workflow):
                window.alert_actions["tracking"].setChecked(tracking)
                window.alert_actions["workflow"].setChecked(workflow)
                for page in (window.run_page, window.editor):
                    self.assertEqual(page._show_tracking_alerts, tracking)
                    self.assertEqual(page._show_workflow_alerts, workflow)
                    self.assertEqual(page.tracking_label.isHidden(), not tracking)
                stored = json.loads((root / "settings.json").read_text(encoding="utf-8"))
                self.assertEqual(stored["alerts"], {"tracking": tracking, "workflow": workflow})
        self.assertFalse(window.dirty)
        self.assertEqual(window.editor.sop.to_dict(), original)
        modified.assert_not_called()

    def test_load_and_station_restore_apply_preferences_without_writing_or_dirtying_sop(self):
        window, _root = self.window({"alerts": {"tracking": True, "workflow": False}})
        for page in (window.run_page, window.editor):
            self.assertTrue(page._show_tracking_alerts)
            self.assertFalse(page._show_workflow_alerts)
        window._apply_sop(definition(), None)
        original = deepcopy(window.editor.sop.to_dict())
        with patch.object(window, "_save_settings") as save:
            window.settings = {"alerts": {"tracking": False, "workflow": True}}
            window._restore_alert_options()
            for page in (window.run_page, window.editor):
                self.assertFalse(page._show_tracking_alerts)
                self.assertTrue(page._show_workflow_alerts)
            self.assertFalse(window.alert_actions["tracking"].isChecked())
            self.assertTrue(window.alert_actions["workflow"].isChecked())
            save.assert_not_called()
        self.assertFalse(window.dirty)
        self.assertEqual(window.editor.sop.to_dict(), original)


if __name__ == "__main__":
    unittest.main()
