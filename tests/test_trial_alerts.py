"""Trial warning presentation follows the production engine's real window."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from sop_app.detection import Detection, FrameResult
from sop_app.engine import Phase
from sop_app.sop_schema import Condition, CycleSettings, ROI, SOPDefinition, Step
from sop_app.ui.editor_validation import EditorValidation


def frame(at, *regions, done=False, valid=True):
    boxes = {"C2": (10, 10, 30, 30), "C3": (130, 10, 150, 30)}
    detections = [Detection("hand", .95, boxes[region]) for region in regions]
    if done:
        detections.append(Detection("done", .95, (50, 10, 70, 30)))
    return FrameResult(200, 100, at, detections, detection_valid=valid)


def definition(*, mode="any", hold=.3, ratio=.8, manual=False, steps=1, cycle=False, interval=0):
    current = Step("左把手安裝", conditions=[] if manual else [Condition(label="done")],
                   forbidden=[Condition(label="hand", roi="C2"), Condition(label="hand", roi="C3")],
                   forbidden_mode=mode, forbidden_hold_sec=hold, forbidden_ratio=ratio,
                   hold_sec=0, trial_notice_interval_sec=interval)
    return SOPDefinition(rois=[ROI("C2", [(0, 0), (.4, 0), (.4, 1), (0, 1)]),
                               ROI("C3", [(.6, 0), (1, 0), (1, 1), (.6, 1)])],
                         steps=[current] + [Step("下一工序") for _ in range(steps - 1)],
                         cycle=CycleSettings(enabled=cycle, auto_restart_sec=.3))


class TrialAlertTests(unittest.TestCase):
    def trial(self, source=None, mode="single", selected=0):
        trial = EditorValidation()
        trial.start(source or definition(), mode, selected, 0)
        return trial

    @staticmethod
    def feed(trial, times, *regions, **kwargs):
        for at in times:
            trial.update(frame(at, *regions, **kwargs))

    @staticmethod
    def formal_violations(trial):
        return [event for event in trial.events if event.kind == "alarm"
                and event.data.get("alarm") == "violation"]

    @staticmethod
    def notices(trial):
        return [event for event in trial.events if event.kind == "trial_notice"]

    def assert_inactive(self, trial):
        self.assertFalse(trial.alert()["active"])
        self.assertEqual(trial.alert()["regions"], ())

    def test_any_changes_c2_to_c3_after_formal_alarm_latches(self):
        source = definition()
        original = copy.deepcopy(source)
        trial = self.trial(source)
        self.feed(trial, (.1, .2, .3, .4), "C2")
        self.assertTrue(trial.alert()["active"])
        self.assertEqual(trial.alert()["regions"], ("C2",))
        self.assertEqual(len(self.formal_violations(trial)), 1)
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(.5, "C3"))
        self.assertTrue(trial.engine.runtimes[0].violation_alarmed)
        self.assertTrue(trial.alert()["active"])
        self.assertEqual(trial.alert()["regions"], ("C3",))
        self.assertIn("hand＠C3", trial.alert()["text"])
        self.assertEqual(len(self.formal_violations(trial)), 1)
        self.assertEqual(len(self.notices(trial)), 1)
        self.assertFalse(trial.engine.cycle_ok)
        self.assertEqual(source, original)

    def test_continuous_same_condition_and_repeated_reads_do_not_flood_log(self):
        trial = self.trial()
        self.feed(trial, (.1, .2, .3, .4, .5, .6), "C2")
        before = list(trial.events)
        for _ in range(100):
            self.assertEqual(trial.alert()["regions"], ("C2",))
        self.assertEqual(list(trial.events), before)
        self.assertEqual(len(self.formal_violations(trial)), 1)
        self.assertEqual(self.notices(trial), [])

    def test_clear_then_reenter_creates_one_nonformal_notice(self):
        trial = self.trial(definition(hold=0))
        trial.update(frame(.1, "C2"))
        trial.update(frame(.2))
        self.assert_inactive(trial)
        self.assertIn("本輪已NG", trial.alert()["text"])
        self.assertIn("最後警報", trial.alert()["text"])
        self.feed(trial, (.3, .4, .5), "C2")
        self.assertTrue(trial.alert()["active"])
        self.assertEqual(len(self.formal_violations(trial)), 1)
        self.assertEqual(len(self.notices(trial)), 1)
        notice = self.notices(trial)[0]
        self.assertEqual(notice.data["notice"], "violation")
        self.assertEqual(notice.data["regions"], ("C2",))
        self.assertIn("hand＠C2", notice.data["summary"])

    def test_same_region_with_different_condition_updates_notice(self):
        source = definition(hold=0)
        source.steps[0].forbidden[1] = Condition(label="tool", roi="C2")
        trial = self.trial(source)
        trial.update(frame(.1, "C2"))
        result = frame(.2)
        result.detections = [Detection("tool", .95, (10, 10, 30, 30))]
        trial.update(result)
        self.assertEqual(trial.alert()["regions"], ("C2",))
        self.assertIn("tool＠C2", trial.alert()["text"])
        self.assertEqual(len(self.notices(trial)), 1)
        self.assertEqual(len(self.formal_violations(trial)), 1)

    def test_all_mode_requires_every_condition(self):
        trial = self.trial(definition(mode="all", hold=0))
        self.feed(trial, (.1, .2, .3), "C2")
        self.assert_inactive(trial)
        self.assertEqual(self.formal_violations(trial), [])
        trial.update(frame(.4, "C2", "C3"))
        self.assertEqual(trial.alert()["regions"], ("C2", "C3"))
        self.assertTrue(trial.alert()["active"])

    def test_current_true_does_not_bypass_ratio_or_observation(self):
        trial = self.trial(definition(hold=1, ratio=.8))
        self.feed(trial, (.1, .2, .3, .4, .5, .6))
        self.feed(trial, (.7, .8, .9, 1.0), "C2")
        self.assert_inactive(trial)
        self.assertEqual(self.formal_violations(trial), [])
        self.feed(trial, (1.1, 1.2, 1.3, 1.4, 1.5), "C2")
        self.assertTrue(trial.alert()["active"])

    def test_latest_false_clears_live_warning_even_with_sufficient_ratio(self):
        trial = self.trial(definition(hold=1, ratio=.5))
        self.feed(trial, (.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0), "C2")
        self.assertTrue(trial.alert()["active"])
        trial.update(frame(1.1))
        self.assertGreaterEqual(trial.engine._forbidden_window._true_ratio(), .5)
        self.assert_inactive(trial)
        self.assertIn("最後警報", trial.alert()["text"])

    def test_invalid_frame_clears_live_warning_and_resets_real_window(self):
        trial = self.trial()
        self.feed(trial, (.1, .2, .3, .4), "C2")
        trial.update(frame(.5, valid=False))
        self.assert_inactive(trial)
        self.assertIn("無法確認", trial.alert()["text"])
        self.assertIn("本輪已NG", trial.alert()["text"])
        trial.update(frame(.6, "C3"))
        self.assert_inactive(trial)
        trial.update(frame(.8, "C3"))
        self.assert_inactive(trial)
        trial.update(frame(1.0, "C3"))
        self.assertTrue(trial.alert()["active"])
        self.assertEqual(trial.alert()["regions"], ("C3",))
        self.assertEqual(len(self.formal_violations(trial)), 1)

    def test_unavailable_without_prior_alarm_does_not_claim_safety(self):
        trial = self.trial()
        trial.update(frame(.1, valid=False))
        self.assert_inactive(trial)
        self.assertIn("無法確認", trial.alert()["text"])
        self.assertNotIn("安全", trial.alert()["text"])

    def test_position_loss_and_recovery_reset_the_same_engine_window(self):
        source = definition()
        for roi in source.rois:
            roi.anchor = "workpiece"
        trial = self.trial(source)
        for at in (.1, .2, .3, .4):
            result = frame(at, "C2")
            result.workpiece_transform = np.array([[1., 0., 0.], [0., 1., 0.]])
            trial.update(result)
        self.assertTrue(trial.alert()["active"])
        trial.update(frame(.5, "C2"))
        self.assert_inactive(trial)
        self.assertIn("無法確認", trial.alert()["text"])
        for at in (.6, .8, 1.0):
            result = frame(at, "C3")
            result.workpiece_transform = np.array([[1., 0., 0.], [0., 1., 0.]])
            trial.update(result)
            if at < 1.0:
                self.assert_inactive(trial)
        self.assertEqual(trial.alert()["regions"], ("C3",))
        self.assertTrue(trial.alert()["active"])
        self.assertEqual(len(self.formal_violations(trial)), 1)

    def test_stop_retains_labelled_history_and_no_live_regions(self):
        trial = self.trial(definition(hold=0))
        trial.update(frame(.1, "C2"))
        trial.stop()
        trial.update(frame(.2, "C3"))
        self.assert_inactive(trial)
        self.assertIn("已停止試跑", trial.alert()["text"])
        self.assertIn("C2", trial.alert()["text"])
        self.assertNotIn("C3", trial.alert()["text"])

    def test_notice_history_uses_short_summary_and_retains_diagnostic_metadata(self):
        trial = self.trial(definition(hold=0))
        trial.update(frame(.1, "C2"))
        trial.update(frame(.2, "C3"))
        notice = self.notices(trial)[0]
        self.assertIn("試跑提示（非正式）", notice.message)
        self.assertEqual(notice.data["regions"], ("C3",))
        trial.stop()
        text = trial.alert()["text"]
        self.assertIn("已停止試跑", text)
        self.assertIn("hand＠C3", text)
        self.assertNotIn("試跑提示", text)
        self.assertNotIn("違規警示", text)
        self.assertEqual(text.count("本輪已NG"), 1)

    def test_start_and_invalidate_clear_previous_warning(self):
        source = definition(hold=0)
        trial = self.trial(source)
        trial.update(frame(.1, "C2"))
        trial.invalidate()
        self.assertEqual(trial.alert(), dict(text="", regions=(), active=False))
        trial.start(source, "single", 0, .2)
        self.assertEqual(trial.alert(), dict(text="", regions=(), active=False))
        self.assertEqual(self.formal_violations(trial), [])
        self.assertEqual(self.notices(trial), [])

    def test_time_rewind_invalidates_warning(self):
        trial = self.trial(definition(hold=0))
        trial.update(frame(.5, "C2"))
        trial.update(frame(.4, "C3"))
        self.assertTrue(trial.stale)
        self.assertEqual(trial.alert(), dict(text="", regions=(), active=False))

    def test_frozen_timestamp_cannot_change_alert_or_log(self):
        trial = self.trial(definition(hold=0))
        trial.update(frame(.1, "C2"))
        before = list(trial.events)
        trial.update(frame(.1, "C3"))
        self.assertEqual(trial.alert()["regions"], ("C2",))
        self.assertEqual(list(trial.events), before)

    def test_same_frame_violation_and_advance_preserve_old_step_history(self):
        trial = self.trial(definition(hold=0, steps=2), mode="flow")
        trial.update(frame(.1, "C2", done=True))
        self.assertEqual(trial.engine.index, 1)
        self.assertEqual(trial.engine.snapshot().forbidden_results, [])
        self.assert_inactive(trial)
        self.assertIn("左把手安裝", trial.alert()["text"])
        self.assertIn("C2", trial.alert()["text"])
        self.assertNotIn("下一工序", trial.alert()["text"])
        self.assertEqual(self.formal_violations(trial)[0].data["regions"], ("C2",))
        self.assertEqual(self.notices(trial), [])

    def test_same_frame_violation_and_single_completion_preserve_ng_history(self):
        trial = self.trial(definition(hold=0))
        trial.update(frame(.1, "C3", done=True))
        self.assertTrue(trial.completed)
        self.assertFalse(trial.running)
        self.assert_inactive(trial)
        self.assertIn("單步驗證完成", trial.alert()["text"])
        self.assertIn("本輪已NG", trial.alert()["text"])
        self.assertIn("C3", trial.alert()["text"])
        self.assertEqual(self.formal_violations(trial)[0].data["regions"], ("C3",))
        self.assertEqual(self.notices(trial), [])

    def test_manual_confirm_updates_warning_to_completed_history(self):
        trial = self.trial(definition(hold=0, manual=True))
        trial.update(frame(.1, "C2"))
        trial.confirm()
        self.assertTrue(trial.completed)
        self.assert_inactive(trial)
        self.assertIn("本輪已NG", trial.alert()["text"])
        self.assertIn("最後警報", trial.alert()["text"])

    def test_new_flow_cycle_clears_last_cycle_alarm(self):
        trial = self.trial(definition(hold=0, cycle=True), mode="flow")
        trial.update(frame(.1, "C2", done=True))
        self.assertEqual(trial.engine.phase, Phase.CYCLE_DONE)
        self.assertIn("本輪已NG", trial.alert()["text"])
        trial.update(frame(.5))
        self.assertEqual(trial.engine.cycle, 2)
        self.assertEqual(trial.alert(), dict(text="", regions=(), active=False))
        self.assertTrue(trial.engine.cycle_ok)

    def test_zero_observation_immediately_uses_real_engine_decision(self):
        trial = self.trial(definition(hold=0, ratio=1))
        trial.update(frame(.1, "C3"))
        self.assertTrue(trial.alert()["active"])
        self.assertEqual(len(self.formal_violations(trial)), 1)

    def test_generic_timeout_alarm_has_clear_history(self):
        source = definition(hold=0)
        source.steps[0].timeout_sec = .2
        trial = self.trial(source)
        trial.update(frame(.2))
        self.assert_inactive(trial)
        self.assertIn("本輪已NG", trial.alert()["text"])
        self.assertIn("超時", trial.alert()["text"])

    def test_generic_sequence_and_too_fast_alarms_have_history(self):
        with self.subTest(alarm="sequence"):
            source = definition(steps=2)
            source.cycle.detect_sequence = True
            source.steps[1].conditions = [Condition(label="hand", roi="C3")]
            source.steps[1].hold_sec = 0
            trial = self.trial(source, mode="flow")
            self.feed(trial, (.1, .2), "C3")
            self.assertIn("順序錯誤", trial.alert()["text"])
            self.assert_inactive(trial)
        with self.subTest(alarm="too_fast"):
            source = definition()
            source.steps[0].min_duration_sec = 3
            trial = self.trial(source)
            trial.update(frame(.1, done=True))
            self.assertIn("工時過短", trial.alert()["text"])
            self.assertIn("本輪已NG", trial.alert()["text"])
            self.assert_inactive(trial)

    def test_single_selected_step_keeps_original_number(self):
        source = definition()
        source.steps.insert(0, Step("之前工序"))
        source.steps[1].forbidden_hold_sec = 0
        trial = self.trial(source, selected=1)
        trial.update(frame(.1, "C2"))
        self.assertIn("工序 2", trial.alert()["text"])

    def test_interval_delays_only_log_while_display_updates_immediately(self):
        trial = self.trial(definition(interval=2))
        self.feed(trial, (.1, .2, .3), "C2")
        self.assertEqual(len(self.formal_violations(trial)), 1)
        self.assertEqual(self.formal_violations(trial)[0].time, .3)
        trial.update(frame(.5, "C3"))
        self.assertTrue(trial.alert()["active"])
        self.assertEqual(trial.alert()["regions"], ("C3",))
        self.assertIn("C3", trial.alert()["text"])
        self.assertEqual(self.notices(trial), [])
        for _ in range(20):
            trial.alert()
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(2.29, "C3"))
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(2.31, "C3"))
        self.assertEqual(len(self.notices(trial)), 1)
        self.assertEqual(self.notices(trial)[0].time, 2.31)
        self.assertEqual(self.notices(trial)[0].data["regions"], ("C3",))
        self.assertEqual(len(self.formal_violations(trial)), 1)

    def test_interval_does_not_repeat_unchanged_violation(self):
        trial = self.trial(definition(hold=0, interval=2))
        self.feed(trial, (.1, .2, 2.2, 5.0, 10.0), "C2")
        self.assertEqual(len(self.formal_violations(trial)), 1)
        self.assertEqual(self.notices(trial), [])

    def test_interval_rapid_switches_keep_only_latest_pending_notice(self):
        trial = self.trial(definition(interval=2))
        self.feed(trial, (.1, .2, .3), "C2")
        for at, region in ((.5, "C3"), (.7, "C2"), (.9, "C3"), (1.1, "C2"), (1.4, "C3")):
            trial.update(frame(at, region))
            self.assertEqual(trial.alert()["regions"], (region,))
            self.assertEqual(self.notices(trial), [])
        trial.update(frame(2.4, "C3"))
        self.assertEqual(len(self.notices(trial)), 1)
        self.assertEqual(self.notices(trial)[0].data["regions"], ("C3",))
        self.feed(trial, (2.5, 4.5, 8.5), "C3")
        self.assertEqual(len(self.notices(trial)), 1)

    def test_interval_is_measured_from_last_actual_record(self):
        trial = self.trial(definition(hold=0, interval=2))
        trial.update(frame(.25, "C2"))
        trial.update(frame(.5, "C3"))
        trial.update(frame(2.25, "C3"))
        self.assertEqual(len(self.notices(trial)), 1)
        trial.update(frame(2.5, "C2"))
        trial.update(frame(4.24, "C2"))
        self.assertEqual(len(self.notices(trial)), 1)
        trial.update(frame(4.25, "C2"))
        self.assertEqual([notice.time for notice in self.notices(trial)], [2.25, 4.25])

    def test_pending_is_cancelled_when_condition_clears_or_is_unavailable(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                trial = self.trial(definition(hold=0, interval=2))
                trial.update(frame(.1, "C2"))
                trial.update(frame(.5, "C3"))
                self.assertIsNotNone(trial._pending_notice)
                trial.update(frame(.6, valid=not invalid))
                self.assertIsNone(trial._pending_notice)
                self.feed(trial, (2.5, 4.5))
                self.assertEqual(self.notices(trial), [])
                self.assert_inactive(trial)

    def test_pending_is_cancelled_on_stop_or_invalidate(self):
        for action in ("stop", "invalidate"):
            with self.subTest(action=action):
                trial = self.trial(definition(hold=0, interval=2))
                trial.update(frame(.1, "C2"))
                trial.update(frame(.5, "C3"))
                getattr(trial, action)()
                self.assertIsNone(trial._pending_notice)
                trial.update(frame(3.0, "C3"))
                self.assertEqual(self.notices(trial), [])
                self.assert_inactive(trial)
                if action == "invalidate":
                    self.assertEqual(trial.alert()["text"], "")

    def test_stopped_history_keeps_latest_unlogged_region(self):
        trial = self.trial(definition(hold=0, interval=2))
        trial.update(frame(.1, "C2"))
        trial.update(frame(.5, "C3"))
        self.assertEqual(self.notices(trial), [])
        self.assertEqual(self.formal_violations(trial)[0].data["regions"], ("C2",))
        trial.stop()
        self.assertIn("已停止試跑", trial.alert()["text"])
        self.assertIn("C3", trial.alert()["text"])
        self.assertNotIn("C2", trial.alert()["text"])
        self.assertEqual(self.notices(trial), [])
        self.assert_inactive(trial)

    def test_pending_is_cancelled_on_step_advance_or_completion(self):
        for steps, mode in ((1, "single"), (2, "flow")):
            with self.subTest(steps=steps):
                trial = self.trial(definition(hold=0, interval=2, steps=steps), mode=mode)
                trial.update(frame(.1, "C2"))
                trial.update(frame(.5, "C3"))
                trial.update(frame(.6, done=True))
                self.assertIsNone(trial._pending_notice)
                trial.update(frame(3.0))
                self.assertEqual(self.notices(trial), [])
                self.assert_inactive(trial)

    def test_interval_reentry_logs_only_after_interval_while_still_present(self):
        trial = self.trial(definition(hold=0, interval=2))
        trial.update(frame(.1, "C2"))
        trial.update(frame(.5))
        trial.update(frame(.8, "C2"))
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(2.2, "C2"))
        self.assertEqual(len(self.notices(trial)), 1)
        self.assertEqual(len(self.formal_violations(trial)), 1)

    def test_restart_clears_pending_and_starts_new_interval(self):
        source = definition(hold=0, interval=2)
        trial = self.trial(source)
        trial.update(frame(.1, "C2"))
        trial.update(frame(.5, "C3"))
        trial.stop()
        trial.start(source, "single", 0, 5.0)
        self.assertIsNone(trial._pending_notice)
        self.assertEqual(trial.alert()["text"], "")
        trial.update(frame(5.1, "C3"))
        self.assertEqual(len(self.formal_violations(trial)), 1)
        trial.update(frame(5.2, "C2"))
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(7.2, "C2"))
        self.assertEqual(len(self.notices(trial)), 1)

    def test_new_step_has_immediate_formal_alarm_and_own_notice_interval(self):
        source = definition(hold=0, interval=2)
        following = copy.deepcopy(source.steps[0])
        following.name = "右把手安裝"
        source.steps.append(following)
        trial = self.trial(source, mode="flow")
        trial.update(frame(.1, "C2"))
        trial.update(frame(.5, "C3"))
        trial.update(frame(.6, done=True))
        trial.update(frame(.7, "C3"))
        self.assertEqual([event.step_index for event in self.formal_violations(trial)], [0, 1])
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(.8, "C2"))
        trial.update(frame(2.2, "C2"))
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(2.8, "C2"))
        self.assertEqual(len(self.notices(trial)), 1)
        self.assertEqual(self.notices(trial)[0].step_index, 1)

    def test_new_cycle_clears_pending_and_starts_new_interval(self):
        trial = self.trial(definition(hold=0, interval=2, cycle=True), mode="flow")
        trial.update(frame(.1, "C2"))
        trial.update(frame(.2, "C3"))
        trial.update(frame(.3, done=True))
        trial.update(frame(.7))
        self.assertEqual(trial.engine.cycle, 2)
        self.assertIsNone(trial._pending_notice)
        self.assertEqual(trial.alert()["text"], "")
        trial.update(frame(.8, "C3"))
        self.assertEqual(len(self.formal_violations(trial)), 2)
        trial.update(frame(.9, "C2"))
        trial.update(frame(2.2, "C2"))
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(2.9, "C2"))
        self.assertEqual(len(self.notices(trial)), 1)
        self.assertEqual(self.notices(trial)[0].cycle, 2)

    def test_message_interval_never_delays_other_formal_alarms(self):
        source = definition(hold=0, interval=50)
        source.steps[0].timeout_sec = .2
        trial = self.trial(source)
        trial.update(frame(.1, "C2"))
        trial.update(frame(.2, "C3"))
        self.assertEqual([event.data["alarm"] for event in trial.events if event.kind == "alarm"],
                         ["violation", "timeout"])
        self.assertEqual(self.notices(trial), [])
        self.assertEqual(trial.alert()["regions"], ("C3",))

    def test_continuing_violation_keeps_timeout_visible_now_and_after_clear(self):
        source = definition(hold=0, interval=2)
        source.steps[0].timeout_sec = .2
        trial = self.trial(source)
        trial.update(frame(.1, "C2"))
        self.assertNotIn("超時", trial.alert()["text"])
        trial.update(frame(.2, "C2"))
        self.assertTrue(trial.alert()["active"])
        self.assertIn("hand＠C2", trial.alert()["text"])
        self.assertIn("超時", trial.alert()["text"])
        self.assertEqual(trial.alert()["text"].count("超時"), 1)
        trial.update(frame(.3, "C3"))
        self.assertIn("hand＠C3", trial.alert()["text"])
        self.assertIn("超時", trial.alert()["text"])
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(.4))
        self.assert_inactive(trial)
        self.assertIn("hand＠C3", trial.alert()["text"])
        self.assertIn("超時", trial.alert()["text"])
        self.assertEqual(trial.alert()["text"].count("超時"), 1)
        self.assertEqual([event.data["alarm"] for event in trial.events if event.kind == "alarm"],
                         ["violation", "timeout"])
        self.assertEqual(self.notices(trial), [])

    def test_timeout_does_not_reset_violation_notice_interval(self):
        source = definition(hold=0, interval=2)
        source.steps[0].timeout_sec = 1.0
        trial = self.trial(source)
        trial.update(frame(.25, "C2"))
        trial.update(frame(.5, "C3"))
        trial.update(frame(1.0, "C3"))
        self.assertIn("超時", trial.alert()["text"])
        self.assertEqual(self.notices(trial), [])
        trial.update(frame(2.25, "C3"))
        self.assertEqual([notice.time for notice in self.notices(trial)], [2.25])
        self.assertIn("超時", trial.alert()["text"])
        trial.stop()
        self.assertIn("超時", trial.alert()["text"])
        self.assertIn("C3", trial.alert()["text"])

    def test_generic_alarm_clears_with_start_cycle_and_invalidate(self):
        for action in ("start", "cycle", "invalidate"):
            with self.subTest(action=action):
                source = definition(hold=0, interval=2, cycle=action == "cycle")
                source.steps[0].timeout_sec = .2
                trial = self.trial(source, mode="flow" if action == "cycle" else "single")
                trial.update(frame(.1, "C2"))
                trial.update(frame(.2, "C2"))
                self.assertIn("超時", trial.alert()["text"])
                if action == "start":
                    trial.start(source, "single", 0, .3)
                elif action == "cycle":
                    trial.update(frame(.3, done=True))
                    trial.update(frame(.7))
                    self.assertEqual(trial.engine.cycle, 2)
                else:
                    trial.invalidate()
                self.assertEqual(trial.alert(), dict(text="", regions=(), active=False))


if __name__ == "__main__":
    unittest.main()
