"""Unavailable causes stay inspectable while tracking notifications are hidden."""
from copy import deepcopy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sop_app.conditions import evaluate_condition
from sop_app.detection import Detection, FrameResult, TRACKING_CONFIRMED, TRACKING_RELOCATING
from sop_app.engine import SOPEngine
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, CycleSettings, ROI, SOPDefinition, Step, Workpiece
from sop_app.ui.condition_diagnostics import condition_error_text
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.editor_validation import EditorValidation
from sop_app.ui.run_page import RunPage


IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])
LEFT = "左把手安裝區"
RIGHT = "右把手安裝區"
POINTS = [(.1, .1), (.4, .1), (.4, .4), (.1, .4)]


def definition():
    return SOPDefinition(
        rois=[ROI(LEFT, POINTS, "workpiece"),
              ROI(RIGHT, [(.6, .1), (.9, .1), (.9, .4), (.6, .4)], "workpiece")],
        workpiece=Workpiece(reference_png="fixture", points=POINTS),
        steps=[Step("安裝 L", conditions=[Condition(label="L_grip_done", roi=LEFT)],
                    forbidden=[Condition(label="R_grip_done", roi=LEFT)], hold_sec=10)],
        cycle=CycleSettings(enabled=False))


def frame(at=.1, *, matrix=None, valid=True, metrics=None, labels=()):
    return FrameResult(200, 100, at,
                       [Detection(label, .95, (25, 15, 65, 35)) for label in labels],
                       workpiece_transform=matrix, detection_valid=valid,
                       tracking_state=TRACKING_RELOCATING if matrix is None else TRACKING_CONFIRMED,
                       tracking_status="重新定位中" if matrix is None else "位置已確認",
                       tracking_metrics=metrics or {})


class ConditionReasonTests(unittest.TestCase):
    def trial(self, sop=None, mode="single"):
        trial = EditorValidation()
        trial.start(sop or definition(), mode, 0, 0)
        return trial

    def reason(self, observed, sop=None):
        sop = sop or definition()
        result = evaluate_condition(sop.steps[0].conditions[0], observed, sop.roi_map())
        return condition_error_text(result, observed, sop)

    def test_structured_causes_distinguish_pending_outside_model_and_configuration(self):
        pending = self.reason(frame())
        self.assertIn(f"「{LEFT}」尚未定位", pending)
        self.assertIn("重新定位中", pending)
        self.assertIn("短暫靜止", pending)
        outside = self.reason(frame(matrix=IDENTITY + np.array([[0., 0., .9], [0., 0., 0.]])))
        self.assertIn("已超出畫面", outside)
        self.assertNotIn("尚未定位", outside)
        self.assertIn("模型尚未就緒", self.reason(frame(valid=False)))
        invalid_reference = self.reason(frame(metrics={"configuration_error": True}))
        self.assertIn("參考工件設定無效", invalid_reference)
        self.assertIn("重新設定參考工件", invalid_reference)
        no_reference = definition()
        no_reference.workpiece = None
        self.assertIn("未設定參考工件", self.reason(frame(), no_reference))
        no_roi = definition()
        no_roi.rois = []
        self.assertIn("找不到區域 " + LEFT, self.reason(frame(), no_roi))
        self.assertIn("選擇有效區域", self.reason(frame(), no_roi))
        self.assertEqual(self.reason(frame(matrix=IDENTITY)), "")

    def test_completion_and_warning_keep_scope_index_class_and_region(self):
        sop = definition()
        sop.steps[0].forbidden[0].roi = RIGHT
        original = deepcopy(sop.to_dict())
        trial = self.trial(sop)
        trial.update(frame())
        before = list(trial.events)
        reason = trial.unavailable_reasons()
        self.assertIn("完成條件 1｜L_grip_done＠" + LEFT, reason)
        self.assertIn("警告條件 1｜R_grip_done＠" + RIGHT, reason)
        self.assertEqual(trial.metrics()["count"], "無法判斷")
        self.assertEqual(list(trial.events), before)
        self.assertTrue(trial.engine.cycle_ok)
        self.assertEqual(sop.to_dict(), original)

    def test_observed_rejections_are_specific_deduplicated_and_do_not_guess_occlusion(self):
        metrics = {"diagnostics": {
            "template": {"reason": "ambiguous", "best_score": .72, "competitor_score": .70},
            "orb": {"reason": "matches_insufficient", "matches": 3},
            "local_orb": {"reason": "matches_insufficient", "matches": 2},
            "sift": {"reason": "frame_features_insufficient", "frame_count": 1},
        }}
        reason = self.reason(frame(metrics=metrics))
        self.assertIn("本幀檢查：有多個相似位置、特徵配對不足", reason)
        self.assertEqual(reason.count("特徵配對不足"), 1)
        self.assertNotIn("當幀特徵不足", reason)  # Keep the inspection short.
        self.assertNotIn("遮擋", reason)
        self.assertNotIn("0.72", reason)  # Per-frame counters should not make the explanation flicker.
        low_similarity = self.reason(frame(metrics={"diagnostics": {
            "template": {"reason": "score_below_threshold", "best_score": .4}}}))
        self.assertIn("輪廓相似度不足", low_similarity)
        geometry = self.reason(frame(metrics={"diagnostics": {
            "orb": {"reason": "geometry_rejected", "inliers": 2}}}))
        self.assertIn("特徵配對的幾何驗證未通過", geometry)

    def test_absent_unknown_and_deferred_diagnostics_preserve_existing_fallback(self):
        baseline = self.reason(frame())
        for diagnostics in (None, [], {}, {"template": None}, {"template": {"reason": "accepted"}},
                            {"orb": {"reason": "compute_budget_deferred"}},
                            {"orb": {"reason": "new_unknown_reason"}}):
            with self.subTest(diagnostics=diagnostics):
                self.assertEqual(self.reason(frame(metrics={"diagnostics": diagnostics})), baseline)

    def test_diagnostic_numeric_changes_keep_the_same_current_reason(self):
        first = self.reason(frame(metrics={"diagnostics": {
            "template": {"reason": "score_below_threshold", "best_score": .4},
            "orb": {"reason": "matches_insufficient", "matches": 3}}}))
        second = self.reason(frame(metrics={"diagnostics": {
            "template": {"reason": "score_below_threshold", "best_score": .5},
            "orb": {"reason": "matches_insufficient", "matches": 5}}}))
        self.assertEqual(first, second)

    def test_warning_only_unavailable_does_not_claim_completion_blocked(self):
        sop = definition()
        sop.steps[0].conditions[0].roi = ""
        trial = self.trial(sop)
        trial.update(frame())
        self.assertEqual(trial.metrics()["count"], "0/1")
        self.assertNotIn("完成條件", trial.unavailable_reasons())
        self.assertIn("警告條件 1", trial.unavailable_reasons())

    def test_any_mode_reports_individual_issue_without_falsely_blocking_group(self):
        sop = definition()
        sop.steps[0].conditions.insert(0, Condition(label="L_grip_done"))
        sop.steps[0].completion_mode = "any"
        sop.steps[0].forbidden = []
        trial = self.trial(sop)
        trial.update(frame(labels=("L_grip_done",)))
        self.assertEqual(trial.metrics()["count"], "1/2")
        self.assertIn("完成條件 2", trial.unavailable_reasons())
        self.assertFalse(trial.engine._completion_invalid)

    def test_cycle_reset_is_explained_as_reset_instead_of_completed_step(self):
        sop = definition()
        sop.steps[0].conditions = []
        sop.steps[0].forbidden = []
        sop.cycle.enabled = True
        sop.cycle.reset_conditions = [Condition(type="disappear", label="L_grip_done", roi=LEFT)]
        trial = self.trial(sop, "flow")
        trial.confirm()
        trial.update(frame())
        self.assertIn("重置條件 1", trial.unavailable_reasons())
        self.assertNotIn("完成條件", trial.unavailable_reasons())
        self.assertEqual(trial.metrics()["count"], "無法判斷")

    def test_current_reason_clears_on_recovery_stop_or_settings_invalidation(self):
        for operation in ("recover", "stop", "invalidate"):
            with self.subTest(operation=operation):
                trial = self.trial()
                trial.update(frame())
                self.assertTrue(trial.unavailable_reasons())
                if operation == "recover":
                    trial.update(frame(.2, matrix=IDENTITY))
                else:
                    getattr(trial, operation)()
                self.assertEqual(trial.unavailable_reasons(), "")


class RunConditionReasonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def page(self, sop):
        page = RunPage()
        self.addCleanup(page.close)
        self.addCleanup(lambda: [timer.stop() for timer in page.findChildren(QTimer)])
        with patch("sop_app.ui.run_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.set_sop(sop)
        page.set_running(True)
        page.set_alert_options(False, False)
        return page

    @staticmethod
    def update(page, observed, engine):
        raw = np.zeros((observed.height, observed.width, 3), np.uint8)
        page.on_packet(FramePacket(raw, raw.copy(), observed, engine.snapshot(), 10., 1.))

    def test_existing_details_explain_both_completion_and_warning_without_alert_overlay(self):
        sop = definition()
        sop.steps[0].conditions[0].label = "<b>part</b>"
        page = self.page(sop)
        engine = SOPEngine(sop)
        engine.start(0.)
        observed = frame()
        engine.update(observed)
        self.update(page, observed, engine)
        text = page.condition_label.text()
        self.assertIn("完成條件 1 暫停判定", text)
        self.assertIn("警告條件 1 無法判斷", text)
        self.assertIn(LEFT, text)
        self.assertIn("&lt;b&gt;part&lt;/b&gt;", text)
        self.assertIn("尚未定位", text)
        self.assertEqual(page.video._alert, "")
        self.assertTrue(engine.cycle_ok)

    def test_cycle_reset_details_do_not_mislabel_unavailable_as_unmet(self):
        sop = definition()
        sop.steps[0].conditions = []
        sop.steps[0].forbidden = []
        sop.cycle.enabled = True
        sop.cycle.reset_conditions = [Condition(type="disappear", label="L_grip_done", roi=LEFT)]
        page = self.page(sop)
        engine = SOPEngine(sop)
        engine.start(0.)
        engine.confirm_manual_step(0.)
        observed = frame()
        engine.update(observed)
        self.update(page, observed, engine)
        self.assertIn("重置條件無法判斷", page.condition_label.text())
        self.assertIn(LEFT, page.condition_label.text())
        self.assertNotIn("未成立", page.condition_label.text())


class ValidationReasonUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def page(self, sop=None):
        page = EditorPage()
        self.addCleanup(page.deleteLater)
        self.addCleanup(page.close)
        self.addCleanup(lambda: [timer.stop() for timer in page.findChildren(QTimer)])
        with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()), \
             patch("sop_app.ui.editor_page.decode_reference", return_value=np.zeros((100, 200, 3), np.uint8)):
            page.set_sop(sop or definition())
        page.set_alert_options(False, False)
        page._validation.start(page.sop, "single", 0, 0)
        page._validation.update(frame())
        page._update_validation_view()
        return page

    def test_reason_is_user_initiated_with_tracking_off_and_preserves_engine(self):
        page = self.page()
        trial = page._validation
        events = list(trial.events)
        original = deepcopy(page.sop.to_dict())
        self.assertIn('href="condition-reasons"', page.validation_count.text())
        self.assertIn("無法判斷（原因）", page.validation_count.text())
        self.assertIn(LEFT, page.validation_count.toolTip())
        self.assertTrue(page.validation_alert.isHidden())
        self.assertEqual(page.video._alert, "")
        with patch("sop_app.ui.editor_page.QToolTip.showText") as popup:
            page._update_validation_view()
            popup.assert_not_called()
            page.validation_count.linkActivated.emit("condition-reasons")
            popup.assert_called_once()
            self.assertIn("完成條件 1", popup.call_args.args[1])
            self.assertIn("尚未定位", popup.call_args.args[1])
        self.assertEqual(list(trial.events), events)
        self.assertEqual(page.sop.to_dict(), original)
        self.assertTrue(trial.engine.cycle_ok)

    def test_recovery_clears_link_tooltip_and_open_reason(self):
        page = self.page()
        with patch("sop_app.ui.editor_page.QToolTip.showText"), \
             patch("sop_app.ui.editor_page.QToolTip.hideText") as close_popup:
            page.validation_count.linkActivated.emit("condition-reasons")
            page._validation.update(frame(.2, matrix=IDENTITY))
            page._update_validation_view()
            close_popup.assert_called_once()
        self.assertNotIn("href=", page.validation_count.text())
        self.assertEqual(page.validation_count.toolTip(), "")
        self.assertEqual(page.validation_count.accessibleDescription(), "")
        self.assertEqual(page._validation_reason_popup_text, "")

    def test_tracking_counters_do_not_close_an_open_reason_on_every_frame(self):
        page = self.page()
        with patch("sop_app.ui.editor_page.QToolTip.showText"), \
             patch("sop_app.ui.editor_page.QToolTip.hideText") as close_popup:
            page.validation_count.linkActivated.emit("condition-reasons")
            observed = frame(.2)
            observed.tracking_status = "重新定位中（2/3）"
            page._validation.update(observed)
            page._update_validation_view()
            close_popup.assert_not_called()

    def test_condition_names_are_escaped_in_inspectable_tooltip(self):
        sop = definition()
        sop.steps[0].conditions[0].label = "<b>custom</b>"
        page = self.page(sop)
        self.assertIn("&lt;b&gt;custom&lt;/b&gt;", page.validation_count.toolTip())
        self.assertNotIn("<b>custom</b>", page.validation_count.toolTip())
        self.assertIn("<b>custom</b>", page.validation_count.accessibleDescription())


if __name__ == "__main__":
    unittest.main()
