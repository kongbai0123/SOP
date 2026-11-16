"""Monitor never invents positions; tracking feedback stays in details and tooltips."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from sop_app.detection import (TRACKING_CONFIRMED, TRACKING_RELOCATING, TRACKING_UNCERTAIN,
                               FrameResult)
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step, Workpiece
from sop_app.tracking import workpiece_key
from sop_app.ui.run_page import RunPage


IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])
POINTS = [(.2, .2), (.4, .2), (.4, .4), (.2, .4)]


def definition(*, reference=True, automatic=True):
    return SOPDefinition(
        name="Tracking feedback",
        workpiece=Workpiece(reference_png="fixture", points=POINTS) if reference else None,
        rois=[ROI("固定料盒", POINTS), ROI("安裝位置", POINTS, "workpiece")],
        steps=[Step("安裝", conditions=[Condition(label="part", roi="安裝位置")] if automatic else [])])


def packet(sop, *, matrix=None, state=TRACKING_RELOCATING, key=None, status="重新定位中"):
    frame = np.zeros((120, 160, 3), np.uint8)
    result = FrameResult(160, 120, 0., workpiece_transform=matrix,
                         tracking_state=state, tracking_status=status,
                         workpiece_key=workpiece_key(sop.workpiece) if key is None else key)
    return FramePacket(frame, frame.copy(), result, None, 10., 5.)


class RunTrackingFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def page(self, sop, *, error=""):
        page = RunPage()
        self.addCleanup(self.dispose, page)
        with patch("sop_app.ui.run_page.WorkpieceLocator",
                   side_effect=ValueError(error) if error else None,
                   return_value=SimpleNamespace()) as constructor:
            page.set_sop(sop)
        if sop.workpiece:
            constructor.assert_called_once_with(page.sop.workpiece, adaptive_budget=True)
        return page

    @staticmethod
    def dispose(page):
        page.close()
        for timer in page.findChildren(QTimer):
            timer.stop()

    @staticmethod
    def names(page):
        return [roi.name for roi in page.video._rois]

    def confirm(self, page, sop):
        matrix = IDENTITY.copy()
        matrix[0, 2] = .1
        page.on_packet(packet(sop, matrix=matrix, state=TRACKING_CONFIRMED, status="位置已確認"))
        return matrix

    def test_never_confirmed_does_not_show_identity_or_call_it_last_position(self):
        sop = definition()
        original = deepcopy(sop.to_dict())
        page = self.page(sop)
        self.assertEqual(self.names(page), ["固定料盒"])
        self.assertEqual(page.video._stale, set())
        self.assertIn("未顯示 1 個跟隨區域", page.tracking_label.text())
        self.assertNotIn("最後", page.tracking_label.text())
        page.on_packet(packet(sop))
        self.assertEqual(self.names(page), ["固定料盒"])
        self.assertIsNone(page._last_transform)
        self.assertEqual(sop.to_dict(), original)

    def test_uncertain_raw_candidate_cannot_become_confirmed_or_create_first_position(self):
        for state in (TRACKING_UNCERTAIN, TRACKING_RELOCATING):
            with self.subTest(state=state):
                sop = definition()
                page = self.page(sop)
                page.on_packet(packet(sop, matrix=IDENTITY.copy(), state=state, status="位置待確認"))
                self.assertIsNone(page._transform)
                self.assertIsNone(page._last_transform)
                self.assertEqual(self.names(page), ["固定料盒"])
                self.assertNotIn("位置跟隨正常", page.tracking_label.text())

    def test_first_uncertain_frame_keeps_only_confirmed_old_position_and_marks_it(self):
        sop = definition()
        page = self.page(sop)
        confirmed = self.confirm(page, sop)
        old_points = page.video._rois[1].points
        candidate = IDENTITY.copy()
        candidate[0, 2] = .4
        page.on_packet(packet(sop, matrix=candidate, state=TRACKING_UNCERTAIN, status="位置待確認"))
        self.assertIsNone(page._transform)
        np.testing.assert_array_equal(page._last_transform, confirmed)
        self.assertEqual(page.video._rois[1].points, old_points)
        self.assertEqual(page.video._stale, {"安裝位置"})
        self.assertIn("舊位置，僅供參考", page.tracking_label.text())
        self.assertIn("位置待確認", page.tracking_label.toolTip())
        self.assertIn("跟隨條件停止累積", page.tracking_label.text())
        self.assertNotIn("位置跟隨正常", page.tracking_label.text())

    def test_different_reference_packet_discards_old_position(self):
        sop = definition()
        page = self.page(sop)
        self.confirm(page, sop)
        page.on_packet(packet(sop, matrix=IDENTITY.copy(), state=TRACKING_CONFIRMED, key="old-reference"))
        self.assertIsNone(page._last_transform)
        self.assertIsNone(page._transform)
        self.assertEqual(self.names(page), ["固定料盒"])
        self.assertIn("同步", page.tracking_label.toolTip())
        self.assertNotIn("最後", page.tracking_label.text())

    def test_malformed_confirmed_position_cannot_persist_an_old_position(self):
        for bad in (np.eye(3), np.full((2, 3), np.nan), [["bad"]]):
            with self.subTest(matrix=repr(bad)):
                sop = definition()
                page = self.page(sop)
                self.confirm(page, sop)
                page.on_packet(packet(sop, matrix=bad, state=TRACKING_CONFIRMED, status="位置已確認"))
                self.assertIsNone(page._last_transform)
                self.assertIsNone(page._transform)
                self.assertEqual(self.names(page), ["固定料盒"])
                self.assertIn("定位資料無效", page.tracking_label.toolTip())
                self.assertNotIn("位置已確認", page.tracking_label.toolTip())

    def test_invalid_reference_keeps_start_guard_without_notice_under_start_button(self):
        sop = definition()
        page = self.page(sop, error="工件特徵不足")
        self.assertTrue(page.details_panel.isHidden())
        self.assertIsNone(page.findChild(QLabel, "trackingNotice"))
        self.assertTrue(page.details_panel.isAncestorOf(page.tracking_label))
        self.assertIn("工件特徵不足", page.start_button.toolTip())
        self.assertIn("編輯 SOP → 作業區域", page.start_button.toolTip())
        self.assertFalse(page.start_button.isEnabled())
        page.on_packet(packet(sop, matrix=IDENTITY.copy(), state=TRACKING_CONFIRMED))
        page.set_display_groups({"objects"})
        self.assertIn("工件特徵不足", page.start_button.toolTip())
        self.assertIsNone(page.findChild(QLabel, "trackingNotice"))
        self.assertIsNone(page._last_transform)

    def test_pipeline_configuration_failure_clears_old_position_and_blocks_start(self):
        sop = definition()
        page = self.page(sop)
        self.confirm(page, sop)
        failed = packet(sop, status="工件定位設定無效：參考品質不足")
        failed.result.tracking_metrics = {"configuration_error": True}
        page.on_packet(failed)
        self.assertIsNone(page._last_transform)
        self.assertFalse(page.start_button.isEnabled())
        self.assertIn("參考品質不足", page.start_button.toolTip())
        self.assertIn("重新設定參考工件", page.start_button.toolTip())
        self.assertEqual(self.names(page), ["固定料盒"])
        self.confirm(page, sop)
        self.assertTrue(page.start_button.isEnabled())
        self.assertNotIn("參考品質不足", page.start_button.toolTip())

    def test_missing_reference_blocks_only_conditions_that_use_following_regions(self):
        sop = definition(reference=False)
        page = self.page(sop)
        self.assertFalse(page.start_button.isEnabled())
        self.assertIn("設定參考工件", page.start_button.toolTip())
        started = Mock()
        page.start_requested.connect(started)
        page._on_start_clicked()  # Guard direct invocation as well as disabled clicks.
        started.assert_not_called()
        for mode in ("manual", "fixed", "whole-frame"):
            with self.subTest(mode=mode):
                usable = definition(reference=False, automatic=mode != "manual")
                if mode != "manual":
                    usable.steps[0].conditions[0].roi = "固定料盒" if mode == "fixed" else ""
                other = self.page(usable)
                self.assertTrue(other.start_button.isEnabled())

    def test_invalid_decorative_reference_does_not_block_manual_or_fixed_conditions(self):
        for automatic in (True, False):
            with self.subTest(automatic=automatic):
                sop = definition(automatic=automatic)
                if automatic:
                    sop.steps[0].conditions[0].roi = "固定料盒"
                page = self.page(sop, error="參考工件無效")
                self.assertTrue(page.start_button.isEnabled())
                started = Mock()
                page.start_requested.connect(started)
                page.start_button.click()
                started.assert_called_once()

    def test_valid_reference_can_start_while_waiting_for_position(self):
        page = self.page(definition())
        self.assertIsNone(page._transform)
        self.assertTrue(page.start_button.isEnabled())
        self.assertIn("暫停累積", page.start_button.toolTip())

    def test_forbidden_and_enabled_cycle_reset_also_require_reference(self):
        for source in ("forbidden", "reset"):
            with self.subTest(source=source):
                sop = definition(reference=False, automatic=False)
                conditions = [Condition(label="part", roi="安裝位置")]
                if source == "forbidden":
                    sop.steps[0].forbidden = conditions
                else:
                    sop.cycle.reset_conditions = conditions
                page = self.page(sop)
                self.assertFalse(page.start_button.isEnabled())
                if source == "reset":
                    sop.cycle.enabled = False
                    page.set_sop(sop)
                    self.assertTrue(page.start_button.isEnabled())

    def test_stop_never_becomes_disabled_when_tracking_setup_is_invalid(self):
        sop = definition()
        page = self.page(sop, error="參考工件無效")
        self.assertFalse(page.start_button.isEnabled())
        page.set_running(True)
        page.on_packet(packet(sop))
        self.assertTrue(page.start_button.isEnabled())
        self.assertEqual(page.start_button.text(), "停止作業")
        stopped = Mock()
        page.stop_requested.connect(stopped)
        with patch("sop_app.ui.run_page.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            page.start_button.click()
        stopped.assert_called_once()

    def test_new_sop_and_cleared_source_never_keep_previous_position(self):
        sop = definition()
        page = self.page(sop)
        self.confirm(page, sop)
        page.clear_video("來源已中斷")
        self.assertIsNone(page._last_transform)
        self.assertEqual(self.names(page), [])
        page.set_display_groups({"operation", "installation"})
        self.assertEqual(self.names(page), ["固定料盒"])
        self.assertNotIn("最後", page.tracking_label.text())

    def test_out_of_frame_is_distinct_from_never_confirmed(self):
        sop = definition()
        page = self.page(sop)
        outside = IDENTITY.copy()
        outside[0, 2] = .8
        page.on_packet(packet(sop, matrix=outside, state=TRACKING_CONFIRMED))
        self.assertEqual(self.names(page), ["固定料盒"])
        self.assertIn("區域超出畫面", page.tracking_label.text())
        self.assertNotIn("尚未取得", page.tracking_label.text())


if __name__ == "__main__":
    unittest.main()
