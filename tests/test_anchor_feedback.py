"""Anchor selection gives local, actionable feedback without changing rejected data."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sop_app.ui.editor_page import EditorPage
from test_follow_roi_editing import definition, packet, POINTS, POSE


class AnchorFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def page(self):
        page = EditorPage()
        def dispose():
            page.close()
            for timer in page.findChildren(QTimer):
                timer.stop()
        self.addCleanup(dispose)
        sop = definition()
        # A fixed region whose inverse coordinates are inside the reference.
        sop.rois[1].points = list(POINTS)
        with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.set_sop(sop)
        page.edit_rois_button.setChecked(True)
        page.on_packet(packet(page.sop))
        return page

    @staticmethod
    def row(page, name="Fixed"):
        return next(row for row in page.roi_rows if row.roi.name == name)

    def choose(self, row, anchor):
        row.anchor_combo.setCurrentIndex(row.anchor_combo.findData(anchor))

    def reject(self, page, reason_words, action_words):
        row = self.row(page)
        before = deepcopy(page.sop.to_dict())
        history = deepcopy((page._roi_undo, page._roi_redo))
        changed = Mock()
        page.modified.connect(changed)
        self.choose(row, "workpiece")
        self.assertIs(self.row(page), row, "Failed selection should keep its row and focus.")
        self.assertEqual(row.anchor_combo.currentData(), "fixed")
        self.assertEqual(page.sop.to_dict(), before)
        self.assertEqual((page._roi_undo, page._roi_redo), history)
        changed.assert_not_called()
        feedback = row.anchor_feedback
        self.assertFalse(feedback.isHidden())
        self.assertTrue(feedback.wordWrap())
        self.assertEqual(feedback.property("state"), "warning")
        message = feedback.text()
        self.assertTrue(any(word in message for word in reason_words), message)
        self.assertTrue(any(word in message for word in action_words), message)
        return message

    def test_existing_modes_have_local_explanation_before_interaction(self):
        page = self.page()
        fixed = self.row(page)
        self.assertEqual(fixed.anchor_combo.currentData(), "fixed")
        self.assertTrue(fixed.anchor_feedback.isHidden())
        following = self.row(page, "Follow").anchor_feedback.text()
        self.assertIn("跟隨工件", following)
        self.assertTrue(any(word in following for word in ("設定", "定位")), following)

    def test_frozen_image_rejection_names_return_live_action_and_preserves_row(self):
        page = self.page()
        page.freeze_check.setChecked(True)
        warning = self.reject(page, ("凍結", "暫停"), ("返回即時影像", "取消凍結"))
        page._update_live()
        page._refresh_roi_rows()
        self.assertEqual(self.row(page).anchor_feedback.text(), warning)

    def test_following_retry_after_returning_live_replaces_failure_with_success(self):
        page = self.page()
        page.freeze_check.setChecked(True)
        self.reject(page, ("凍結", "暫停"), ("返回即時影像", "取消凍結"))
        page.live_button.click()
        page.on_packet(packet(page.sop, at=1))
        self.choose(self.row(page), "workpiece")
        row = self.row(page)
        self.assertEqual(row.anchor_combo.currentData(), "workpiece")
        self.assertEqual(row.anchor_feedback.property("state"), "ok")
        self.assertIn("已啟用跟隨工件", row.anchor_feedback.text())
        self.assertNotIn("未切換", row.anchor_feedback.text())

    def test_reference_edit_rejection_names_return_live_action(self):
        page = self.page()
        page.reference_button.setChecked(True)
        self.reject(page, ("參考", "跟隨區域編輯"), ("返回即時影像",))

    def test_reference_capture_rejection_explains_how_to_finish(self):
        page = self.page()
        page.freeze_check.setChecked(True)
        page._capture_target = True
        self.reject(page, ("參考工件", "框選", "凍結"), ("完成", "取消", "返回即時影像"))

    def test_missing_reference_rejection_gives_reference_setup_action(self):
        page = self.page()
        page.sop.workpiece = None
        page._locator = page._reference_frame = None
        self.reject(page, ("參考工件",), ("框選參考工件", "設定參考工件"))

    def test_invalid_reference_rejection_gives_retake_action(self):
        page = self.page()
        page._locator = page._reference_frame = None
        self.reject(page, ("參考工件", "參考影像"), ("重新", "框選參考工件", "設定參考工件"))

    def test_no_live_image_rejection_names_image_connection_or_wait(self):
        page = self.page()
        page._packet = page._preview_result = None
        self.reject(page, ("即時影像",), ("等待", "連接", "啟動"))

    def test_lost_pose_rejection_names_reposition_or_wait(self):
        page = self.page()
        page._preview_result.workpiece_transform = None
        self.reject(page, ("定位",), ("等待", "工件", "重新"))

    def test_invalid_pose_never_raises_or_mutates_and_explains_recovery(self):
        for matrix in (np.eye(3), np.full((2, 3), np.nan),
                       np.array([[np.inf, 0., 0.], [0., 1., 0.]]),
                       np.zeros((2, 3))):
            with self.subTest(matrix=matrix):
                page = self.page()
                page._preview_result.workpiece_transform = matrix
                self.reject(page, ("定位",), ("等待", "重新", "調整"))

    def test_out_of_bounds_rejection_keeps_original_geometry_and_next_action(self):
        page = self.page()
        page._preview_result.workpiece_transform = np.array([[1., 0., .9], [0., 1., .9]])
        self.reject(page, ("超出", "範圍"), ("調整", "重新"))

    def test_running_monitor_rejection_gives_stop_monitor_action(self):
        page = self.page()
        page.set_production_running(True)
        self.reject(page, ("執行", "監控", "作業"), ("停止",))

    def test_success_switches_preserve_current_geometry_and_report_configured_state(self):
        page = self.page()
        original = deepcopy(page.sop.to_dict())
        original_points = np.asarray(self.row(page).roi.points).copy()
        changed = Mock()
        page.modified.connect(changed)
        self.choose(self.row(page), "workpiece")
        row = self.row(page)
        self.assertEqual(row.anchor_combo.currentData(), "workpiece")
        self.assertEqual(row.roi.anchor, "workpiece")
        np.testing.assert_allclose(np.asarray(row.roi.points) @ POSE[:, :2].T + POSE[:, 2], original_points)
        self.assertEqual(page.sop.to_dict()["workpiece"], original["workpiece"])
        self.assertEqual(page.sop.to_dict()["steps"], original["steps"])
        self.assertEqual(page.sop.to_dict()["cycle"], original["cycle"])
        self.assertEqual(row.anchor_feedback.property("state"), "ok")
        self.assertIn("已啟用跟隨工件", row.anchor_feedback.text())
        self.assertTrue(any(word in row.anchor_feedback.text() for word in ("定位狀態", "設定")))
        self.assertEqual(len(page._roi_undo), 1)
        self.assertEqual(changed.call_count, 1)
        page._update_live()
        row.set_status("本幀無效：定位遺失")
        self.assertIn("已啟用跟隨工件", row.anchor_feedback.text())
        page._refresh_roi_rows()
        self.assertIn("已啟用跟隨工件", self.row(page).anchor_feedback.text())
        self.choose(self.row(page), "fixed")
        row = self.row(page)
        self.assertEqual(row.anchor_combo.currentData(), "fixed")
        np.testing.assert_allclose(row.roi.points, original_points)
        self.assertIn("已切換為固定畫面", row.anchor_feedback.text())
        self.assertEqual(row.anchor_feedback.property("state"), "ok")
        self.assertEqual(changed.call_count, 2)

    def test_new_sop_cannot_inherit_previous_same_named_region_warning(self):
        page = self.page()
        page.freeze_check.setChecked(True)
        warning = self.reject(page, ("凍結", "暫停"), ("返回即時影像", "取消凍結"))
        with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.set_sop(definition())
        row = self.row(page)
        self.assertNotEqual(row.anchor_feedback.text(), warning)
        self.assertEqual(row.anchor_combo.currentData(), "fixed")
        self.assertTrue(row.anchor_feedback.isHidden())
        self.assertNotEqual(row.anchor_feedback.property("state"), "warning")

    def test_undo_removes_success_message_from_restored_fixed_region(self):
        page = self.page()
        self.choose(self.row(page), "workpiece")
        self.assertIn("已啟用跟隨工件", self.row(page).anchor_feedback.text())
        page._undo_roi()
        row = self.row(page)
        self.assertEqual(row.roi.anchor, "fixed")
        self.assertEqual(row.anchor_combo.currentData(), "fixed")
        self.assertNotIn("已啟用跟隨工件", row.anchor_feedback.text())
        self.assertTrue(row.anchor_feedback.isHidden())

    def test_duplicate_does_not_copy_the_original_region_warning(self):
        page = self.page()
        page.freeze_check.setChecked(True)
        row = self.row(page)
        warning = self.reject(page, ("凍結", "暫停"), ("返回即時影像", "取消凍結"))
        page._duplicate_roi(row.roi)
        duplicate = page.roi_rows[-1]
        self.assertNotEqual(duplicate.roi.name, "Fixed")
        self.assertNotEqual(duplicate.anchor_feedback.text(), warning)
        self.assertEqual(duplicate.anchor_combo.currentData(), "fixed")
        self.assertTrue(duplicate.anchor_feedback.isHidden())
        self.assertNotEqual(duplicate.anchor_feedback.property("state"), "warning")


if __name__ == "__main__":
    unittest.main()
