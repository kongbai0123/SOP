"""Repair tracking on the same saved photo without changing region coordinates.

These are editor integration tests: locator outcomes are controlled explicitly,
and no camera, model, production SOP file or tracking accuracy is involved.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sop_app.detection import FrameResult
from sop_app.native_runtime import cv2
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step, Workpiece
from sop_app.tracking import workpiece_key
from sop_app.ui.editor_page import EditorPage


OLD_TARGET = [(.43, .16), (.51, .16), (.51, .39), (.43, .39)]
NEW_TARGET = [(.12, .1), (.88, .1), (.88, .78), (.12, .78)]
REGION = [(.26, .3), (.36, .3), (.36, .4), (.26, .4)]


def pixels(seed=1):
    return np.random.default_rng(seed).integers(0, 256, (180, 320, 3), dtype=np.uint8)


def definition():
    encoded = cv2.imencode(".png", pixels())[1].tobytes()
    return SOPDefinition(
        name="Reference recovery fixture",
        workpiece=Workpiece(name="Fixture workpiece", reference_png=base64.b64encode(encoded).decode("ascii"),
                            points=list(OLD_TARGET)),
        rois=[ROI("installation", list(REGION), "workpiece", "installation"),
              ROI("fixed bin", list(REGION), "fixed")],
        steps=[Step("Install", conditions=[Condition(label="done", roi="installation")],
                    forbidden=[Condition(label="wrong", roi="fixed bin")])])


def frame_packet(sop, seed=2):
    image = pixels(seed)
    result = FrameResult(320, 180, float(seed), workpiece_transform=None,
                         tracking_status="lost", workpiece_key=workpiece_key(sop.workpiece))
    return FramePacket(image, image.copy(), result, None, 30., 5., model_display=image.copy())


class TrackingEditorRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def page(self, sop=None, invalid=True):
        page = EditorPage()
        def dispose():
            for timer in page.findChildren(QTimer):
                timer.stop()
            page.close()
        self.addCleanup(dispose)
        with patch("sop_app.ui.editor_page.WorkpieceLocator",
                   side_effect=ValueError("參考自我驗證失敗") if invalid else None,
                   return_value=SimpleNamespace()) as locator:
            page.set_sop(sop or definition())
        if page.sop.workpiece is not None and page._reference_frame is not None:
            locator.assert_called_once_with(page.sop.workpiece, adaptive_budget=True)
        return page

    def test_invalid_reference_explains_same_photo_recovery_and_allows_original_editing(self):
        page = self.page()
        self.assertIsNone(page._locator)
        self.assertIsNotNone(page._reference_frame)
        self.assertTrue(page.reference_button.isEnabled())
        self.assertIn("框選參考工件", page.tracking_label.text())
        self.assertIn("原始照片", page.tracking_label.text())
        page.reference_button.setChecked(True)
        page.original_frame_button.click()
        self.assertTrue(page._following_edit_ready())
        np.testing.assert_array_equal(page._edit_frame, pixels())

    def test_reframe_requires_no_camera_and_keeps_original_coordinate_space(self):
        page = self.page()
        before = deepcopy(page.sop.to_dict())
        self.assertIsNone(page._packet)
        page.target_button.click()
        self.assertTrue(page._capture_target)
        self.assertTrue(page.video._draw_mode)
        self.assertEqual(page._edit_frame_kind, "reference")
        np.testing.assert_array_equal(page._target_frame, pixels())
        np.testing.assert_array_equal(page._edit_matrix, np.eye(2, 3))
        self.assertIn("原始參考照片", page.region_creation_label.text())
        self.assertEqual(page.sop.to_dict(), before)

    def test_success_replaces_only_polygon_and_validates_production_strategy(self):
        page = self.page()
        before = deepcopy(page.sop.to_dict())
        changes = Mock()
        page.workpiece_changed.connect(changes)
        page.target_button.click()
        with patch("sop_app.ui.editor_page.capture_workpiece") as capture, \
             patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()) as locator:
            page.video.roi_drawn.emit(NEW_TARGET)
        capture.assert_not_called()
        locator.assert_called_once_with(page.sop.workpiece, adaptive_budget=True)
        expected = deepcopy(before)
        expected["workpiece"]["points"] = list(NEW_TARGET)
        self.assertEqual(page.sop.to_dict(), expected)
        self.assertFalse(page._capture_target)
        self.assertIsNone(page._target_frame)
        self.assertIsNone(page._target_original)
        self.assertEqual(page.target_button.text(), "框選參考工件")
        self.assertTrue(page.following_frame_tools.isEnabled())
        self.assertIn("已驗證", page.tracking_label.text())
        changes.assert_called_once_with(page.sop.workpiece)
        self.assertNotEqual(workpiece_key(page.sop.workpiece), workpiece_key(definition().workpiece))
        self.assertTrue(page._following_edit_ready())
        np.testing.assert_array_equal(page._edit_frame, pixels())

    def test_invalid_retry_does_not_mutate_sop_and_can_then_succeed(self):
        page = self.page()
        before = deepcopy(page.sop.to_dict())
        changes = Mock()
        page.workpiece_changed.connect(changes)
        page.target_button.click()
        with patch("sop_app.ui.editor_page.WorkpieceLocator", side_effect=ValueError("紋理不足")):
            page.video.roi_drawn.emit(OLD_TARGET)
        self.assertEqual(page.sop.to_dict(), before)
        changes.assert_not_called()
        self.assertTrue(page._capture_target)
        self.assertTrue(page.video._draw_mode)
        self.assertIn("紋理不足", page.region_creation_label.text())
        with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.video.roi_drawn.emit(NEW_TARGET)
        self.assertEqual(page.sop.workpiece.reference_png, before["workpiece"]["reference_png"])
        self.assertEqual(page.sop.rois[0].points, REGION)
        changes.assert_called_once()

    def test_arriving_live_image_cannot_replace_saved_photo_during_reframe(self):
        page = self.page()
        before = deepcopy(page.sop.to_dict())
        page.on_packet(frame_packet(page.sop))
        page.target_button.click()
        page.on_packet(frame_packet(page.sop, seed=3))
        np.testing.assert_array_equal(page._target_frame, pixels())
        np.testing.assert_array_equal(page._edit_frame, pixels())
        with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.video.roi_drawn.emit(NEW_TARGET)
        self.assertEqual(page.sop.workpiece.reference_png, before["workpiece"]["reference_png"])

    def test_valid_reference_also_reframes_same_photo_with_existing_regions(self):
        page = self.page(invalid=False)
        page.on_packet(frame_packet(page.sop))
        page.target_button.click()
        self.assertTrue(page._capture_target)
        np.testing.assert_array_equal(page._target_frame, pixels())
        self.assertIsNotNone(page._target_original)

    def test_new_following_region_can_repair_invalid_reference_and_continue_without_extra_mode(self):
        page = self.page()
        page.region_anchor_combo.setCurrentIndex(page.region_anchor_combo.findData("workpiece"))
        page.draw_button.click()
        self.assertTrue(page._capture_target)
        self.assertIn("原始參考照片", page.region_creation_label.text())
        with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.video.roi_drawn.emit(NEW_TARGET)
        self.assertTrue(page.video._draw_mode)
        self.assertFalse(page._capture_target)
        page.video.roi_drawn.emit(REGION)
        self.assertEqual(len(page.sop.rois), 3)
        self.assertEqual(page.sop.rois[-1].anchor, "workpiece")
        np.testing.assert_allclose(page.sop.rois[-1].points, REGION)
        self.assertEqual(page.sop.rois[0].points, REGION)

    def test_missing_or_damaged_original_does_not_apply_existing_regions_to_new_photo(self):
        for broken in (None, Workpiece(reference_png="broken", points=list(OLD_TARGET))):
            with self.subTest(original=broken):
                sop = definition()
                sop.workpiece = broken
                page = self.page(sop)
                page.on_packet(frame_packet(page.sop))
                before = deepcopy(page.sop.to_dict())
                with patch("sop_app.ui.editor_page.capture_workpiece") as capture:
                    page.target_button.click()
                capture.assert_not_called()
                self.assertFalse(page._capture_target)
                self.assertFalse(page.video._draw_mode)
                self.assertIn("不能直接套用新照片", page.region_creation_label.text())
                self.assertEqual(page.sop.to_dict(), before)

    def test_cancel_and_lifecycle_changes_release_recovery_snapshot_without_mutation(self):
        for action in ("live", "target", "source", "production", "sop"):
            with self.subTest(action=action):
                page = self.page()
                before = deepcopy(page.sop.to_dict())
                page.target_button.click()
                if action == "live":
                    page.live_button.click()
                elif action == "target":
                    page.target_button.click()
                elif action == "source":
                    page.clear_video("source disconnected")
                elif action == "production":
                    page.set_production_running(True)
                else:
                    with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
                        page.set_sop(definition())
                self.assertFalse(page._capture_target)
                self.assertIsNone(page._target_frame)
                self.assertIsNone(page._target_original)
                self.assertFalse(page.video._draw_mode)
                self.assertEqual(page.sop.to_dict(), before)
                if action != "production":
                    self.assertTrue(page.following_frame_tools.isEnabled())
                else:
                    page.set_production_running(False)
                    self.assertTrue(page.following_frame_tools.isEnabled())

    def test_production_does_not_start_reference_recovery(self):
        page = self.page()
        before = deepcopy(page.sop.to_dict())
        page.set_production_running(True)
        page._begin_target()
        self.assertFalse(page._capture_target)
        self.assertEqual(page.sop.to_dict(), before)

    def test_uncertain_external_pose_cannot_edit_or_convert_following_coordinates(self):
        page = self.page(invalid=False)
        packet = frame_packet(page.sop)
        packet.result.workpiece_transform = np.eye(2, 3)
        packet.result.tracking_state = "uncertain"
        page.on_packet(packet)
        self.assertIsNone(page._latest_matrix)
        self.assertFalse(page._latest_following_frame_valid())
        page.reference_button.setChecked(True)
        self.assertFalse(page.use_frame_button.isEnabled())
        self.assertFalse(page._following_edit_ready())
        self.assertFalse(page._use_latest_following_frame())

    def test_confirmed_external_pose_remains_available_for_following_editor(self):
        page = self.page(invalid=False)
        packet = frame_packet(page.sop)
        packet.result.workpiece_transform = np.eye(2, 3)
        packet.result.tracking_state = "confirmed"
        page.on_packet(packet)
        self.assertTrue(page._latest_following_frame_valid())
        page.reference_button.setChecked(True)
        self.assertTrue(page._following_edit_ready())

    def test_recovery_cannot_switch_display_to_unrelated_live_frame(self):
        page = self.page()
        page.on_packet(frame_packet(page.sop))
        page.target_button.click()
        self.assertFalse(page.following_frame_tools.isEnabled())
        page.on_packet(frame_packet(page.sop, seed=3))
        page._retake_following_frame()
        self.assertFalse(page._use_latest_following_frame())
        self.assertFalse(page._use_original_reference())
        self.assertTrue(page._capture_target)
        self.assertTrue(page.video._draw_mode)
        self.assertEqual(page._edit_frame_kind, "reference")
        np.testing.assert_array_equal(page._edit_frame, pixels())
        np.testing.assert_array_equal(page._target_frame, pixels())


if __name__ == "__main__":
    unittest.main()
