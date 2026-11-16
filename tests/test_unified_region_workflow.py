"""The region pane starts fixed/following drawing without a separate setup entry.

Synthetic image/pose pairs exercise UI state and coordinate ownership. These
tests do not start a model, camera, production session, or save any SOP data.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sop_app.detection import FrameResult
from sop_app.native_runtime import cv2
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import ROI, SOPDefinition, Step, Workpiece
from sop_app.tracking import capture_workpiece, workpiece_key
from sop_app.ui.editor_page import EditorPage


POINTS = [(.2, .2), (.4, .2), (.4, .4), (.2, .4)]
TARGET = [(.08, .08), (.9, .08), (.9, .9), (.08, .9)]
POSE = np.array([[.8, -.15, .2], [.15, .8, .03]])


def image(seed=9):
    # A genuine reference capture needs texture and contour evidence.
    return np.random.default_rng(seed).integers(0, 256, (180, 320, 3), dtype=np.uint8)


def definition(with_reference=True):
    target = None
    if with_reference:
        encoded = cv2.imencode(".png", image())[1].tobytes()
        target = Workpiece(reference_png=base64.b64encode(encoded).decode("ascii"), points=list(TARGET))
    return SOPDefinition(name="Unified region fixture", workpiece=target, steps=[Step("Install")])


def frame_packet(sop, matrix=POSE, seed=1, key=None):
    raw = image(seed)
    preview = np.zeros_like(raw)
    result = FrameResult(320, 180, float(seed), workpiece_transform=matrix,
                         tracking_status="valid" if matrix is not None else "lost",
                         workpiece_key=workpiece_key(sop.workpiece) if key is None else key)
    return FramePacket(raw, preview.copy(), result, None, 30., 5., model_display=preview)


class UnifiedRegionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def page(self, with_reference=True):
        page = EditorPage()
        def dispose():
            for timer in page.findChildren(QTimer):
                timer.stop()
            page.close()
        self.addCleanup(dispose)
        with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.set_sop(definition(with_reference))
        return page

    def choose_anchor(self, page, anchor):
        index = page.region_anchor_combo.findData(anchor)
        self.assertGreaterEqual(index, 0)
        page.region_anchor_combo.setCurrentIndex(index)

    def start_following(self, page):
        self.choose_anchor(page, "workpiece")
        page.draw_button.click()

    def test_tracking_controls_have_one_parent_inside_region_pane(self):
        page = self.page()
        self.assertTrue(page.regions_panel.isAncestorOf(page.region_anchor_combo))
        self.assertTrue(page.regions_panel.isAncestorOf(page.tracking_settings_button))
        self.assertTrue(page.regions_panel.isAncestorOf(page.tracking_tools))
        self.assertFalse(page.monitor_pane.isAncestorOf(page.tracking_tools))
        for widget in (page.target_button, page.reference_button, page.clear_target_button,
                       page.retake_frame_button, page.use_frame_button, page.original_frame_button):
            self.assertTrue(page.tracking_tools.isAncestorOf(widget))
        self.assertTrue(page.tracking_tools.isHidden())
        page.tracking_settings_button.click()
        self.assertFalse(page.tracking_tools.isHidden())
        page.tracking_settings_button.click()
        self.assertTrue(page.tracking_tools.isHidden())

    def test_fixed_creation_needs_no_reference_and_preserves_frame_coordinates(self):
        page = self.page(False)
        packet = frame_packet(page.sop, matrix=None)
        page.on_packet(packet)
        self.choose_anchor(page, "fixed")
        with patch("sop_app.ui.editor_page.capture_workpiece") as capture:
            page.draw_button.click()
            self.assertTrue(page.video._draw_mode)
            self.assertFalse(page._capture_target)
            self.assertFalse(page.reference_button.isChecked())
            page.video.roi_drawn.emit(POINTS)
        capture.assert_not_called()
        self.assertIsNone(page.sop.workpiece)
        self.assertEqual([(roi.anchor, roi.points) for roi in page.sop.rois], [("fixed", POINTS)])

    def test_following_creation_uses_latest_matching_pose_without_separate_setup(self):
        page = self.page()
        packet = frame_packet(page.sop)
        original_workpiece = deepcopy(page.sop.workpiece)
        page.on_packet(packet)
        self.start_following(page)
        self.assertFalse(page._capture_target)
        self.assertTrue(page._following_edit_ready())
        self.assertTrue(page.video._draw_mode)
        self.assertEqual(page._edit_frame_kind, "latest")
        np.testing.assert_array_equal(page._edit_frame, packet.frame)
        np.testing.assert_array_equal(page._edit_matrix, POSE)
        # New packets may update the live cache, never the image/pose being drawn.
        page.on_packet(frame_packet(page.sop, matrix=np.eye(2, 3), seed=2))
        shown = np.asarray(POINTS) @ POSE[:, :2].T + POSE[:, 2]
        page.video.roi_drawn.emit(shown.tolist())
        self.assertEqual(len(page.sop.rois), 1)
        self.assertEqual(page.sop.rois[0].anchor, "workpiece")
        np.testing.assert_allclose(page.sop.rois[0].points, POINTS, atol=1e-12)
        self.assertEqual(page.sop.workpiece, original_workpiece)

    def test_first_following_region_captures_reference_then_continues_drawing(self):
        page = self.page(False)
        packet = frame_packet(page.sop, matrix=None)
        page.on_packet(packet)
        self.start_following(page)
        self.assertTrue(page._capture_target)
        self.assertTrue(page.video._draw_mode)
        self.assertEqual(page.sop.rois, [])
        np.testing.assert_array_equal(page._packet.frame, packet.frame)
        # Capture validates a real textured frame; only the editor's already
        # validated locator is stubbed so no background feature tracker runs.
        with patch("sop_app.ui.editor_page.capture_workpiece", wraps=capture_workpiece) as capture, \
             patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.video.roi_drawn.emit(TARGET)
        capture.assert_called_once()
        np.testing.assert_array_equal(capture.call_args.args[0], packet.frame)
        self.assertIsNotNone(page.sop.workpiece)
        self.assertEqual(page.sop.rois, [], "reference capture itself must not create an ROI")
        self.assertFalse(page._capture_target)
        self.assertTrue(page._following_edit_ready())
        self.assertEqual(page._edit_frame_kind, "reference")
        np.testing.assert_array_equal(page._edit_frame, packet.frame)
        np.testing.assert_array_equal(page._edit_matrix, np.eye(2, 3))
        self.assertTrue(page.video._draw_mode, "no second setup or add-region click should be required")
        page.video.roi_drawn.emit(POINTS)
        self.assertEqual([(roi.anchor, roi.points) for roi in page.sop.rois], [("workpiece", POINTS)])

    def test_invalid_reference_capture_can_retry_and_cancel_without_creating_region(self):
        page = self.page(False)
        page.on_packet(frame_packet(page.sop, matrix=None))
        self.start_following(page)
        with patch("sop_app.ui.editor_page.capture_workpiece", side_effect=ValueError("紋理不足")), \
             patch("sop_app.ui.editor_page.QMessageBox.warning"):
            page.video.roi_drawn.emit(TARGET)
        self.assertTrue(page._capture_target)
        self.assertTrue(page.video._draw_mode)
        self.assertIsNone(page.sop.workpiece)
        self.assertEqual(page.sop.rois, [])
        page.live_button.click()
        self.assertFalse(page._capture_target)
        self.assertFalse(page.video._draw_mode)
        self.assertFalse(page.draw_button.isChecked())
        self.assertIsNone(page.sop.workpiece)
        self.assertEqual(page.sop.rois, [])

    def test_lost_pose_does_not_silently_use_original_reference(self):
        page = self.page()
        page.on_packet(frame_packet(page.sop, matrix=None))
        before = deepcopy(page.sop.to_dict())
        self.start_following(page)
        self.assertFalse(page._capture_target, "a valid reference must not be overwritten when tracking is lost")
        self.assertFalse(page._following_edit_ready())
        self.assertEqual(page._edit_frame_kind, "selecting")
        self.assertIsNone(page._edit_matrix)
        self.assertFalse(page.video._draw_mode)
        self.assertFalse(page.use_frame_button.isEnabled())
        self.assertEqual(page.sop.to_dict(), before)
        page.original_frame_button.click()
        self.assertEqual(page._edit_frame_kind, "reference")
        self.assertTrue(page.video._draw_mode)
        page.video.roi_drawn.emit(POINTS)
        self.assertEqual(page.sop.rois[0].anchor, "workpiece")
        self.assertEqual(page.sop.to_dict()["workpiece"], before["workpiece"])

    def test_waiting_for_pose_only_resumes_after_explicit_frame_selection(self):
        page = self.page()
        page.on_packet(frame_packet(page.sop, matrix=None))
        self.start_following(page)
        packet = frame_packet(page.sop, seed=3)
        page.on_packet(packet)
        self.assertFalse(page.video._draw_mode)
        self.assertFalse(page._following_edit_ready())
        self.assertTrue(page.use_frame_button.isEnabled())
        page.use_frame_button.click()
        self.assertTrue(page._following_edit_ready())
        self.assertTrue(page.video._draw_mode)
        self.assertEqual(page._edit_frame_kind, "latest")
        np.testing.assert_array_equal(page._edit_frame, packet.frame)

    def test_previous_workpiece_pose_is_not_used_for_new_following_region(self):
        page = self.page()
        page.on_packet(frame_packet(page.sop, key="previous-workpiece"))
        self.start_following(page)
        self.assertFalse(page._following_edit_ready())
        self.assertFalse(page.video._draw_mode)
        self.assertIsNone(page._edit_matrix)
        self.assertEqual(page.sop.rois, [])

    def test_pending_reference_flow_is_canceled_by_live_source_sop_and_production_changes(self):
        for action in ("live", "source", "sop", "production"):
            with self.subTest(action=action):
                page = self.page(False)
                page.on_packet(frame_packet(page.sop, matrix=None))
                self.start_following(page)
                self.assertTrue(page._capture_target)
                if action == "live":
                    page.live_button.click()
                elif action == "source":
                    page.clear_video("source changed")
                elif action == "sop":
                    page.set_sop(definition(False))
                else:
                    page.set_production_running(True)
                self.assertFalse(page._capture_target)
                self.assertFalse(page.video._draw_mode)
                self.assertFalse(page.draw_button.isChecked())
                self.assertIsNone(page._pending_region_anchor)
                self.assertEqual(page.sop.rois, [])

    def test_canceled_following_intent_cannot_leak_into_later_fixed_region(self):
        page = self.page(False)
        page.on_packet(frame_packet(page.sop, matrix=None))
        self.start_following(page)
        page.live_button.click()
        page.on_packet(frame_packet(page.sop, matrix=None, seed=4))
        self.choose_anchor(page, "fixed")
        page.draw_button.click()
        self.assertFalse(page._capture_target)
        self.assertFalse(page.reference_button.isChecked())
        self.assertIsNone(page._pending_region_anchor)
        page.video.roi_drawn.emit(POINTS)
        self.assertEqual([(roi.anchor, roi.points) for roi in page.sop.rois], [("fixed", POINTS)])
        self.assertIsNone(page.sop.workpiece)

    def test_canceled_waiting_request_is_not_revived_by_later_reference_selection(self):
        page = self.page()
        page.on_packet(frame_packet(page.sop, matrix=None))
        self.start_following(page)
        self.assertFalse(page._following_edit_ready())
        page.live_button.click()
        page.reference_button.setChecked(True)
        page.original_frame_button.click()
        self.assertTrue(page._following_edit_ready())
        self.assertIsNone(page._pending_region_anchor)
        self.assertFalse(page.video._draw_mode)
        self.assertFalse(page.draw_button.isChecked())
        self.assertEqual(page.sop.rois, [])

    def test_changing_creation_mode_cancels_reference_capture(self):
        page = self.page(False)
        page.on_packet(frame_packet(page.sop, matrix=None))
        self.start_following(page)
        self.assertTrue(page._capture_target)
        self.choose_anchor(page, "fixed")
        self.assertFalse(page._capture_target)
        self.assertFalse(page.video._draw_mode)
        self.assertFalse(page.reference_button.isChecked())
        self.assertIsNone(page._pending_region_anchor)
        page.on_packet(frame_packet(page.sop, matrix=None, seed=4))
        page.draw_button.click()
        page.video.roi_drawn.emit(POINTS)
        self.assertIsNone(page.sop.workpiece)
        self.assertEqual([(roi.anchor, roi.points) for roi in page.sop.rois], [("fixed", POINTS)])

    def test_existing_following_regions_prevent_implicit_reference_replacement(self):
        for target in (None, Workpiece(reference_png="damaged-reference", points=list(TARGET))):
            with self.subTest(reference=target):
                page = self.page(False)
                sop = definition(False)
                sop.workpiece = target
                sop.rois = [ROI("Existing following", list(POINTS), "workpiece")]
                page.set_sop(sop)
                page.on_packet(frame_packet(page.sop, matrix=None))
                before = deepcopy(page.sop.to_dict())
                history = deepcopy(page._roi_undo)
                with patch("sop_app.ui.editor_page.QMessageBox.information"), \
                     patch("sop_app.ui.editor_page.capture_workpiece") as capture:
                    self.start_following(page)
                capture.assert_not_called()
                self.assertFalse(page._capture_target)
                self.assertFalse(page.video._draw_mode)
                self.assertIsNone(page._pending_region_anchor)
                self.assertEqual(page.sop.to_dict(), before)
                self.assertEqual(page._roi_undo, history)

    def test_cancel_after_reference_capture_keeps_reference_without_creating_region(self):
        page = self.page(False)
        page.on_packet(frame_packet(page.sop, matrix=None))
        self.start_following(page)
        with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.video.roi_drawn.emit(TARGET)
        target = deepcopy(page.sop.workpiece)
        self.assertIsNotNone(target)
        self.assertTrue(page.video._draw_mode)
        page.live_button.click()
        self.assertFalse(page.video._draw_mode)
        self.assertIsNone(page._pending_region_anchor)
        self.assertEqual(page.sop.workpiece, target)
        self.assertEqual(page.sop.rois, [])


if __name__ == "__main__":
    unittest.main()
