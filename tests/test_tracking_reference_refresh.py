"""Update reference pixels and rebase all following regions as one transaction."""
from __future__ import annotations

import base64
from copy import deepcopy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import time
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
from sop_app.tracking import capture_workpiece, decode_reference, workpiece_key
from sop_app.ui.editor_page import EditorPage


POINTS = [(.2,.2),(.4,.2),(.4,.4),(.2,.4)]
TARGET = [(.05,.05),(.95,.05),(.95,.95),(.05,.95)]
POSE = np.array([[.88,-.15,.16],[.15,.88,-.02]])


def image(seed=1):
    return np.random.default_rng(seed).integers(0,256,(180,320,3),dtype=np.uint8)


def definition():
    encoded = cv2.imencode(".png",image())[1].tobytes()
    return SOPDefinition(name="Refresh fixture",
        workpiece=Workpiece(name="Named fixture",reference_png=base64.b64encode(encoded).decode("ascii"),points=list(TARGET)),
        rois=[ROI("Follow A",list(POINTS),"workpiece","installation"),
              ROI("Follow B",[(x+.25,y+.1) for x,y in POINTS],"workpiece","operation"),
              ROI("Fixed",list(POINTS),"fixed","operation")],
        steps=[Step("Install",conditions=[Condition(label="done",roi="Follow A")],
                    forbidden=[Condition(label="wrong",roi="Follow B")])])


def packet(sop, matrix=POSE, seed=2, key=None, state="confirmed"):
    raw = image(seed)
    result = FrameResult(320,180,float(seed),workpiece_transform=matrix,
        workpiece_key=workpiece_key(sop.workpiece) if key is None else key,
        tracking_state=state,tracking_status=state)
    return FramePacket(raw,np.zeros_like(raw),result,None,30.,5.,model_display=np.zeros_like(raw))


class TrackingReferenceRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def page(self):
        page = EditorPage()
        def dispose():
            for timer in page.findChildren(QTimer):
                timer.stop()
            page.close()
        self.addCleanup(dispose)
        with patch("sop_app.ui.editor_page.WorkpieceLocator",return_value=SimpleNamespace()):
            page.set_sop(definition())
        return page

    def complete(self,page):
        with patch("sop_app.ui.editor_page.WorkpieceLocator",return_value=SimpleNamespace()) as locator:
            page.video.roi_drawn.emit(TARGET)
        return locator

    def test_action_is_in_single_region_tracking_pane_and_explains_no_frame(self):
        page = self.page()
        self.assertTrue(page.tracking_tools.isAncestorOf(page.refresh_reference_button))
        self.assertEqual(page.refresh_reference_button.text(),"更新參考影像")
        self.assertFalse(page.refresh_reference_button.isEnabled())
        self.assertIn("影像來源",page.reference_update_label.text())
        before = deepcopy(page.sop.to_dict())
        self.assertFalse(page._begin_reference_refresh())
        self.assertEqual(page.sop.to_dict(),before)

    def test_only_fresh_matching_confirmed_invertible_pose_enables_update(self):
        for kind in ("mismatch","uncertain","relocating","legacy","missing","singular","nan","shape","stale"):
            with self.subTest(kind=kind):
                page = self.page()
                source = packet(page.sop)
                if kind == "mismatch": source.result.workpiece_key = "previous-reference"
                elif kind in ("uncertain","relocating"): source.result.tracking_state = kind
                elif kind == "legacy": source.result.tracking_state = ""
                elif kind == "missing": source.result.workpiece_transform = None
                elif kind == "singular": source.result.workpiece_transform = np.zeros((2,3))
                elif kind == "nan": source.result.workpiece_transform = np.full((2,3),np.nan)
                elif kind == "shape": source.result.workpiece_transform = np.eye(3)
                page.on_packet(source)
                if kind == "stale":
                    page._latest_arrived_at = time.monotonic()-4
                    page._check_validation_feed()
                self.assertFalse(page.refresh_reference_button.isEnabled())
                before = deepcopy(page.sop.to_dict())
                self.assertFalse(page._begin_reference_refresh())
                self.assertFalse(page._capture_target)
                self.assertEqual(page.sop.to_dict(),before)
                self.assertTrue(page.reference_update_label.text())

    def test_freezes_exact_raw_frame_and_confirmed_pose_despite_new_packets(self):
        page = self.page()
        source = packet(page.sop)
        raw = source.frame.copy()
        page.on_packet(source)
        self.assertTrue(page.refresh_reference_button.isEnabled())
        self.assertTrue(page._begin_reference_refresh())
        self.assertTrue(page._capture_target)
        self.assertTrue(page.video._draw_mode)
        self.assertFalse(page.following_frame_tools.isEnabled())
        np.testing.assert_array_equal(page._target_frame,raw)
        np.testing.assert_array_equal(page._target_refresh_matrix,POSE)
        source.frame[:]=0
        page.on_packet(packet(page.sop,np.eye(2,3),seed=3))
        np.testing.assert_array_equal(page._target_frame,raw)
        np.testing.assert_array_equal(page._target_refresh_matrix,POSE)
        # VideoWidget owns the same frozen raw pixels, not incoming overlays.
        self.assertEqual(page.video._image.pixelColor(0,0).red(),int(raw[0,0,2]))

    def test_commit_rebases_all_following_regions_preserving_names_conditions_and_fixed_region(self):
        page = self.page()
        before = deepcopy(page.sop.to_dict())
        fixed = page.sop.rois[-1]
        source = packet(page.sop)
        page.on_packet(source)
        changed = Mock()
        page.workpiece_changed.connect(changed)
        self.assertTrue(page._begin_reference_refresh())
        locator = self.complete(page)
        locator.assert_called_once_with(page.sop.workpiece,adaptive_budget=True)
        self.assertEqual(page.sop.workpiece.name,"Named fixture")
        self.assertNotEqual(page.sop.workpiece.reference_png,before["workpiece"]["reference_png"])
        np.testing.assert_array_equal(decode_reference(page.sop.workpiece),source.frame)
        for index in (0,1):
            old = before["rois"][index]
            np.testing.assert_allclose(page.sop.rois[index].points,
                np.asarray(old["points"]) @ POSE[:,:2].T+POSE[:,2],atol=1e-12)
            self.assertEqual(page.sop.rois[index].name,old["name"])
            self.assertEqual(page.sop.rois[index].display_group,old["display_group"])
            self.assertEqual(page.sop.rois[index].anchor,"workpiece")
        self.assertIs(page.sop.rois[-1],fixed)
        self.assertEqual(page.sop.to_dict()["steps"],before["steps"])
        self.assertEqual(page.sop.to_dict()["cycle"],before["cycle"])
        self.assertEqual(page._roi_undo,[])
        changed.assert_called_once_with(page.sop.workpiece)
        self.assertFalse(page._capture_target)
        self.assertIsNone(page._target_refresh_matrix)
        self.assertEqual(page._target_source_key,"")
        self.assertIn("已更新",page.region_creation_label.text())
        self.assertTrue(page._following_edit_ready())
        self.assertFalse(page.refresh_reference_button.isEnabled(),"old-key pose cannot refresh new reference")

    def test_one_outside_region_rejects_entire_commit_and_preserves_history_and_signals(self):
        page = self.page()
        page.sop.rois[1].points=[(.9,.1),(1.,.1),(1.,.2),(.9,.2)]
        page.on_packet(packet(page.sop,np.array([[1.,0.,.15],[0.,1.,0.]])))
        before = deepcopy(page.sop.to_dict())
        changed = Mock()
        page.modified.connect(changed)
        self.assertTrue(page._begin_reference_refresh())
        with patch("sop_app.ui.editor_page.capture_workpiece") as capture:
            page.video.roi_drawn.emit(TARGET)
        capture.assert_not_called()
        self.assertEqual(page.sop.to_dict(),before)
        self.assertEqual(page._roi_undo,[])
        changed.assert_not_called()
        self.assertTrue(page._capture_target)
        self.assertIn("Follow B",page.region_creation_label.text())
        self.assertIn("超出",page.region_creation_label.text())

    def test_invalid_frozen_pose_cannot_fall_back_to_replacing_photo_without_rebasing(self):
        for matrix in (None,np.zeros((2,3)),np.full((2,3),np.inf)):
            with self.subTest(matrix=matrix):
                page = self.page()
                page.on_packet(packet(page.sop))
                self.assertTrue(page._begin_reference_refresh())
                page._target_refresh_matrix=matrix
                before = deepcopy(page.sop.to_dict())
                with patch("sop_app.ui.editor_page.capture_workpiece") as capture:
                    page.video.roi_drawn.emit(TARGET)
                capture.assert_not_called()
                self.assertEqual(page.sop.to_dict(),before)

    def test_invalid_new_capture_preserves_all_old_data_and_can_retry(self):
        page = self.page()
        page.on_packet(packet(page.sop))
        before = deepcopy(page.sop.to_dict())
        self.assertTrue(page._begin_reference_refresh())
        with patch("sop_app.ui.editor_page.capture_workpiece",side_effect=ValueError("紋理不足")):
            page.video.roi_drawn.emit(TARGET)
        self.assertEqual(page.sop.to_dict(),before)
        self.assertTrue(page.video._draw_mode)
        self.complete(page)
        self.assertFalse(page._capture_target)
        self.assertNotEqual(page.sop.to_dict(),before)

    def test_canonical_configuration_change_during_capture_rejects_old_pose(self):
        page = self.page()
        page.on_packet(packet(page.sop))
        self.assertTrue(page._begin_reference_refresh())
        page.sop.workpiece.points=list(POINTS)
        before = deepcopy(page.sop.to_dict())
        with patch("sop_app.ui.editor_page.capture_workpiece") as capture:
            page.video.roi_drawn.emit(TARGET)
        capture.assert_not_called()
        self.assertEqual(page.sop.to_dict(),before)

    def test_cancel_source_sop_and_production_changes_discard_refresh_without_rebasing(self):
        for action in ("live","target","source","sop","production"):
            with self.subTest(action=action):
                page = self.page()
                page.on_packet(packet(page.sop))
                before = deepcopy(page.sop.to_dict())
                self.assertTrue(page._begin_reference_refresh())
                if action == "live": page.live_button.click()
                elif action == "target": page.target_button.click()
                elif action == "source": page.clear_video("source stopped")
                elif action == "sop":
                    with patch("sop_app.ui.editor_page.WorkpieceLocator",return_value=SimpleNamespace()):
                        page.set_sop(definition())
                else:
                    page.set_production_running(True)
                    page.set_production_running(False)
                self.assertEqual(page.sop.to_dict(),before)
                self.assertFalse(page._capture_target)
                self.assertIsNone(page._target_refresh_matrix)
                self.assertIsNone(page._target_frame)
                self.assertEqual(page._target_source_key,"")
                self.assertTrue(page.freeze_check.isEnabled())
                self.assertTrue(page.following_frame_tools.isEnabled())
                self.assertFalse(page.video._draw_mode)
                if action in ("live","target","production"):
                    self.assertTrue(page.refresh_reference_button.isEnabled())
                else:
                    self.assertFalse(page.refresh_reference_button.isEnabled())

    def test_production_rejects_refresh_even_with_valid_frame(self):
        page = self.page()
        page.on_packet(packet(page.sop))
        page.set_production_running(True)
        before=deepcopy(page.sop.to_dict())
        self.assertFalse(page._begin_reference_refresh())
        self.assertIn("停止正式作業",page.reference_update_label.text())
        self.assertEqual(page.sop.to_dict(),before)

    def test_frozen_capture_stays_valid_after_source_pose_age_expires(self):
        page=self.page()
        page.on_packet(packet(page.sop))
        self.assertTrue(page._begin_reference_refresh())
        page._latest_arrived_at=time.monotonic()-10
        self.complete(page)
        self.assertFalse(page._capture_target)
        np.testing.assert_allclose(page.sop.rois[0].points,np.asarray(POINTS)@POSE[:,:2].T+POSE[:,2])

    def test_fresh_new_key_pose_can_enable_later_refresh(self):
        page=self.page()
        page.on_packet(packet(page.sop))
        page._begin_reference_refresh()
        self.complete(page)
        page.live_button.click()
        page.on_packet(packet(page.sop,np.eye(2,3),seed=3))
        self.assertTrue(page.refresh_reference_button.isEnabled())
        self.assertTrue(page._begin_reference_refresh())


if __name__=="__main__":
    unittest.main()
