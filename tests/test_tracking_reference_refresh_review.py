"""Independent boundary review of reference replacement with existing ROIs."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import base64
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sop_app.detection import FrameResult, TRACKING_CONFIRMED, TRACKING_UNCERTAIN
from sop_app.native_runtime import cv2
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step, Workpiece
from sop_app.tracking import workpiece_key
from sop_app.ui.editor_page import EditorPage


POINTS = [(.2, .2), (.4, .2), (.4, .4), (.2, .4)]
NEW_TARGET = [(.1,.1),(.85,.1),(.85,.85),(.1,.85)]
MATRIX = np.array([[1.,0.,.1],[0.,1.,.05]])


def image(seed):
    return np.random.default_rng(seed).integers(0,256,(180,320,3),dtype=np.uint8)


def reference(seed=1, points=POINTS):
    encoded = cv2.imencode(".png",image(seed))[1].tobytes()
    return Workpiece(name="Reference to retain",reference_png=base64.b64encode(encoded).decode("ascii"),points=list(points))


def sop():
    result = SOPDefinition(workpiece=reference(),
                           rois=[ROI("fixed",list(POINTS)),ROI("moving",list(POINTS),"workpiece","installation")],
                           steps=[Step("install",conditions=[Condition(label="done",roi="moving")],
                                       forbidden=[Condition(label="wrong",roi="fixed")])])
    result.cycle.reset_conditions = [Condition(type="disappear",label="done",roi="moving")]
    return result


def packet(definition, *, state=TRACKING_CONFIRMED, key=None, matrix=MATRIX, seed=2):
    pixels=image(seed)
    result=FrameResult(320,180,0.,workpiece_transform=matrix,tracking_state=state,
                       workpiece_key=workpiece_key(definition.workpiece) if key is None else key)
    return FramePacket(pixels,pixels.copy(),result,None,30,5)


class ReferenceRefreshIndependentReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def page(self):
        page=EditorPage()
        self.addCleanup(self.dispose,page)
        with patch("sop_app.ui.editor_page.WorkpieceLocator",return_value=SimpleNamespace()):
            page.set_sop(sop())
        return page

    @staticmethod
    def dispose(page):
        for timer in page.findChildren(QTimer):
            timer.stop()
        page.close()

    def begin(self,page):
        page.on_packet(packet(page.sop))
        self.assertTrue(page._begin_reference_refresh())

    def commit(self,page,*,error=None):
        target=reference(seed=2,points=NEW_TARGET)
        with patch("sop_app.ui.editor_page.capture_workpiece",return_value=target), \
             patch("sop_app.ui.editor_page.WorkpieceLocator",return_value=SimpleNamespace(),side_effect=error):
            page._on_roi_drawn(NEW_TARGET)

    def test_refresh_freezes_pixels_and_pose_before_new_packets_or_owner_mutation(self):
        page=self.page()
        original=packet(page.sop,matrix=MATRIX.copy())
        page.on_packet(original)
        self.assertTrue(page._begin_reference_refresh())
        original.frame[:]=0
        original.result.workpiece_transform[:]=0
        page.on_packet(packet(page.sop,matrix=np.eye(2,3),seed=3))
        np.testing.assert_array_equal(page._target_frame,image(2))
        np.testing.assert_array_equal(page._target_refresh_matrix,MATRIX)

    def test_commit_maps_only_following_rois_and_keeps_all_condition_references(self):
        page=self.page()
        before=deepcopy(page.sop.to_dict())
        fixed=page.sop.rois[0]
        modified,changed=Mock(),Mock()
        page.modified.connect(modified)
        page.workpiece_changed.connect(changed)
        self.begin(page)
        self.commit(page)
        self.assertIs(page.sop.rois[0],fixed)
        np.testing.assert_allclose(page.sop.rois[1].points,np.asarray(POINTS)+[.1,.05])
        self.assertEqual(page.sop.rois[1].anchor,"workpiece")
        self.assertEqual(page.sop.rois[1].display_group,"installation")
        self.assertEqual(page.sop.to_dict()["steps"],before["steps"])
        self.assertEqual(page.sop.to_dict()["cycle"],before["cycle"])
        self.assertEqual(page.sop.workpiece.name,before["workpiece"]["name"])
        self.assertFalse(page._capture_target)
        self.assertIsNone(page._target_refresh_matrix)
        changed.assert_called_once()
        modified.assert_called_once()

    def test_failed_new_reference_validation_preserves_everything_and_allows_retry(self):
        page=self.page()
        before=deepcopy(page.sop.to_dict())
        changed=Mock()
        page.workpiece_changed.connect(changed)
        self.begin(page)
        self.commit(page,error=ValueError("ambiguous target"))
        self.assertEqual(page.sop.to_dict(),before)
        self.assertTrue(page._capture_target)
        self.assertTrue(page.video._draw_mode)
        self.assertIn("ambiguous target",page.region_creation_label.text())
        changed.assert_not_called()
        self.commit(page)
        changed.assert_called_once()

    def test_one_out_of_frame_roi_prevents_all_reference_and_roi_changes(self):
        page=self.page()
        page.sop.rois.append(ROI("edge",[(.88,.1),(.98,.1),(.98,.3),(.88,.3)],"workpiece"))
        before=deepcopy(page.sop.to_dict())
        changed=Mock()
        page.workpiece_changed.connect(changed)
        self.begin(page)
        self.commit(page)
        self.assertEqual(page.sop.to_dict(),before)
        self.assertIn("edge",page.region_creation_label.text())
        self.assertIn("超出新照片",page.region_creation_label.text())
        self.assertTrue(page._capture_target)
        changed.assert_not_called()

    def test_bad_frozen_matrix_cannot_commit_partially(self):
        for matrix in (np.zeros((2,3)),np.full((2,3),np.nan),np.eye(3)):
            with self.subTest(matrix=repr(matrix)):
                page=self.page()
                self.begin(page)
                before=deepcopy(page.sop.to_dict())
                page._target_refresh_matrix=matrix
                self.commit(page)
                self.assertEqual(page.sop.to_dict(),before)
                self.assertTrue(page._capture_target)
                self.assertIn("已無效",page.region_creation_label.text())

    def test_wrong_key_unconfirmed_bad_pose_and_expired_frame_cannot_begin(self):
        for kind in ("wrong-key","uncertain","singular","expired"):
            with self.subTest(kind=kind):
                page=self.page()
                before=deepcopy(page.sop.to_dict())
                sample=packet(page.sop,key="another" if kind=="wrong-key" else None,
                               state=TRACKING_UNCERTAIN if kind=="uncertain" else TRACKING_CONFIRMED,
                               matrix=np.zeros((2,3)) if kind=="singular" else MATRIX)
                page.on_packet(sample)
                if kind=="expired":
                    page._latest_arrived_at-=4
                self.assertFalse(page._begin_reference_refresh())
                self.assertFalse(page._capture_target)
                self.assertEqual(page.sop.to_dict(),before)

    def test_legacy_external_pose_can_edit_existing_regions_but_cannot_replace_reference(self):
        page=self.page()
        page.on_packet(packet(page.sop,state=""))
        self.assertIsNotNone(page._latest_matrix)
        self.assertTrue(page._latest_following_frame_valid())
        self.assertFalse(page._begin_reference_refresh())
        self.assertFalse(page.refresh_reference_button.isEnabled())
        page.reference_button.setChecked(True)
        self.assertTrue(page._following_edit_ready())
        self.assertFalse(page._begin_reference_refresh())

    def test_reference_changed_during_capture_prevents_rebase(self):
        for kind in ("cache-key","workpiece-content"):
            with self.subTest(kind=kind):
                page=self.page()
                self.begin(page)
                if kind=="cache-key":
                    page._workpiece_key="changed"
                else:
                    page.sop.workpiece=replace(page.sop.workpiece,points=NEW_TARGET)
                before=deepcopy(page.sop.to_dict())
                self.commit(page)
                self.assertEqual(page.sop.to_dict(),before)
                self.assertTrue(page._capture_target)
                self.assertIn("原參考設定已無效",page.region_creation_label.text())

    def test_cancel_source_production_and_sop_change_clear_frozen_refresh_transaction(self):
        for action in ("cancel","source","production","sop"):
            with self.subTest(action=action):
                page=self.page()
                before=deepcopy(page.sop.to_dict())
                self.begin(page)
                if action=="cancel":
                    page._return_live()
                elif action=="source":
                    page.clear_video("source changed")
                elif action=="production":
                    page.set_production_running(True)
                else:
                    with patch("sop_app.ui.editor_page.WorkpieceLocator",return_value=SimpleNamespace()):
                        page.set_sop(sop())
                self.assertFalse(page._capture_target)
                self.assertIsNone(page._target_frame)
                self.assertIsNone(page._target_refresh_matrix)
                self.assertEqual(page._target_source_key,"")
                self.assertFalse(page.video._draw_mode)
                self.assertEqual(page.sop.to_dict(),before)


if __name__=="__main__":
    unittest.main()
