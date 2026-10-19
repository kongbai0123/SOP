"""Following ROI editing uses one explicit image/pose pair, preserving the SOP."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import base64
from copy import deepcopy
import gc
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sop_app.detection import FrameResult
from sop_app.native_runtime import cv2
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step, Workpiece
from sop_app.tracking import workpiece_key
from sop_app.ui.editor_page import EditorPage


POSE = np.array([[.8, -.2, .18], [.2, .8, .03]])
IDENTITY = np.eye(2, 3)
POINTS = [(.2, .2), (.45, .2), (.45, .4), (.2, .4)]


def definition():
    """Tiny CPU-only fixture, also usable by isolated native-window previews."""
    reference = np.full((180, 320, 3), (90, 35, 15), np.uint8)
    cv2.putText(reference, "ORIGINAL", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    encoded = cv2.imencode(".png", reference)[1].tobytes()
    sop = SOPDefinition(name="Follow ROI fixture", workpiece=Workpiece(
        reference_png=base64.b64encode(encoded).decode("ascii"),
        points=[(.1, .1), (.8, .1), (.8, .8), (.1, .8)]),
        rois=[ROI("Follow", list(POINTS), "workpiece"),
              ROI("Fixed", [(.05, .05), (.15, .05), (.15, .15), (.05, .15)])],
        steps=[Step("Install", conditions=[Condition(label="part", roi="Follow")],
                    forbidden=[Condition(label="tool", roi="Follow")])])
    sop.cycle.reset_conditions = [Condition(type="disappear", label="part", roi="Follow")]
    return sop


def packet(sop, *, matrix=POSE, color=(20, 120, 210), at=0., key=None):
    image = np.full((180, 320, 3), color, np.uint8)
    cv2.putText(image, "LIVE " + str(at), (15, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    preview = image.copy()
    preview[:12] = (0, 200, 0)
    evidence = image.copy()
    evidence[:12] = (0, 0, 255)
    result = FrameResult(320, 180, at, workpiece_transform=matrix,
                         tracking_status="valid" if matrix is not None else "lost",
                         workpiece_key=workpiece_key(sop.workpiece) if key is None else key)
    return FramePacket(image, evidence, result, None, 30., 5., model_display=preview)


def forward(points, matrix=POSE):
    return np.asarray(points) @ matrix[:, :2].T + matrix[:, 2]


class FollowROIEditingTests(unittest.TestCase):
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
        with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
            page.set_sop(definition())
        return page

    def assert_no_change(self, page, before, changed):
        self.assertEqual(page.sop.to_dict(), before)
        self.assertEqual(page._roi_undo, [])
        self.assertEqual(page._roi_redo, [])
        changed.assert_not_called()

    def test_add_edit_return_share_region_action_group(self):
        page = self.page()
        layout = page.region_tools.layout()
        self.assertEqual([layout.itemAt(i).widget() for i in range(layout.count())],
                         [page.draw_button, page.edit_rois_button, page.live_button])
        self.assertEqual(page.target_button.text(), "框選參考工件")

    def test_latest_raw_frame_and_exact_pose_used_without_replacing_reference_or_conditions(self):
        page = self.page()
        frame = packet(page.sop)
        before = deepcopy(page.sop.to_dict())
        changed = Mock()
        page.modified.connect(changed)
        page.on_packet(frame)
        page.edit_rois_button.setChecked(True)
        page.reference_button.setChecked(True)
        self.assertEqual(page._edit_frame_kind, "latest")
        np.testing.assert_array_equal(page._edit_frame, frame.frame)
        self.assertNotEqual(page.video._image.pixelColor(0, 0).green(), 200)
        self.assertIn("最新", page.following_frame_label.text())
        self.assertEqual([roi.name for roi in page.video._rois], ["Follow"])
        np.testing.assert_allclose(page.video._rois[0].points, forward(POINTS))
        self.assertEqual(page.video._editable, {"Follow"})
        self.assertIsNone(page._packet)
        self.assertIsNone(page._preview_result)
        self.assert_no_change(page, before, changed)

    def test_scaled_rotated_translated_polygon_inverse_roundtrip_add_edit_undo_redo(self):
        page = self.page()
        page.on_packet(packet(page.sop))
        page.reference_button.setChecked(True)
        original = deepcopy(page.sop.to_dict())
        canonical = [(.27, .24), (.53, .28), (.48, .55), (.23, .48)]
        page._edit_roi_points("Follow", forward(canonical))
        np.testing.assert_allclose(page.sop.rois[0].points, canonical, atol=1e-14)
        np.testing.assert_allclose(page.video._rois[0].points, forward(canonical), atol=1e-14)
        page._undo_roi()
        self.assertEqual(page.sop.to_dict(), original)
        page._redo_roi()
        np.testing.assert_allclose(page.sop.rois[0].points, canonical, atol=1e-14)
        page.draw_button.setChecked(True)
        page._on_roi_drawn(forward(canonical))
        self.assertEqual(page.sop.rois[-1].anchor, "workpiece")
        np.testing.assert_allclose(page.sop.rois[-1].points, canonical, atol=1e-14)
        after = page.sop.to_dict()
        self.assertEqual(after["workpiece"], original["workpiece"])
        self.assertEqual(after["steps"], original["steps"])
        self.assertEqual(after["cycle"], original["cycle"])
        self.assertEqual(workpiece_key(page.sop.workpiece), page._workpiece_key)

    def test_following_edit_does_not_retain_packet_or_detection_result_owners(self):
        page = self.page()
        frame = packet(page.sop)
        packet_owner, result_owner = weakref.ref(frame), weakref.ref(frame.result)
        page.on_packet(frame)
        page.reference_button.setChecked(True)
        del frame
        gc.collect()
        self.assertIsNone(packet_owner())
        self.assertIsNone(result_owner())
        self.assertIsNotNone(page._latest_frame)
        self.assertIsNotNone(page._edit_frame)

    def test_retake_is_read_only_until_explicit_freeze_and_later_frame_cannot_replace_selection(self):
        page = self.page()
        page.on_packet(packet(page.sop, at=1))
        page.reference_button.setChecked(True)
        page.retake_frame_button.click()
        self.assertFalse(page._following_edit_ready())
        page.edit_rois_button.setChecked(True)
        self.assertEqual(page.video._editable, set())
        before = deepcopy(page.sop.to_dict())
        changed = Mock()
        page.modified.connect(changed)
        next_frame = packet(page.sop, at=2, matrix=IDENTITY, color=(80, 40, 20))
        page.on_packet(next_frame)
        self.assertEqual(page._edit_frame_kind, "selecting")
        self.assertIsNone(page._edit_frame)
        page.draw_button.setChecked(True)
        self.assertFalse(page.draw_button.isChecked())
        page._on_roi_drawn(POINTS)
        page._edit_roi_points("Follow", POINTS)
        self.assert_no_change(page, before, changed)
        page.use_frame_button.click()
        self.assertTrue(page._following_edit_ready())
        np.testing.assert_array_equal(page._edit_frame, next_frame.frame)
        np.testing.assert_array_equal(page._edit_matrix, IDENTITY)
        selected = page._edit_frame.copy()
        page.on_packet(packet(page.sop, at=3, color=(10, 20, 30)))
        np.testing.assert_array_equal(page._edit_frame, selected)
        np.testing.assert_array_equal(page._edit_matrix, IDENTITY)
        page.live_button.click()
        self.assertIsNone(page._edit_frame)
        self.assertEqual(page.video._image.pixelColor(0, 0).green(), 200)

    def test_missing_stale_malformed_nonfinite_and_singular_pose_do_not_mutate(self):
        for kind in ("none", "old_key", "stale", "shape", "nan", "singular"):
            with self.subTest(kind=kind):
                page = self.page()
                matrices = {"none": None, "shape": np.eye(3), "nan": np.full((2, 3), np.nan),
                            "singular": np.zeros((2, 3))}
                frame = packet(page.sop, matrix=matrices.get(kind, POSE),
                               key="old-workpiece" if kind == "old_key" else None)
                page.on_packet(frame)
                if kind == "stale":
                    page._latest_arrived_at = time.monotonic() - 4
                before = deepcopy(page.sop.to_dict())
                changed = Mock()
                page.modified.connect(changed)
                page.reference_button.setChecked(True)
                self.assertEqual(page._edit_frame_kind, "selecting")
                self.assertFalse(page._following_edit_ready())
                self.assertFalse(page._use_latest_following_frame())
                page._on_roi_drawn(POINTS)
                page._edit_roi_points("Follow", POINTS)
                self.assert_no_change(page, before, changed)

    def test_latest_invalid_frame_never_reuses_previous_valid_pose(self):
        page = self.page()
        page.on_packet(packet(page.sop))
        page.on_packet(packet(page.sop, matrix=None, at=1))
        page.reference_button.setChecked(True)
        self.assertEqual(page._edit_frame_kind, "selecting")
        self.assertFalse(page.use_frame_button.isEnabled())
        self.assertEqual(page.video._rois, [])

    def test_return_live_restores_matching_fresh_overlay_without_reviving_evaluation(self):
        page = self.page()
        page.on_packet(packet(page.sop))
        page.reference_button.setChecked(True)
        latest = packet(page.sop, at=2, matrix=IDENTITY)
        page.on_packet(latest)
        with patch("sop_app.ui.editor_page.evaluate_step") as step_eval, \
             patch("sop_app.ui.editor_page.evaluate_conditions") as condition_eval:
            page.live_button.click()
            step_eval.assert_not_called()
            condition_eval.assert_not_called()
        visible = next(roi for roi in page.video._rois if roi.name == "Follow")
        np.testing.assert_allclose(visible.points, POINTS)
        self.assertEqual(page.video._image.pixelColor(0, 0).green(), 200)
        self.assertIsNone(page._preview_result)
        self.assertIsNone(page._packet)
        self.assertFalse(page._validation.running)
        self.assertIn("等待即時影像", page._validation_blocker())
        page.reference_button.setChecked(True)
        page._latest_arrived_at = time.monotonic() - 4
        page.live_button.click()
        self.assertEqual([roi.name for roi in page.video._rois], ["Fixed"])

    def test_unfreeze_restores_pixels_before_using_latest_pose_for_display(self):
        page = self.page()
        page.on_packet(packet(page.sop))
        page.freeze_check.setChecked(True)
        page.on_packet(packet(page.sop, matrix=IDENTITY, at=2, color=(40, 80, 120)))
        page.freeze_check.setChecked(False)
        visible = next(roi for roi in page.video._rois if roi.name == "Follow")
        np.testing.assert_allclose(visible.points, POINTS)
        self.assertEqual(page.video._image.pixelColor(20, 20).red(), 120)
        self.assertIsNone(page._preview_result)

    def test_original_reference_is_explicit_and_does_not_replace_canonical_data(self):
        page = self.page()
        before = deepcopy(page.sop.to_dict())
        page.reference_button.setChecked(True)
        self.assertFalse(page._following_edit_ready())
        page.original_frame_button.click()
        self.assertEqual(page._edit_frame_kind, "reference")
        self.assertIn("原始參考影像", page.following_frame_label.text())
        np.testing.assert_array_equal(page._edit_frame, page._reference_frame)
        np.testing.assert_array_equal(page._edit_matrix, IDENTITY)
        page._edit_roi_points("Follow", POINTS)
        self.assertEqual(page.sop.to_dict(), before)

    def test_inverse_outside_reference_is_rejected_without_clamping_history_or_signal(self):
        page = self.page()
        page.on_packet(packet(page.sop, matrix=np.array([[.5, 0., .4], [0., .5, .4]])))
        page.reference_button.setChecked(True)
        before = deepcopy(page.sop.to_dict())
        changed = Mock()
        page.modified.connect(changed)
        page._edit_roi_points("Follow", POINTS)
        page._on_roi_drawn(POINTS)
        page._edit_roi_points("Follow", [(np.nan, .2)] * 4)
        self.assert_no_change(page, before, changed)

    def test_source_and_sop_reset_discard_frame_and_pose_caches(self):
        page = self.page()
        for reset in (lambda: page.clear_video("disconnected"),
                      lambda: page.set_sop(definition())):
            page.on_packet(packet(page.sop))
            page.reference_button.setChecked(True)
            with patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=SimpleNamespace()):
                reset()
            self.assertIsNone(page._latest_frame)
            self.assertIsNone(page._edit_frame)
            self.assertIsNone(page._latest_matrix)
            page.reference_button.setChecked(True)
            self.assertFalse(page._following_edit_ready())
            self.assertEqual(page._edit_frame_kind, "selecting")


if __name__ == "__main__":
    unittest.main()
