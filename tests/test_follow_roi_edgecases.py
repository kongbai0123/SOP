"""Following-region frame ownership and cross-mode guards use synthetic frames only."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sop_app.detection import FrameResult
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step, Workpiece
from sop_app.ui.editor_page import EditorPage


FOLLOWING = [(0.1, 0.1), (0.25, 0.1), (0.25, 0.25), (0.1, 0.25)]
FIXED = [(0.6, 0.6), (0.8, 0.6), (0.8, 0.8), (0.6, 0.8)]
POSE_A = np.array([[1., 0., .05], [0., 1., .1]])
POSE_B = np.array([[.8, -.1, .2], [.1, .8, .15]])


class FollowROIFrameEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.editor = EditorPage()
        self.addCleanup(self.dispose)
        self.load_sop()

    def dispose(self):
        for timer in self.editor.findChildren(QTimer):
            timer.stop()
        self.editor.close()

    def load_sop(self):
        definition = SOPDefinition(
            workpiece=Workpiece(reference_png="canonical-reference-fixture",
                                points=[(.05, .05), (.95, .05), (.95, .95), (.05, .95)]),
            rois=[ROI("跟隨區", list(FOLLOWING), "workpiece"), ROI("固定區", list(FIXED), "fixed")],
            steps=[Step("安裝", conditions=[Condition(label="tool", roi="跟隨區")],
                        forbidden=[Condition(label="part", roi="固定區")])])
        reference = np.full((100, 160, 3), 7, np.uint8)
        # Coordinate/state tests do not invoke native feature extraction or any model.
        with patch("sop_app.ui.editor_page.decode_reference", return_value=reference), \
             patch("sop_app.ui.editor_page.WorkpieceLocator", return_value=object()):
            self.editor.set_sop(definition)

    def packet(self, value, matrix):
        raw = np.full((100, 160, 3), value, np.uint8)
        preview = np.full_like(raw, value + 1)
        result = FrameResult(160, 100, float(value), workpiece_transform=matrix,
                             workpiece_key=self.editor._workpiece_key, tracking_status="定位有效")
        return FramePacket(raw, preview, result, None, 30., 5., model_display=preview)

    def assert_following_display(self, matrix):
        visible = next(roi for roi in self.editor.video._rois if roi.name == "跟隨區")
        expected = np.asarray(FOLLOWING) @ matrix[:, :2].T + matrix[:, 2]
        np.testing.assert_allclose(visible.points, expected)

    def test_frozen_pixels_and_pose_survive_owner_mutation_and_new_packets(self):
        matrix = POSE_A.copy()
        packet = self.packet(10, matrix)
        self.editor.on_packet(packet)
        self.assertTrue(self.editor._begin_roi_edit("跟隨區"))
        self.assertTrue(self.editor._following_edit_ready())
        self.assertIsNone(self.editor._packet)
        self.assertIsNone(self.editor._preview_result)
        packet.frame[:] = 90
        packet.preview_display[:] = 91
        matrix[:] = POSE_B
        self.editor.on_packet(self.packet(40, POSE_B.copy()))
        np.testing.assert_array_equal(self.editor._edit_frame, np.full((100, 160, 3), 10, np.uint8))
        np.testing.assert_array_equal(self.editor._edit_matrix, POSE_A)
        self.assertEqual(self.editor.video._image.pixelColor(5, 5).red(), 10)
        self.assert_following_display(POSE_A)
        self.editor.retake_frame_button.click()
        self.assertFalse(self.editor._following_edit_ready())
        self.assertFalse(self.editor.video._editable)
        self.editor.use_frame_button.click()
        np.testing.assert_array_equal(self.editor._edit_matrix, POSE_B)
        self.assertEqual(self.editor.video._image.pixelColor(5, 5).red(), 40)

    def test_fixed_edit_after_following_edit_uses_frame_coordinates_and_undo(self):
        self.editor.on_packet(self.packet(10, POSE_A.copy()))
        canonical = deepcopy(self.editor.sop.workpiece)
        conditions = deepcopy(self.editor.sop.steps)
        self.assertTrue(self.editor._begin_roi_edit("跟隨區"))
        attempted = [(x - .1, y - .1) for x, y in FIXED]
        self.editor.video.roi_edited.emit("固定區", attempted)
        self.assertEqual(self.editor.sop.rois[1].points, FIXED)
        self.editor.live_button.click()
        self.editor.on_packet(self.packet(40, POSE_B.copy()))
        self.assertTrue(self.editor._begin_roi_edit("固定區"))
        self.assertFalse(self.editor.reference_button.isChecked())
        self.assertTrue(self.editor.freeze_check.isChecked())
        self.assertIn("固定區", self.editor.video._editable)
        self.assertNotIn("跟隨區", self.editor.video._editable)
        self.editor.video.roi_edited.emit("固定區", attempted)
        self.assertEqual(self.editor.sop.rois[1].points, attempted)
        self.assertEqual(self.editor.sop.rois[0].points, FOLLOWING)
        self.assertEqual(self.editor.sop.workpiece, canonical)
        self.assertEqual(self.editor.sop.steps, conditions)
        self.editor._undo_roi()
        self.assertEqual(self.editor.sop.rois[1].points, FIXED)
        self.editor._return_live()
        self.editor.on_packet(self.packet(40, POSE_B.copy()))
        self.assertTrue(self.editor._begin_roi_edit("跟隨區"))
        self.assert_following_display(POSE_B)

    def test_production_start_keeps_pixels_and_overlay_consistent_and_blocks_late_signals(self):
        self.editor.on_packet(self.packet(10, POSE_A.copy()))
        self.assertTrue(self.editor._begin_roi_edit("跟隨區"))
        self.editor.draw_button.setChecked(True)
        self.editor.on_packet(self.packet(40, POSE_B.copy()))
        definition = deepcopy(self.editor.sop)
        undo_depth = len(self.editor._roi_undo)
        modified = Mock()
        self.editor.modified.connect(modified)
        self.editor.set_production_running(True)
        self.assertFalse(self.editor._following_edit_ready())
        self.assertFalse(self.editor.video._editable)
        self.assertFalse(self.editor.video._draw_mode)
        pixel = self.editor.video._image.pixelColor(5, 5).red()
        if pixel == 10:
            self.assert_following_display(POSE_A)
        else:
            self.assertEqual(pixel, 41, "live mode must show the latest preview, not a different frame")
            self.assert_following_display(POSE_B)
        self.editor.video.roi_edited.emit("跟隨區", FOLLOWING)
        self.editor.video.roi_drawn.emit(FOLLOWING)
        self.assertFalse(self.editor._begin_roi_edit("跟隨區"))
        self.assertEqual(self.editor.sop, definition)
        self.assertEqual(len(self.editor._roi_undo), undo_depth)
        modified.assert_not_called()

    def test_source_and_sop_reset_cannot_reuse_a_previous_frame_even_with_same_key(self):
        for reset in (lambda: self.editor.clear_video("來源已切換"), self.load_sop):
            with self.subTest(reset=reset.__name__):
                self.editor._return_live()
                self.editor.on_packet(self.packet(10, POSE_A.copy()))
                self.assertTrue(self.editor._begin_roi_edit("跟隨區"))
                key = self.editor._workpiece_key
                reset()
                self.assertEqual(self.editor._workpiece_key, key)
                self.assertIsNone(self.editor._latest_frame)
                self.assertIsNone(self.editor._latest_matrix)
                self.assertIsNone(self.editor._edit_frame)
                self.assertIsNone(self.editor._edit_matrix)
                self.assertFalse(self.editor.reference_button.isChecked())
                self.editor.reference_button.setChecked(True)
                self.assertFalse(self.editor._following_edit_ready())
                self.assertFalse(self.editor.use_frame_button.isEnabled())
                self.assertFalse(self.editor._use_latest_following_frame())
                self.assertFalse(self.editor.video._editable)

    def test_inverse_roundtrip_keeps_legal_reference_edges_and_rejects_real_overflow(self):
        angle = np.deg2rad(17.)
        linear = .63 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        matrix = np.column_stack((linear, np.array([.5, .5]) - linear @ np.array([.5, .5])))
        self.editor.on_packet(self.packet(10, matrix))
        self.assertTrue(self.editor._begin_roi_edit("跟隨區"))
        canonical = np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
        displayed = canonical @ matrix[:, :2].T + matrix[:, 2]
        converted = self.editor._following_points_to_reference(displayed)
        self.assertIsNotNone(converted, "round-trip floating-point noise must not reject valid 0/1 boundaries")
        np.testing.assert_allclose(converted, canonical, atol=1e-12)
        self.assertGreaterEqual(np.min(converted), 0.)
        self.assertLessEqual(np.max(converted), 1.)
        outside = canonical.copy()
        outside[0, 0] = -.00001
        invalid = outside @ matrix[:, :2].T + matrix[:, 2]
        before = deepcopy(self.editor.sop)
        undo_depth = len(self.editor._roi_undo)
        self.assertIsNone(self.editor._following_points_to_reference(invalid))
        self.assertEqual(self.editor.sop, before)
        self.assertEqual(len(self.editor._roi_undo), undo_depth)


if __name__ == "__main__":
    unittest.main()
