"""Guided search must retain geometric validation and full-frame recovery."""
import unittest
from unittest.mock import patch
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from sop_app.detection import Detection
from sop_app.detection import FrameResult
from sop_app.pipeline import FramePacket
from sop_app.tracking import WorkpieceLocator, capture_workpiece
from sop_app.tracking_debug import TrackingDebug
from test_tracking import TARGET, scene


class ReacquisitionTests(unittest.TestCase):
    def setUp(self):
        self.frame = scene()
        self.locator = WorkpieceLocator(capture_workpiece(self.frame, TARGET))

    def test_detector_box_keeps_original_verified_pose(self):
        detection = Detection('handlebar', .95, (140, 90, 510, 390))
        matrix, _ = self.locator.locate(self.frame, detections=[detection])
        self.assertIsNotNone(matrix)
        self.assertEqual(self.locator.metrics['source'], 'orb')

    def test_detector_guides_recovery_only_after_original_methods_fail(self):
        detection = Detection('handlebar', .95, (140, 90, 510, 390))
        original = self.locator._estimate_features
        def feature_search(gray, box=None):
            return original(gray, box) if box is not None else (None, 0)
        with patch.object(self.locator, '_estimate_features', side_effect=feature_search), \
             patch.object(self.locator, '_estimate_sift', return_value=(None, 0)), \
             patch.object(self.locator, '_estimate_template', return_value=(None, 0)):
            matrix, _ = self.locator.locate(self.frame, detections=[detection])
        self.assertIsNotNone(matrix)
        self.assertEqual(self.locator.metrics['source'], 'local_orb')
        self.assertEqual(self.locator.metrics['diagnostics']['local_orb']['reason'], 'accepted')

    def test_wrong_detector_box_falls_back_to_global_matching(self):
        wrong = Detection('handlebar', .95, (0, 0, 100, 100))
        matrix, _ = self.locator.locate(self.frame, detections=[wrong])
        self.assertIsNotNone(matrix)
        self.assertNotIn(self.locator.metrics['source'], ('local_orb', 'local_sift'))

    def test_detector_alone_never_produces_valid_pose(self):
        blank = np.zeros_like(self.frame)
        detection = Detection('handlebar', .95, (140, 90, 510, 390))
        matrix, _ = self.locator.locate(blank, detections=[detection])
        self.assertIsNone(matrix)
        self.assertEqual(self.locator.metrics['source'], 'none')
        self.assertIn('reason', self.locator.metrics['diagnostics']['local_orb'])

    def test_local_reacquisition_after_occlusion_is_immediate(self):
        detection = Detection('handlebar', .95, (140, 90, 510, 390))
        self.locator.locate(self.frame, detections=[detection])
        for _ in range(4):
            self.assertIsNone(self.locator.locate(np.zeros_like(self.frame), detections=[detection])[0])
        matrix, _ = self.locator.locate(self.frame, detections=[detection])
        self.assertIsNotNone(matrix)
        self.assertEqual(self.locator.metrics['source'], 'orb')

    def test_visual_last_pose_is_logged_separately_from_decision_pose(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'SOP_TRACKING_DEBUG': folder}):
            debug = TrackingDebug()
            result = FrameResult(640, 480, 1., [])
            packet = FramePacket(self.frame, self.frame, result,
                                 SimpleNamespace(results=[SimpleNamespace(error='position unavailable')]),
                                 30., 5.)
            debug.capture(packet, np.eye(2, 3), 'lost', {'source': 'none'}, running=True)
            row = json.loads((Path(folder) / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()[0])
            self.assertTrue(row['display_valid'])
            self.assertFalse(row['decision_valid'])
            self.assertTrue(row['decision_blocked'])


if __name__ == '__main__':
    unittest.main()
