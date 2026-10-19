"""Adaptive work limits must preserve current-frame geometry and fail closed."""
import base64
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sop_app.detection import Detection
from sop_app.sop_schema import Workpiece
from sop_app.tracking import WorkpieceLocator
from sop_app.tracking_budget import TrackingBudget
from test_tracking import scene, TARGET, IDENTITY


class AdaptiveTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = scene()
        _, encoded = cv2.imencode('.png', cls.reference)
        cls.workpiece = Workpiece(reference_png=base64.b64encode(encoded).decode('ascii'), points=TARGET)

    def locator(self):
        return WorkpieceLocator(self.workpiece, adaptive_budget=True)

    def test_detector_local_first_and_wrong_hint_global_fallback(self):
        good = Detection('handlebar', .95, (140, 90, 510, 390))
        locator = self.locator()
        self.assertIsNotNone(locator.locate(self.reference, detections=[good])[0])
        self.assertEqual(locator.metrics['source'], 'local_orb')
        self.assertNotIn('orb', locator.metrics['stages_ms'])
        wrong = Detection('handlebar', .95, (0, 0, 80, 80))
        locator = self.locator()
        self.assertIsNotNone(locator.locate(self.reference, detections=[wrong])[0])
        self.assertEqual(locator.metrics['source'], 'orb')
        self.assertIn('local_orb', locator.metrics['stages_ms'])

    def test_valid_flow_periodic_verification_uses_global_reference(self):
        locator = self.locator()
        locator._flow_gray = np.zeros((480, 640), np.uint8)
        locator._flow_age = locator.MAX_RECHECK_FRAMES
        good = Detection('handlebar', .95, (140, 90, 510, 390))
        with patch.object(locator, '_estimate_flow', return_value=(IDENTITY.copy(), 30)), \
             patch.object(locator, '_estimate_features', return_value=(IDENTITY.copy(), 30)) as estimate:
            locator.locate(self.reference, detections=[good])
        estimate.assert_called_once()
        self.assertEqual(len(estimate.call_args.args), 1)
        self.assertEqual(locator.metrics['source'], 'orb')

    def test_one_expensive_stage_even_when_all_retries_forced(self):
        locator = self.locator()
        selected = []
        with patch.object(locator, '_estimate_features', return_value=(None, 0)), \
             patch.object(locator, '_estimate_sift', return_value=(None, 0)), \
             patch.object(locator, '_estimate_template', return_value=(None, 0)), \
             patch.object(locator.budget, 'allow', return_value=True):
            for _ in range(6):
                self.assertIsNone(locator.locate(self.reference)[0])
                stages = set(locator.metrics['stages_ms']) & {'sift', 'template'}
                self.assertEqual(len(stages), 1)
                selected.extend(stages)
        self.assertEqual(selected, ['sift', 'template'] * 3)

    def test_rotation_candidate_search_progresses_without_old_frame_pose(self):
        locator = self.locator()
        affine = cv2.getRotationMatrix2D((320, 240), 6, 1)
        affine[:, 2] += [18, 9]
        moved = cv2.warpAffine(self.reference, affine, (640, 480))
        with patch.object(locator, '_estimate_features', return_value=(None, 0)), \
             patch.object(locator, '_estimate_sift', return_value=(None, 0)), \
             patch.object(locator.budget, 'allow', return_value=True):
            actual = None
            for _ in range(6):
                actual, _ = locator.locate(moved)
                self.assertLessEqual(locator.metrics.get('template_candidates', 0), 6)
                if actual is not None:
                    break
        self.assertIsNotNone(actual)
        expected = np.diag([1/640, 1/480]) @ affine @ np.diag([640, 480, 1])
        np.testing.assert_allclose(actual, expected, atol=.015)

    def test_loss_never_returns_stale_pose_and_orb_recovers_when_budget_denies(self):
        locator = self.locator()
        self.assertIsNotNone(locator.locate(self.reference, timestamp=1.)[0])
        with patch.object(locator.budget, 'allow', return_value=False):
            for index in range(5):
                self.assertIsNone(locator.locate(np.zeros_like(self.reference), timestamp=2. + index)[0])
            self.assertIsNotNone(locator.locate(self.reference, timestamp=8.)[0])
        self.assertEqual(locator.metrics['source'], 'orb')

    def test_flow_seed_cap_keeps_spatial_coverage(self):
        locator = self.locator()
        x, y = np.meshgrid(np.linspace(160, 450, 35), np.linspace(120, 345, 30))
        source = np.float32(np.column_stack([x.ravel(), y.ravel()]))
        locator._seed_flow(np.zeros((480, 640), np.uint8), source, source + 10)
        self.assertEqual(len(locator._flow_src), locator.MAX_FLOW_POINTS)
        self.assertTrue((np.ptp(locator._flow_src, axis=0) >= np.ptp(source, axis=0) * .8).all())
        np.testing.assert_allclose(locator._flow_dst - locator._flow_src, 10, atol=1e-4)

    def test_resized_crop_points_return_to_original_frame_coordinates(self):
        class Detector:
            def detectAndCompute(self, image, mask):
                self.shape = image.shape
                return [cv2.KeyPoint(10, 20, 3)], np.ones((1, 32), np.uint8)
        detector = Detector()
        points, _ = WorkpieceLocator._keypoints_in_box(detector, np.zeros((401, 803), np.uint8),
                                                      (101, 37, 702, 338), 300)
        self.assertLessEqual(max(detector.shape), 300)
        np.testing.assert_allclose(points[0].pt, [101 + 10 * 601 / 300, 37 + 20 * 301 / 150], atol=1e-4)

    def test_components_cover_actual_features_matching_geometry_and_flow(self):
        locator = self.locator()
        locator.locate(self.reference)
        first = locator.metrics['components_ms']
        for name in ('preprocess', 'orb.features', 'orb.matching', 'orb.geometry', 'motion_filter'):
            self.assertGreaterEqual(first[name], 0.)
        locator.locate(self.reference)
        second = locator.metrics['components_ms']
        for name in ('flow.forward', 'flow.backward', 'flow.geometry'):
            self.assertGreaterEqual(second[name], 0.)
        self.assertGreaterEqual(locator.metrics['tracking_ms'], sum(second.values()))


class RecoveryBudgetSchedulingTests(unittest.TestCase):
    def test_forced_due_stages_are_staggered_and_fair(self):
        budget = TrackingBudget()
        budget.record('sift', 0., 100.)
        budget.record('template', 0., 100.)
        budget.begin_frame()
        self.assertEqual(budget.select_stage(['sift', 'template'], .3, 100.), 'sift')
        self.assertIsNone(budget.select_stage(['sift', 'template'], .3, 100.))
        budget.begin_frame()
        self.assertEqual(budget.select_stage(['sift', 'template'], .31, 100.), 'template')


if __name__ == '__main__':
    unittest.main()
