"""核對頻率同時受來源時間與幀數限制；低 FPS、倒帶及壞時間戳不能延長可信光流。"""
import base64
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sop_app.detection import TRACKING_CONFIRMED, TRACKING_UNCERTAIN
from sop_app.sop_schema import Workpiece
from sop_app.tracking import WorkpieceLocator, capture_workpiece


TARGET = [(0.25, 0.25), (0.7, 0.25), (0.7, 0.72), (0.25, 0.72)]
IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


class TrackingTimeRecheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = np.zeros((480, 640, 3), np.uint8)
        rng = np.random.default_rng(42)
        for _ in range(400):
            x, y = rng.integers([180, 140], [430, 330])
            cv2.circle(cls.reference, (int(x), int(y)), int(rng.integers(2, 6)),
                       tuple(int(v) for v in rng.integers(70, 255, 3)), -1)
        _, encoded = cv2.imencode('.png', cls.reference)
        cls.workpiece = Workpiece(reference_png=base64.b64encode(encoded).decode('ascii'), points=TARGET)

    def locator(self, adaptive):
        return WorkpieceLocator(self.workpiece, adaptive_budget=adaptive)

    def test_five_fps_rechecks_on_first_frame_at_or_after_half_second(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                locator = self.locator(adaptive)
                with patch.object(locator, '_estimate_features', wraps=locator._estimate_features) as absolute:
                    for timestamp, source in ((0., 'orb'), (.2, 'flow'), (.4, 'flow'), (.6, 'orb'),
                                              (.8, 'flow'), (1., 'flow'), (1.2, 'orb')):
                        self.assertIsNotNone(locator.locate(self.reference, timestamp=timestamp)[0])
                        self.assertEqual(locator.metrics['source'], source)
                    self.assertEqual(absolute.call_count, 3)
                    self.assertEqual(locator.metrics['time_recheck_reason'], 'time_limit')

    def test_high_fps_still_rechecks_after_fifteen_frames(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                locator = self.locator(adaptive)
                locator.locate(self.reference, timestamp=0.)
                with patch.object(locator, '_estimate_features', wraps=locator._estimate_features) as absolute:
                    for index in range(1, locator.MAX_RECHECK_FRAMES):
                        locator.locate(self.reference, timestamp=index / 120)
                        self.assertEqual(locator.metrics['source'], 'flow')
                    absolute.assert_not_called()
                    locator.locate(self.reference, timestamp=locator.MAX_RECHECK_FRAMES / 120)
                    absolute.assert_called_once()
                    self.assertEqual(locator.metrics['source'], 'orb')

    def test_timestamp_discontinuities_force_absolute_measurement(self):
        for adaptive in (False, True):
            for timestamp in (10., 9., 1e9, float('nan'), float('inf')):
                with self.subTest(adaptive=adaptive, timestamp=timestamp):
                    locator = self.locator(adaptive)
                    locator.locate(self.reference, timestamp=10.)
                    with patch.object(locator, '_estimate_features', wraps=locator._estimate_features) as absolute:
                        matrix, _ = locator.locate(self.reference, timestamp=timestamp)
                    absolute.assert_called_once()
                    self.assertEqual(locator.metrics['source'], 'orb')
                    self.assertEqual(locator.state, TRACKING_CONFIRMED)
                    self.assertTrue(np.isfinite(matrix).all())

    def test_failed_time_recheck_blocks_output_until_next_success(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                locator = self.locator(adaptive)
                locator.locate(self.reference, timestamp=0.)
                with ExitStack() as patches:
                    absolute = patches.enter_context(patch.object(locator, '_estimate_features', return_value=(None, 0)))
                    patches.enter_context(patch.object(locator, '_estimate_sift', return_value=(None, 0)))
                    patches.enter_context(patch.object(locator, '_estimate_template', return_value=(None, 0)))
                    for timestamp in (.2, .4):
                        self.assertIsNotNone(locator.locate(self.reference, timestamp=timestamp)[0])
                    absolute.assert_not_called()
                    for timestamp in (.6, .61):
                        matrix, _ = locator.locate(self.reference, timestamp=timestamp)
                        self.assertIsNone(matrix)
                        self.assertEqual(locator.state, TRACKING_UNCERTAIN)
                    self.assertEqual(absolute.call_count, 2)
                self.assertIsNotNone(locator.locate(self.reference, timestamp=.62)[0])
                self.assertEqual(locator.state, TRACKING_CONFIRMED)
                self.assertEqual(locator.metrics['source'], 'orb')

    def test_bad_timestamp_followed_by_rewind_cannot_reuse_previously_confirmed_flow(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                locator = self.locator(adaptive)
                locator.locate(self.reference, timestamp=10.)
                with patch.object(locator, '_estimate_features', return_value=(None, 0)), \
                     patch.object(locator, '_estimate_sift', return_value=(None, 0)), \
                     patch.object(locator, '_estimate_template', return_value=(None, 0)):
                    with patch.object(locator, '_estimate_flow', return_value=(None, 0)):
                        self.assertIsNone(locator.locate(self.reference, timestamp=float('nan'))[0])
                    # 光流從舊上下文恢復也必須核對，不能因壞時間戳抹去原本的來源時間而放行。
                    with patch.object(locator, '_estimate_flow', return_value=(IDENTITY.copy(), 30)):
                        self.assertIsNone(locator.locate(self.reference, timestamp=1.)[0])
                    self.assertEqual(locator.metrics['time_recheck_reason'], 'timestamp_discontinuity')
                    self.assertEqual(locator.state, TRACKING_UNCERTAIN)

    def test_reset_discards_previous_source_clock(self):
        locator = self.locator(True)
        locator.locate(self.reference, timestamp=10.)
        locator.reset()
        locator.locate(self.reference, timestamp=0.)
        locator.locate(self.reference, timestamp=.1)
        self.assertEqual(locator.metrics['source'], 'flow')
        self.assertEqual(locator.metrics['time_recheck_reason'], '')

    def test_verified_large_motion_is_not_rejected_due_to_flow_conflict(self):
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                locator = self.locator(adaptive)
                locator.locate(self.reference, timestamp=0.)
                moved = IDENTITY.copy()
                moved[0, 2] = .25
                with patch.object(locator, '_estimate_flow', return_value=(IDENTITY.copy(), 30)), \
                     patch.object(locator, '_estimate_features', return_value=(moved, 30)):
                    matrix, _ = locator.locate(self.reference, timestamp=.6)
                np.testing.assert_allclose(matrix, moved)
                self.assertEqual(locator.state, TRACKING_CONFIRMED)

    def test_capture_checks_the_same_adaptive_strategy_as_runtime(self):
        with patch('sop_app.tracking.WorkpieceLocator') as locator:
            workpiece = capture_workpiece(self.reference, TARGET)
        locator.assert_called_once_with(workpiece, adaptive_budget=True)


if __name__ == '__main__':
    unittest.main()
