import unittest
from unittest.mock import patch

from sop_app.tracking_budget import TrackingBudget
from sop_app.tracking import WorkpieceLocator, capture_workpiece
from test_tracking import scene, TARGET
from sop_app.pipeline import FramePacket
from sop_app.detection import FrameResult


class BudgetTests(unittest.TestCase):
    def test_video_timestamps_are_not_compared_to_wall_clock(self):
        frame = scene()
        packet = FramePacket(frame, frame, FrameResult(640, 480, 1.), None, 30., 10., processing_ms=15.)
        with patch('sop_app.pipeline.time.monotonic', return_value=100.):
            self.assertEqual(packet.tracking_load_ms, 15.)
            packet.capture_age_ms = 10.
            packet.result.timestamp = 99.95
            self.assertAlmostEqual(packet.tracking_load_ms, 50.)

    def test_unknown_cost_probed_then_expensive_retry_bounded(self):
        budget = TrackingBudget()
        self.assertTrue(budget.allow('sift', 0., 40.))
        budget.record('sift', 0., 80.)
        self.assertFalse(budget.allow('sift', .1, 40.))
        self.assertTrue(budget.allow('sift', .25, 40.))

    def test_pressure_requires_sustained_overload_and_recovers(self):
        budget = TrackingBudget()
        budget.observe_frame(100.)
        self.assertFalse(budget.overloaded)
        for _ in range(30):
            budget.observe_frame(60.)
        self.assertTrue(budget.overloaded)
        for _ in range(80):
            budget.observe_frame(5.)
        self.assertFalse(budget.overloaded)

    def test_available_budget_allows_early_retry(self):
        budget = TrackingBudget()
        budget.record('sift', 0., 8.)
        self.assertTrue(budget.allow('sift', .04, 10.))
        self.assertFalse(budget.allow('sift', .04, 30.))

    def test_budget_skips_never_reuse_old_transform_and_orb_recovers_immediately(self):
        frame = scene()
        locator = WorkpieceLocator(capture_workpiece(frame, TARGET), adaptive_budget=True)
        self.assertIsNotNone(locator.locate(frame)[0])
        with patch.object(locator, '_estimate_flow', return_value=(None, 0)), \
             patch.object(locator, '_estimate_features', return_value=(None, 0)), \
             patch.object(locator.budget, 'allow', return_value=False), \
             patch.object(locator, '_estimate_sift') as sift, \
             patch.object(locator, '_estimate_template') as template:
            for _ in range(5):
                self.assertIsNone(locator.locate(frame, other_ms=90.)[0])
            sift.assert_not_called()
            template.assert_not_called()
            self.assertEqual(locator.metrics['skipped'], ['sift', 'template'])
        # Loss resets must not erase the cost history or prevent a valid ORB match.
        self.assertIn('orb', locator.budget.costs)
        self.assertIsNotNone(locator.locate(frame, other_ms=90.)[0])
        self.assertTrue(locator.metrics['valid'])


if __name__ == '__main__':
    unittest.main()
