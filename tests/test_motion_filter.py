import unittest
import numpy as np

from sop_app.motion_filter import AdaptiveMotionFilter


class MotionFilterTests(unittest.TestCase):
    def setUp(self):
        self.filter = AdaptiveMotionFilter([(.2,.2),(.8,.2),(.8,.8),(.2,.8)])
        self.identity = np.array([[1.,0.,0.],[0.,1.,0.]])

    def test_stationary_noise_is_reduced(self):
        rng = np.random.default_rng(5)
        raw, smooth = [], []
        for i in range(150):
            matrix = self.identity.copy()
            matrix[:,2] = rng.normal(0, 2, 2) / [640,480]
            result = self.filter.update(matrix, i/30, 640,480)
            if i > 30:
                raw.append(matrix[:,2] * [640,480])
                smooth.append(result[:,2] * [640,480])
        self.assertLess(np.std(smooth), np.std(raw) * .8)

    def test_small_stationary_jitter_keeps_anchor(self):
        self.filter.update(self.identity, 0, 640, 480)
        rng = np.random.default_rng(4)
        for i in range(1, 90):
            matrix = self.identity.copy()
            matrix[:, 2] = rng.uniform(-.6, .6, 2) / [640, 480]
            result = self.filter.update(matrix, i / 30, 640, 480)
            np.testing.assert_allclose(result, self.identity, atol=1e-12)

    def test_slow_motion_accumulates_against_anchor(self):
        self.filter.update(self.identity, 0, 640, 480)
        for i in range(1, 121):
            moved = self.identity.copy()
            moved[0, 2] = i * .2 / 640
            result = self.filter.update(moved, i / 30, 640, 480)
        self.assertLess(abs(result[0, 2] - moved[0, 2]) * 640, 4.)
        self.assertGreater(result[0, 2] * 640, 20.)

    def test_rotation_releases_stationary_anchor(self):
        self.filter.update(self.identity, 0, 640, 480)
        angle = .04
        moved = np.array([[np.cos(angle), -np.sin(angle), 0.],
                          [np.sin(angle), np.cos(angle), 0.]])
        result = self.filter.update(moved, 1 / 30, 640, 480)
        np.testing.assert_allclose(result, moved, atol=1e-12)
        self.assertFalse(self.filter.stationary)

    def test_stops_and_settles_without_prediction_drift(self):
        for i in range(30):
            moved = self.identity.copy()
            moved[0, 2] = i / 640 * 3
            self.filter.update(moved, i / 30, 640, 480)
        for i in range(30, 90):
            result = self.filter.update(moved, i / 30, 640, 480)
        self.assertTrue(self.filter.stationary)
        np.testing.assert_allclose(result, moved, atol=1e-12)

    def test_verified_large_jump_corrects_immediately(self):
        self.filter.update(self.identity, 0, 640,480)
        moved = self.identity.copy()
        moved[0,2] = .2
        np.testing.assert_allclose(self.filter.update(moved, 1/30,640,480,verified=True), moved)

    def test_occlusion_never_returns_prediction_and_recovery_has_no_lag(self):
        for i in range(10):
            moved = self.identity.copy()
            moved[0,2] = i * .005
            self.filter.update(moved, i/30,640,480)
        self.assertIsNone(self.filter.update(None, .4,640,480))
        np.testing.assert_allclose(self.filter.update(self.identity,.5,640,480),self.identity)

    def test_irregular_timestamps_and_resolution_reset(self):
        for t in (0, .03, .09, .1, .19, .23):
            moved = self.identity.copy()
            moved[0,2] = t * .1
            result = self.filter.update(moved,t,640,480)
        self.assertLess(abs(result[0,2] - moved[0,2]), .01)
        np.testing.assert_allclose(self.filter.update(self.identity, .3,1280,720),self.identity)
        np.testing.assert_allclose(self.filter.update(moved, .1,1280,720),moved)
