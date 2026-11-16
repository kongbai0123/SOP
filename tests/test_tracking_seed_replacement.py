"""輪廓重新定位不得在建立新光流失敗時留下舊位置的追蹤點。"""
import base64
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sop_app.sop_schema import Workpiece
from sop_app.tracking import WorkpieceLocator


TARGET = [(0.25, 0.25), (0.7, 0.25), (0.7, 0.72), (0.25, 0.72)]
IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


class TemplateFlowReplacementTests(unittest.TestCase):
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

    def test_failed_template_seed_discards_old_points(self):
        gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        for points in (None, np.zeros((9, 1, 2), np.float32), np.zeros((20, 1, 2), np.float32)):
            with self.subTest(points=None if points is None else len(points)):
                locator = WorkpieceLocator(self.workpiece)
                locator.locate(self.reference, timestamp=0.)
                self.assertIsNotNone(locator._flow_gray)
                with patch('sop_app.tracking.cv2.goodFeaturesToTrack', return_value=points):
                    locator._seed_template_flow(gray, IDENTITY)
                self.assertIsNone(locator._flow_gray)
                self.assertIsNone(locator._flow_src)
                self.assertIsNone(locator._flow_dst)

    def test_absolute_relocation_without_seed_cannot_snap_back_to_old_flow(self):
        moved = cv2.warpAffine(self.reference, np.float32([[1, 0, 40], [0, 1, 20]]), (640, 480))
        expected = IDENTITY.copy()
        expected[:, 2] = [40 / 640, 20 / 480]
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive):
                locator = WorkpieceLocator(self.workpiece, adaptive_budget=adaptive)
                locator.locate(self.reference, timestamp=0.)
                if adaptive:
                    # 本幀排到輪廓核對；預算選擇本身已有獨立測試。
                    budget = patch.object(locator.budget, 'select_stage', return_value='template')
                else:
                    budget = nullcontext()
                with patch.object(locator, '_estimate_features', return_value=(None, 0)), \
                     patch.object(locator, '_estimate_sift', return_value=(None, 0)), \
                     patch('sop_app.tracking.cv2.goodFeaturesToTrack', return_value=None), budget:
                    with patch.object(locator, '_estimate_flow', return_value=(IDENTITY.copy(), 30)) as old_flow:
                        actual, _ = locator.locate(moved, timestamp=.6)
                    old_flow.assert_called_once()
                    np.testing.assert_allclose(actual, expected, atol=.012)
                    self.assertEqual(locator.metrics['source'], 'template')
                    self.assertIsNone(locator._flow_gray)
                    with patch.object(locator, '_estimate_flow', return_value=(IDENTITY.copy(), 30)) as old_flow:
                        actual, _ = locator.locate(moved, timestamp=.61)
                    old_flow.assert_not_called()
                    np.testing.assert_allclose(actual, expected, atol=.012)
                    self.assertEqual(locator.metrics['source'], 'template')


if __name__ == '__main__':
    unittest.main()
