"""Object-only motion against a stationary background, and absence protection."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sop_app.tracking import WorkpieceLocator, capture_workpiece


POINTS = [(0.22, .12), (.78, .12), (.78, .67), (.22, .67)]


def moved_part(background, mask, affine):
    moved = background.copy()
    moved[cv2.warpAffine(mask, affine, (640, 480)) > 128] = 215
    return cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)


def normalized(affine):
    return np.diag([1 / 640, 1 / 480]) @ affine @ np.diag([640, 480, 1])


def fixture():
    background = np.full((480, 640, 3), 140, np.uint8)
    cv2.line(background, (0, 75), (639, 75), (30, 30, 30), 3)
    cv2.rectangle(background, (20, 365), (170, 465), (80, 80, 80), 3)
    mask = np.zeros(background.shape[:2], np.uint8)
    cv2.line(mask, (320, 105), (320, 240), 255, 30)
    cv2.line(mask, (205, 240), (435, 240), 255, 30)
    cv2.ellipse(mask, (205, 260), (22, 38), 0, 90, 270, 255, 23)
    cv2.ellipse(mask, (435, 260), (22, 38), 0, -90, 90, 255, 23)
    reference = background.copy()
    reference[mask > 0] = 215
    return background, reference, mask


class ContourRecoveryTests(unittest.TestCase):
    def test_verified_contour_rechecks_before_orb_can_replace_its_flow_seed(self):
        _, reference, _ = fixture()
        points = [(0.22, .12), (.78, .12), (.78, .67), (.22, .67)]
        locator = WorkpieceLocator(capture_workpiece(reference, points), adaptive_budget=True)
        locator._last_absolute_stage = 'template'
        measured = np.array([[1., 0., .02], [0., 1., .03]])
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        # Both a normal mandatory recheck and recovery after complete flow loss
        # must measure the whole contour before accepting local ORB points.
        for flowed in ((measured, 68), (None, 0)):
            locator._verification_failed = False
            with patch.object(locator, '_estimate_template', return_value=(measured, -90)) as contour, \
                 patch.object(locator, '_estimate_features', return_value=(measured, 18)) as orb:
                actual, count = locator._estimate_adaptive(gray, None, flowed)
            np.testing.assert_array_equal(actual, measured)
            self.assertEqual(count, -90)
            contour.assert_called_once()
            orb.assert_not_called()

    def test_angle_hint_is_a_fresh_search_and_recovers_after_loss(self):
        background, reference, mask = fixture()
        points = [(0.22, .12), (.78, .12), (.78, .67), (.22, .67)]
        locator = WorkpieceLocator(capture_workpiece(reference, points), adaptive_budget=True)
        affine = cv2.getRotationMatrix2D((320, 220), 24, 1.)
        affine[:, 2] += (18, 35)
        moved = background.copy()
        moved[cv2.warpAffine(mask, affine, (640, 480)) > 128] = 215
        gray = cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)
        # The second bounded search contains this rotation. Thereafter either
        # group must independently verify the angle, avoiding periodic gaps.
        outputs = [locator._estimate_contour(gray)[0] for _ in range(4)]
        self.assertTrue(all(value is not None for value in outputs[1:]))
        locator.reset()
        self.assertIsNone(locator._estimate_contour(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))[0])
        self.assertIsNotNone(locator._estimate_contour(gray)[0])

    def test_rotated_part_recovers_without_rotating_background(self):
        background, reference, mask = fixture()
        points = [(0.22, .12), (.78, .12), (.78, .67), (.22, .67)]
        locator = WorkpieceLocator(capture_workpiece(reference, points))
        affine = cv2.getRotationMatrix2D((320, 220), 24, 1.)
        affine[:, 2] += (18, 35)
        moved = background.copy()
        moved[cv2.warpAffine(mask, affine, (640, 480)) > 128] = 215
        # Exact reference descriptors are deliberately unavailable: recovery
        # must compare current contours rather than prolong a previous pose.
        with patch.object(locator, '_estimate_features', return_value=(None, 0)), \
             patch.object(locator, '_estimate_sift', return_value=(None, 0)):
            actual, _ = locator.locate(moved, 0.)
        self.assertIsNotNone(actual)
        expected = np.diag([1 / 640, 1 / 480]) @ affine @ np.diag([640, 480, 1])
        np.testing.assert_allclose(actual, expected, atol=.015)
        self.assertEqual(locator.metrics['diagnostics']['template']['method'], 'contour_distance')
        self.assertIsNotNone(locator._flow_src)
        # Remove the actual object while leaving the stationary background.
        self.assertIsNone(locator.locate(background, 1.)[0])
        self.assertIsNone(locator.locate(background, 2.)[0])
        # A previous position cannot make absence valid; the rotated object
        # must be freshly acquired after flow has been lost.
        self.assertIsNotNone(locator.locate(moved, 3.)[0])

    def test_repeated_background_lines_are_not_a_unique_contour(self):
        _, reference, _ = fixture()
        points = [(0.22, .12), (.78, .12), (.78, .67), (.22, .67)]
        locator = WorkpieceLocator(capture_workpiece(reference, points))
        distractor = np.full_like(reference, 140)
        for y in range(30, 480, 25):
            cv2.line(distractor, (0, y), (639, y), (215, 215, 215), 7)
        for x in range(30, 640, 55):
            cv2.line(distractor, (x, 0), (x, 479), (215, 215, 215), 7)
        gray = cv2.cvtColor(distractor, cv2.COLOR_BGR2GRAY)
        # Also covers every full-circle group reached after repeated failures.
        for _ in range(8):
            self.assertIsNone(locator._estimate_contour(gray)[0])

    def test_part_resting_beyond_bounded_angles_is_found_after_failures(self):
        background, reference, mask = fixture()
        locator = WorkpieceLocator(capture_workpiece(reference, POINTS), adaptive_budget=True)
        affine = cv2.getRotationMatrix2D((320, 220), 120, 1.)
        affine[:, 2] += (10, 20)
        gray = moved_part(background, mask, affine)
        outputs = [locator._estimate_contour(gray)[0] for _ in range(7)]
        # The two bounded groups cannot see it; one of the three full-circle
        # groups, interleaved after two failures, must find it unambiguously.
        self.assertIsNone(outputs[0])
        self.assertIsNone(outputs[1])
        found = [value for value in outputs[2:] if value is not None]
        self.assertTrue(found)
        np.testing.assert_allclose(found[0], normalized(affine), atol=.015)
        # Thereafter the accepted angle is searched on every call.
        self.assertIsNotNone(locator._estimate_contour(gray)[0])

    def test_flow_angle_is_searched_beyond_bounded_angles(self):
        background, reference, mask = fixture()
        locator = WorkpieceLocator(capture_workpiece(reference, POINTS), adaptive_budget=True)
        affine = cv2.getRotationMatrix2D((320, 220), 60, 1.)
        gray = moved_part(background, mask, affine)
        self.assertIsNone(locator._estimate_contour(gray)[0])
        # Flow only proposes its current angle; the pose is matched afresh.
        locator._flow_matrix = normalized(affine) + [[0., 0., .1], [0., 0., -.1]]
        actual = locator._estimate_contour(gray)[0]
        self.assertIsNotNone(actual)
        np.testing.assert_allclose(actual, normalized(affine), atol=.015)

    def test_partly_out_of_frame_part_is_found_from_visible_contour(self):
        background, reference, mask = fixture()
        locator = WorkpieceLocator(capture_workpiece(reference, POINTS), adaptive_budget=True)
        affine = np.float64([[1, 0, 0], [0, 1, -120]])   # stem top leaves the frame
        actual = locator._estimate_contour(moved_part(background, mask, affine))[0]
        self.assertIsNotNone(actual)
        np.testing.assert_allclose(actual, normalized(affine), atol=.015)
        self.assertLess(locator.metrics['diagnostics']['template']['visible'], 1.)

    def test_mostly_out_of_frame_part_is_not_accepted(self):
        background, reference, mask = fixture()
        locator = WorkpieceLocator(capture_workpiece(reference, POINTS), adaptive_budget=True)
        gray = moved_part(background, mask, np.float64([[1, 0, 0], [0, 1, -235]]))
        for _ in range(3):
            self.assertIsNone(locator._estimate_contour(gray)[0])

    def test_contour_template_cache_is_bounded(self):
        _, reference, _ = fixture()
        locator = WorkpieceLocator(capture_workpiece(reference, POINTS), adaptive_budget=True)
        cache = locator._contour_reference((360, 480))
        for angle in range(-180, 180):
            locator._contour_entry(cache, angle, 1.)
        self.assertLessEqual(len(cache['entries']), WorkpieceLocator.CONTOUR_CACHE_LIMIT)
        self.assertIs(locator._contour_entry(cache, 179, 1.), cache['entries'][(179., 1.)])


if __name__ == '__main__':
    unittest.main()
