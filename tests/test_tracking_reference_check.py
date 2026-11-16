"""輪廓比對前處理、競爭位置歧義檢查與參考自我驗證的回歸測試。"""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sop_app.tracking import WorkpieceLocator, capture_workpiece

WIDE_TARGET = [(0.35, 0.3), (0.6, 0.3), (0.6, 0.56), (0.35, 0.56)]
IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


def wide_scene():
    """1280×720 淺色桌面上一塊帶孔位與槽孔的工件；比對時一定會縮小到 480／640 寬。"""
    frame = np.full((720, 1280, 3), 190, np.uint8)
    cv2.line(frame, (0, 90), (1279, 90), (120, 120, 120), 3)          # 桌緣
    cv2.rectangle(frame, (460, 225), (765, 400), (215, 215, 215), -1)
    rng = np.random.default_rng(7)
    for _ in range(9):
        x, y = rng.integers([480, 240], [720, 360])
        w, h = rng.integers(14, 40, 2)
        shade = tuple(int(v) for v in rng.integers(70, 120, 3))
        if rng.random() < .5:
            cv2.circle(frame, (int(x), int(y)), int(min(w, h) // 2), shade, -1)
        else:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h // 2)), shade, -1)
    return frame


def tube_scene():
    """白色直管橫過畫面、上方有平行桌緣：框一小段管身時沒有任何不重複的結構。"""
    frame = np.full((720, 1280, 3), 185, np.uint8)
    cv2.line(frame, (0, 120), (1279, 120), (120, 120, 120), 3)
    cv2.rectangle(frame, (150, 300), (1130, 330), (240, 240, 240), -1)
    cv2.line(frame, (150, 300), (1130, 300), (150, 150, 150), 2)
    cv2.line(frame, (150, 330), (1130, 330), (150, 150, 150), 2)
    return frame


def moved(frame, angle=0., shift=(0., 0.)):
    centre = np.mean(np.asarray(WIDE_TARGET) * [frame.shape[1], frame.shape[0]], axis=0)
    affine = cv2.getRotationMatrix2D(tuple(centre), angle, 1.)
    affine[:, 2] += shift
    image = cv2.warpAffine(frame, affine, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    h, w = frame.shape[:2]
    return image, np.diag([1 / w, 1 / h]) @ affine @ np.diag([w, h, 1])


def corner_error(locator, actual, expected, size=(1280, 720)):
    corners = locator.corners
    return float(np.max(np.linalg.norm(((corners @ actual[:, :2].T + actual[:, 2])
                                        - (corners @ expected[:, :2].T + expected[:, 2])) * size, axis=1)))


class TemplatePreprocessingTests(unittest.TestCase):
    def test_identical_reference_is_found_at_its_own_position(self):
        # 舊作法先取邊緣再縮小，正確位置只得約 0.1 分，卻在別處找到高分並接受。
        reference = wide_scene()
        for adaptive in (False, True):
            locator = WorkpieceLocator(capture_workpiece(reference, WIDE_TARGET), adaptive_budget=adaptive)
            actual, count = locator._estimate_template(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
            self.assertIsNotNone(actual, locator.metrics['diagnostics'])
            self.assertLess(corner_error(locator, actual, IDENTITY), 1.)
            self.assertGreaterEqual(-count, 90)

    def test_subpixel_shift_and_small_rotation_follow_the_part(self):
        reference = wide_scene()
        locator = WorkpieceLocator(capture_workpiece(reference, WIDE_TARGET))
        for angle, shift in ((0, (25., 15.)), (0, (1., 1.)), (6, (10., 5.)), (-12, (-7., 3.))):
            image, expected = moved(reference, angle, shift)
            locator.reset()
            actual, _ = locator._estimate_template(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            self.assertIsNotNone(actual, (angle, shift, locator.metrics['diagnostics']))
            self.assertLess(corner_error(locator, actual, expected), 3., (angle, shift))

    def test_adaptive_rotation_groups_always_compare_zero_degrees(self):
        # 只比旋轉候選時，錯位置可能勝過沒有被比到的 0° 正確解，並蓋掉正確的光流。
        reference = wide_scene()
        locator = WorkpieceLocator(capture_workpiece(reference, WIDE_TARGET), adaptive_budget=True)
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        with patch.object(locator, '_template_candidate', wraps=locator._template_candidate) as candidate:
            for _ in range(3):
                candidate.reset_mock()
                locator._estimate_template(gray)
                self.assertIn(0, {call.args[1] for call in candidate.call_args_list})


class TemplateAmbiguityTests(unittest.TestCase):
    def test_two_equally_good_positions_fail_closed(self):
        reference = wide_scene()
        locator = WorkpieceLocator(capture_workpiece(reference, WIDE_TARGET))
        live = reference.copy()
        live[230:410, 790:1110] = reference[230:410, 450:770]   # 旁邊放了一件外觀相同的工件
        actual, _ = locator._estimate_template(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY))
        self.assertIsNone(actual)
        diagnosis = locator.metrics['diagnostics']['template']
        self.assertEqual(diagnosis['reason'], 'ambiguous')
        self.assertLess(diagnosis['best_score'] - diagnosis['competitor_score'], locator.TEMPLATE_MIN_MARGIN)

    def test_accepted_match_reports_competing_score(self):
        reference = wide_scene()
        locator = WorkpieceLocator(capture_workpiece(reference, WIDE_TARGET))
        image, _ = moved(reference, shift=(18., 9.))
        self.assertIsNotNone(locator._estimate_template(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[0])
        diagnosis = locator.metrics['diagnostics']['template']
        self.assertEqual(diagnosis['reason'], 'accepted')
        self.assertGreaterEqual(diagnosis['best_score'] - diagnosis['competitor_score'], locator.TEMPLATE_MIN_MARGIN)


class ReferenceSelfCheckTests(unittest.TestCase):
    def test_reference_without_unique_structure_is_rejected_with_reasons(self):
        # 只框到一段光滑直管：沿管身滑動都一樣像，不能啟用，並說明原因。
        target = [(500 / 1280, 290 / 720), (650 / 1280, 290 / 720), (650 / 1280, 340 / 720), (500 / 1280, 340 / 720)]
        with self.assertRaises(ValueError) as raised:
            capture_workpiece(tube_scene(), target)
        message = str(raised.exception)
        self.assertIn('找不回原位', message)
        self.assertIn('輪廓：畫面中有相似位置', message)

    def test_distinctive_reference_passes_and_keeps_all_methods(self):
        locator = WorkpieceLocator(capture_workpiece(wide_scene(), WIDE_TARGET))
        self.assertEqual(locator.reference_check['template'], 'ok')
        self.assertEqual(locator._disabled_stages, set())

    def test_method_that_cannot_find_itself_is_never_used(self):
        reference = wide_scene()
        wrong = IDENTITY.copy()
        wrong[0, 2] = .2
        with patch.object(WorkpieceLocator, '_estimate_template', return_value=(wrong, -90)):
            locator = WorkpieceLocator(capture_workpiece(reference, WIDE_TARGET))
        self.assertEqual(locator.reference_check['template'], 'wrong_position')
        self.assertIn('template', locator._disabled_stages)
        with patch.object(locator, '_estimate_features', return_value=(None, 0)), \
             patch.object(locator, '_estimate_sift', return_value=(None, 0)), \
             patch.object(locator, '_estimate_template', return_value=(wrong, -90)) as template:
            for _ in range(4):
                self.assertIsNone(locator.locate(reference)[0])
        template.assert_not_called()

    def test_method_that_cannot_find_itself_is_skipped_in_adaptive_mode(self):
        reference = wide_scene()
        wrong = IDENTITY.copy()
        wrong[1, 2] = .2
        with patch.object(WorkpieceLocator, '_estimate_template', return_value=(wrong, -90)):
            locator = WorkpieceLocator(capture_workpiece(reference, WIDE_TARGET), adaptive_budget=True)
        with patch.object(locator, '_estimate_features', return_value=(None, 0)), \
             patch.object(locator, '_estimate_sift', return_value=(None, 0)), \
             patch.object(locator, '_estimate_template', return_value=(wrong, -90)) as template, \
             patch.object(locator.budget, 'allow', return_value=True):
            for _ in range(4):
                self.assertIsNone(locator.locate(reference)[0])
        template.assert_not_called()

    def test_identical_frame_replay_never_drifts(self):
        # 同一張參考連續送入：位置理應完全不變，包含每隔數幀以輪廓重新核對（輪流比對各組角度）。
        reference = wide_scene()
        locator = WorkpieceLocator(capture_workpiece(reference, WIDE_TARGET), adaptive_budget=True)
        valid = 0
        with patch.object(locator, '_estimate_features', return_value=(None, 0)), \
             patch.object(locator, '_estimate_sift', return_value=(None, 0)), \
             patch.object(locator.budget, 'allow', return_value=True):
            for index in range(45):
                actual, _ = locator.locate(reference, timestamp=index / 30)
                if actual is not None:   # 每幀只做一種昂貴搜尋，輪到尺度特徵的幀可能暫時無結果
                    valid += 1
                    self.assertLess(corner_error(locator, actual, IDENTITY), 1., index)
        self.assertGreaterEqual(valid, 40)


if __name__ == '__main__':
    unittest.main()
