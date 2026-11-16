"""定位狀態：已確認、位置待確認、重新定位中；判定只用已確認的位置。"""
from contextlib import nullcontext
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from sop_app.conditions import evaluate_condition
from sop_app.detection import (TRACKING_CONFIRMED, TRACKING_RELOCATING, TRACKING_UNCERTAIN, Detection,
                               FrameResult)
from sop_app.engine import Phase, SOPEngine
from sop_app.pipeline import VideoPipeline
from sop_app.recorder import Recorder
from sop_app.sop_schema import ROI, Condition, SOPDefinition, Step
from sop_app.tracking import WorkpieceLocator, capture_workpiece

TARGET = [(0.25, 0.25), (0.7, 0.25), (0.7, 0.72), (0.25, 0.72)]
IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])
NO_MATCH = (None, 0)


def scene():
    frame = np.zeros((480, 640, 3), np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(400):
        x, y = rng.integers([180, 140], [430, 330])
        cv2.circle(frame, (int(x), int(y)), int(rng.integers(2, 6)),
                   tuple(int(v) for v in rng.integers(70, 255, 3)), -1)
    return frame


def absolute_stages_fail(locator):
    return (patch.object(locator, '_estimate_features', return_value=NO_MATCH),
            patch.object(locator, '_estimate_sift', return_value=NO_MATCH),
            patch.object(locator, '_estimate_template', return_value=NO_MATCH))


class LocatorStateTests(unittest.TestCase):
    def follow_until_recheck(self, locator, reference):
        matrix, status = locator.locate(reference)
        self.assertEqual(locator.state, TRACKING_CONFIRMED)
        self.assertIn('位置已確認（特徵', status)
        for _ in range(locator.MAX_RECHECK_FRAMES - 1):
            matrix, status = locator.locate(reference)
            self.assertIsNotNone(matrix)
            self.assertEqual(locator.state, TRACKING_CONFIRMED)
        self.assertEqual(locator.metrics['source'], 'flow')
        self.assertIn('光流', status)
        self.assertNotIn('特徵', status)   # 光流點數不是與參考比對成功的特徵

    def check_failed_recheck_is_uncertain(self, adaptive):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET), adaptive_budget=adaptive)
        self.follow_until_recheck(locator, reference)
        failing = absolute_stages_fail(locator)
        budget = patch.object(locator.budget, 'allow', return_value=True) if adaptive else nullcontext()
        with failing[0], failing[1], failing[2], budget:
            matrix, status = locator.locate(reference)
            self.assertIsNone(matrix)
            self.assertEqual(locator.state, TRACKING_UNCERTAIN)
            self.assertIn('位置待確認', status)
            self.assertNotIn('穩定', status)
            self.assertFalse(locator.metrics['valid'])
            # 待確認期間每幀重新核對，仍不通過就維持待確認；光流沒有中斷，也不算遺失。
            for _ in range(locator.MIN_RECHECK_FRAMES - 1):
                self.assertIsNone(locator.locate(reference)[0])
                self.assertEqual(locator.state, TRACKING_UNCERTAIN)
            self.assertEqual(locator.misses, 0)
            self.assertIsNotNone(locator._flow_gray)
        # 下一次重新核對通過，立刻恢復已確認。
        matrix, status = locator.locate(reference)
        self.assertIsNotNone(matrix)
        self.assertEqual(locator.state, TRACKING_CONFIRMED)
        self.assertNotEqual(locator.metrics['source'], 'flow')
        np.testing.assert_allclose(matrix, IDENTITY, atol=.01)

    def test_failed_recheck_is_uncertain_until_verified_again(self):
        self.check_failed_recheck_is_uncertain(adaptive=False)

    def test_failed_recheck_is_uncertain_in_adaptive_mode(self):
        self.check_failed_recheck_is_uncertain(adaptive=True)

    def test_lost_position_is_relocating(self):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
        self.assertEqual(locator.state, TRACKING_RELOCATING)
        self.assertIsNotNone(locator.locate(reference)[0])
        blank = np.zeros_like(reference)
        statuses = [locator.locate(blank)[1] for _ in range(3)]
        self.assertEqual(locator.state, TRACKING_RELOCATING)
        self.assertEqual(statuses[:2], ['重新定位中（1/3）', '重新定位中（2/3）'])
        self.assertEqual(statuses[2], '位置跟隨已中斷，正在重新定位')
        self.assertIsNotNone(locator.locate(reference)[0])
        self.assertEqual(locator.state, TRACKING_CONFIRMED)


class ConfirmedPositionOnlyTests(unittest.TestCase):
    roi = ROI('作業點', [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)], 'workpiece')

    def frame(self, state, timestamp=0.):
        matrix = IDENTITY.copy()
        matrix[0, 2] = 0.4
        return FrameResult(100, 100, timestamp, [Detection('tool', 0.9, (55, 15, 65, 25))], matrix,
                           tracking_state=state)

    def test_only_confirmed_or_external_transform_reaches_conditions(self):
        for state, usable in (('', True), (TRACKING_CONFIRMED, True),
                              (TRACKING_UNCERTAIN, False), (TRACKING_RELOCATING, False)):
            frame = self.frame(state)
            self.assertEqual(frame.confirmed_transform is not None, usable, state)
            result = evaluate_condition(Condition(label='tool', roi=self.roi.name), frame, {self.roi.name: self.roi})
            self.assertEqual(result.met, usable, state)
            self.assertEqual(result.error_kind, '' if usable else 'unavailable', state)

    def test_engine_does_not_complete_on_uncertain_position(self):
        sop = SOPDefinition(rois=[self.roi], steps=[Step('放置', conditions=[Condition(label='tool', roi='作業點')],
                                                         hold_sec=.2)])
        engine = SOPEngine(sop)
        engine.start(0.)
        for index in range(20):
            engine.update(self.frame(TRACKING_UNCERTAIN, index * .05))
        self.assertEqual(engine.phase, Phase.RUNNING)
        self.assertEqual(engine.snapshot().progress, 0.)
        for index in range(20, 30):
            engine.update(self.frame(TRACKING_CONFIRMED, index * .05))
        self.assertEqual(engine.phase, Phase.CYCLE_DONE)


class PipelineStateTests(unittest.TestCase):
    def test_pipeline_passes_locator_state_into_frame_result(self):
        frame = scene()
        sop = SOPDefinition(workpiece=capture_workpiece(frame, TARGET), rois=[ROI('點', TARGET, 'workpiece')],
                            steps=[Step('放置', conditions=[Condition(label='tool', roi='點')])])
        detector = SimpleNamespace(classes=['tool'], info=SimpleNamespace(model_version_id='test'),
                                   detect=lambda image: [])
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(Path(folder))
            pipe = VideoPipeline(Path(folder))
            pipe._source = pipe._engine = pipe._locator = None
            pipe._detector = detector
            pipe._now = 0
            pipe._last_display = None
            pipe._show_masks, pipe._min_score = False, 0.5
            pipe._fps, pipe._last_frame_at = 0, None
            packets = []
            pipe.packet_ready.connect(packets.append)
            pipe._handle_command('start', sop, recorder)
            for timestamp, image in ((0., frame), (.1, np.zeros_like(frame))):
                pipe.acknowledge_packet()
                pipe._process(image, timestamp, recorder)
            self.assertEqual([p.result.tracking_state for p in packets], [TRACKING_CONFIRMED, TRACKING_RELOCATING])
            self.assertEqual(packets[0].result.tracking_metrics['state'], TRACKING_CONFIRMED)
            pipe._handle_command('stop', None, recorder)
            recorder.close()


if __name__ == '__main__':
    unittest.main()
