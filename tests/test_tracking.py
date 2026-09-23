"""定位、遺失保護、SOP 保存與編輯工作流程的回歸測試。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from sop_app.tracking import WorkpieceLocator, capture_workpiece, transform_roi
from sop_app.sop_schema import ROI, Condition, Step, SOPDefinition, load_sop, save_sop
from sop_app.detection import Detection, FrameResult
from sop_app.conditions import evaluate_condition
from sop_app.engine import SOPEngine, Phase
from sop_app.pipeline import FramePacket, VideoPipeline
from sop_app.recorder import Recorder, query
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.run_page import RunPage
from sop_app.ui.records_page import RecordsPage


def scene():
    frame = np.zeros((480, 640, 3), np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(400):
        x, y = rng.integers([180, 140], [430, 330])
        cv2.circle(frame, (int(x), int(y)), int(rng.integers(2, 6)),
                   tuple(int(v) for v in rng.integers(70, 255, 3)), -1)
    return frame


TARGET = [(0.25, 0.25), (0.7, 0.25), (0.7, 0.72), (0.25, 0.72)]
IDENTITY = np.array([[1., 0., 0.], [0., 1., 0.]])


class TrackingTests(unittest.TestCase):
    def test_translation_rotation_scale_and_loss_recovery(self):
        reference = scene()
        locator = WorkpieceLocator(capture_workpiece(reference, TARGET))
        affine = cv2.getRotationMatrix2D((320, 240), 18, 0.85)
        affine[:, 2] += [55, 25]
        moved = cv2.warpAffine(reference, affine, (640, 480))
        self.assertIsNone(locator.locate(moved)[0])
        self.assertIsNone(locator.locate(moved)[0])
        actual, _ = locator.locate(moved)
        self.assertIsNotNone(actual)
        expected = np.diag([1/640, 1/480]) @ affine @ np.diag([640, 480, 1])
        np.testing.assert_allclose(actual, expected, atol=0.01)
        self.assertIsNone(locator.locate(np.zeros_like(reference))[0])
        self.assertIsNone(locator.locate(moved)[0])
        locator.locate(moved)
        self.assertIsNotNone(locator.locate(moved)[0])

    def test_low_texture_rejected(self):
        with self.assertRaises(ValueError):
            capture_workpiece(np.zeros((480, 640, 3), np.uint8), TARGET)

    def test_condition_follows_region_and_never_disappears_when_lost(self):
        roi = ROI("作業點", [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)], "workpiece")
        matrix = IDENTITY.copy()
        matrix[0, 2] = 0.4
        frame = FrameResult(100, 100, 0, [Detection("tool", 0.9, (55, 15, 65, 25))], matrix)
        self.assertTrue(evaluate_condition(Condition(label="tool", roi=roi.name), frame, {roi.name: roi}).met)
        frame.workpiece_transform = None
        frame.detections = []
        result = evaluate_condition(Condition("disappear", "tool", roi.name), frame, {roi.name: roi})
        self.assertFalse(result.met)
        self.assertTrue(result.error)
        matrix[0, 2] = 0.8
        self.assertIsNone(transform_roi(roi, matrix))

    def test_loss_clears_hold_window_and_logs_once(self):
        roi = ROI("作業點", [(0, 0), (1, 0), (1, 1), (0, 1)], "workpiece")
        sop = SOPDefinition(rois=[roi], steps=[Step("移除工具", conditions=[Condition("disappear", "tool", roi.name)])])
        engine = SOPEngine(sop)
        events = engine.start(0)
        for t in (0, 0.5, 0.9):
            events += engine.update(FrameResult(100, 100, t, [], IDENTITY))
        for t in (1, 2, 3):
            events += engine.update(FrameResult(100, 100, t))
        self.assertEqual(sum(e.kind == "position_lost" for e in events), 1)
        events += engine.update(FrameResult(100, 100, 3.1, [], IDENTITY))
        self.assertEqual(engine.phase, Phase.RUNNING)
        events += engine.update(FrameResult(100, 100, 4.2, [], IDENTITY))
        self.assertEqual(engine.phase, Phase.CYCLE_DONE)
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(Path(folder))
            recorder.handle(events, "測試", 1, "model")
            recorder.close()
            self.assertEqual(len(query(Path(folder) / "records.db", "SELECT * FROM positioning_events")), 2)

    def test_schema_roundtrip_and_legacy(self):
        sop = SOPDefinition(workpiece=capture_workpiece(scene(), TARGET),
                            rois=[ROI("點", TARGET, "workpiece")], steps=[Step("人工")])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sop.json"
            save_sop(sop, path)
            restored = load_sop(path)
            self.assertEqual(sop, restored)
            self.assertEqual(restored.validate()[0], [])
        legacy = SOPDefinition.from_dict({"rois": [{"name": "old", "points": TARGET}]})
        self.assertEqual(legacy.rois[0].anchor, "fixed")
        self.assertIsNone(legacy.workpiece)

    def test_model_failure_cannot_satisfy_disappear(self):
        result = evaluate_condition(Condition("disappear", "tool"),
                                    FrameResult(100, 100, 1, detection_valid=False), {})
        self.assertFalse(result.met)

    def test_reset_and_forbidden_do_not_treat_loss_as_removal(self):
        roi = ROI("作業點", TARGET, "workpiece")
        disappear = Condition("disappear", "tool", roi.name)
        sop = SOPDefinition(rois=[roi], steps=[Step("人工", forbidden=[disappear], hold_sec=0)])
        sop.cycle.reset_conditions = [disappear]
        sop.cycle.reset_hold_sec = 0
        engine = SOPEngine(sop)
        engine.start(0)
        events = engine.update(FrameResult(640, 480, 2))
        self.assertFalse(any(e.kind == "alarm" for e in events))
        engine.confirm_step(3)
        engine.update(FrameResult(640, 480, 10))
        self.assertEqual(engine.cycle, 1)
        engine.update(FrameResult(640, 480, 11, [], IDENTITY))
        self.assertEqual(engine.cycle, 2)


class EditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_capture_bind_preview_and_reload(self):
        editor = EditorPage()
        editor.set_classes(["tool"])
        editor.set_sop(SOPDefinition(steps=[Step("壓合")]))
        frame = scene()
        packet = FramePacket(frame, frame.copy(), FrameResult(640, 480, 0), None, 10, 5)
        editor.on_packet(packet)
        editor._begin_target()
        editor._on_roi_drawn(TARGET)
        self.assertTrue(editor.reference_button.isChecked())
        with patch("sop_app.ui.editor_page.QInputDialog.getText", return_value=("壓合位置", True)):
            editor._on_roi_drawn([(0.4, 0.4), (0.5, 0.4), (0.5, 0.5), (0.4, 0.5)])
        self.assertEqual(editor.sop.rois[0].anchor, "workpiece")
        editor.sop.steps[0].conditions = [Condition(label="tool", roi="壓合位置")]
        editor.reference_button.setChecked(False)
        for _ in range(3):
            editor.on_packet(packet)
        self.assertIsNotNone(editor._preview_result.workpiece_transform)
        saved = SOPDefinition.from_dict(editor.sop.to_dict())
        editor.set_sop(saved)
        self.assertTrue(editor.reference_button.isEnabled())
        run = RunPage()
        run.set_sop(saved)
        run.set_running(True)
        engine = SOPEngine(saved)
        engine.start(0)
        engine.update(packet.result)
        packet.snapshot = engine.snapshot()
        run.on_packet(packet)
        self.assertEqual(run.video._rois, [])
        editor.close()
        run.close()

    def test_pipeline_locates_and_records_current_region(self):
        from types import SimpleNamespace
        frame = scene()
        sop = SOPDefinition(workpiece=capture_workpiece(frame, TARGET), rois=[ROI("點", TARGET, "workpiece")],
                            steps=[Step("放置", conditions=[Condition(label="tool", roi="點")], hold_sec=0)])
        detector = SimpleNamespace(classes=["tool"], info=SimpleNamespace(model_version_id="test"),
                                   detect=lambda image: [Detection("tool", 0.9, (250, 200, 280, 230))])
        with tempfile.TemporaryDirectory() as folder:
            recorder = Recorder(Path(folder))
            pipe = VideoPipeline(Path(folder))
            pipe._source = None
            pipe._engine = None
            pipe._locator = None
            pipe._detector = detector
            pipe._now = 0
            pipe._last_display = None
            pipe._show_masks, pipe._min_score = False, 0.5
            pipe._fps, pipe._last_frame_at = 0, None
            packets = []
            pipe.packet_ready.connect(packets.append)
            pipe._handle_command("start", sop, recorder)
            for t in (0, 0.1, 0.2):
                pipe.acknowledge_packet()
                pipe._process(frame, t, recorder)
            self.assertEqual(pipe._engine.phase, Phase.CYCLE_DONE)
            self.assertIsNotNone(packets[-1].result.workpiece_transform)
            self.assertTrue(list((Path(folder) / "snapshots").glob("*.jpg")))
            page = RecordsPage(Path(folder))
            page.refresh()
            page.cycles.selectRow(0)
            self.assertEqual(page.positioning.rowCount(), 2)
            page.close()
            pipe._handle_command("stop", None, recorder)
            self.assertIsNone(pipe._locator)
            recorder.close()
