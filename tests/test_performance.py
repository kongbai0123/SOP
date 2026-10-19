"""Observability must not change frame timestamps, event ordering or durability."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PySide6.QtWidgets import QApplication

from sop_app.detection import Detection, FrameResult
from sop_app.engine import SOPEngine
from sop_app.performance import PerformanceWindow, observed_mapping
from sop_app.pipeline import VideoPipeline
from sop_app.sop_schema import Condition, SOPDefinition, Step


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_window_is_bounded_and_does_not_invent_missing_stage_samples(self):
        window = PerformanceWindow(3)
        window.observe({"frame": 100, "event": 100})
        window.observe({"frame": 1, "bad": float("nan")})
        window.observe({"frame": 2, "bad": -1})
        window.observe({"frame": 3, "event": 9})
        summary = window.snapshot()
        self.assertEqual(summary["frames"], 3)
        self.assertEqual(summary["stages_ms"]["frame"]["p50"], 2)
        self.assertEqual(summary["stages_ms"]["event"]["count"], 1)
        self.assertNotIn("bad", summary["stages_ms"])
        summary["stages_ms"]["frame"]["p50"] = 999
        self.assertEqual(window.snapshot()["stages_ms"]["frame"]["p50"], 2)

    def test_optional_observer_failure_cannot_fail_a_business_operation(self):
        class BrokenObserver:
            @property
            def runtime_info(self):
                raise ValueError("unavailable telemetry")
        self.assertEqual(observed_mapping(BrokenObserver(), "runtime_info"), {})
        self.assertEqual(observed_mapping(SimpleNamespace(runtime_info=None), "runtime_info"), {})

    def test_pipeline_timings_keep_all_engine_samples_when_gui_is_busy(self):
        sop = SOPDefinition(steps=[Step("A", conditions=[Condition(label="part")], hold_sec=.2)])
        pipe = VideoPipeline(Path("unused"))
        detections = [Detection("part", .9, (2, 2, 9, 9))]
        pipe._detector = SimpleNamespace(detect=lambda _: detections, classes=("part",),
            info=SimpleNamespace(model_version_id="M"),
            last_metrics={"model_inference_ms": 1.5,
                          "runtime": {"device": "cuda:0", "dtype": "torch.float16"}})
        pipe._engine = SOPEngine(sop)
        expected = SOPEngine(sop)
        pipe._engine.start(0.)
        expected.start(0.)
        recorder = Mock()
        recorder.last_enqueue_metrics = {"encode_ms": 2., "pending_commit_ms": 1.}
        packets, actual, reference = [], [], []
        pipe.packet_ready.connect(packets.append)
        pipe.events_ready.connect(actual.extend)
        image = np.zeros((20, 20, 3), np.uint8)
        for timestamp in (0., .1, .2, .3):
            reference.extend(expected.update(FrameResult(20, 20, timestamp, detections)))
            pipe._process(image, timestamp, recorder)
        self.assertEqual(actual, reference)
        self.assertEqual(pipe.performance_snapshot()["frames"], 4)
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].result.timestamp, 0.)
        self.assertEqual(packets[0].inference_runtime["device"], "cuda:0")
        self.assertIn("engine_ms", packets[0].stage_ms)
        pipe.acknowledge_packet()
        pipe._process(image, .4, recorder)
        self.assertEqual([r.timestamp for r in packets[-1].all_results()], [.1, .2, .3, .4])
        self.assertEqual(pipe.performance_snapshot()["stages_ms"]["record_encode_ms"]["count"], 1)

    def test_failed_durable_acceptance_stops_engine_and_has_no_stale_record_metrics(self):
        pipe = VideoPipeline(Path("unused"))
        pipe._detector = SimpleNamespace(detect=lambda _: [], classes=(),
                                        info=SimpleNamespace(model_version_id="M"))
        pipe._engine = SOPEngine(SOPDefinition(steps=[Step("A", hold_sec=0,
                                                    conditions=[Condition("disappear", "part")])]))
        pipe._engine.start(0.)
        recorder = Mock()
        recorder.handle.side_effect = OSError("disk failed")
        recorder.last_enqueue_metrics = {"encode_ms": 999}
        errors = []
        pipe.error.connect(errors.append)
        pipe._process(np.zeros((20, 20, 3), np.uint8), .1, recorder)
        self.assertTrue(errors)
        self.assertNotIn("record_encode_ms", pipe.performance_snapshot()["stages_ms"])
        self.assertEqual(pipe._engine.phase.value, "idle")


if __name__ == "__main__":
    unittest.main()
