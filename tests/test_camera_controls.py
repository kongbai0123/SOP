import unittest
import threading
from collections import deque
from unittest.mock import Mock

import cv2

from sop_app.camera_controls import apply_controls, read_controls
from sop_app.pipeline import CameraSource, VideoPipeline
from sop_app.engine import Phase


class CameraControlsTests(unittest.TestCase):
    def capture(self):
        capture = Mock()
        capture.get.side_effect = lambda prop: cv2.CAP_DSHOW if prop == cv2.CAP_PROP_BACKEND else -6
        capture.set.return_value = True
        return capture

    def test_read_keeps_valid_negative_exposure_without_writes(self):
        capture = self.capture()
        self.assertEqual(read_controls(capture)["exposure"], -6)
        capture.set.assert_not_called()

    def test_manual_exposure_disables_auto_before_writing(self):
        capture = self.capture()
        messages = apply_controls(capture, {"exposure": ("manual", -7)})
        self.assertEqual([call.args for call in capture.set.call_args_list],
                         [(cv2.CAP_PROP_AUTO_EXPOSURE, 0), (cv2.CAP_PROP_EXPOSURE, -7)])
        self.assertIn("要求 -7，驅動回報 -6", messages[0])

    def test_auto_does_not_write_manual_value(self):
        capture = self.capture()
        apply_controls(capture, {"focus": ("auto", 42)})
        capture.set.assert_called_once_with(cv2.CAP_PROP_AUTOFOCUS, 1)

    def test_failed_mode_does_not_write_manual_value_and_other_rows_continue(self):
        capture = self.capture()
        capture.set.side_effect = [False, True]
        messages = apply_controls(capture, {"exposure": ("manual", -7), "brightness": ("manual", 10)})
        self.assertIn("未完成", messages[0])
        self.assertEqual([call.args for call in capture.set.call_args_list],
                         [(cv2.CAP_PROP_AUTO_EXPOSURE, 0), (cv2.CAP_PROP_BRIGHTNESS, 10)])

    def test_unsupported_property_is_reported(self):
        capture = self.capture()
        capture.set.return_value = False
        self.assertIn("未完成", apply_controls(capture, {"gain": ("manual", 10)})[0])

    def test_other_backend_does_not_receive_dshow_mode_values(self):
        capture = self.capture()
        capture.get.side_effect = None
        capture.get.return_value = cv2.CAP_MSMF
        self.assertIn("後端", apply_controls(capture, {"exposure": ("auto", 0)})[0])
        capture.set.assert_not_called()

    def test_read_failure_does_not_abort_remaining_controls(self):
        capture = self.capture()
        def read(prop):
            if prop == cv2.CAP_PROP_BRIGHTNESS:
                raise cv2.error("unsupported")
            return 0
        capture.get.side_effect = read
        values = read_controls(capture)
        self.assertIsNone(values["brightness"])
        self.assertEqual(values["focus"], 0)

    def test_controls_command_never_confirms_engine_step(self):
        pipeline = VideoPipeline(".")
        pipeline._source = Mock(spec=CameraSource)
        pipeline._source.controls.return_value = ({"exposure": -6}, ["ok"])
        pipeline._engine = Mock()
        received = []
        pipeline.camera_controls_ready.connect(lambda values, messages: received.append((values, messages)))
        pipeline._handle_command("camera_controls", {}, Mock())
        self.assertEqual(received, [({"exposure": -6}, ["ok"])])
        pipeline._engine.confirm_step.assert_not_called()

    def test_resolution_is_applied_before_fps_and_reads_actual_size(self):
        capture = self.capture()
        capture.get.side_effect = lambda prop: {cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720, cv2.CAP_PROP_FPS: 30}.get(prop, 0)
        messages = apply_controls(capture, {"fps": ("manual", 60),
                                            "resolution": ("manual", (1920, 1080))})
        self.assertEqual([call.args for call in capture.set.call_args_list],
                         [(cv2.CAP_PROP_FRAME_WIDTH, 1920), (cv2.CAP_PROP_FRAME_HEIGHT, 1080),
                          (cv2.CAP_PROP_FPS, 60)])
        self.assertIn("驅動回報 1280 × 720", messages[0])
        self.assertIn("驅動回報 30", messages[1])

    def test_resolution_failure_still_sends_height_and_reads_actual(self):
        capture = self.capture()
        capture.set.side_effect = [False, True]
        messages = apply_controls(capture, {"resolution": ("manual", (1920, 1080))})
        self.assertEqual(capture.set.call_count, 2)
        self.assertIn("未完成", messages[0])

    def test_invalid_format_does_not_write_device(self):
        capture = self.capture()
        messages = apply_controls(capture, {"resolution": ("manual", (0, 720)), "fps": ("manual", 0)})
        capture.set.assert_not_called()
        self.assertTrue(all("未完成" in message for message in messages))

    def test_running_engine_blocks_format_changes_but_allows_brightness(self):
        pipeline = VideoPipeline(".")
        pipeline._source = Mock(spec=CameraSource)
        pipeline._source.controls.return_value = ({}, [])
        pipeline._engine = Mock(phase=Phase.RUNNING)
        received = []
        pipeline.camera_controls_ready.connect(lambda values, messages: received.extend(messages))
        pipeline._handle_command("camera_controls", {"resolution": ("manual", (640, 480)),
            "fps": ("manual", 60), "brightness": ("manual", 20)}, Mock())
        pipeline._source.controls.assert_called_once_with({"brightness": ("manual", 20)})
        self.assertIn("請先停止作業", received[0])
        pipeline._engine.confirm_step.assert_not_called()

    def test_capture_fps_measures_arrivals_and_format_change_clears_old_frames(self):
        source = CameraSource.__new__(CameraSource)
        source._lock = threading.Lock()
        source._capture_lock = threading.Lock()
        source._frame_times = deque([1.0, 1.05, 1.1])
        source._latest = (object(), 1.1)
        source.capture = self.capture()
        self.assertAlmostEqual(source.capture_fps, 20)
        source.controls({"fps": ("manual", 30)})
        self.assertIsNone(source.capture_fps)
        self.assertIsNone(source.read())


if __name__ == "__main__":
    unittest.main()
