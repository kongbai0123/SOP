import unittest
import threading
from collections import deque
from unittest.mock import Mock

import cv2

from sop_app.camera_controls import (accepted_camera_preferences, apply_controls, read_controls,
                                     saved_camera_changes)
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

    def test_fourcc_and_auto_modes_are_separate_from_sampled_control_values(self):
        capture = self.capture()
        capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_BACKEND: cv2.CAP_DSHOW,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
            cv2.CAP_PROP_AUTOFOCUS: 2, cv2.CAP_PROP_AUTO_WB: 1,
            cv2.CAP_PROP_AUTO_EXPOSURE: -1,
            cv2.CAP_PROP_EXPOSURE: -7, cv2.CAP_PROP_FOCUS: 25,
        }.get(prop, 0)
        values = read_controls(capture)
        self.assertEqual(values["pixel_format"], "MJPG")
        self.assertEqual(values["control_modes"],
                         {"exposure": None, "wb_temperature": "auto", "focus": "manual"})
        self.assertEqual(values["exposure"], -7)
        self.assertEqual(values["focus"], 25)
        capture.set.assert_not_called()

    def test_final_format_readback_happens_after_fps_and_fourcc_negotiation(self):
        capture = Mock()
        state = {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
                 cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"YUY2")}
        operations = []
        def write(prop, value):
            operations.append(("set", prop, value))
            state[prop] = value
            if prop == cv2.CAP_PROP_FPS:
                state[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*"YUY2")
            elif prop == cv2.CAP_PROP_FOURCC:
                state.update({cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720,
                              cv2.CAP_PROP_FPS: 59.94})
            return True
        def read(prop):
            operations.append(("get", prop))
            return state.get(prop, -1)
        capture.set.side_effect, capture.get.side_effect = write, read
        changes = {"pixel_format": ("manual", "MJPG"), "fps": ("manual", 60),
                   "resolution": ("manual", (1920, 1080))}
        messages = apply_controls(capture, changes)
        writes = [call for call in operations if call[0] == "set"]
        self.assertEqual([item[1] for item in writes], [cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC,
            cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC])
        self.assertIn("驅動回報 1280 × 720", messages[0])
        self.assertIn("驅動回報 MJPG", messages[1])
        self.assertIn("驅動回報 59.94", messages[2])
        profile = accepted_camera_preferences({}, changes, read_controls(capture), messages)
        self.assertEqual(profile, {"resolution": [1280, 720], "fps": 59.94, "pixel_format": "MJPG"})

    def test_true_setter_does_not_claim_requested_fps_and_rejected_items_keep_preferences(self):
        capture = self.capture()
        capture.get.side_effect = lambda prop: {cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"YUY2")}.get(prop, -1)
        capture.set.side_effect = lambda prop, value: prop != cv2.CAP_PROP_FOURCC
        changes = {"fps": ("manual", 60), "pixel_format": ("manual", "MJPG")}
        messages = apply_controls(capture, changes)
        self.assertIn("要求 60，驅動回報 30", messages[1])
        previous = {"fps": 24, "pixel_format": "YUY2", "controls": {"gain": {"mode": "manual", "value": 7}}}
        profile = accepted_camera_preferences(previous, changes, read_controls(capture), messages)
        self.assertEqual(profile["fps"], 30)
        self.assertEqual(profile["pixel_format"], "YUY2")
        self.assertEqual(profile["controls"], previous["controls"])
        self.assertEqual(previous["fps"], 24)  # merge never mutates prior station settings

    def test_auto_persistence_stores_mode_without_variable_sample_value(self):
        capture = self.capture()
        changes = {"exposure": ("auto", -6)}
        messages = apply_controls(capture, changes)
        profile = accepted_camera_preferences({}, changes, read_controls(capture), messages)
        self.assertEqual(profile, {"controls": {"exposure": {"mode": "auto"}}})
        self.assertEqual(saved_camera_changes(profile), {"exposure": ("auto", 0.)})
        self.assertIn("模式無法讀取", messages[0])

    def test_driver_mode_mismatch_is_rejected_and_unknown_fps_not_persisted(self):
        capture = self.capture()
        capture.get.side_effect = lambda prop: {cv2.CAP_PROP_BACKEND: cv2.CAP_DSHOW,
            cv2.CAP_PROP_AUTOFOCUS: 2, cv2.CAP_PROP_FPS: 0}.get(prop, -1)
        changes = {"focus": ("auto", 20), "fps": ("manual", 60)}
        messages = apply_controls(capture, changes)
        self.assertTrue(all("未完成" in message for message in messages))
        previous = {"fps": 25, "controls": {"focus": {"mode": "manual", "value": 20}}}
        self.assertEqual(accepted_camera_preferences(previous, changes, read_controls(capture), messages), previous)

    def test_legacy_preferences_normalize_without_invalid_or_automatic_sample_values(self):
        legacy = {"width": 1280, "height": 720, "fps": 60, "fourcc": cv2.VideoWriter_fourcc(*"MJPG"),
            "controls": {"white_balance": {"auto": True, "value": 4000},
                         "exposure": {"auto": False, "value": -6}, "gain": 12,
                         "unknown": {"mode": "manual", "value": 99}}}
        changes = saved_camera_changes(legacy)
        self.assertEqual(changes, {"resolution": ("manual", (1280, 720)), "fps": ("manual", 60.),
            "pixel_format": ("manual", "MJPG"), "wb_temperature": ("auto", 0.),
            "exposure": ("manual", -6.), "gain": ("manual", 12.)})
        self.assertEqual(saved_camera_changes({"resolution": [True, 480], "fps": float("nan"),
            "pixel_format": "NOT_A_FORMAT", "controls": {"gain": {"value": 10 ** 1000}}}), {})
        self.assertEqual(saved_camera_changes(None), {})

    def test_unknown_or_malformed_controls_report_failure_without_writes(self):
        capture = self.capture()
        messages = apply_controls(capture, {"pixel_format": ("manual", "MJPEG"),
            "mystery": ("manual", 1), "gain": None, "fps": ("auto", 60)})
        self.assertEqual(len(messages), 4)
        self.assertTrue(all("未完成" in message for message in messages))
        capture.set.assert_not_called()

    def mode_capture(self, fps=30, pixel="MJPG"):
        capture = Mock()
        state = {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720,
                 cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*pixel)}
        capture.get.side_effect = lambda prop: state.get(prop, -1)
        return capture, state

    def test_fps_only_preserves_existing_dimensions_and_fourcc_after_graph_reset(self):
        capture, state = self.mode_capture()
        def write(prop, value):
            state[prop] = value
            if prop == cv2.CAP_PROP_FPS:
                state[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*"YUY2")
            return True
        capture.set.side_effect = write
        changes = {"fps": ("manual", 60)}
        messages = apply_controls(capture, changes)
        self.assertEqual([call.args[0] for call in capture.set.call_args_list],
                         [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                          cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC])
        actual = read_controls(capture)
        self.assertEqual((actual["fps"], actual["pixel_format"], actual["resolution"]),
                         (60, "MJPG", (1280, 720)))
        self.assertEqual(len(messages), 1)  # only requested rows, unless preservation fails
        self.assertEqual(accepted_camera_preferences({}, changes, actual, messages),
                         {"fps": 60., "pixel_format": "MJPG", "resolution": [1280, 720]})

    def test_pixel_only_preserves_fps_with_one_bounded_repair(self):
        capture, state = self.mode_capture(fps=60, pixel="YUY2")
        selected = []
        def write(prop, value):
            state[prop] = value
            if prop == cv2.CAP_PROP_FPS:
                state[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*"YUY2")
            elif prop == cv2.CAP_PROP_FOURCC:
                selected.append(value)
                if len(selected) == 1:
                    state[cv2.CAP_PROP_FPS] = 30
            return True
        capture.set.side_effect = write
        messages = apply_controls(capture, {"pixel_format": ("manual", "MJPG")})
        self.assertEqual(len(selected), 2)
        self.assertEqual(capture.set.call_count, 6)
        self.assertEqual(read_controls(capture)["fps"], 60)
        self.assertEqual(len(messages), 1)
        self.assertIn("驅動回報 MJPG", messages[0])

    def test_failed_implicit_format_preservation_is_reported_without_overwriting_preference(self):
        capture, state = self.mode_capture()
        def write(prop, value):
            if prop == cv2.CAP_PROP_FOURCC:
                return False
            state[prop] = value
            if prop == cv2.CAP_PROP_FPS:
                state[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*"YUY2")
            return True
        capture.set.side_effect = write
        changes = {"fps": ("manual", 60)}
        messages = apply_controls(capture, changes)
        self.assertTrue(any(message.startswith("像素格式：未完成") for message in messages))
        previous = {"fps": 30, "pixel_format": "MJPG"}
        profile = accepted_camera_preferences(previous, changes, read_controls(capture), messages)
        self.assertEqual(profile["fps"], 60)
        self.assertEqual(profile["pixel_format"], "MJPG")
        self.assertLessEqual(capture.set.call_count, 6)

    def test_unknown_current_fourcc_is_not_guessed_during_fps_change(self):
        capture, state = self.mode_capture()
        state[cv2.CAP_PROP_FOURCC] = -1
        capture.set.side_effect = lambda prop, value: state.update({prop: value}) or True
        apply_controls(capture, {"fps": ("manual", 60)})
        self.assertNotIn(cv2.CAP_PROP_FOURCC, [call.args[0] for call in capture.set.call_args_list])

    def test_failed_implicit_fps_preservation_does_not_save_lower_fps_as_success(self):
        capture, state = self.mode_capture(fps=60, pixel="YUY2")
        def write(prop, value):
            state[prop] = value
            if prop == cv2.CAP_PROP_FOURCC:
                state[cv2.CAP_PROP_FPS] = 30
            return True
        capture.set.side_effect = write
        changes = {"pixel_format": ("manual", "MJPG")}
        messages = apply_controls(capture, changes)
        self.assertTrue(any(message.startswith("攝影機 FPS：未完成") for message in messages))
        profile = accepted_camera_preferences({"fps": 60}, changes, read_controls(capture), messages)
        self.assertEqual(profile["fps"], 60)
        self.assertEqual(profile["pixel_format"], "MJPG")

    def test_repair_clears_initial_rejection_only_when_final_readback_matches(self):
        for final_matches in (True, False):
            with self.subTest(final_matches=final_matches):
                capture, state = self.mode_capture(fps=30, pixel="YUY2")
                attempts = []
                def write(prop, value):
                    if prop == cv2.CAP_PROP_FOURCC:
                        attempts.append(value)
                        if len(attempts) == 1:
                            return False
                        if final_matches:
                            state[prop] = value
                        return True
                    state[prop] = value
                    return True
                capture.set.side_effect = write
                changes = {"pixel_format": ("manual", "MJPG")}
                messages = apply_controls(capture, changes)
                self.assertEqual(len(attempts), 2)
                profile = accepted_camera_preferences({"pixel_format": "YUY2"}, changes,
                                                       read_controls(capture), messages)
                if final_matches:
                    self.assertNotIn("未完成", messages[0])
                    self.assertIn("驅動回報 MJPG", messages[0])
                    self.assertEqual(profile["pixel_format"], "MJPG")
                else:
                    self.assertIn("未完成", messages[0])
                    self.assertEqual(profile["pixel_format"], "YUY2")


if __name__ == "__main__":
    unittest.main()
