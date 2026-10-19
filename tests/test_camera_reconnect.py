"""Camera reconnect, station persistence and rate-cap integration without hardware."""
from contextlib import ExitStack
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from sop_app.engine import Phase
from sop_app.frame_pacing import InferencePacer
from sop_app.pipeline import CameraSource, VideoPipeline
from sop_app.recorder import connect
from sop_app.stations import StationStore
from sop_app.ui.main_window import MainWindow


class CameraReconnectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def window(self, settings=None, stations=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = StationStore(root) if stations else None
        station = store.initialize() if store else None
        records = station.records if station else root / "records"
        records.mkdir(parents=True, exist_ok=True)
        connect(records / "records.db").close()
        settings_path = station.settings if station else root / "settings.json"
        settings_path.write_text(json.dumps(settings or {}), encoding="utf-8")
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, path in (("RECORDS_DIR", records), ("PROJECT_ROOT", root),
                           ("SOPS_DIR", root / "sops"), ("APP_SETTINGS", settings_path)):
            stack.enter_context(patch(f"sop_app.ui.main_window.{name}", path))
        stack.enter_context(patch.object(MainWindow, "_bind_station_logging"))
        stack.enter_context(patch.object(MainWindow, "_open_initial_sop"))
        stack.enter_context(patch("sop_app.ui.main_window.cv2.require"))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.start"))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.shutdown", return_value=True))
        window = MainWindow(defer_initial_sop=True, station_store=store)
        def close():
            window.dirty = False
            window.close()
            for timer in window.findChildren(QTimer):
                timer.stop()
        self.addCleanup(close)
        window.pipeline._drain_commands(Mock())
        return window, root

    @staticmethod
    def configuration(token, fps=60):
        return {"source_key": "camera:0", "settings_token": token,
                "changes": {"fps": ("manual", fps)},
                "values": {"fps": 30.},
                "messages": [f"攝影機 FPS：要求 {fps}，驅動回報 30"]}

    def test_startup_preferences_and_readback_complete_before_capture_thread_starts(self):
        events = []
        capture = Mock()
        capture.isOpened.return_value = True
        reader = Mock()
        reader.start.side_effect = lambda: events.append("thread")
        def apply(_capture, changes):
            events.append("apply")
            self.assertEqual(changes["resolution"], ("manual", (1920, 1080)))
            self.assertEqual(changes["exposure"], ("manual", -7.))
            return ["攝影機 FPS：要求 60，驅動回報 30"]
        def read(_capture):
            events.append("readback")
            return {"fps": 30.}
        with patch("sop_app.pipeline.cv2.VideoCapture", return_value=capture), \
                patch("sop_app.pipeline.threading.Thread", return_value=reader), \
                patch("sop_app.camera_controls.apply_controls", side_effect=apply), \
                patch("sop_app.camera_controls.read_controls", side_effect=read):
            source = CameraSource(0, preferences={"resolution": [1920, 1080], "fps": 60,
                "controls": {"exposure": {"mode": "manual", "value": -7}}}, settings_token=41)
            self.assertEqual(events, ["apply", "readback", "thread"])
            self.assertEqual(source.reported_controls["fps"], 30.)
            self.assertEqual((source.source_key, source.settings_token), ("camera:0", 41))
            capture.read.assert_not_called()
            source.close()

    def test_failed_startup_releases_capture_without_starting_reader(self):
        capture = Mock()
        capture.isOpened.return_value = True
        with patch("sop_app.pipeline.cv2.VideoCapture", return_value=capture), \
                patch("sop_app.pipeline.threading.Thread") as reader, \
                patch("sop_app.camera_controls.apply_controls", side_effect=OSError("driver failed")):
            with self.assertRaisesRegex(OSError, "driver failed"):
                CameraSource(0)
            capture.release.assert_called_once()
            reader.assert_not_called()

    def test_queued_open_owns_preferences_and_keeps_connection_identity(self):
        pipeline = VideoPipeline("unused")
        prefs = {"controls": {"exposure": {"mode": "manual", "value": -7}}}
        pipeline.open_camera(2, prefs, 9)
        prefs["controls"]["exposure"]["value"] = -1
        _command, factory = pipeline._commands.get_nowait()
        with patch("sop_app.pipeline.CameraSource") as camera:
            factory()
            self.assertEqual(camera.call_args.args, (2,))
            self.assertEqual(camera.call_args.kwargs["preferences"]["controls"]["exposure"]["value"], -7)
            self.assertEqual(camera.call_args.kwargs["settings_token"], 9)

    def test_legacy_mode_is_reapplied_and_only_final_accepted_values_are_saved(self):
        legacy = {"camera_modes": {"camera:0": {"width": 1920, "height": 1080, "fps": 60}},
                  "camera_controls": {"camera:0": {"white_balance": {"auto": True, "value": 4000},
                                                     "exposure": {"auto": False, "value": -7}}}}
        window, _root = self.window(legacy)
        with patch.object(window.pipeline, "open_camera") as open_camera:
            window._connect_source()
            prefs = open_camera.call_args.args[1]
            self.assertEqual(prefs["width"], 1920)
            self.assertEqual(prefs["controls"]["exposure"]["value"], -7)
            self.assertEqual(open_camera.call_args.args[2], window._camera_settings_token)
        payload = self.configuration(window._camera_settings_token)
        payload["changes"].update({"exposure": ("manual", -5), "wb_temperature": ("auto", 0)})
        payload["values"].update({"exposure": -5, "wb_temperature": 4200,
                                    "control_modes": {"wb_temperature": "auto"}})
        payload["messages"].extend(["曝光：未完成（不支援）", "白平衡色溫：已切換自動模式"])
        window._on_camera_configuration(payload)
        saved = json.loads(window._settings_path.read_text(encoding="utf-8"))
        profile = saved["camera_profiles"]["camera:0"]
        self.assertEqual(profile["fps"], 30.)
        self.assertEqual(profile["controls"]["exposure"], {"mode": "manual", "value": -7.})
        self.assertEqual(profile["controls"]["wb_temperature"], {"mode": "auto"})
        self.assertEqual(saved["camera_modes"], legacy["camera_modes"])

    def test_disconnected_camera_reply_is_ignored_and_failed_persistence_rolls_back(self):
        window, _root = self.window({"camera_profiles": {"camera:0": {"fps": 25}}})
        token = window._camera_settings_token
        before = window._settings_path.read_bytes()
        window._disconnect_source()
        window._on_camera_configuration(self.configuration(token))
        self.assertEqual(window._settings_path.read_bytes(), before)
        with patch.object(window, "_atomic_settings", side_effect=OSError("disk full")), \
                patch.object(window, "_notify_error") as notify:
            window._on_camera_configuration(self.configuration(window._camera_settings_token))
            self.assertTrue(notify.called)
        self.assertEqual(window.settings["camera_profiles"]["camera:0"]["fps"], 25)
        self.assertEqual(window._settings_path.read_bytes(), before)

    def test_station_switch_reloads_own_profile_and_rejects_previous_station_reply(self):
        window, _root = self.window({"camera_profiles": {"camera:0": {"fps": 60}}}, stations=True)
        second = window.station_store.create("包裝站")
        second.settings.write_text(json.dumps({"camera_profiles": {"camera:0": {"fps": 15}},
                                               "inference_fps": 12}), encoding="utf-8")
        connect(second.records / "records.db").close()
        old_token = window._camera_settings_token
        with patch.object(window.pipeline, "isRunning", return_value=True):
            window._switch_to_station(second)
        window._on_station_switched(str(second.records), "")
        before = second.settings.read_bytes()
        window._on_camera_configuration(self.configuration(old_token))
        self.assertEqual(second.settings.read_bytes(), before)
        with patch.object(window.pipeline, "open_camera") as camera:
            window._connect_source()
            self.assertEqual(camera.call_args.args[1]["fps"], 15)
        self.assertEqual(window.inference_fps.value(), 12)

    def test_failed_station_switch_keeps_old_camera_profile_saveable(self):
        window, _root = self.window({"camera_profiles": {"camera:0": {"fps": 25}}}, stations=True)
        first = window.current_station
        second = window.station_store.create("檢查站")
        with patch.object(window.pipeline, "open_camera"):
            window._connect_source()
        connected_token = window._camera_settings_token
        with patch.object(window.pipeline, "isRunning", return_value=True):
            window._switch_to_station(second)
        self.assertNotEqual(window._camera_settings_token, connected_token)
        # Replies received while a station transition is pending are ignored.
        window._on_camera_configuration(self.configuration(connected_token))
        self.assertEqual(window.settings["camera_profiles"]["camera:0"]["fps"], 25)
        with patch("sop_app.ui.main_window.QMessageBox.warning"):
            window._on_station_switched(str(second.records), "target is locked")
        self.assertEqual(window.current_station.id, first.id)
        self.assertEqual(window._camera_settings_token, connected_token)
        # The old source remains connected after this failure; subsequent
        # accepted driver changes must still persist into its original station.
        window._on_camera_configuration(self.configuration(connected_token))
        saved = json.loads(first.settings.read_text(encoding="utf-8"))
        self.assertEqual(saved["camera_profiles"]["camera:0"]["fps"], 30.)
        self.assertIsNone(window._pending_camera_settings_token)

    def test_rate_cap_persists_only_after_ack_and_rejects_changes_during_operation(self):
        window, _root = self.window({"inference_fps": 20})
        window.inference_fps.setValue(15)
        self.assertEqual(window.settings["inference_fps"], 20)
        window.pipeline._drain_commands(Mock())
        self.assertEqual(window.settings["inference_fps"], 15)
        self.assertEqual(json.loads(window._settings_path.read_text())["inference_fps"], 15)
        window._on_engine_state(True)
        self.assertFalse(window.inference_fps.isEnabled())
        window.pipeline._engine = Mock(phase=Phase.RUNNING)
        window.inference_fps.setValue(30)  # Programmatic change must also be guarded in the worker.
        window.pipeline._drain_commands(Mock())
        self.assertEqual(window.inference_fps.value(), 15)
        self.assertEqual(window.settings["inference_fps"], 15)

    def test_fraction_below_minimum_is_normalized_before_queueing(self):
        window, _root = self.window()
        window.inference_fps.setValue(.5)
        self.assertEqual(window.inference_fps.value(), 1.)
        window.pipeline._drain_commands(Mock())
        self.assertEqual(window.pipeline._pacer.fps, 1.)
        self.assertFalse(window.inference_fps.keyboardTracking())

    def test_pacing_waits_before_reading_latest_camera_slot_without_queueing_old_frames(self):
        pipeline = VideoPipeline("unused")
        source = CameraSource.__new__(CameraSource)
        source._lock = threading.Lock()
        source._latest = (np.array([0], dtype=np.uint8), 0.)
        source.close = Mock()
        pipeline._source = source
        pipeline._pacer = InferencePacer(10)
        clock, arrivals, processed = [0.], [0], []
        def sleep(seconds):
            clock[0] += seconds
            arrivals[0] += 1
            source._latest = (np.array([arrivals[0]], dtype=np.uint8), clock[0])
        def process(image, timestamp, recorder, **_kwargs):
            processed.append((int(image[0]), timestamp))
            if len(processed) == 3:
                pipeline._alive = False
        with patch("sop_app.pipeline.AsyncRecorder"), \
                patch("sop_app.pipeline.time.monotonic", side_effect=lambda: clock[0]), \
                patch("sop_app.pipeline.time.sleep", side_effect=sleep), \
                patch.object(pipeline, "_process", side_effect=process):
            pipeline.run()
        self.assertEqual(len(processed), 3)
        self.assertGreater(processed[1][0], 1)  # Intermediate camera arrivals were replaced, not buffered.
        self.assertGreaterEqual(processed[1][1] - processed[0][1], .1 - 1e-9)
        self.assertGreaterEqual(processed[2][1] - processed[1][1], .1 - 1e-9)
        source.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
