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
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from sop_app.auxiliary_preview import PreviewFrame
from sop_app.camera_discovery import CameraDevice, CameraDiscoveryResult
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

    def window(self, settings=None, stations=False, *, load_initial_sop=False):
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
        if load_initial_sop:
            stack.enter_context(patch("sop_app.ui.main_window.initial_sop_path", return_value=None))
        else:
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

    def test_discovery_lists_connected_cameras_and_preserves_source_without_opening_it(self):
        window, _root = self.window({"source": "camera:6"})
        with patch.object(window._camera_discovery, "request", return_value=True) as search, \
                patch.object(window.pipeline, "open_camera") as open_camera:
            window._search_cameras()
            search.assert_called_once()
            self.assertFalse(window.connect_source_button.isEnabled())
            window._on_cameras_found(CameraDiscoveryResult((CameraDevice(0, "Built-in"), CameraDevice(6, "USB"))))
        self.assertEqual([window.source_combo.itemData(i) for i in range(window.source_combo.count())],
                         ["camera:0", "camera:6", "file"])
        self.assertEqual(window.source_combo.currentData(), "camera:6")
        self.assertIn("USB", window.source_combo.currentText())
        self.assertTrue(window.connect_source_button.isEnabled())
        open_camera.assert_not_called()

        window._on_cameras_found(CameraDiscoveryResult())
        self.assertEqual(window.source_combo.currentData(), "file")
        self.assertEqual(window.source_combo.count(), 1)
        self.assertEqual(window.settings["source"], "camera:6")

    def test_startup_automatically_connects_the_saved_camera_once(self):
        window, _root = self.window({"source": "camera:6", "camera_profiles": {"camera:6": {"fps": 25}}},
                                    load_initial_sop=True)
        result = CameraDiscoveryResult((CameraDevice(0, "Built-in"), CameraDevice(6, "USB")))
        with patch.object(window._camera_discovery, "request", return_value=True), \
                patch.object(window.pipeline, "open_camera") as open_camera:
            window._open_initial_sop(auto_connect_camera=True)
            open_camera.assert_not_called()
            window._on_cameras_found(result)
            open_camera.assert_called_once_with(6, {"fps": 25}, window._camera_settings_token,
                                                directshow_only=True)

            window._on_source_change_finished()
            window._search_cameras()
            window._on_cameras_found(result)
            self.assertEqual(open_camera.call_count, 1)

    def test_startup_connects_an_available_camera_without_replacing_missing_preference(self):
        window, _root = self.window({"source": "camera:6"}, load_initial_sop=True)
        original = window._settings_path.read_bytes()
        with patch.object(window._camera_discovery, "request", return_value=True), \
                patch.object(window.pipeline, "open_camera") as open_camera:
            window._open_initial_sop(auto_connect_camera=True)
            window._on_cameras_found(CameraDiscoveryResult((CameraDevice(2, "USB"),)))
            open_camera.assert_called_once_with(2, {}, window._camera_settings_token, directshow_only=True)
        self.assertEqual(window.source_combo.currentData(), "camera:2")
        self.assertEqual(window._settings_path.read_bytes(), original)

    def test_startup_leaves_image_source_idle_when_it_should_not_connect(self):
        camera = CameraDiscoveryResult((CameraDevice(0, "USB"),))
        for settings, result, cancel in [
            ({"source": "file"}, camera, False),
            ({"source": "camera:0"}, CameraDiscoveryResult(), False),
            ({"source": "camera:0"}, CameraDiscoveryResult(error="Unavailable"), False),
            ({"source": "camera:0"}, camera, True),
        ]:
            with self.subTest(settings=settings, result=result, cancel=cancel):
                window, _root = self.window(settings, load_initial_sop=True)
                with patch.object(window._camera_discovery, "request", return_value=True), \
                        patch.object(window.pipeline, "open_camera") as open_camera, \
                        patch("sop_app.ui.main_window.QFileDialog.getOpenFileName") as file_dialog:
                    window._open_initial_sop(auto_connect_camera=True)
                    if cancel:
                        window._disconnect_source()
                        window._on_source_change_finished()
                    window._on_cameras_found(result)
                    open_camera.assert_not_called()
                    file_dialog.assert_not_called()
                self.assertEqual(window._source_label, "")

    def test_selected_preview_stays_separate_and_releases_before_becoming_primary(self):
        window, _root = self.window()
        window._on_cameras_found(CameraDiscoveryResult((CameraDevice(0, "Main"), CameraDevice(1, "Second"))))
        frame = PreviewFrame(image=np.zeros((100, 100, 3), dtype=np.uint8))
        with patch.object(window._auxiliary_preview, "set_cameras") as select, \
                patch.object(window._auxiliary_preview, "snapshot", return_value={1: frame}), \
                patch.object(window._auxiliary_preview, "is_released", return_value=False) as released, \
                patch.object(window.pipeline, "open_camera") as open_camera:
            camera_action = next(action for action in window.multi_screen_menu.actions() if "Second" in action.text())
            camera_action.trigger()
            self.assertEqual(set(select.call_args.args[0]), {1})
            self.assertFalse(window.run_page.preview_pane.auxiliary.isHidden())
            self.assertFalse(window.editor.preview_pane.auxiliary.isHidden())
            with patch.object(window.run_page.preview_pane.auxiliary, "set_frame") as draw:
                window._update_auxiliary_previews()
                self.assertEqual(draw.call_args.args[0], 1)
                self.assertIs(draw.call_args.args[1], frame.image)
            open_camera.assert_not_called()

            window.source_combo.setCurrentIndex(window.source_combo.findData("camera:1"))
            window._connect_source()
            open_camera.assert_not_called()
            self.assertEqual(set(select.call_args.args[0]), set())
            self.assertTrue(window.run_page.preview_pane.auxiliary.isHidden())
            self.assertFalse(window.connect_source_button.isEnabled())

            released.return_value = True
            window._update_auxiliary_previews()
            open_camera.assert_called_once()
            self.assertEqual(open_camera.call_args.args[0], 1)
            self.assertTrue(open_camera.call_args.kwargs["directshow_only"])
            window._on_source_changed("")
            window._on_source_changed("攝影機 1")
            window._on_source_change_finished()
            self.assertTrue(window.connect_source_button.isEnabled())
            self.assertTrue(window.run_page.preview_pane.auxiliary.isHidden())

    def test_discovered_camera_does_not_fall_back_to_a_different_backend_index(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch("sop_app.pipeline.cv2.VideoCapture", return_value=capture) as video_capture:
            with self.assertRaisesRegex(RuntimeError, "無法開啟攝影機"):
                CameraSource(1, directshow_only=True)
        self.assertEqual(video_capture.call_count, 1)
        capture.release.assert_called_once()

    def test_preview_layout_syncs_pages_and_saves_the_ratio_for_each_direction(self):
        window, _root = self.window()
        monitor = window.run_page.preview_pane
        editor = window.editor.preview_pane
        actions = window._preview_orientation_actions
        self.assertTrue(actions["horizontal"].isChecked())
        monitor.split_ratio_changed.emit(.65)
        actions["vertical"].trigger()
        self.assertFalse(actions["horizontal"].isChecked())
        self.assertTrue(actions["vertical"].isChecked())
        for pane in (monitor, editor):
            self.assertEqual(pane._splitter.orientation(), Qt.Orientation.Vertical)
        editor.split_ratio_changed.emit(.58)
        window._flush_preview_layout()
        saved = json.loads(window._settings_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["preview_layout"], {"orientation": "vertical", "horizontal": .65, "vertical": .58})

        reopened, _other_root = self.window(saved)
        self.assertTrue(reopened._preview_orientation_actions["vertical"].isChecked())
        for pane in (reopened.run_page.preview_pane, reopened.editor.preview_pane):
            self.assertEqual(pane._splitter.orientation(), Qt.Orientation.Vertical)
            self.assertEqual(pane._ratio, .58)
        reopened._preview_orientation_actions["horizontal"].trigger()
        self.assertEqual(reopened.run_page.preview_pane._ratio, .65)
        self.assertEqual(reopened.editor.preview_pane._ratio, .65)

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
        self.assertIsNone(window.inference_fps_preset.currentData())
        self.assertFalse(window.inference_fps.isHidden())

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
        self.assertFalse(window.inference_fps_preset.isEnabled())
        window.pipeline._engine = Mock(phase=Phase.RUNNING)
        window.inference_fps.setValue(30)  # Programmatic change must also be guarded in the worker.
        window.pipeline._drain_commands(Mock())
        self.assertEqual(window.inference_fps.value(), 15)
        self.assertIsNone(window.inference_fps_preset.currentData())  # Keep the chosen custom mode.
        self.assertEqual(window.settings["inference_fps"], 15)

    def test_rate_presets_apply_after_worker_ack_and_keep_custom_entry_available(self):
        window, _root = self.window({"inference_fps": 0})
        self.assertEqual(window.inference_fps_preset.currentData(), 0)
        self.assertTrue(window.inference_fps.isHidden())
        window.inference_fps_preset.setCurrentIndex(window.inference_fps_preset.findData(30))
        self.assertEqual(window.settings["inference_fps"], 0)
        window.pipeline._drain_commands(Mock())
        self.assertEqual(window.pipeline._pacer.fps, 30)
        self.assertEqual(json.loads(window._settings_path.read_text())["inference_fps"], 30)
        window.inference_fps_preset.setCurrentIndex(window.inference_fps_preset.findData(None))
        self.assertFalse(window.inference_fps.isHidden())
        self.assertEqual(window.inference_fps.value(), 30)
        self.assertEqual(window.settings["inference_fps"], 30)
        window.inference_fps.setValue(12.5)
        window.pipeline._drain_commands(Mock())
        self.assertEqual(window.pipeline._pacer.fps, 12.5)
        self.assertIsNone(window.inference_fps_preset.currentData())
        self.assertFalse(window.inference_fps.isHidden())
        window.inference_fps_preset.setCurrentIndex(window.inference_fps_preset.findData(0))
        window.pipeline._drain_commands(Mock())
        self.assertEqual(window.settings["inference_fps"], 0)
        self.assertTrue(window.inference_fps.isHidden())

    def test_saved_custom_rate_is_not_replaced_by_a_preset_on_startup(self):
        window, _root = self.window({"inference_fps": 29.9})
        self.assertIsNone(window.inference_fps_preset.currentData())
        self.assertFalse(window.inference_fps.isHidden())
        self.assertEqual(window.inference_fps.value(), 29.9)
        window.pipeline._drain_commands(Mock())
        self.assertEqual(window.pipeline._pacer.fps, 29.9)
        self.assertEqual(window.settings["inference_fps"], 29.9)

    def test_older_rate_replies_do_not_overwrite_a_newer_custom_choice(self):
        window, _root = self.window()
        for fps in (30, 60, None):
            window.inference_fps_preset.setCurrentIndex(window.inference_fps_preset.findData(fps))
        replies = []
        window.pipeline.inference_fps_ready.connect(
            lambda _fps: replies.append((window.inference_fps.value(),
                                         window.inference_fps_preset.currentData())))
        window.pipeline._drain_commands(Mock())
        self.assertEqual(replies, [(60, None)] * 2)
        self.assertFalse(window.inference_fps.isHidden())
        window.inference_fps.setValue(12.5)
        window.inference_fps.setValue(15)
        window.pipeline._drain_commands(Mock())
        self.assertEqual(replies[-2:], [(15, None)] * 2)
        self.assertEqual(window.settings["inference_fps"], 15)
        self.assertFalse(window.inference_fps.isHidden())

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
