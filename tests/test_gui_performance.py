"""UI coalescing and preview correctness; temporary files, no model/camera."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import ExitStack
from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sop_app.detection import Detection, FrameResult
from sop_app.engine import SOPEngine
from sop_app.pipeline import FramePacket
from sop_app.recorder import connect
from sop_app.sop_schema import Condition, SOPDefinition, Step
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.main_window import MainWindow
from sop_app.ui.run_page import RunPage


def packet(at=0., snapshot=None, results=()):
    image = np.zeros((50, 100, 3), np.uint8)
    result = FrameResult(100, 50, at, [Detection("part", .95, (10, 10, 30, 30))])
    return FramePacket(image, image, result, snapshot, 20., 5., results=results)


class GuiPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def dispose(widget):
        widget.close()
        for timer in widget.findChildren(QTimer):
            timer.stop()

    def window(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        connect(root / "records.db").close()
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, path in (("RECORDS_DIR", root), ("PROJECT_ROOT", root),
                           ("SOPS_DIR", root / "sops"), ("APP_SETTINGS", root / "settings.json")):
            stack.enter_context(patch(f"sop_app.ui.main_window.{name}", path))
        stack.enter_context(patch.object(MainWindow, "_load_settings", return_value={}))
        stack.enter_context(patch.object(MainWindow, "_save_settings"))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.start"))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.shutdown", return_value=True))
        window = MainWindow(defer_initial_sop=True)
        self.addCleanup(self.dispose, window)
        return window, root

    def test_today_stats_notifications_merge_and_explicit_refresh_cancels_timer(self):
        window, root = self.window()
        window._record_stats_timer.setInterval(30)
        with connect(root / "records.db") as db:
            db.execute("INSERT INTO cycles(sop_name,started_at,result) VALUES('test',?,'OK')",
                       (date.today().isoformat() + " 10:00:00",))
        db.close()
        with patch.object(window, "_refresh_today_stats", wraps=window._refresh_today_stats) as refresh:
            for _ in range(30):
                window._on_records_changed()
            self.assertEqual(refresh.call_count, 0)
            QTest.qWait(70)
            self.assertEqual(refresh.call_count, 1)
            self.assertEqual(window.run_page._today, {"OK": 1})
            window._on_records_changed()
            window._refresh_today_stats()
            QTest.qWait(70)
            self.assertEqual(refresh.call_count, 2)
            self.assertFalse(window._today_stats_dirty)

    def test_performance_tooltip_is_sampled_once_per_second_and_explains_runtime(self):
        window, _root = self.window()
        frame = packet()
        frame.stage_ms = {"inference_model_inference_ms": 4., "inference_rpc_ms": 5.}
        frame.inference_runtime = {"backend": "ultralytics", "device": "cuda:0", "dtype": "torch.float16",
                                   "fp16": True, "input_shape": (1, 3, 480, 800), "shape_stage": "network_input"}
        with patch.object(window.pipeline, "performance_snapshot", return_value={
                "frames": 300, "stages_ms": {"inference_rpc_ms": {"p95": 8.}}}) as summary:
            with patch("sop_app.ui.main_window.time.monotonic", return_value=10.):
                window._update_performance_tooltip(frame)
                window._update_performance_tooltip(frame)
            self.assertEqual(summary.call_count, 1)
            self.assertIn("480, 800", window.fps_status.toolTip())
            self.assertIn("torch.float16", window.fps_status.toolTip())
            self.assertIn("非純 GPU kernel", window.fps_status.toolTip())
            self.assertIn("8.00 ms", window.fps_status.toolTip())
            with patch("sop_app.ui.main_window.time.monotonic", return_value=11.1):
                window._update_performance_tooltip(frame)
            self.assertEqual(summary.call_count, 2)

    def test_running_packet_refreshes_regions_once_and_reuses_start_button_style(self):
        page = RunPage()
        self.addCleanup(self.dispose, page)
        sop = SOPDefinition(steps=[Step("test", conditions=[Condition(label="part")], hold_sec=2.)])
        page.set_sop(sop)
        page.set_running(True)
        engine = SOPEngine(sop)
        engine.start(0.)
        engine.update(packet(.1).result)
        with patch.object(page, "_refresh_regions", wraps=page._refresh_regions) as regions, \
                patch.object(page.start_button, "setStyleSheet", wraps=page.start_button.setStyleSheet) as style:
            page.on_packet(packet(.1, engine.snapshot()))
            self.assertEqual(regions.call_count, 1)
            self.assertFalse(style.called)

    def test_editor_text_sampling_does_not_drop_trial_frames(self):
        page = EditorPage()
        self.addCleanup(self.dispose, page)
        page.set_sop(SOPDefinition(steps=[Step("test", conditions=[Condition(label="part")], hold_sec=.4)]))
        page.set_classes(["part"])
        page.set_model_state("available")
        with patch("sop_app.ui.editor_page.time.monotonic", return_value=10.):
            page.on_packet(packet())
            page._start_validation()
            self.assertTrue(page._validation.running)
            with patch.object(page, "_update_live", wraps=page._update_live) as live, \
                    patch.object(page._validation, "update", wraps=page._validation.update) as update:
                for index in (1, 2):
                    page.on_packet(packet(index / 10))
                pending = tuple(packet(index / 10).result for index in range(3, 7))
                page.on_packet(packet(.6, results=pending))
                self.assertFalse(live.called)
                # The trial completes at .4 inside the pending result batch.
                self.assertEqual(update.call_count, 4)
                self.assertTrue(page._validation.completed)
        with patch.object(page, "_update_live", wraps=page._update_live) as live, \
                patch("sop_app.ui.editor_page.time.monotonic", return_value=10.21):
            page.on_packet(packet(.7))
            self.assertEqual(live.call_count, 1)


if __name__ == "__main__":
    unittest.main()
