"""ROI visibility changes presentation only, including burned-in evidence paths."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import base64
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sop_app.detection import Detection, FrameResult
from sop_app.engine import SOPEngine
from sop_app.native_runtime import cv2
from sop_app.overlay import draw_detections
from sop_app.pipeline import FramePacket, VideoPipeline
from sop_app.recorder import connect
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step, Workpiece
from sop_app.tracking import workpiece_key
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.main_window import MainWindow
from sop_app.ui.run_page import RunPage
from sop_app.ui.video_widget import VideoWidget


def rgb(image):
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return pixels[:, :image.width() * 4].reshape(image.height(), image.width(), 4)[:, :, :3].copy()


def definition(following=False):
    roi = ROI("REGION", [(.55, .25), (.95, .25), (.95, .60), (.55, .60)],
              "workpiece" if following else "fixed")
    sop = SOPDefinition(name="visibility", rois=[roi],
                        steps=[Step("A", conditions=[Condition(label="part", roi="REGION")], hold_sec=0.)])
    if following:
        reference = cv2.imencode(".png", np.zeros((180, 320, 3), np.uint8))[1].tobytes()
        sop.workpiece = Workpiece(reference_png=base64.b64encode(reference).decode("ascii"),
                                  points=[(.1, .1), (.9, .1), (.9, .9), (.1, .9)])
    return sop


def packet(sop=None, *, classes=("part",), at=.1):
    image = np.zeros((180, 320, 3), np.uint8)
    detection = Detection("part", .95, (30, 100, 80, 150), np.ones((50, 50), dtype=bool))
    model = draw_detections(image, [detection], ("part",), .5, True)
    evidence = model.copy()
    cv2.line(evidence, (176, 45), (304, 45), (0, 220, 255), 2)
    result = FrameResult(320, 180, at, [detection], np.array([[1., 0., 0.], [0., 1., 0.]]))
    result.tracking_status = "valid"
    result.workpiece_key = workpiece_key(sop.workpiece) if sop else ""
    return FramePacket(image, evidence, result, None, 20., 5., classes=classes, model_display=model)


class RegionVisibilityTests(unittest.TestCase):
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
        return window

    def page(self, kind, sop):
        page = kind()
        self.addCleanup(self.dispose, page)
        # Reference-mode UI needs valid reference pixels, not a feature search.
        target = "sop_app.ui.editor_page.WorkpieceLocator" if kind is EditorPage else "sop_app.ui.run_page.WorkpieceLocator"
        with patch(target, return_value=SimpleNamespace()):
            page.set_sop(sop)
        return page

    def test_toolbar_toggles_both_pages_immediately_without_changing_sop_or_dirty(self):
        window = self.window()
        sop = definition()
        window._apply_sop(sop, None)
        window.run_page.on_packet(packet(sop))
        window.editor.on_packet(packet(sop))
        expected = deepcopy(window.editor.sop.to_dict())
        modified = Mock()
        window.editor.modified.connect(modified)
        masks = window.mask_check.isChecked()
        self.assertTrue(window.regions_check.isChecked())
        window.regions_check.setChecked(False)
        self.assertFalse(window.run_page.video.regions_visible)
        self.assertFalse(window.editor.video.regions_visible)
        self.assertEqual(len(window.run_page.video._rois), 1)
        self.assertEqual(len(window.editor.video._rois), 1)
        window.regions_check.setChecked(True)
        self.assertTrue(window.run_page.video.regions_visible)
        self.assertTrue(window.editor.video.regions_visible)
        self.assertEqual(window.editor.sop.to_dict(), expected)
        self.assertFalse(window.dirty)
        modified.assert_not_called()
        self.assertEqual(window.mask_check.isChecked(), masks)

    def test_hidden_rois_remove_lines_names_and_stale_fill_but_keep_model_and_mask(self):
        widget = VideoWidget()
        self.addCleanup(self.dispose, widget)
        widget.resize(640, 360)
        frame = packet()
        widget.set_frame(frame.preview_display, frame.result.detections, ("part",))
        widget.show()
        self.app.processEvents()
        baseline = rgb(widget.grab().toImage())
        rois = definition().rois + [ROI("STALE", [(.55, .7), (.95, .7), (.95, .95), (.55, .95)])]
        widget.set_rois(rois, selected="REGION", stale=["STALE"])
        widget.set_editable(["REGION"])
        visible = rgb(widget.grab().toImage())
        self.assertFalse(np.array_equal(visible, baseline))
        np.testing.assert_array_equal(visible[:, :300], baseline[:, :300])
        widget.set_regions_visible(False)
        hidden = rgb(widget.grab().toImage())
        np.testing.assert_array_equal(hidden, baseline)
        self.assertGreater(np.count_nonzero(hidden[200:300, 60:160]), 0)  # model mask/box remains
        renamed, edited = Mock(), Mock()
        widget.roi_rename_requested.connect(renamed)
        widget.roi_edited.connect(edited)
        QTest.mouseDClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(400, 140))
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(400, 140))
        QTest.mouseMove(widget, QPoint(420, 150))
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(420, 150))
        renamed.assert_not_called()
        edited.assert_not_called()
        self.assertEqual(widget._rois, rois)
        widget.set_regions_visible(True)
        np.testing.assert_array_equal(rgb(widget.grab().toImage()), visible)

    def test_editor_edit_and_draw_override_hidden_choice_then_restore_it(self):
        sop = definition()
        page = self.page(EditorPage, sop)
        page.on_packet(packet(sop))
        original = deepcopy(sop.to_dict())
        changed = Mock()
        page.modified.connect(changed)
        page.set_regions_visible(False)
        page.edit_rois_button.setChecked(True)
        self.assertTrue(page.video.regions_visible)
        page.edit_rois_button.setChecked(False)
        self.assertFalse(page.video.regions_visible)
        page.draw_button.setChecked(True)
        self.assertTrue(page.video.regions_visible)
        page.draw_button.setChecked(False)
        self.assertFalse(page.video.regions_visible)
        page._return_live()
        self.assertFalse(page.video.regions_visible)
        self.assertEqual(sop.to_dict(), original)
        changed.assert_not_called()

    def test_reference_exit_restores_hidden_choice_and_model_only_live_frame(self):
        sop = definition(following=True)
        page = self.page(EditorPage, sop)
        frame = packet(sop)
        page.on_packet(frame)
        page.set_regions_visible(False)
        np.testing.assert_array_equal(rgb(page.video._image), frame.preview_display[:, :, ::-1])
        page.reference_button.setChecked(True)
        self.assertTrue(page.video.regions_visible)
        np.testing.assert_array_equal(rgb(page.video._image), np.zeros((180, 320, 3), np.uint8))
        page.reference_button.setChecked(False)
        self.assertFalse(page.video.regions_visible)
        np.testing.assert_array_equal(rgb(page.video._image), frame.preview_display[:, :, ::-1])
        self.assertFalse(np.array_equal(rgb(page.video._image), frame.display[:, :, ::-1]))

    def test_monitor_all_and_missing_classes_never_use_burned_evidence_preview(self):
        for classes in (("part",), ()):
            with self.subTest(classes=classes):
                sop = definition()
                page = self.page(RunPage, sop)
                if classes:
                    page.display_mode.setCurrentIndex(page.display_mode.findData("all"))
                frame = packet(sop, classes=classes)
                page.on_packet(frame)
                page.set_regions_visible(False)
                np.testing.assert_array_equal(rgb(page.video._image), frame.preview_display[:, :, ::-1])
                self.assertGreater(np.count_nonzero(rgb(page.video._image)[100:150, 30:80]), 0)
                page.set_regions_visible(True)
                self.assertEqual(len(page.video._rois), 1)

    def test_pipeline_preview_is_model_only_while_completed_step_keeps_roi_evidence(self):
        sop = definition(following=True)
        sop.steps[0].conditions[0].roi = ""  # completion input is separate from the visible evidence polygon
        pipe = VideoPipeline(Path("unused"))
        detection = Detection("part", .95, (30, 100, 80, 150))
        pipe._detector = SimpleNamespace(detect=lambda _: [detection], classes=("part",),
                                        info=SimpleNamespace(model_version_id="fixture"))
        pipe._locator = SimpleNamespace(locate=lambda *args, **kwargs: (
            np.array([[1., 0., 0.], [0., 1., 0.]]), "valid"), metrics={})
        pipe._locator_key = workpiece_key(sop.workpiece)
        pipe._engine = SOPEngine(sop)
        recorder = Mock()
        pipe._emit_events(pipe._engine.start(0.), recorder)
        recorder.reset_mock()
        packets = []
        pipe.packet_ready.connect(packets.append)
        image = np.zeros((180, 320, 3), np.uint8)
        pipe._process(image, .1, recorder)
        self.assertEqual(len(packets), 1)
        frame = packets[0]
        expected = draw_detections(image, [detection], ("part",), .5, True)
        np.testing.assert_array_equal(frame.preview_display, expected)
        yellow = np.all(frame.display == (0, 220, 255), axis=2)
        self.assertGreater(np.count_nonzero(yellow), 100)
        self.assertFalse(np.any(np.all(frame.preview_display == (0, 220, 255), axis=2)))
        np.testing.assert_array_equal(recorder.handle.call_args.args[4], frame.display)
        np.testing.assert_array_equal(pipe._last_display, frame.display)
        self.assertIn("step_completed", [event.kind for event in recorder.handle.call_args.args[0]])
        self.assertFalse(np.shares_memory(frame.preview_display, frame.display))

    def test_unready_model_clears_both_preview_images_and_batched_detections(self):
        window = self.window()
        frame = packet()
        frame.results = (frame.result,)
        window.model_ready = False
        with patch.object(window.run_page, "on_packet") as consume:
            window.tabs.setCurrentWidget(window.run_page)
            window._on_packet(frame)
        shown = consume.call_args.args[0]
        np.testing.assert_array_equal(shown.display, frame.frame)
        np.testing.assert_array_equal(shown.preview_display, frame.frame)
        self.assertEqual(shown.result.detections, [])
        self.assertFalse(shown.result.detection_valid)
        self.assertEqual(shown.results[0].detections, [])
        self.assertFalse(shown.results[0].detection_valid)
        self.assertEqual(len(frame.result.detections), 1)  # input packet is not mutated


if __name__ == "__main__":
    unittest.main()
