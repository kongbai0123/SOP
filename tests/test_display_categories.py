"""Shared display groups change preview pixels, never SOP evaluation or evidence."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import ExitStack
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sop_app.detection import Detection, FrameResult
from sop_app.display_options import filter_display_rois, roi_display_group
from sop_app.engine import Phase, SOPEngine
from sop_app.native_runtime import cv2
from sop_app.overlay import draw_detections
from sop_app.pipeline import FramePacket
from sop_app.recorder import connect
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.main_window import MainWindow
from sop_app.ui.run_page import RunPage


GROUPS = {"operation", "objects", "installation"}
CHOICES = tuple(frozenset(choice) for size in range(4) for choice in combinations(sorted(GROUPS), size))


def pixels(image):
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    buffer = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return buffer[:, :image.width() * 4].reshape(image.height(), image.width(), 4)[:, :, :3].copy()


def definition():
    return SOPDefinition(name="Display categories", rois=[
        ROI("C1盒", [(.05, .05), (.3, .05), (.3, .3), (.05, .3)]),
        ROI("左把手安裝區", [(.55, .05), (.9, .05), (.9, .3), (.55, .3)]),
        ROI("工作區", [(.1, .7), (.9, .7), (.9, .95), (.1, .95)]),
    ], steps=[
        Step("取料", conditions=[Condition(label="part", roi="C1盒")], hold_sec=1.),
        Step("安裝", conditions=[Condition(label="part_done", roi="左把手安裝區")], hold_sec=1.),
    ])


def packet(*, at=.1, shade=0, model_preview=True, classes=("part", "unrelated")):
    raw = np.full((180, 320, 3), shade, np.uint8)
    detections = [Detection("part", .95, (30, 85, 80, 135), np.ones((50, 50), dtype=bool)),
                  Detection("unrelated", .7, (130, 85, 180, 135))]
    model = draw_detections(raw, detections, classes, .5, True)
    evidence = model.copy()
    cv2.line(evidence, (176, 45), (304, 45), (0, 220, 255), 2)
    result = FrameResult(320, 180, at, detections)
    return FramePacket(raw, evidence, result, None, 20., 5., classes=classes,
                       model_display=model if model_preview else None)


class DisplayCategoriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def dispose(widget):
        widget.close()
        for timer in widget.findChildren(QTimer):
            timer.stop()

    def page(self, kind, sop=None):
        page = kind()
        self.addCleanup(self.dispose, page)
        page.set_sop(sop or definition())
        return page

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

    def test_all_eight_group_combinations_match_both_pages_without_changing_inputs(self):
        for kind in (RunPage, EditorPage):
            with self.subTest(page=kind.__name__):
                sop = definition()
                page = self.page(kind, sop)
                frame = packet()
                page.on_packet(frame)
                original = deepcopy(page.sop.to_dict())
                raw, evidence = frame.frame.copy(), frame.display.copy()
                detections = deepcopy(frame.result.detections)
                for groups in CHOICES:
                    with self.subTest(groups=groups):
                        page.set_display_groups(groups)
                        expected_names = {"C1盒", "工作區"} if "operation" in groups else set()
                        if "installation" in groups:
                            expected_names.add("左把手安裝區")
                        shown = {roi.name for roi in page.video._rois} if page.video.regions_visible else set()
                        self.assertEqual(shown, expected_names)
                        expected = frame.model_display if "objects" in groups else frame.frame
                        np.testing.assert_array_equal(pixels(page.video._image), expected[:, :, ::-1])
                self.assertEqual(page.sop.to_dict(), original)
                np.testing.assert_array_equal(frame.frame, raw)
                np.testing.assert_array_equal(frame.display, evidence)
                self.assertEqual([d.label for d in frame.result.detections], [d.label for d in detections])
                for before, after in zip(detections, frame.result.detections):
                    np.testing.assert_array_equal(after.mask, before.mask)

    def test_default_all_and_objects_include_classes_unrelated_to_current_step(self):
        for kind in (RunPage, EditorPage):
            with self.subTest(page=kind.__name__):
                page = self.page(kind)
                frame = packet()
                page.on_packet(frame)
                self.assertEqual({r.name for r in page.video._rois}, {"C1盒", "左把手安裝區", "工作區"})
                np.testing.assert_array_equal(pixels(page.video._image), frame.model_display[:, :, ::-1])
                self.assertGreater(np.count_nonzero(pixels(page.video._image)[85:135, 130:180]), 0)

    def test_selected_regions_stay_visible_after_step_changes_and_highlight_moves(self):
        sop = definition()
        page = self.page(RunPage, sop)
        engine = SOPEngine(sop)
        engine.start(0.)
        page.set_running(True)
        frame = packet()
        frame.snapshot = engine.snapshot()
        page.on_packet(frame)
        self.assertEqual(page.video._highlight, {"C1盒"})
        first_names = [roi.name for roi in page.video._rois]
        next_frame = packet(at=.2)
        next_frame.snapshot = deepcopy(frame.snapshot)
        next_frame.snapshot.current_index = 1
        before = deepcopy(next_frame.snapshot)
        page.on_packet(next_frame)
        self.assertEqual([roi.name for roi in page.video._rois], first_names)
        self.assertEqual(page.video._highlight, {"左把手安裝區"})
        page.set_display_groups({"operation", "installation"})
        self.assertEqual([roi.name for roi in page.video._rois], first_names)
        self.assertEqual(next_frame.snapshot, before)

    def test_toolbar_is_shared_and_applies_while_other_page_is_hidden(self):
        window = self.window()
        window._apply_sop(definition(), None)
        frame = packet()
        for page in (window.run_page, window.editor):
            page.on_packet(frame)
        original = deepcopy(window.editor.sop.to_dict())
        changed = Mock()
        window.editor.modified.connect(changed)
        self.assertEqual(set(window.display_category_actions), GROUPS)
        self.assertTrue(all(action.isChecked() for action in window.display_category_actions.values()))
        self.assertIs(window.display_menu_button.menu(), window.display_menu)
        for active_page in (window.run_page, window.editor):
            window.tabs.setCurrentWidget(active_page)
            for groups in CHOICES:
                for group, action in window.display_category_actions.items():
                    action.setChecked(group in groups)
                for page in (window.run_page, window.editor):
                    expected = frame.model_display if "objects" in groups else frame.frame
                    np.testing.assert_array_equal(pixels(page.video._image), expected[:, :, ::-1])
                    shown = {roi.name for roi in page.video._rois} if page.video.regions_visible else set()
                    expected_names = {"C1盒", "工作區"} if "operation" in groups else set()
                    if "installation" in groups:
                        expected_names.add("左把手安裝區")
                    self.assertEqual(shown, expected_names)
        self.assertEqual(window.editor.sop.to_dict(), original)
        self.assertFalse(window.dirty)
        changed.assert_not_called()

    def test_menu_keeps_open_for_several_clicks_and_keyboard_selection(self):
        window = self.window()
        menu = window.display_menu
        menu.popup(QPoint(20, 20))
        self.app.processEvents()
        self.addCleanup(menu.close)
        for group in ("operation", "installation"):
            action = window.display_category_actions[group]
            QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center())
            self.assertFalse(action.isChecked())
            self.assertTrue(menu.isVisible())
        objects = window.display_category_actions["objects"]
        menu.setActiveAction(objects)
        QTest.keyClick(menu, Qt.Key.Key_Space)
        self.assertFalse(objects.isChecked())
        self.assertTrue(menu.isVisible())
        window._select_display_categories(GROUPS)
        self.assertTrue(all(action.isChecked() for action in window.display_category_actions.values()))
        window._select_display_categories(set())
        self.assertFalse(any(action.isChecked() for action in window.display_category_actions.values()))

    def test_step_checkbox_filters_both_pages_and_restores_without_changing_inputs(self):
        window = self.window()
        sop = definition()
        sop.steps[0].forbidden = [Condition(label="wrong", roi="左把手安裝區")]
        window._apply_sop(sop, None)
        frame = packet()
        frame.result.detections.append(Detection("wrong", .95, (220, 85, 270, 135)))
        frame.model_display = draw_detections(frame.frame, frame.result.detections, frame.classes, .5, True)
        for page in (window.run_page, window.editor):
            page.on_packet(frame)
        original = deepcopy(window.editor.sop.to_dict())
        labels = [d.label for d in frame.result.detections]
        evidence = frame.display.copy()
        modified = Mock()
        window.editor.modified.connect(modified)
        self.assertEqual(window.step_display_check.text(), "逐一任務")
        self.assertFalse(window.step_display_check.isChecked())
        self.assertEqual(set(window.display_category_actions), GROUPS)
        window.step_display_check.setChecked(True)
        expected = draw_detections(frame.frame, [d for d in frame.result.detections
                                               if d.label in {"part", "wrong"}], frame.classes, .5, True)
        for page in (window.run_page, window.editor):
            np.testing.assert_array_equal(pixels(page.video._image), expected[:, :, ::-1])
            self.assertEqual({r.name for r in page.video._rois}, {"C1盒", "左把手安裝區"})
        window._select_display_categories({"operation"})
        for page in (window.run_page, window.editor):
            np.testing.assert_array_equal(pixels(page.video._image), frame.frame[:, :, ::-1])
            self.assertEqual({r.name for r in page.video._rois}, {"C1盒"})
        window._select_display_categories(GROUPS)
        window.step_display_check.setChecked(False)
        for page in (window.run_page, window.editor):
            np.testing.assert_array_equal(pixels(page.video._image), frame.model_display[:, :, ::-1])
            self.assertEqual({r.name for r in page.video._rois}, {"C1盒", "左把手安裝區", "工作區"})
        self.assertEqual(window.editor.sop.to_dict(), original)
        self.assertEqual([d.label for d in frame.result.detections], labels)
        np.testing.assert_array_equal(frame.display, evidence)
        self.assertFalse(window.dirty)
        modified.assert_not_called()

    def test_run_step_display_follows_current_step_and_cycle_reset_with_no_class_list(self):
        sop = definition()
        sop.cycle.reset_conditions = [Condition(label="unrelated", roi="工作區")]
        page = self.page(RunPage, sop)
        engine = SOPEngine(sop)
        engine.start(0.)
        page.set_running(True)
        page.set_step_display(True)
        first = packet(classes=())
        first.snapshot = engine.snapshot()
        page.on_packet(first)
        self.assertEqual({r.name for r in page.video._rois}, {"C1盒"})
        expected = draw_detections(first.frame, first.result.detections[:1], (), .5, True)
        np.testing.assert_array_equal(pixels(page.video._image), expected[:, :, ::-1])
        second = packet(at=.2, classes=())
        second.snapshot = deepcopy(first.snapshot)
        second.snapshot.current_index = 1
        page.on_packet(second)
        self.assertEqual({r.name for r in page.video._rois}, {"左把手安裝區"})
        np.testing.assert_array_equal(pixels(page.video._image), second.frame[:, :, ::-1])
        reset = packet(at=.3, classes=())
        reset.snapshot = deepcopy(second.snapshot)
        reset.snapshot.phase = Phase.CYCLE_DONE
        page.on_packet(reset)
        self.assertEqual({r.name for r in page.video._rois}, {"工作區"})
        expected = draw_detections(reset.frame, reset.result.detections[1:], (), .5, True)
        np.testing.assert_array_equal(pixels(page.video._image), expected[:, :, ::-1])

    def test_editor_step_display_changes_immediately_with_selection_and_flow(self):
        page = self.page(EditorPage)
        frame = packet()
        page.on_packet(frame)
        page.set_step_display(True)
        self.assertEqual({r.name for r in page.video._rois}, {"C1盒"})
        page.validation_step.setCurrentIndex(1)
        self.assertEqual({r.name for r in page.video._rois}, {"左把手安裝區"})
        np.testing.assert_array_equal(pixels(page.video._image), frame.frame[:, :, ::-1])
        page._validation.start(page.sop, "flow", 0, frame.result.timestamp)
        page.on_packet(packet(at=.2))
        self.assertEqual({r.name for r in page.video._rois}, {"C1盒"})
        # Advance the actual trial engine with observed samples in C1.
        for at in (.3, .6, .9, 1.2, 1.4):
            observed = packet(at=at)
            observed.result.detections = [Detection("part", .95, (20, 15, 60, 45))]
            observed.model_display = None
            page.on_packet(observed)
        self.assertEqual(page._validation.engine.index, 1)
        self.assertEqual({r.name for r in page.video._rois}, {"左把手安裝區"})
        np.testing.assert_array_equal(pixels(page.video._image), observed.frame[:, :, ::-1])

    def test_toolbar_score_and_mask_settings_redraw_both_cached_pages(self):
        window = self.window()
        window._apply_sop(definition(), None)
        frame = packet()
        for page in (window.run_page, window.editor):
            page.on_packet(frame)
        window.tabs.setCurrentWidget(window.run_page)
        window.mask_check.setChecked(False)
        window.display_score.setValue(.8)
        expected = draw_detections(frame.frame, frame.result.detections, frame.classes, .8, False)
        for page in (window.run_page, window.editor):
            np.testing.assert_array_equal(pixels(page.video._image), expected[:, :, ::-1])
        self.assertEqual(len(frame.result.detections), 2)
        self.assertFalse(window.dirty)

    def test_score_and_masks_redraw_cached_images_and_restore_from_raw(self):
        for kind in (RunPage, EditorPage):
            for model_preview in (True, False):
                with self.subTest(page=kind.__name__, model_preview=model_preview):
                    page = self.page(kind)
                    frame = packet(model_preview=model_preview)
                    page.on_packet(frame)
                    for masks, score in ((False, .5), (True, .99), (True, .5)):
                        page.set_overlay_options(masks, score)
                        expected = draw_detections(frame.frame, frame.result.detections, frame.classes, score, masks)
                        np.testing.assert_array_equal(pixels(page.video._image), expected[:, :, ::-1])
                        page.set_display_groups({"operation"})
                        np.testing.assert_array_equal(pixels(page.video._image), frame.frame[:, :, ::-1])
                        page.set_display_groups(GROUPS)
                        np.testing.assert_array_equal(pixels(page.video._image), expected[:, :, ::-1])

    def test_missing_class_list_rebuilds_model_preview_without_evidence_roi(self):
        for kind in (RunPage, EditorPage):
            with self.subTest(page=kind.__name__):
                page = self.page(kind)
                frame = packet(model_preview=False, classes=())
                page.on_packet(frame)
                expected = draw_detections(frame.frame, frame.result.detections, (), .5, True)
                np.testing.assert_array_equal(pixels(page.video._image), expected[:, :, ::-1])
                self.assertFalse(np.array_equal(pixels(page.video._image), frame.display[:, :, ::-1]))

    def test_clear_source_then_change_groups_does_not_revive_old_pixels(self):
        for kind in (RunPage, EditorPage):
            with self.subTest(page=kind.__name__):
                page = self.page(kind)
                page.on_packet(packet())
                page.clear_video("Disconnected")
                for groups in CHOICES:
                    page.set_display_groups(groups)
                    page.set_overlay_options(False, .7)
                    self.assertIsNone(page.video._image)

    def test_editor_frozen_pixels_keep_own_detections_and_return_to_latest_frame(self):
        page = self.page(EditorPage)
        old, latest = packet(shade=10), packet(at=.2, shade=60)
        page.on_packet(old)
        page.freeze_check.setChecked(True)
        page.on_packet(latest)
        page.set_display_groups({"objects"})
        np.testing.assert_array_equal(pixels(page.video._image), old.model_display[:, :, ::-1])
        page.set_overlay_options(False, .8)
        expected_old = draw_detections(old.frame, old.result.detections, old.classes, .8, False)
        np.testing.assert_array_equal(pixels(page.video._image), expected_old[:, :, ::-1])
        page.set_display_groups(set())
        np.testing.assert_array_equal(pixels(page.video._image), old.frame[:, :, ::-1])
        page._return_live()
        np.testing.assert_array_equal(pixels(page.video._image), latest.frame[:, :, ::-1])
        page.set_display_groups({"objects"})
        expected_latest = draw_detections(latest.frame, latest.result.detections, latest.classes, .8, False)
        np.testing.assert_array_equal(pixels(page.video._image), expected_latest[:, :, ::-1])

    def test_editor_temporarily_reveals_selected_hidden_region_without_changing_groups(self):
        page = self.page(EditorPage)
        page.on_packet(packet())
        page.set_display_groups({"operation"})
        original = deepcopy(page.sop.to_dict())
        modified = Mock()
        page.modified.connect(modified)
        page._select_roi("左把手安裝區")
        self.assertNotIn("左把手安裝區", {r.name for r in page.video._rois})
        page.edit_rois_button.setChecked(True)
        self.assertIn("左把手安裝區", {r.name for r in page.video._rois})
        page.edit_rois_button.setChecked(False)
        self.assertNotIn("左把手安裝區", {r.name for r in page.video._rois})
        self.assertEqual(page.sop.to_dict(), original)
        modified.assert_not_called()

    def test_editor_explicit_region_category_survives_duplicate_and_undo(self):
        page = self.page(EditorPage)
        page.on_packet(packet())
        roi = page.sop.rois[0]
        page._change_roi_display_group(roi, "installation")
        self.assertEqual(roi_display_group(roi), "installation")
        page._duplicate_roi(roi)
        self.assertEqual(page.sop.rois[-1].display_group, "installation")
        restored = SOPDefinition.from_dict(page.sop.to_dict())
        self.assertEqual(restored.rois[-1].display_group, "installation")
        page._undo_roi()
        self.assertEqual(len(page.sop.rois), 3)
        self.assertEqual(page.sop.rois[0].display_group, "installation")
        page._undo_roi()
        self.assertEqual(page.sop.rois[0].display_group, "auto")

    def test_active_trial_alert_reveals_hidden_region_until_violation_clears(self):
        sop = definition()
        sop.steps[0].conditions = [Condition(label="part_done")]
        sop.steps[0].forbidden = [Condition(label="hand", roi="C1盒")]
        sop.steps[0].forbidden_hold_sec = 0.
        page = self.page(EditorPage, sop)
        page.set_display_groups({"objects"})
        page._validation.start(page.sop, "single", 0, 0.)
        active = packet()
        active.result.detections = [Detection("hand", .95, (20, 15, 60, 45))]
        active.model_display = draw_detections(active.frame, active.result.detections, ("hand",), .5, True)
        active.classes = ("hand", "part_done")
        page.on_packet(active)
        self.assertTrue(page._validation.alert()["active"])
        self.assertEqual(page.video._alert_regions, {"C1盒"})
        self.assertEqual({roi.name for roi in page.video._rois}, {"C1盒"})
        clear = packet(at=.2)
        clear.result.detections = []
        clear.model_display = clear.frame.copy()
        page.on_packet(clear)
        self.assertFalse(page._validation.alert()["active"])
        self.assertTrue(page._validation.alert()["text"])  # History remains visible.
        self.assertEqual(page.video._alert_regions, set())
        self.assertEqual(page.video._rois, [])


class DisplayRegionClassificationTests(unittest.TestCase):
    def test_old_names_default_to_operations_except_installation_locations(self):
        for name in ("工作區", "C1盒", "C2盒", "C3盒", "未命名區域", "取料區"):
            with self.subTest(name=name):
                self.assertEqual(roi_display_group(ROI(name, [])), "operation")
        for name in ("左把手安裝區", "右把手安裝位置", "主軸安裝區"):
            with self.subTest(name=name):
                self.assertEqual(roi_display_group(ROI(name, [])), "installation")

    def test_explicit_category_is_independent_of_name_and_survives_serialization(self):
        sop = SOPDefinition(rois=[ROI("治具 A", [], display_group="installation"),
                                  ROI("安裝工具料盒", [], display_group="operation")])
        restored = SOPDefinition.from_dict(sop.to_dict())
        self.assertEqual([roi_display_group(roi) for roi in restored.rois], ["installation", "operation"])
        self.assertEqual([roi.display_group for roi in restored.rois], ["installation", "operation"])
        self.assertEqual([roi.name for roi in filter_display_rois(restored.rois, {"installation"})], ["治具 A"])
        self.assertEqual([roi.name for roi in filter_display_rois(restored.rois, {"objects"})], [])

    def test_legacy_sop_load_keeps_rules_and_auto_category(self):
        original = definition().to_dict()
        for roi in original["rois"]:
            roi.pop("display_group", None)
        sop = SOPDefinition.from_dict(original)
        self.assertTrue(all(roi.display_group == "auto" for roi in sop.rois))
        self.assertEqual([condition.roi for step in sop.steps for condition in step.conditions],
                         ["C1盒", "左把手安裝區"])
        self.assertEqual([roi.name for roi in filter_display_rois(sop.rois, {"installation"})], ["左把手安裝區"])


if __name__ == "__main__":
    unittest.main()
