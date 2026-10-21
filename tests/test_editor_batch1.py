"""Editor batch-one regressions: real Qt widgets, isolated files, no camera/model required."""
from __future__ import annotations

import copy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from sop_app.detection import Detection, FrameResult
from sop_app.engine import SOPEngine
from sop_app.model_bundle import ModelBundleError, ModelInfo
from sop_app.pipeline import FramePacket, VideoPipeline
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step, load_sop, save_sop
from sop_app.ui.condition_table import ConditionTable
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.main_window import MainWindow


def model_info(path: Path, classes=("grip",)) -> ModelInfo:
    return ModelInfo(path, "yolo26n_detect", "YOLO", "ultralytics", False, classes,
                     640, .5, "v1", "test")


def packet(timestamp=0.0, detected=True, valid=True) -> FramePacket:
    image = np.zeros((100, 200, 3), np.uint8)
    detections = [Detection("grip", .9, (10, 10, 30, 30))] if detected else []
    result = FrameResult(200, 100, timestamp, detections, detection_valid=valid)
    return FramePacket(image, image.copy(), result, None, 10, 5)


class QtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # The offscreen Windows platform has no system font inventory by default.
        # Load actual UI fonts for optional visual QA, not for production settings.
        if os.environ.get("SOP_EDITOR_TEST_ARTIFACT_DIR") and os.name == "nt":
            font_directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for filename in ("msjh.ttc", "msjhbd.ttc"):
                font_id = QFontDatabase.addApplicationFont(str(font_directory / filename))
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    cls.app.setFont(QFont(families[0], 9))

    def widget(self, widget):
        self.addCleanup(widget.deleteLater)
        self.addCleanup(widget.close)
        return widget


class ConditionDeleteTests(QtTests):
    def setUp(self):
        self.editor = self.widget(ConditionTable())
        self.editor.set_options(["grip", "part"], ["left"])
        self.original = [Condition(label="grip"), Condition(label="part", roi="left", min_count=2,
                                                          min_score=.85, roi_overlap=.75)]
        self.editor.set_conditions(self.original)
        self.changes = []
        self.editor.changed.connect(lambda: self.changes.append(True))

    def test_no_selection_cannot_delete_or_emit_change(self):
        self.assertFalse(self.editor.remove_button.isEnabled())
        self.editor._remove()
        self.assertEqual(self.editor.conditions(), self.original)
        self.assertEqual(self.changes, [])

    def test_current_row_without_selection_is_not_authority_to_delete(self):
        self.editor.table.selectRow(0)
        self.editor.table.clearSelection()
        self.assertEqual(self.editor.table.currentRow(), 0)
        self.assertFalse(self.editor.remove_button.isEnabled())
        self.editor._remove()
        self.assertEqual(self.editor.conditions(), self.original)

    def test_selected_delete_and_undo_restore_all_fields(self):
        self.editor.table.selectRow(1)
        self.assertTrue(self.editor.remove_button.isEnabled())
        self.editor.remove_button.click()
        self.assertEqual(self.editor.conditions(), self.original[:1])
        self.assertFalse(self.editor.remove_button.isEnabled())
        self.assertTrue(self.editor.undo_delete_button.isEnabled())
        self.editor.undo_delete_button.click()
        self.assertEqual(self.editor.conditions(), self.original)
        self.assertEqual(len(self.changes), 2)
        self.assertFalse(self.editor.undo_delete_button.isEnabled())

    def test_undo_does_not_overwrite_later_edits_to_other_conditions(self):
        self.editor.table.selectRow(1)
        self.editor._remove()
        self.editor._set(0, "min_score", .65)
        self.editor._undo_delete()
        self.assertEqual(self.editor.conditions()[0].min_score, .65)
        self.assertEqual(self.editor.conditions()[1], self.original[1])

    def test_multiple_deletions_restore_in_reverse_order(self):
        self.editor.table.selectRow(0)
        self.editor._remove()
        self.editor.table.selectRow(0)
        self.editor._remove()
        self.assertEqual(self.editor.conditions(), [])
        self.editor._undo_delete()
        self.editor._undo_delete()
        self.assertEqual(self.editor.conditions(), self.original)

    def test_new_context_cannot_restore_previous_steps_condition(self):
        self.editor.table.selectRow(0)
        self.editor._remove()
        other = [Condition(label="other")]
        self.editor.set_conditions(other)
        self.editor._undo_delete()
        self.assertEqual(self.editor.conditions(), other)
        self.assertFalse(self.editor.undo_delete_button.isEnabled())

    def test_option_rebuild_never_silently_fills_missing_label(self):
        self.editor.set_conditions([Condition()])
        self.editor.set_options(["grip"], [])
        self.assertEqual(self.editor.conditions()[0].label, "")
        self.assertEqual(self.editor.table.cellWidget(0, 1).currentData(), "")


class EditorPreviewAndSummaryTests(QtTests):
    def make_editor(self, sop=None):
        editor = self.widget(EditorPage())
        editor.set_sop(sop or SOPDefinition(steps=[Step("install", conditions=[Condition(label="grip")])]))
        editor.set_classes(["grip"])
        return editor

    def test_brief_flash_is_only_one_frame_not_engine_completion(self):
        editor = self.make_editor()
        engine = SOPEngine(copy.deepcopy(editor.sop))
        engine.start(0)
        engine.update(packet(0, False).result)
        frame = packet(.1)
        editor.on_packet(frame)
        events = engine.update(frame.result)
        self.assertIn("本幀成立，尚未代表工序完成", editor.step_editors[0].live_label.text())
        self.assertIn("時間窗", editor.step_editors[0].live_label.text())
        self.assertIn("80%", editor.step_editors[0].live_label.text())
        self.assertIn("未累積時間", editor.step_editors[0].live_label.text())
        self.assertNotIn("step_completed", [event.kind for event in events])

    def test_sustained_detection_uses_engine_hold_but_preview_never_claims_completion(self):
        editor = self.make_editor()
        engine = SOPEngine(copy.deepcopy(editor.sop))
        engine.start(0)
        events = []
        for index in range(13):
            frame = packet(index / 10)
            editor.on_packet(frame)
            events.extend(engine.update(frame.result))
        self.assertIn("step_completed", [event.kind for event in events])
        self.assertIn("尚未代表工序完成", editor.step_editors[0].live_label.text())

    def test_invalid_detection_shows_unavailable_not_disappearance(self):
        editor = self.make_editor(SOPDefinition(steps=[Step(conditions=[Condition("disappear", "grip")])]))
        editor.on_packet(packet(detected=False, valid=False))
        # Same state as the engine: the frame blocks accumulation instead of meeting "disappear".
        self.assertIn("無法判斷", editor.step_editors[0].live_label.text())
        self.assertNotIn("本幀成立", editor.step_editors[0].live_label.text())
        self.assertIn("模型尚未就緒", editor.step_editors[0].live_label.text())

    def test_manual_step_is_explicit_not_missing_condition_error(self):
        editor = self.make_editor(SOPDefinition(steps=[Step("manual")]))
        self.assertIn("人工確認", editor.step_editors[0].manual_label.text())
        self.assertNotIn("⚠", editor.step_editors[0].summary_label.text())
        editor.on_packet(packet())
        self.assertIn("需手動確認", editor.step_editors[0].live_label.text())

    def test_summary_checks_every_step_and_referenced_fields(self):
        editor = self.make_editor(SOPDefinition(steps=[
            Step("auto", conditions=[Condition(label="grip")]),
            Step("manual"),
            Step("missing object", conditions=[Condition()]),
            Step("missing region", conditions=[Condition(label="wrong", roi="gone")]),
        ]))
        self.assertIn("2/4", editor.setup_summary.text())
        self.assertIn("待修正 2", editor.setup_summary.text())
        self.assertIn("未選物件", editor.step_editors[2].summary_label.text())
        self.assertIn("模型無類別 wrong", editor.step_editors[3].summary_label.text())
        self.assertIn("區域不存在", editor.step_editors[3].summary_label.text())

    def test_following_region_without_reference_is_marked_incomplete(self):
        region = ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)], "workpiece")
        editor = self.make_editor(SOPDefinition(rois=[region], steps=[
            Step(conditions=[Condition(label="grip", roi="left")])]))
        self.assertIn("區域缺少參考工件", editor.step_editors[0].summary_label.text())
        self.assertIn("待修正 1", editor.setup_summary.text())

    def test_region_binding_updates_the_visible_step_without_navigation(self):
        region = ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)])
        editor = self.make_editor(SOPDefinition(rois=[region], steps=[Step("install")]))
        step = editor.step_editors[0]
        editor._open_step_settings(step.step)
        step.completion_method.setCurrentIndex(step.completion_method.findData("auto"))
        cards = step.condition_table
        self.assertTrue(cards.isEnabled())
        self.assertIs(editor.step_settings.current_editor, step)
        label = cards.field_widget(0, "label")
        label.setCurrentIndex(label.findData("grip"))
        field = cards.field_widget(0, "roi")
        field.setCurrentIndex(field.findData("left"))
        self.assertEqual(editor.sop.steps[0].conditions[0].roi, "left")
        self.assertIn("自動", editor.step_editors[0].manual_label.text())
        self.assertIn("自動 1", editor.setup_summary.text())
        self.assertIs(editor.step_settings.current_editor, step)

    def test_delete_and_restore_synchronize_sop_list_and_summary(self):
        editor = self.make_editor()
        editor.step_editors[0].condition_table.select_condition(0)
        editor.step_editors[0].condition_table._remove_row(0)
        self.assertEqual(editor.sop.steps[0].conditions, [])
        self.assertIn("人工確認 1", editor.setup_summary.text())
        editor.step_editors[0].condition_table.undo_delete_button.click()
        self.assertEqual(editor.sop.steps[0].conditions[0].label, "grip")
        self.assertIn("自動 1", editor.setup_summary.text())

    def test_missing_file_is_failure_and_existing_path_is_not_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            editor = self.make_editor(SOPDefinition(model_path=str(Path(directory) / "missing.zip")))
            self.assertEqual(editor._model_state, "failed")
            editor.set_sop(SOPDefinition(model_path=directory))
            self.assertEqual(editor._model_state, "selected")
            self.assertIn("尚未載入", editor.setup_summary.text())
            for state, text in [("loading", "載入中"), ("available", "可用"), ("failed", "失敗")]:
                editor.set_model_state(state, "details")
                self.assertIn(text, editor.setup_summary.text())
                self.assertEqual(editor.setup_summary.toolTip(), "details")

    def test_forbidden_condition_is_also_labelled_one_frame(self):
        editor = self.make_editor(SOPDefinition(steps=[Step(conditions=[Condition(label="grip")],
                                                          forbidden=[Condition(label="grip")])]))
        editor.on_packet(packet())
        self.assertIn("本幀違規條件成立", editor.step_editors[0].live_label.text())



class MainWindowBatchOneTests(QtTests):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for target, kwargs in [
            ("sop_app.ui.main_window.RECORDS_DIR", {"new": self.root / "records"}),
            ("sop_app.ui.main_window.MainWindow._load_settings", {"return_value": {}}),
            ("sop_app.ui.main_window.MainWindow._save_settings", {}),
            ("sop_app.ui.main_window.VideoPipeline.start", {}),
        ]:
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        with patch("sop_app.ui.main_window.cv2.require", return_value=Mock()):
            self.window = MainWindow(defer_initial_sop=True)
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.close_window)
        self.window._apply_sop(SOPDefinition(name="draft", version=3,
                                           steps=[Step("install", conditions=[Condition(label="grip")])]), None)
        self.window._on_modified()
        self.results = []
        self.window.save_finished.connect(self.results.append)

    def close_window(self):
        self.window.dirty = False
        self.window.close()

    def test_pending_save_ignores_repeated_requests_while_dialog_is_open(self):
        calls = []

        def dialog(*_args, **_kwargs):
            calls.append(True)
            self.assertTrue(self.window._save_pending)
            self.assertFalse(self.window.save_button.isEnabled())
            self.assertFalse(self.window._request_save())   # e.g. Ctrl+S during the modal dialog
            return "", ""

        with patch("sop_app.ui.main_window.QFileDialog.getSaveFileName", side_effect=dialog):
            self.window.save_button.click()
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.results, ["cancelled"])
        self.assertFalse(self.window._save_pending)
        self.assertTrue(self.window.save_button.isEnabled())

    def test_cancel_save_as_keeps_guide_content_dirty_and_path(self):
        before = copy.deepcopy(self.window.editor.sop)
        with patch("sop_app.ui.main_window.QFileDialog.getSaveFileName", return_value=("", "")):
            self.window.save_button.click()
        self.assertEqual(self.results, ["cancelled"])
        self.assertFalse(self.window._save_pending)
        self.assertEqual(self.window.editor.sop, before)
        self.assertTrue(self.window.dirty)
        self.assertIsNone(self.window.sop_path)
        self.assertIn("已取消儲存", self.window.editor.guide_label.text())

    def test_failed_atomic_write_keeps_version_then_retry_increments_once(self):
        path = self.root / "existing.json"
        save_sop(self.window.editor.sop, path)
        self.window.sop_path = path
        original_object = self.window.editor.sop
        with patch("sop_app.sop_schema.os.replace", side_effect=PermissionError("test denied")), \
             patch("sop_app.ui.main_window.QMessageBox.warning"):
            self.window.save_button.click()
        self.assertEqual(self.results, ["failed"])
        self.assertEqual(self.window.editor.sop.version, 3)
        self.assertEqual(load_sop(path).version, 3)
        self.assertEqual(self.window.editor.version_label.text(), "3")
        self.assertFalse(self.window._save_pending)
        self.assertTrue(self.window.dirty)
        self.window.save_button.click()
        self.assertEqual(self.results, ["failed", "saved"])
        self.assertIs(self.window.editor.sop, original_object)
        self.assertEqual(self.window.editor.sop.version, 4)
        self.assertEqual(load_sop(path).version, 4)
        self.assertEqual(self.window.editor.version_label.text(), "4")
        self.assertFalse(self.window._save_pending)
        self.assertFalse(self.window.dirty)

    def test_new_save_succeeds_without_bumping_version(self):
        path = self.root / "new.json"
        with patch("sop_app.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(path), "")):
            self.window.save_button.click()
        self.assertEqual(load_sop(path).version, 3)
        self.assertEqual(self.window.sop_path, path)
        self.assertEqual(self.results, ["saved"])
        self.assertFalse(self.window._save_pending)
        self.assertFalse(self.window.dirty)

    def test_failed_save_as_never_replaces_current_path(self):
        original = self.root / "original.json"
        self.window.sop_path = original
        with patch("sop_app.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(self.root / "new.json"), "")), \
             patch("sop_app.ui.main_window.save_sop", side_effect=OSError("write failed")), \
             patch("sop_app.ui.main_window.QMessageBox.warning"):
            self.assertFalse(self.window._save_as())
        self.assertEqual(self.window.sop_path, original)
        self.assertEqual(self.window.editor.sop.version, 3)
        self.assertTrue(self.window.dirty)

    def test_serialization_error_is_reported_and_keeps_dirty(self):
        with patch("sop_app.ui.main_window.save_sop", side_effect=TypeError("bad draft")), \
             patch("sop_app.ui.main_window.QMessageBox.warning"):
            self.assertFalse(self.window._write(self.root / "new.json"))
        self.assertEqual(self.results, ["failed"])
        self.assertTrue(self.window.dirty)

    def test_clean_save_does_not_increment_existing_version(self):
        path = self.root / "existing.json"
        save_sop(self.window.editor.sop, path)
        self.window.dirty = False
        self.assertTrue(self.window._write(path))
        self.assertEqual(load_sop(path).version, 3)

    def choose_model(self, path=None):
        path = path or self.root
        self.window.editor.sop.model_path = str(path)
        info = model_info(path)
        with patch("sop_app.ui.main_window.read_model_info", return_value=info), \
             patch.object(self.window.pipeline, "load_model") as loader:
            self.window._load_model(str(path))
            loader.assert_called_once_with(str(path))
        return info

    def test_model_is_usable_only_after_success_callback_and_failure_allows_retry(self):
        info = self.choose_model()
        self.assertFalse(self.window.model_ready)
        self.assertEqual(self.window.editor._model_state, "loading")
        self.window._on_model_loaded(info, "CPU", str(self.root))
        self.assertTrue(self.window.model_ready)
        self.assertEqual(self.window.editor._model_state, "available")
        self.window._on_model_loaded(None, "", str(self.root))
        self.assertFalse(self.window.model_ready)
        self.assertIsNone(self.window.model_info)
        self.assertIsNone(self.window._requested_model)
        self.assertEqual(self.window.editor._model_state, "failed")
        self.choose_model()
        self.assertEqual(self.window.editor._model_state, "loading")

    def test_invalid_new_bundle_does_not_keep_previous_model_ready(self):
        info = self.choose_model()
        self.window._on_model_loaded(info, "CPU", str(self.root))
        missing = self.root / "missing.zip"
        self.window.editor.sop.model_path = str(missing)
        with patch("sop_app.ui.main_window.QMessageBox.warning"):
            self.window._load_model(str(missing))
        self.assertFalse(self.window.model_ready)
        self.assertIsNone(self.window.model_info)
        self.assertEqual(self.window.editor._model_state, "failed")
        self.assertIn("找不到模型", self.window.editor.setup_summary.toolTip())
        self.assertEqual(self.window.editor.classes, [])

    def test_stale_load_events_cannot_override_latest_model_selection(self):
        info = self.choose_model()
        newer = self.root / "newer"
        self.window.editor.sop.model_path = str(newer)
        self.window.editor.set_model_state("loading", "new request")
        self.window._on_model_loaded(info, "CPU", str(self.root))
        self.window._on_model_loaded(None, "", str(self.root))
        self.window._on_model_progress("載入失敗", str(self.root))
        self.assertFalse(self.window.model_ready)
        self.assertEqual(self.window.editor._model_state, "loading")
        self.assertEqual(self.window.editor._model_detail, "new request")

    def test_model_json_selection_matches_bundle_directory_source(self):
        chosen = self.root / "model.json"
        self.window.editor.sop.model_path = str(chosen)
        self.window._on_model_loaded(model_info(self.root), "CPU", str(chosen))
        self.assertTrue(self.window.model_ready)

    def test_native_runtime_failure_is_explicit_and_does_not_queue_model(self):
        self.window._vision_error = "OpenCV unavailable (test)"
        self.window.editor.sop.model_path = str(self.root)
        with patch("sop_app.ui.main_window.read_model_info", return_value=model_info(self.root)), \
             patch.object(self.window.pipeline, "load_model") as loader:
            self.window._load_model(str(self.root))
        loader.assert_not_called()
        self.assertFalse(self.window.model_ready)
        self.assertEqual(self.window.editor._model_state, "failed")
        self.assertIn("OpenCV unavailable", self.window.editor.setup_summary.toolTip())

    def test_opening_sop_without_model_clears_previous_state(self):
        info = self.choose_model()
        self.window._on_model_loaded(info, "CPU", str(self.root))
        with patch.object(self.window.pipeline, "clear_model") as clear:
            self.window._apply_sop(SOPDefinition(steps=[Step("manual")]), None)
        clear.assert_called_once()
        self.assertFalse(self.window.model_ready)
        self.assertEqual(self.window.editor._model_state, "unselected")
        self.assertEqual(self.window.editor.classes, [])

    def test_pending_clear_never_previews_previous_models_detections(self):
        self.window.tabs.setCurrentWidget(self.window.editor)
        with patch.object(self.window.editor, "on_packet") as preview, \
             patch.object(self.window.pipeline, "acknowledge_packet") as acknowledge:
            self.window._on_packet(packet())
        frame = preview.call_args.args[0]
        self.assertFalse(frame.result.detection_valid)
        self.assertEqual(frame.result.detections, [])
        acknowledge.assert_called_once()

    def editing_workflow(self, width, height):
        """Edit individual steps, draw/bind a region, and save the actual UI draft."""
        definition = SOPDefinition(name="workflow", steps=[
            Step(f"step {index}", conditions=[Condition(label="grip")]) for index in range(1, 5)])
        self.window._apply_sop(definition, None)
        editor = self.window.editor
        editor.sop.model_path = str(self.root)
        self.window._on_model_loaded(model_info(self.root, ("grip", "part")), "CPU (simulated)", str(self.root))
        self.window.tabs.setCurrentWidget(editor)
        self.window.resize(width, height)
        self.window.show()
        self.app.processEvents()
        editor._open_step_settings(editor.sop.steps[0])
        self.app.processEvents()
        QTest.mouseClick(editor.step_editors[0].duplicate_button, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertEqual(len(editor.sop.steps), 5)
        editor.step_settings.close_button.click()
        target = editor.step_editors[1]
        editor._open_step_settings(target.step)
        self.app.processEvents()
        field = target.condition_table.field_widget(0, "label")
        editor.step_settings.scroll.ensureWidgetVisible(field)
        field.setFocus()
        QTest.keyClick(field, Qt.Key.Key_End)
        self.assertEqual(editor.sop.steps[1].conditions[0].label, "part")
        self.assertEqual(editor.sop.steps[0].conditions[0].label, "grip")
        editor.step_settings.close_button.click()
        editor.on_packet(packet())
        editor.freeze_check.setChecked(True)
        editor.draw_button.click()
        self.app.processEvents()
        start = editor.video._to_widget(editor.video._image_rect(), .05, .1).toPoint()
        end = editor.video._to_widget(editor.video._image_rect(), .25, .5).toPoint()
        QTest.mousePress(editor.video, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(editor.video, end)
        QTest.mouseRelease(editor.video, Qt.MouseButton.LeftButton, pos=end)
        self.assertEqual(len(editor.sop.rois), 1)
        row = editor.roi_rows[0]
        row.name_edit.setText("new position")
        row.name_edit.editingFinished.emit()
        editor._open_step_settings(editor.sop.steps[1])
        field = editor.step_editors[1].condition_table.field_widget(0, "roi")
        field.setCurrentIndex(field.findData("new position"))
        self.assertEqual(editor.sop.steps[1].conditions[0].roi, "new position")
        editor.step_settings.close_button.click()
        editor._return_live()
        frame = packet()
        frame.result.detections = [Detection("part", .9, (10, 10, 30, 30))]
        editor.on_packet(frame)
        self.app.processEvents()
        self.assertIn("本幀成立", editor.step_editors[1].live_label.text())
        self.assertEqual((self.window.width(), self.window.height()), (width, height))
        saved_path = self.root / "workflow.json"
        with patch("sop_app.ui.main_window.QFileDialog.getSaveFileName", return_value=(str(saved_path), "")):
            self.window.save_button.click()
        restored = load_sop(saved_path)
        self.assertEqual(len(restored.steps), 5)
        self.assertEqual(restored.steps[1].conditions[0].roi, "new position")
        self.assertFalse(self.window.dirty)
        self.assertFalse(self.window._save_pending)

    def test_mouse_keyboard_workflow_at_1366x768(self):
        self.editing_workflow(1366, 768)

    def test_mouse_keyboard_workflow_at_1920x1080(self):
        self.editing_workflow(1920, 1080)


class PipelineModelStateTests(QtTests):
    def test_clear_model_command_removes_detector_without_confirming_step(self):
        pipeline = VideoPipeline(Path("unused"))
        detector, engine = Mock(), Mock()
        pipeline._detector, pipeline._engine = detector, engine
        pipeline.clear_model()
        command, payload = pipeline._commands.get_nowait()
        pipeline._handle_command(command, payload, Mock())
        self.assertIsNone(pipeline._detector)
        engine.confirm_step.assert_not_called()

    def test_load_success_emits_source_for_progress_and_readiness(self):
        pipeline = VideoPipeline(Path("unused"))
        path = Path("example-model")
        info = model_info(path)
        detector = Mock(device_name="CPU")
        worker = Mock()
        worker.load_model.return_value = detector
        pipeline._inference_worker = worker
        progress, loaded = [], []
        pipeline.model_progress.connect(lambda *args: progress.append(args))
        pipeline.model_loaded.connect(lambda *args: loaded.append(args))
        with patch("sop_app.pipeline.read_model_info", return_value=info):
            pipeline._load_model(str(path))
        self.assertEqual(loaded, [(info, "CPU", str(path))])
        self.assertTrue(progress)
        self.assertTrue(all(source == str(path) for _stage, source in progress))

    def test_load_failure_emits_its_source_and_clears_detector(self):
        pipeline = VideoPipeline(Path("unused"))
        loaded = []
        pipeline.model_loaded.connect(lambda *args: loaded.append(args))
        with patch("sop_app.pipeline.read_model_info", side_effect=ModelBundleError("bad model")):
            with self.assertRaisesRegex(RuntimeError, "bad model"):
                pipeline._load_model("bad.zip")
        self.assertEqual(loaded, [(None, "", "bad.zip")])
        self.assertIsNone(pipeline._detector)

    def test_inference_failure_emits_source_and_invalid_frame_without_crashing(self):
        pipeline = VideoPipeline(Path("unused"))
        pipeline._detector = Mock(info=model_info(Path("failed-model")))
        pipeline._detector.detect.side_effect = RuntimeError("inference failed")
        pipeline._source = pipeline._engine = pipeline._locator = None
        pipeline._show_masks, pipeline._min_score = False, .5
        pipeline._fps, pipeline._last_frame_at = 0, None
        loaded, frames = [], []
        pipeline.model_loaded.connect(lambda *args: loaded.append(args))
        pipeline.packet_ready.connect(frames.append)
        pipeline._process(np.zeros((100, 200, 3), np.uint8), 0, Mock())
        self.assertEqual(loaded, [(None, "", "failed-model")])
        self.assertIsNone(pipeline._detector)
        self.assertFalse(frames[0].result.detection_valid)
        self.assertEqual(frames[0].result.detections, [])


if __name__ == "__main__":
    unittest.main()
