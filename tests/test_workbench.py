"""Native workbench layout and isolated validation regressions."""
from __future__ import annotations

import copy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication, QListWidget, QStackedWidget, QTabWidget, QToolButton

from sop_app.detection import Detection, FrameResult
from sop_app.engine import Phase, SOPEngine
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, CycleSettings, ROI, SOPDefinition, Step
from sop_app.ui.condition_rows import ConditionRows
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.editor_validation import EditorValidation
from sop_app.ui.main_window import MainWindow



def result(timestamp, *labels, valid=True):
    return FrameResult(200, 100, timestamp, [Detection(label, .9, (10, 10, 30, 30)) for label in labels],
                       detection_valid=valid)


def packet(timestamp, *labels, valid=True):
    image = np.zeros((100, 200, 3), np.uint8)
    return FramePacket(image, image.copy(), result(timestamp, *labels, valid=valid), None, 10, 5)


def shown_field(table, row, field):
    """The input, or the reason shown in its place when the field does not apply."""
    widget = table.field_widget(row, field)
    note = table._fields[row].get(f"{field}_note")
    return note if note is not None and note.isVisible() else widget


def sop(*steps, **cycle):
    return SOPDefinition(name="試跑測試", steps=list(steps) or [Step("安裝", conditions=[Condition(label="grip")])],
                         cycle=CycleSettings(**cycle))


class ValidationAdapterTests(unittest.TestCase):
    def test_single_uses_same_engine_and_does_not_mutate_sop(self):
        source = sop(Step("第一道"), Step("第二道", conditions=[Condition(label="grip")], hold_sec=.5))
        original = copy.deepcopy(source)
        trial = EditorValidation()
        trial.start(source, "single", 1, 0)
        self.assertIsInstance(trial.engine, SOPEngine)
        self.assertEqual(trial.original_indices, [1])
        for timestamp in (.1, .2, .3, .4, .5):
            trial.update(result(timestamp, "grip"))
        self.assertTrue(trial.completed)
        self.assertFalse(trial.running)
        self.assertEqual(source, original)
        self.assertIn("工序 2", trial.metrics()["target"])
        self.assertIn("達標", trial.metrics()["ratio"])

    def test_brief_flash_cannot_complete_hold_window(self):
        trial = EditorValidation()
        trial.start(sop(), "single", 0, 0)
        trial.update(result(.1, "grip"))
        trial.update(result(.2))
        self.assertTrue(trial.running)
        self.assertFalse(trial.completed)
        self.assertEqual(trial.metrics()["ratio"], "50% / ≥ 80%")

    def test_any_completion_mode_is_preserved(self):
        source = sop(Step(conditions=[Condition(label="grip"), Condition(label="tool")],
                          completion_mode="any", hold_sec=0))
        trial = EditorValidation()
        trial.start(source, "single", 0, 0)
        trial.update(result(.1, "grip"))
        self.assertTrue(trial.completed)
        self.assertEqual(trial.metrics()["count"], "1/2")

    def test_flow_preserves_existing_auto_restart_rules(self):
        source = sop(Step("一", conditions=[Condition(label="grip")], hold_sec=0),
                     Step("二", conditions=[Condition(label="tool")], hold_sec=0), auto_restart_sec=.3)
        trial = EditorValidation()
        trial.start(source, "flow", 1, 0)
        trial.update(result(.1, "grip"))
        trial.update(result(.2, "tool"))
        self.assertEqual(trial.engine.phase, Phase.CYCLE_DONE)
        self.assertTrue(trial.running)
        trial.update(result(.4))
        self.assertEqual(trial.engine.cycle, 1)
        trial.update(result(.6))
        self.assertEqual(trial.engine.cycle, 2)
        self.assertEqual(trial.engine.index, 0)

    def test_flow_reset_never_treats_model_failure_as_disappearance(self):
        source = sop(Step(conditions=[Condition(label="grip")], hold_sec=0),
                     reset_conditions=[Condition("disappear", "grip")], reset_hold_sec=.5)
        trial = EditorValidation()
        trial.start(source, "flow", 0, 0)
        trial.update(result(.1, "grip"))
        trial.update(result(.6, valid=False))
        self.assertEqual(trial.engine.cycle, 1)
        self.assertEqual(trial.metrics()["count"], "無法判斷")
        self.assertEqual(trial.metrics()["ratio"], "—")
        for t in (.7, .8, .9, 1, 1.1, 1.2, 1.3):
            trial.update(result(t))
        self.assertEqual(trial.engine.cycle, 2)

    def test_repeated_timestamps_do_not_accumulate_samples(self):
        trial = EditorValidation()
        trial.start(sop(), "single", 0, 0)
        for _ in range(30):
            trial.update(result(0, "grip"))
        self.assertEqual(len(trial.engine._window._samples), 0)
        self.assertFalse(trial.completed)

    def test_time_rewind_invalidates_trial(self):
        trial = EditorValidation()
        trial.start(sop(), "single", 0, 5)
        trial.update(result(4, "grip"))
        self.assertTrue(trial.stale)
        self.assertFalse(trial.running)
        self.assertEqual(trial.metrics()["ratio"], "—")

    def test_manual_confirm_never_skips_an_automatic_step(self):
        trial = EditorValidation()
        trial.start(sop(), "single", 0, 0)
        trial.confirm()
        self.assertTrue(trial.running)
        self.assertFalse(trial.completed)
        self.assertTrue(trial.engine.cycle_ok)
        trial.start(sop(Step("人工")), "single", 0, 0)
        trial.confirm()
        self.assertTrue(trial.completed)
        self.assertIn("OK", trial.message)

    def test_timeout_and_violation_events_use_original_engine(self):
        source = sop(Step(conditions=[Condition(label="grip")], forbidden=[Condition(label="tool")],
                          hold_sec=.1, timeout_sec=.2))
        trial = EditorValidation()
        trial.start(source, "single", 0, 0)
        for t in (.1, .2, .3):
            trial.update(result(t, "tool"))
        alarms = [event.data.get("alarm") for event in trial.events if event.kind == "alarm"]
        self.assertIn("violation", alarms)
        self.assertIn("timeout", alarms)
        self.assertFalse(trial.engine.cycle_ok)

    def test_stop_and_invalidate_preserve_events_but_hide_old_values(self):
        trial = EditorValidation()
        trial.start(sop(), "single", 0, 0)
        trial.update(result(.1, "grip"))
        before = len(trial.events)
        trial.invalidate()
        self.assertGreaterEqual(len(trial.events), before)
        self.assertEqual(trial.metrics()["count"], "—")
        self.assertIn("已過期", trial.metrics()["target"])


class WorkbenchQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if os.name == "nt":
            font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msjh.ttc")
            families = QFontDatabase.applicationFontFamilies(font_id)
            QFontDatabase.addApplicationFont("C:/Windows/Fonts/msjhbd.ttc")
            QFontDatabase.addApplicationFont("C:/Windows/Fonts/seguisym.ttf")
            if families:
                cls.app.setFont(QFont(families[0], 9))

    def make_editor(self, definition=None, available=True):
        editor = EditorPage()
        self.addCleanup(editor.deleteLater)
        self.addCleanup(editor.close)
        editor.set_sop(definition or sop())
        editor.set_classes(["grip", "tool"])
        if available:
            editor.set_model_state("available")
        editor.resize(1300, 740)
        editor.show()
        self.app.processEvents()
        editor.on_packet(packet(0, "grip"))
        return editor

    def test_no_automatic_trial_and_each_step_uses_one_settings_page(self):
        editor = self.make_editor(sop(Step("第一道"), Step("第二道")))
        self.assertIsNone(editor._validation.engine)
        self.assertFalse(editor._validation.running)
        self.assertEqual(len(editor.step_editors), 2)
        self.assertTrue(editor.page_scroll.widget().isVisible())
        self.assertFalse(editor.findChildren(QTabWidget))
        self.assertFalse(editor.findChildren(QStackedWidget))
        self.assertFalse(editor.findChildren(QListWidget))
        self.assertFalse(editor.findChildren(QToolButton))
        for card in editor.step_editors:
            self.assertTrue(card.step_name_edit.isVisible())
            self.assertIn("完成", card.header_row.toolTip())
        editor.edit_steps_button.setChecked(True)
        for card in editor.step_editors:
            self.assertTrue(card.body.isHidden())
            editor._open_step_settings(card.step)
            self.app.processEvents()
            self.assertIs(editor.step_settings.current_editor, card)
            self.assertTrue(editor.step_settings.scroll.isAncestorOf(card.body))
            self.assertTrue(card.instruction_edit.isVisible())
            self.assertTrue(card.forbidden_table.isVisible())
        editor.step_settings.close_button.click()

    def test_persistent_trial_survives_scrolling_between_settings(self):
        editor = self.make_editor()
        editor.validation_start.click()
        engine = editor._validation.engine
        for index, widget in enumerate((editor.name_edit, editor.step_editors[0], editor.reset_table,
                                        editor.validation_box)):
            scroll = editor.tools_scroll if widget is editor.validation_box else editor.page_scroll
            scroll.ensureWidgetVisible(widget)
            self.app.processEvents()
            editor.on_packet(packet(.1 + .1 * index, "grip"))
            self.assertTrue(editor.validation_box.isVisible())
            self.assertIs(editor._validation.engine, engine)
            self.assertTrue(editor._validation.running)

    def test_edits_invalidate_old_result_and_retry_captures_current_settings(self):
        editor = self.make_editor()
        editor._open_step_settings(editor.sop.steps[0])
        editor.validation_start.click()
        editor.on_packet(packet(.1, "grip"))
        original_engine = editor._validation.engine
        field = editor.step_editors[0].condition_table.field_widget(0, "min_count")
        field.setValue(2)
        self.assertTrue(editor._validation.stale)
        self.assertIn("設定已變更", editor.validation_status.text())
        self.assertIn("—", editor.validation_ratio.text())
        self.assertEqual(original_engine.sop.steps[0].conditions[0].min_count, 1)
        editor.validation_start.click()
        self.assertFalse(editor._validation.stale)
        self.assertEqual(editor._validation.engine.sop.steps[0].conditions[0].min_count, 2)

    def test_single_target_does_not_follow_editor_selection(self):
        editor = self.make_editor(sop(Step("一", conditions=[Condition(label="grip")]), Step("二")))
        editor.validation_start.click()
        editor.step_editors[1].step_name_edit.setFocus()
        self.app.processEvents()
        self.assertIn("工序 1", editor.validation_target.text())
        self.assertTrue(editor._validation.running)
        self.assertEqual(editor.step_editors[1].step_name_edit.text(), "二")

    def test_flow_starts_at_first_step_even_if_another_is_selected(self):
        editor = self.make_editor(sop(Step("一", conditions=[Condition(label="grip")]), Step("二")))
        editor.validation_step.setCurrentIndex(1)
        editor.validation_mode.setCurrentIndex(1)
        editor.validation_start.click()
        self.assertEqual(editor._validation.original_indices, [0, 1])
        self.assertIn("工序 1", editor.validation_target.text())

    def test_failed_restart_does_not_display_previous_completed_values(self):
        editor = self.make_editor(sop(Step("有效", conditions=[Condition(label="grip")], hold_sec=0),
                                      Step("缺類別", conditions=[Condition(label="unknown")], hold_sec=0)))
        editor.validation_start.click()
        editor.on_packet(packet(.1, "grip"))
        self.assertTrue(editor._validation.completed)
        editor.validation_step.setCurrentIndex(1)
        editor.validation_start.click()
        self.assertIn("無法開始試跑", editor.validation_status.text())
        self.assertTrue(editor._validation.stale)
        self.assertIn("—", editor.validation_ratio.text())
        label = editor.step_editors[1].condition_table.field_widget(0, "label")
        editor._open_step_settings(editor.sop.steps[1])
        label.setCurrentIndex(label.findData("grip"))
        editor.validation_start.click()
        self.assertTrue(editor._validation.running)
        self.assertIn("工序 2", editor.validation_target.text())

    def test_missing_model_blocks_auto_trial_but_preserves_editing(self):
        editor = self.make_editor(available=False)
        self.assertFalse(editor.validation_start.isEnabled())
        editor._start_validation()
        self.assertIsNone(editor._validation.engine)
        self.assertIn("模型尚未就緒", editor.validation_status.text())
        editor._open_step_settings(editor.sop.steps[0])
        editor.step_editors[0].condition_table._add()
        self.assertEqual(len(editor.sop.steps[0].conditions), 2)

    def test_manual_confirmation_uses_engine_without_model(self):
        editor = self.make_editor(sop(Step("人工")), available=False)
        editor.on_packet(packet(.1, valid=False))
        editor.validation_start.click()
        self.assertTrue(editor.validation_confirm.isVisible())
        editor.validation_confirm.click()
        self.assertTrue(editor._validation.completed)
        self.assertIn("OK", editor.validation_status.text())

    def test_freeze_model_change_and_source_change_invalidate(self):
        editor = self.make_editor()
        editor.validation_start.click()
        editor.freeze_check.setChecked(True)
        self.assertTrue(editor._validation.stale)
        self.assertFalse(editor.validation_start.isEnabled())
        editor._return_live()
        editor.on_packet(packet(.1, "grip"))
        editor.validation_start.click()
        editor.set_model_state("failed")
        self.assertTrue(editor._validation.stale)
        editor.set_model_state("available")
        editor.validation_start.click()
        editor.clear_video("source ended")
        self.assertTrue(editor._validation.stale)
        self.assertFalse(editor.validation_start.isEnabled())

    def test_silent_source_stall_is_not_counted_as_hold_time(self):
        editor = self.make_editor()
        editor.validation_start.click()
        with patch("sop_app.ui.editor_page.time.monotonic", return_value=editor._packet_arrived_at + 4):
            editor._check_validation_feed()
        self.assertTrue(editor._validation.stale)
        self.assertIn("影像停止更新", editor.validation_status.text())

    def test_production_monitor_blocks_editor_trial(self):
        editor = self.make_editor()
        editor.validation_start.click()
        editor.set_production_running(True)
        self.assertTrue(editor._validation.stale)
        self.assertFalse(editor.validation_start.isEnabled())
        editor.set_production_running(False)
        self.assertTrue(editor.validation_start.isEnabled())

    def test_trial_writes_no_records_and_is_independent_of_production_engine(self):
        editor = self.make_editor()
        production = SOPEngine(copy.deepcopy(editor.sop))
        production.start(0)
        before = production.snapshot()
        with patch("sop_app.recorder.Recorder.handle") as record:
            editor.validation_start.click()
            for index in range(1, 13):
                editor.on_packet(packet(index / 10, "grip"))
        record.assert_not_called()
        self.assertEqual(production.snapshot(), before)
        self.assertTrue(editor._validation.completed)

    def test_hidden_editor_stops_trial_and_new_sop_discards_old_snapshot(self):
        editor = self.make_editor()
        editor.validation_start.click()
        editor.hide()
        self.assertFalse(editor._validation.running)
        editor.show()
        editor.set_sop(sop(Step("新工序")))
        self.assertIsNone(editor._validation.engine)
        self.assertIsNone(editor.video._image)

    def test_validation_details_are_always_available_on_the_same_page(self):
        editor = self.make_editor()
        self.assertTrue(editor.validation_details.isVisible())
        self.assertFalse(editor.findChildren(QToolButton))
        editor.tools_scroll.ensureWidgetVisible(editor.validation_details)
        self.app.processEvents()
        self.assertGreater(editor.validation_details.height(), 60)
        self.assertEqual(editor.page_scroll.horizontalScrollBar().maximum(), 0)

    def test_status_is_readable_and_preview_has_useful_height(self):
        editor = self.make_editor()
        self.assertLess(editor.setup_summary.height(), 100)
        self.assertGreaterEqual(editor.video.height(), 180)
        self.assertEqual(editor.page_scroll.horizontalScrollBar().maximum(), 0)

    def test_option_refresh_keeps_all_fields_visible_with_real_height(self):
        editor = self.make_editor()
        editor._open_step_settings(editor.sop.steps[0])
        self.app.processEvents()
        table = editor.step_editors[0].condition_table
        score = table.field_widget(0, "min_score")
        editor._refresh_options()
        self.app.processEvents()
        self.assertIs(table.field_widget(0, "min_score"), score)
        for field in ("label", "type", "roi", "min_count", "min_score", "roi_overlap"):
            widget = shown_field(table, 0, field)
            self.assertTrue(widget.isVisible())
            self.assertGreaterEqual(widget.height(), 16)

    def test_each_step_trial_button_uses_its_own_step(self):
        editor = self.make_editor(sop(Step("一"), Step("二")))
        editor._open_step_settings(editor.sop.steps[1])
        editor.step_editors[1].trial_button.click()
        self.assertEqual(editor._validation.original_indices, [1])
        self.assertIn("工序 2", editor.validation_target.text())
        self.assertTrue(editor.step_editors[0].step_name_edit.isVisible())
        self.assertTrue(editor.step_editors[1].step_name_edit.isVisible())

    def test_cycle_ui_explains_real_rules_instead_of_adding_manual_restart_mode(self):
        editor = self.make_editor()
        self.assertTrue(editor.auto_restart_spin.isEnabled())
        self.assertFalse(editor.reset_hold_spin.isEnabled())
        editor.reset_table._add()
        self.assertFalse(editor.auto_restart_spin.isEnabled())
        self.assertTrue(editor.reset_hold_spin.isEnabled())
        self.assertIn("80%", editor.cycle_rule_summary.text())

    def test_card_delete_undo_and_missing_object_are_explicit(self):
        cards = ConditionRows()
        self.addCleanup(cards.close)
        values = [Condition(), Condition(label="grip", min_count=2, min_score=.8)]
        cards.set_options(["grip"], [])
        cards.set_conditions(values)
        self.assertEqual(cards.conditions()[0].label, "")
        self.assertIn("尚未選擇", cards.field_widget(0, "error").text())
        cards.field_widget(1, "remove").click()
        self.assertEqual(cards.conditions(), values[:1])
        cards.undo_delete_button.click()
        self.assertEqual(cards.conditions(), values)
        cards.set_conditions([Condition(label="other")])
        cards._undo_delete()
        self.assertEqual(len(cards.conditions()), 1)
        self.assertIn("目前模型沒有", cards.field_widget(0, "error").text())

    def test_condition_thresholds_are_immediately_editable(self):
        editor = self.make_editor()
        editor._open_step_settings(editor.sop.steps[0])
        self.app.processEvents()
        table = editor.step_editors[0].condition_table
        score = table.field_widget(0, "min_score")
        self.assertTrue(score.isVisible())
        self.assertGreaterEqual(score.height(), 20)
        score.setValue(.75)
        self.assertEqual(editor.sop.steps[0].conditions[0].min_score, .75)
        self.assertTrue(shown_field(table, 0, "roi_overlap").isVisible())
        self.assertEqual(table.field_widget(0, "roi_overlap_note").text(), "不適用")
        self.assertFalse(editor.findChildren(QToolButton))

    def test_dark_palette_keeps_cards_and_validation_visible(self):
        editor = self.make_editor()
        editor._open_step_settings(editor.sop.steps[0])
        original = QPalette(self.app.palette())
        self.addCleanup(self.app.setPalette, original)
        palette = QPalette(original)
        palette.setColor(QPalette.ColorRole.Window, QColor("#171e28"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#e2eaf5"))
        self.app.setPalette(palette)
        self.app.processEvents()
        self.assertTrue(editor._theme_dark)
        self.assertTrue(editor.validation_box.isVisible())
        self.assertGreater(editor.step_editors[0].condition_table.field_widget(0, "label").height(), 20)

    def test_card_edits_preserve_focus_and_highlight_bound_region(self):
        definition = sop()
        definition.rois = [ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)])]
        definition.steps[0].conditions[0].roi = "left"
        editor = self.make_editor(definition)
        editor._open_step_settings(editor.sop.steps[0])
        self.app.processEvents()
        widget = editor.step_editors[0].condition_table.field_widget(0, "min_count")
        widget.setFocus()
        widget.setValue(2)
        self.assertTrue(widget.hasFocus())
        self.assertEqual(editor.video._selected, "left")
        self.assertEqual(editor.sop.steps[0].conditions[0].min_count, 2)

    def test_native_window_layout_at_both_resolutions_throughout_single_page(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch("sop_app.ui.main_window.RECORDS_DIR", Path(folder)), \
             patch.object(MainWindow, "_load_settings", return_value={}), \
             patch.object(MainWindow, "_save_settings"), \
             patch("sop_app.ui.main_window.VideoPipeline.start"), \
             patch("sop_app.ui.main_window.cv2.require", return_value=Mock()):
            window = MainWindow(defer_initial_sop=True)
            try:
                definition = sop(Step("取 grip", conditions=[Condition(label="grip")]),
                                 Step("左側豎管把手", conditions=[Condition(label="grip")]),
                                 Step("取 grip", conditions=[Condition(label="grip")]),
                                 Step("右側豎管把手", conditions=[Condition(label="grip")]))
                window._apply_sop(definition, None)
                editor = window.editor
                editor.set_classes(["grip", "tool"])
                window.tabs.setCurrentWidget(editor)
                window.show()
                for width, height in ((1366, 768), (1920, 1080)):
                    window.resize(width, height)
                    self.app.processEvents()
                    self.assertEqual((window.width(), window.height()), (width, height))
                    self.assertTrue(editor.validation_details.isVisible())
                    self.assertGreaterEqual(editor.video.height(), 180)
                    self.assertEqual(editor.page_scroll.horizontalScrollBar().maximum(), 0)
                    self.assertEqual(len(editor.step_editors), 4)
                    image_geometry = editor.video.geometry()
                    for card in editor.step_editors:
                        self.assertTrue(card.body.isHidden())
                        editor._open_step_settings(card.step)
                        self.app.processEvents()
                        self.assertIs(editor.step_settings.current_editor, card)
                        self.assertTrue(card.instruction_edit.isVisible())
                        self.assertTrue(card.forbidden_table.isVisible())
                        self.assertEqual(editor.video.geometry(), image_geometry)
                        self.assertEqual(editor.step_settings.scroll.horizontalScrollBar().maximum(), 0)
                        for field in ("label", "type", "roi", "min_count", "min_score", "roi_overlap"):
                            self.assertGreaterEqual(shown_field(card.condition_table, 0, field).height(), 16)
                        editor.step_settings.close_button.click()
                    for location, target in (("top", editor.name_edit), ("bottom", editor.validation_details)):
                        scroll = editor.tools_scroll if location == "bottom" else editor.page_scroll
                        scroll.ensureWidgetVisible(target)
                        self.app.processEvents()
                        self.assertEqual(editor.page_scroll.horizontalScrollBar().maximum(), 0)
                        artifact_directory = os.environ.get("SOP_EDITOR_TEST_ARTIFACT_DIR")
                        if artifact_directory:
                            directory = Path(artifact_directory)
                            directory.mkdir(parents=True, exist_ok=True)
                            self.assertTrue(window.grab().save(str(directory / f"workbench-{width}x{height}-{location}.png")))
            finally:
                window.dirty = False
                window.close()


if __name__ == "__main__":
    unittest.main()
