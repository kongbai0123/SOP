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
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sop_app.detection import Detection, FrameResult
from sop_app.engine import Phase, SOPEngine
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, CycleSettings, ROI, SOPDefinition, Step
from sop_app.ui.condition_cards import ConditionCards
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.editor_validation import EditorValidation
from sop_app.ui.main_window import MainWindow


def result(timestamp, *labels, valid=True):
    return FrameResult(200, 100, timestamp, [Detection(label, .9, (10, 10, 30, 30)) for label in labels],
                       detection_valid=valid)


def packet(timestamp, *labels, valid=True):
    image = np.zeros((100, 200, 3), np.uint8)
    return FramePacket(image, image.copy(), result(timestamp, *labels, valid=valid), None, 10, 5)


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

    def test_no_automatic_trial_and_three_primary_workspaces(self):
        editor = self.make_editor()
        self.assertIsNone(editor._validation.engine)
        self.assertFalse(editor._validation.running)
        self.assertEqual([button.text() for button in editor.mode_group.buttons()],
                         ["作業準備", "工序編輯", "循環設定"])
        self.assertEqual(editor._editor_mode, "steps")

    def test_persistent_trial_survives_workspace_switches(self):
        editor = self.make_editor()
        editor.validation_start.click()
        engine = editor._validation.engine
        for mode in ("overview", "positions", "cycle", "steps"):
            editor._set_editor_mode(mode)
            self.app.processEvents()
            editor.on_packet(packet(.1 + .1 * list(("overview", "positions", "cycle", "steps")).index(mode), "grip"))
            self.assertTrue(editor.validation_box.isVisible())
            self.assertIs(editor._validation.engine, engine)
            self.assertTrue(editor._validation.running)

    def test_edits_invalidate_old_result_and_retry_captures_current_settings(self):
        editor = self.make_editor()
        editor.validation_start.click()
        editor.on_packet(packet(.1, "grip"))
        original_engine = editor._validation.engine
        field = editor.condition_table.field_widget(0, "min_count")
        field.setValue(2)
        self.assertTrue(editor._validation.stale)
        self.assertIn("設定已變更", editor.validation_status.text())
        self.assertIn("—", editor.validation_ratio.text())
        self.assertEqual(original_engine.sop.steps[0].conditions[0].min_count, 1)
        editor.validation_restart.click()
        self.assertFalse(editor._validation.stale)
        self.assertEqual(editor._validation.engine.sop.steps[0].conditions[0].min_count, 2)

    def test_single_target_does_not_follow_editor_selection(self):
        editor = self.make_editor(sop(Step("一", conditions=[Condition(label="grip")]), Step("二")))
        editor.validation_start.click()
        editor.step_list.setCurrentRow(1)
        self.assertIn("工序 1", editor.validation_target.text())
        self.assertTrue(editor._validation.running)
        self.assertEqual(editor._current_step().name, "二")

    def test_flow_starts_at_first_step_even_if_another_is_selected(self):
        editor = self.make_editor(sop(Step("一", conditions=[Condition(label="grip")]), Step("二")))
        editor.step_list.setCurrentRow(1)
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
        editor.step_list.setCurrentRow(1)
        editor.validation_restart.click()
        self.assertIn("無法開始試跑", editor.validation_status.text())
        self.assertTrue(editor._validation.stale)
        self.assertIn("—", editor.validation_ratio.text())
        label = editor.condition_table.field_widget(0, "label")
        label.setCurrentIndex(label.findData("grip"))
        editor.validation_restart.click()
        self.assertTrue(editor._validation.running)
        self.assertIn("工序 2", editor.validation_target.text())

    def test_missing_model_blocks_auto_trial_but_preserves_editing(self):
        editor = self.make_editor(available=False)
        self.assertFalse(editor.validation_start.isEnabled())
        editor._start_validation()
        self.assertIsNone(editor._validation.engine)
        self.assertIn("模型尚未就緒", editor.validation_status.text())
        editor.condition_table._add()
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
        editor.validation_restart.click()
        editor.set_model_state("failed")
        self.assertTrue(editor._validation.stale)
        editor.set_model_state("available")
        editor.validation_restart.click()
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

    def test_details_and_user_pane_widths_are_preserved_across_workspaces(self):
        editor = self.make_editor()
        editor.splitter.setSizes([220, 650, 390])
        self.app.processEvents()
        widths = editor.splitter.sizes()
        editor.validation_details_section.set_expanded(True)
        editor._set_editor_mode("cycle")
        self.app.processEvents()
        self.assertTrue(editor.validation_details.isVisible())
        self.assertEqual(editor.splitter.sizes(), widths)

    def test_top_status_and_hint_do_not_consume_image_space(self):
        editor = self.make_editor()
        self.assertLess(editor.setup_summary.height(), 65)
        self.assertLess(editor.mode_hint.height(), 45)
        self.assertGreater(editor.video.height(), 180)
        self.assertLess(editor.splitter.geometry().top(), 180)

    def test_condition_fields_keep_real_height_after_hidden_page_rebuild(self):
        editor = self.make_editor()
        editor._set_editor_mode("positions")
        editor._refresh_options()
        editor._set_editor_mode("steps")
        self.app.processEvents()
        for field in ("label", "type", "roi", "min_count"):
            widget = editor.condition_table.field_widget(0, field)
            self.assertTrue(widget.isVisible())
            self.assertGreaterEqual(widget.height(), 20)
        self.assertGreaterEqual(editor.condition_table._cards[0].height(), 150)

    def test_single_click_opens_step_from_project_preparation(self):
        editor = self.make_editor(sop(Step("一"), Step("二")))
        editor._set_editor_mode("overview")
        rectangle = editor.step_list.visualItemRect(editor.step_list.item(1))
        QTest.mouseClick(editor.step_list.viewport(), Qt.MouseButton.LeftButton, pos=rectangle.center())
        self.assertEqual(editor._editor_mode, "steps")
        self.assertEqual(editor._current_step().name, "二")

    def test_cycle_ui_explains_real_rules_instead_of_adding_manual_restart_mode(self):
        editor = self.make_editor()
        editor._set_editor_mode("cycle")
        self.assertTrue(editor.auto_restart_spin.isEnabled())
        self.assertFalse(editor.reset_hold_spin.isEnabled())
        editor.reset_table._add()
        self.assertFalse(editor.auto_restart_spin.isEnabled())
        self.assertTrue(editor.reset_hold_spin.isEnabled())
        self.assertIn("80%", editor.cycle_rule_summary.text())

    def test_card_delete_undo_and_missing_object_are_explicit(self):
        cards = ConditionCards()
        self.addCleanup(cards.close)
        values = [Condition(), Condition(label="grip", min_count=2, min_score=.8)]
        cards.set_options(["grip"], [])
        cards.set_conditions(values)
        self.assertEqual(cards.conditions()[0].label, "")
        self.assertFalse(cards.remove_button.isEnabled())
        cards._remove()
        self.assertEqual(cards.conditions(), values)
        cards.select_condition(1)
        cards._remove()
        self.assertEqual(cards.conditions(), values[:1])
        cards._undo_delete()
        self.assertEqual(cards.conditions(), values)
        cards.set_conditions([Condition(label="other")])
        cards._undo_delete()
        self.assertEqual(len(cards.conditions()), 1)

    def test_card_advanced_fields_remain_usable_when_expanded(self):
        editor = self.make_editor()
        section = editor.condition_table.field_widget(0, "advanced")
        section.set_expanded(True)
        self.app.processEvents()
        score = editor.condition_table.field_widget(0, "min_score")
        self.assertTrue(score.isVisible())
        self.assertGreaterEqual(score.height(), 20)
        score.setValue(.75)
        self.assertEqual(editor.sop.steps[0].conditions[0].min_score, .75)
        section.set_expanded(False)
        self.assertFalse(score.isVisible())

    def test_dark_palette_keeps_cards_and_validation_visible(self):
        editor = self.make_editor()
        original = QPalette(self.app.palette())
        self.addCleanup(self.app.setPalette, original)
        palette = QPalette(original)
        palette.setColor(QPalette.ColorRole.Window, QColor("#171e28"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#e2eaf5"))
        self.app.setPalette(palette)
        self.app.processEvents()
        self.assertTrue(editor._theme_dark)
        self.assertTrue(editor.validation_box.isVisible())
        self.assertGreater(editor.condition_table.field_widget(0, "label").height(), 20)

    def test_card_edits_preserve_focus_and_highlight_bound_region(self):
        definition = sop()
        definition.rois = [ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)])]
        definition.steps[0].conditions[0].roi = "left"
        editor = self.make_editor(definition)
        widget = editor.condition_table.field_widget(0, "min_count")
        widget.setFocus()
        widget.setValue(2)
        self.assertTrue(widget.hasFocus())
        self.assertEqual(editor.video._selected, "left")
        self.assertEqual(editor.sop.steps[0].conditions[0].min_count, 2)

    def test_native_window_layout_at_both_resolutions_all_workspaces(self):
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
                    for mode in ("steps", "overview", "follow", "positions", "cycle"):
                        editor._set_editor_mode(mode)
                        window.resize(width, height)
                        self.app.processEvents()
                        self.assertEqual((window.width(), window.height()), (width, height), mode)
                        self.assertTrue(editor.validation_box.isVisible())
                        self.assertGreater(editor.video.height(), 179)
                        self.assertLess(editor.setup_summary.height(), 65)
                        if mode == "steps":
                            editor._refresh_options()
                            self.app.processEvents()
                            for field in ("label", "type", "roi", "min_count"):
                                self.assertGreaterEqual(editor.condition_table.field_widget(0, field).height(), 20)
                        artifact_directory = os.environ.get("SOP_EDITOR_TEST_ARTIFACT_DIR")
                        if artifact_directory:
                            directory = Path(artifact_directory)
                            directory.mkdir(parents=True, exist_ok=True)
                            self.assertTrue(window.grab().save(str(directory / f"workbench-{width}x{height}-{mode}.png")))
                    editor._set_editor_mode("steps")
                    editor.validation_details_section.set_expanded(True)
                    self.app.processEvents()
                    self.assertEqual((window.width(), window.height()), (width, height), "details")
                    editor.validation_details_section.set_expanded(False)
            finally:
                window.dirty = False
                window.close()


if __name__ == "__main__":
    unittest.main()
