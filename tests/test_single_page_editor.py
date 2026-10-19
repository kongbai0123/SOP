"""Whole-document editing and preview/engine regressions without devices or records."""
from __future__ import annotations

import copy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QListWidget, QMessageBox, QScrollArea, QTabWidget

from sop_app.detection import Detection, FrameResult
from sop_app.engine import SOPEngine
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step, load_sop, save_sop
from sop_app.ui.editor_page import EditorPage


def packet(at=1, labels=("part",)):
    image = np.zeros((100, 200, 3), np.uint8)
    detections = [Detection(label, .95, (10, 10, 30, 30)) for label in labels]
    result = FrameResult(200, 100, at, detections)
    return FramePacket(image, image.copy(), result, None, 10, 5)


class SinglePageEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def editor(self, sop, expanded=True):
        editor = EditorPage()
        self.addCleanup(editor.deleteLater)
        self.addCleanup(editor.close)
        editor.set_sop(sop)
        editor.set_classes(["part", "other", "never"])
        editor.set_model_state("available")
        editor.resize(1000, 720)
        editor.show()
        if expanded:
            editor.edit_steps_button.setChecked(True)
        self.app.processEvents()
        return editor

    def test_all_settings_remain_available_when_trial_target_changes(self):
        editor = self.editor(SOPDefinition(steps=[Step("automatic", conditions=[Condition(label="part")]), Step("manual")]))
        self.assertTrue(all(card.expanded for card in editor.step_editors))
        # Steps on the left and tools on the right scroll on their own, never nested.
        scrolls = editor.findChildren(QScrollArea)
        self.assertEqual(set(scrolls), {editor.page_scroll, editor.tools_scroll})
        self.assertFalse(editor.page_scroll.isAncestorOf(editor.tools_scroll))
        self.assertFalse(editor.tools_scroll.isAncestorOf(editor.page_scroll))
        self.assertFalse(editor.findChildren(QTabWidget))
        self.assertFalse(editor.findChildren(QListWidget))
        for card in editor.step_editors:
            for widget in (card.step_name_edit, card.instruction_edit, card.condition_table,
                           card.forbidden_table, card.forbidden_hold_spin, card.timeout_spin):
                self.assertTrue(widget.isVisible())
        editor.validation_step.setCurrentIndex(1)
        self.app.processEvents()
        self.assertTrue(all(card.isVisible() for card in editor.step_editors))
        self.assertIn("人工確認", editor.step_editors[1].manual_label.text())
        self.assertTrue(editor.validation_confirm.isVisible())
        self.assertFalse(editor.validation_confirm.isEnabled())

    def test_loaded_steps_collapse_to_complete_rule_summaries(self):
        editor = self.editor(SOPDefinition(steps=[
            Step("pick", conditions=[Condition(label="part"), Condition(type="disappear", label="other")],
                 completion_mode="any", hold_sec=.5, ratio=.7, timeout_sec=20,
                 forbidden=[Condition(label="never")], forbidden_hold_sec=.3),
            Step("manual")]), expanded=False)
        first, second = editor.step_editors
        for card in (first, second):
            self.assertFalse(card.expanded)
            self.assertEqual(card.expand_button.text(), "展開")
            for widget in (card.step_name_edit, card.drag_handle):
                self.assertTrue(widget.isVisible())
            # The description and rules are a tooltip on the name row, not text on screen.
            self.assertFalse(any(label.isVisible() and "違規：" in label.text()
                                 for label in card.findChildren(QLabel)))
            for widget in (card.instruction_edit, card.completion_row, card.condition_table,
                           card.forbidden_row, card.forbidden_table):
                self.assertFalse(widget.isVisible())
        rules = first.header_row.toolTip()
        self.assertTrue(rules.startswith("工序 1 pick"))
        for text in ("完成（任一成立）：part 出現、other 消失", "觀察 0.5 秒、成立比例 70%", "超時 20 秒",
                     "違規（全部成立）：never 出現，觀察 0.3 秒、成立比例 70%"):
            self.assertIn(text, rules)
        self.assertIn("完成：人工確認", second.header_row.toolTip())
        self.assertIn("違規：未設定", second.header_row.toolTip())
        editor.edit_steps_button.click()
        self.app.processEvents()
        self.assertTrue(first.condition_table.isVisible())
        self.assertEqual(first.expand_button.text(), "收合")
        self.assertTrue(first.instruction_edit.isVisible())
        self.assertTrue(second.condition_table.isVisible())
        first.hold_spin.setValue(1.5)
        first.expand_button.click()
        self.assertIn("觀察 1.5 秒", first.header_row.toolTip())
        editor.edit_steps_button.setChecked(False)
        editor.edit_steps_button.setChecked(True)
        self.assertTrue(all(card.expanded for card in editor.step_editors))
        editor.edit_steps_button.setChecked(False)
        self.assertTrue(all(card.body.isHidden() for card in editor.step_editors))

    def test_new_and_duplicated_steps_open_alone_and_moves_keep_state(self):
        editor = self.editor(SOPDefinition(steps=[Step("A"), Step("B")]), expanded=False)
        editor.edit_steps_button.click()
        editor.step_editors[0].set_expanded(False, notify=True)
        editor.step_editors[1].up_button.click()
        self.assertEqual([s.name for s in editor.sop.steps], ["B", "A"])
        self.assertEqual([card.expanded for card in editor.step_editors], [True, False])
        editor.add_step_button.click()
        self.assertEqual([card.expanded for card in editor.step_editors], [False, False, True])
        self.assertTrue(editor.step_editors[2].step_name_edit.hasFocus())
        editor.step_editors[1].duplicate_button.click()
        self.assertEqual([s.name for s in editor.sop.steps], ["B", "A", "A（複製）", "工序 3"])
        self.assertEqual([card.expanded for card in editor.step_editors], [False, False, True, False])
        editor.set_sop(SOPDefinition(steps=[Step("only")]))
        self.assertFalse(editor.step_editors[0].expanded)
        self.assertTrue(editor.step_editors[0].body.isHidden())

    def test_adding_and_deleting_keep_the_scroll_position(self):
        steps = [Step(f"s{i}", conditions=[Condition(label="part")], forbidden=[Condition(label="part")])
                 for i in range(6)]
        editor = self.editor(SOPDefinition(steps=steps))
        editor.resize(1366, 700)
        bar = editor.page_scroll.verticalScrollBar()

        def settle():
            for _ in range(4):
                self.app.processEvents()

        def keeps(action):
            editor.page_scroll.ensureWidgetVisible(editor.step_editors[3], 0, 0)
            settle()
            before = bar.value()
            self.assertGreater(before, 0)
            action()
            settle()
            self.assertEqual(bar.value(), before)

        settle()
        keeps(lambda: editor.step_editors[3].condition_table.add_button.click())
        keeps(lambda: editor.step_editors[3].condition_table.field_widget(0, "remove").click())
        keeps(lambda: editor.step_editors[3].condition_table.undo_delete_button.click())
        with patch("sop_app.ui.editor_page.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            keeps(lambda: editor.step_editors[1].delete_button.click())
        keeps(lambda: editor.step_editors[3].up_button.click())
        editor.add_step_button.click()
        settle()
        added = editor.step_editors[-1]
        top = added.mapTo(editor.page_scroll.viewport(), added.rect().topLeft()).y()
        self.assertGreaterEqual(top, 0)
        self.assertLess(top, editor.page_scroll.viewport().height())

    def test_image_is_the_centre_and_framing_hides_steps_to_enlarge_it(self):
        editor = self.editor(SOPDefinition(steps=[Step("A")]))
        editor.resize(1489, 820)
        editor.on_packet(packet())
        self.app.processEvents()
        # Steps left, image centre, tools and trial right; the image is the widest column.
        left = editor.settings_pane.mapTo(editor, editor.settings_pane.rect().topLeft()).x()
        centre = editor.video.mapTo(editor, editor.video.rect().topLeft()).x()
        right = editor.tools_pane.mapTo(editor, editor.tools_pane.rect().topLeft()).x()
        self.assertLess(left, centre)
        self.assertLess(centre, right)
        self.assertGreater(editor.video.width(), editor.settings_pane.width())
        self.assertGreater(editor.video.width(), editor.tools_pane.width())
        for widget in (editor.draw_button, editor.empty_rois_label, editor.validation_box):
            self.assertTrue(editor.tools_pane.isAncestorOf(widget))
        # Workpiece tracking is one horizontal bar directly under the image.
        bar = editor.tracking_tools
        self.assertTrue(editor.monitor_pane.isAncestorOf(bar))
        video_bottom = editor.video.mapTo(editor, editor.video.rect().bottomLeft()).y()
        bar_top = bar.mapTo(editor, bar.rect().topLeft()).y()
        self.assertGreater(bar_top, video_bottom)
        ys = {button.mapTo(editor, button.rect().topLeft()).y()
              for button in (editor.target_button, editor.clear_target_button, editor.reference_button)}
        self.assertEqual(len(ys), 1)
        normal = editor.video.width()
        editor.draw_button.click()
        self.app.processEvents()
        self.assertFalse(editor.settings_pane.isVisible())
        self.assertTrue(editor.tools_pane.isVisible())
        self.assertGreater(editor.video.width(), normal * 1.3)
        self.assertEqual(editor.focus_button.text(), "還原版面")
        editor._on_roi_drawn([(.2, .2), (.4, .2), (.4, .4), (.2, .4)])
        self.app.processEvents()
        # The new region is named in the tools column while the image stays large.
        self.assertFalse(editor.settings_pane.isVisible())
        self.assertTrue(editor.roi_rows[0].name_edit.isVisible())
        self.assertEqual(editor.roi_rows[0].name_edit.selectedText(), "區域1")
        self.assertIn("區域1（固定畫面）", editor.roi_rows[0].toolTip())
        editor.live_button.click()
        self.app.processEvents()
        self.assertTrue(editor.settings_pane.isVisible())
        self.assertEqual(editor.video.width(), normal)
        # A manual enlargement for checking tracking stays until it is restored.
        editor.focus_button.click()
        editor.draw_button.click()
        editor.live_button.click()
        self.app.processEvents()
        self.assertFalse(editor.settings_pane.isVisible())
        editor.focus_button.click()
        self.app.processEvents()
        self.assertTrue(editor.settings_pane.isVisible())
        self.assertEqual(editor.focus_button.text(), "放大影像")
        editor.focus_button.click()
        editor.set_production_running(True)
        self.assertTrue(editor.settings_pane.isVisible())

    def test_nonselected_step_edits_and_settings_survive_save_reload(self):
        sop = SOPDefinition(steps=[Step("first"), Step("second", conditions=[Condition(label="part")])])
        editor = self.editor(sop)
        card = editor.step_editors[1]
        card.step_name_edit.selectAll()
        QTest.keyClicks(card.step_name_edit, "renamed")
        card.instruction_edit.setPlainText("Install second part")
        card.hold_spin.setValue(2.5)
        card.ratio_spin.setValue(65)
        card.timeout_spin.setValue(17)
        card.min_duration_spin.setValue(3)
        card.completion_mode.setCurrentIndex(card.completion_mode.findData("any"))
        card.forbidden_table.add_button.click()
        card.forbidden_mode.setCurrentIndex(card.forbidden_mode.findData("any"))
        card.forbidden_hold_spin.setValue(.3)
        card.forbidden_ratio_spin.setValue(40)
        card.condition_table.field_widget(0, "min_score").setValue(.85)
        self.assertEqual(editor.validation_step.currentIndex(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sop.json"
            save_sop(editor.sop, path)
            saved = load_sop(path)
        step = saved.steps[1]
        self.assertEqual(saved.to_dict(), editor.sop.to_dict())
        self.assertEqual((step.name, step.instruction, step.hold_sec, step.ratio, step.timeout_sec,
                          step.min_duration_sec, step.completion_mode, step.forbidden_mode,
                          step.forbidden_hold_sec, step.forbidden_ratio, step.conditions[0].min_score),
                         ("renamed", "Install second part", 2.5, .65, 17, 3, "any", "any", .3, .4, .85))

    def test_reorder_duplicate_delete_then_edit_correct_step(self):
        editor = self.editor(SOPDefinition(steps=[Step("A"), Step("B"), Step("C")]))
        editor.step_editors[2].up_button.click()
        self.assertEqual([s.name for s in editor.sop.steps], ["A", "C", "B"])
        editor.step_editors[1].duplicate_button.click()
        self.assertEqual([s.name for s in editor.sop.steps], ["A", "C", "C（複製）", "B"])
        editor.step_editors[2].condition_table.add_button.click()
        editor.step_editors[2].condition_table.field_widget(0, "label").setCurrentIndex(2)
        self.assertFalse(editor.sop.steps[1].conditions)
        self.assertEqual(editor.sop.steps[2].conditions[0].label, "other")
        with patch("sop_app.ui.editor_page.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            editor.step_editors[0].delete_button.click()
        editor.step_editors[2].instruction_edit.setPlainText("B instruction")
        self.assertEqual([s.name for s in editor.sop.steps], ["C", "C（複製）", "B"])
        self.assertEqual(editor.sop.steps[2].instruction, "B instruction")
        self.assertEqual(editor.sop.steps[1].conditions[0].label, "other")

    def test_every_step_previews_without_selecting_it(self):
        editor = self.editor(SOPDefinition(steps=[Step("A", conditions=[Condition(label="part")]),
                                                 Step("B", conditions=[Condition(label="other")])]))
        cards = tuple(editor.step_editors)
        fields = tuple(card.condition_table.field_widget(0, "label") for card in cards)
        editor.on_packet(packet(labels=("part", "other")))
        self.assertTrue(all("本幀成立，尚未代表工序完成" in card.live_label.text() for card in cards))
        self.assertTrue(all(card.live_label.property("state") == "ok" for card in cards))
        editor.on_packet(packet(2, labels=()))
        self.assertTrue(all("本幀條件未成立" in card.live_label.text() for card in cards))
        self.assertEqual(tuple(editor.step_editors), cards)
        self.assertEqual(tuple(card.condition_table.field_widget(0, "label") for card in cards), fields)

    def test_completion_and_forbidden_all_any_match_production_engine(self):
        for completion_mode in ("all", "any"):
            for forbidden_mode in ("all", "any"):
                with self.subTest(completion=completion_mode, forbidden=forbidden_mode):
                    step = Step("check", conditions=[Condition(label="part"), Condition(label="never")],
                                forbidden=[Condition(label="part"), Condition(label="other")],
                                completion_mode=completion_mode, forbidden_mode=forbidden_mode,
                                hold_sec=0, forbidden_hold_sec=0)
                    editor = self.editor(SOPDefinition(steps=[step]))
                    frame = packet()
                    engine = SOPEngine(editor.sop)
                    engine.start(0)
                    events = engine.update(frame.result)
                    editor.on_packet(frame)
                    preview = editor.step_editors[0].live_label.text()
                    self.assertEqual("本幀成立，尚未代表工序完成" in preview, completion_mode == "any")
                    self.assertEqual("本幀違規條件成立" in preview, forbidden_mode == "any")
                    self.assertEqual(any(event.data.get("alarm") == "violation" for event in events), forbidden_mode == "any")
                    self.assertEqual(any(event.kind == "step_completed" for event in events), completion_mode == "any")

    def test_preview_states_match_engine_for_blocked_and_partial_frames(self):
        """Every verdict shown in the editor is the one the production engine reaches on that frame."""
        moving = ROI("moving", [(0, 0), (.5, 0), (.5, .5), (0, .5)], "workpiece")  # no pose → unavailable
        frames = {"part": (("part",), True), "both": (("part", "other"), True),
                  "none": ((), True), "model not ready": ((), False)}
        for completion_mode in ("all", "any"):
            for forbidden_mode in ("all", "any"):
                for name, (labels, valid) in frames.items():
                    with self.subTest(completion=completion_mode, forbidden=forbidden_mode, frame=name):
                        step = Step("check", conditions=[Condition(label="part"), Condition(label="part", roi="moving")],
                                    forbidden=[Condition(label="part"), Condition(label="other")],
                                    completion_mode=completion_mode, forbidden_mode=forbidden_mode,
                                    hold_sec=0, forbidden_hold_sec=0)
                        sop = SOPDefinition(rois=[moving], steps=[step, Step("next")])
                        editor = self.editor(copy.deepcopy(sop))
                        frame = packet(labels=labels)
                        frame.result.detection_valid = valid
                        engine = SOPEngine(sop)
                        engine.start(0)
                        events = engine.update(frame.result)
                        editor.on_packet(frame)
                        preview = editor.step_editors[0].live_label.text()
                        kinds = [event.kind for event in events]
                        self.assertEqual("本幀成立，尚未代表工序完成" in preview, "step_completed" in kinds, preview)
                        self.assertEqual("本幀違規條件成立" in preview,
                                         any(event.data.get("alarm") == "violation" for event in events), preview)
                        self.assertEqual("無法判斷" in preview, "position_lost" in kinds, preview)

    def test_region_rename_undo_redo_preserve_references_after_later_edit(self):
        region = ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)])
        sop = SOPDefinition(rois=[region], steps=[Step("A", conditions=[Condition(label="part", roi="left")],
                                                      forbidden=[Condition(label="other", roi="left")])])
        sop.cycle.reset_conditions = [Condition(type="disappear", label="part", roi="left")]
        editor = self.editor(sop)
        editor.roi_rows[0].name_edit.setText("right")
        editor.roi_rows[0].name_edit.editingFinished.emit()
        self.assertTrue(all(c.roi == "right" for c in editor._all_conditions()))
        editor.undo_roi_button.click()
        self.assertTrue(all(c.roi == "left" for c in editor._all_conditions()))
        editor.redo_roi_button.click()
        self.assertTrue(all(c.roi == "right" for c in editor._all_conditions()))
        editor.step_editors[0].hold_spin.setValue(3)
        self.assertTrue(all(c.roi == "right" for c in editor._all_conditions()))
        self.assertEqual(editor.step_editors[0].condition_table.field_widget(0, "roi").currentData(), "right")

    def test_referenced_region_cannot_be_deleted_and_no_loading_mutation(self):
        sop = SOPDefinition(rois=[ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)])],
                            steps=[Step("A", forbidden=[Condition(label="part", roi="left")])])
        original = copy.deepcopy(sop)
        editor = self.editor(sop)
        self.assertEqual(editor.sop, original)
        with patch("sop_app.ui.editor_page.QMessageBox.information") as notice:
            editor.roi_rows[0].delete_button.click()
        notice.assert_called_once()
        self.assertEqual(editor.sop, original)

    def test_manual_target_trial_uses_snapshot_and_invalidates_on_edits(self):
        editor = self.editor(SOPDefinition(steps=[Step("A"), Step("B")]))
        editor.on_packet(packet())
        editor.step_editors[1].trial_button.click()
        self.assertTrue(editor._validation.running)
        self.assertEqual(editor._validation.original_indices, [1])
        self.assertEqual(editor._validation.engine.sop.steps[0].name, "B")
        self.assertTrue(editor.validation_confirm.isEnabled())
        editor.validation_confirm.click()
        self.assertTrue(editor._validation.completed)
        editor.validation_start.click()
        editor.step_editors[0].instruction_edit.setPlainText("changed")
        self.assertFalse(editor._validation.running)
        self.assertTrue(editor._validation.stale)
        self.assertIn("設定已變更", editor.validation_status.text())

    def test_production_locks_settings_and_trial_and_restores_after_stop(self):
        editor = self.editor(SOPDefinition(steps=[Step("A")]))
        editor.on_packet(packet())
        editor.validation_start.click()
        self.assertTrue(editor._validation.running)
        editor.set_production_running(True)
        self.assertFalse(editor._validation.running)
        self.assertFalse(editor.name_edit.isEnabled())
        self.assertFalse(editor.step_editors[0].condition_table.isEnabled())
        self.assertFalse(editor.step_editors[0].trial_button.isEnabled())
        self.assertFalse(editor.validation_start.isEnabled())
        self.assertFalse(editor.draw_button.isEnabled())
        editor.set_production_running(False)
        self.assertTrue(editor.step_editors[0].condition_table.isEnabled())
        self.assertTrue(editor.validation_start.isEnabled())

    def test_save_results_are_reported_without_losing_edits(self):
        editor = self.editor(SOPDefinition(steps=[Step("A")]))
        editor.step_editors[0].instruction_edit.setPlainText("kept")
        editor.on_save_result("cancelled")
        self.assertIn("已取消儲存", editor.guide_label.text())
        editor.on_save_result("failed")
        self.assertIn("可重試儲存", editor.guide_label.text())
        self.assertEqual(editor.sop.steps[0].instruction, "kept")

    def test_draw_region_creates_inline_name_without_dialog(self):
        editor = self.editor(SOPDefinition(steps=[Step("A")]))
        editor.on_packet(packet())
        editor.draw_button.click()
        points = [(.2, .2), (.4, .2), (.4, .4), (.2, .4)]
        editor._on_roi_drawn(points)
        self.assertEqual(editor.sop.rois[0].name, "區域1")
        self.assertEqual(editor.roi_rows[0].name_edit.selectedText(), "區域1")
        editor.roi_rows[0].name_edit.setText("installation")
        editor.roi_rows[0].name_edit.editingFinished.emit()
        editor._on_roi_drawn(points)
        self.app.processEvents()
        self.assertEqual([r.name for r in editor.sop.rois], ["installation", "區域1"])
        self.assertTrue(all(row.name_edit.isVisible() for row in editor.roi_rows))

    def test_anchor_conversion_keeps_geometry_and_rejects_out_of_bounds_pose(self):
        region = ROI("position", [(.2, .2), (.4, .2), (.4, .4), (.2, .4)])
        editor = self.editor(SOPDefinition(rois=[region], steps=[Step("A")]))
        original = copy.deepcopy(region)
        # Supply the already computed background pose; no tracker is run on the UI thread.
        editor._locator = object()
        editor._preview_result = FrameResult(200, 100, 1, workpiece_transform=np.array([[1., 0, .1], [0, 1., .1]]))
        editor._change_roi_anchor(region, "workpiece")
        self.assertEqual(region.anchor, "workpiece")
        np.testing.assert_allclose(region.points, np.array(original.points) - .1)
        editor._change_roi_anchor(region, "fixed")
        self.assertEqual(region.anchor, "fixed")
        np.testing.assert_allclose(region.points, original.points)
        editor._preview_result.workpiece_transform = np.array([[1., 0, .9], [0, 1., .9]])
        editor._change_roi_anchor(region, "workpiece")
        self.assertEqual(region.anchor, "fixed")
        np.testing.assert_allclose(region.points, original.points)
        self.assertIn("原設定已保留", editor.guide_label.text())


if __name__ == "__main__":
    unittest.main()
