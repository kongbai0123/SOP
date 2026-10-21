"""Compact editing, reorder persistence, and optional next-cycle behavior."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sop_app.detection import Detection, FrameResult
from sop_app.engine import Phase, SOPEngine
from sop_app.sop_schema import Condition, CycleSettings, ROI, SOPDefinition, Step
from sop_app.ui.editor_page import EditorPage


class EditorSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def page(self, sop):
        page = EditorPage()
        self.addCleanup(page.deleteLater)
        self.addCleanup(page.close)
        page.set_sop(sop)
        page.show()
        self.app.processEvents()
        return page

    def test_new_steps_stay_compact_and_management_never_opens_forms(self):
        page = self.page(SOPDefinition(steps=[Step("A"), Step("B")]))
        page.add_step_button.click()
        self.assertFalse(page.edit_steps_button.isChecked())
        self.assertTrue(all(card.body.isHidden() and card.actions_row.isHidden() for card in page.step_editors))
        self.assertTrue(all(card.edit_button.isVisible() for card in page.step_editors))
        self.assertIsNone(page.step_settings.current_editor)
        page.add_step_button.click()
        self.assertTrue(all(card.body.isHidden() for card in page.step_editors))
        moved = page.sop.steps[3]
        page._open_step_settings(moved)
        page._move_step(-1, 3)
        self.assertEqual([card.body.isVisible() for card in page.step_editors], [False, False, True, False])
        self.assertIs(page.step_settings.current_editor.step, moved)
        page.step_settings.close_button.click()
        self.assertTrue(all(card.body.isHidden() and card.actions_row.isHidden() for card in page.step_editors))
        page._duplicate_step(0)
        self.assertFalse(page.edit_steps_button.isChecked())
        self.assertTrue(all(card.body.isHidden() for card in page.step_editors))
        self.assertIsNone(page.step_settings.current_editor)
        page.edit_steps_button.click()
        self.assertTrue(all(card.body.isHidden() for card in page.step_editors))
        self.assertTrue(all(card.actions_row.isVisible() for card in page.step_editors))
        page.add_step_button.click()
        self.assertTrue(page.edit_steps_button.isChecked())
        self.assertTrue(all(card.actions_row.isVisible() for card in page.step_editors))
        page.edit_steps_button.click()
        self.assertTrue(all(card.body.isHidden() and card.actions_row.isHidden() for card in page.step_editors))

    def test_region_tools_and_history_belong_to_region_editing(self):
        roi = ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)])
        page = self.page(SOPDefinition(rois=[roi], steps=[Step("A", conditions=[Condition(roi="left")])]))
        self.assertEqual(page.draw_button.text(), "新增區域")
        self.assertEqual(page.edit_rois_button.text(), "編輯區域")
        self.assertTrue(page.roi_edit_toolbar.isHidden())
        self.assertTrue(page.roi_rows[0].actions_row.isHidden())
        self.assertTrue(page.roi_rows[0].anchor_combo.isHidden())
        page.edit_rois_button.click()
        self.assertTrue(page.roi_edit_toolbar.isVisible())
        self.assertTrue(page.roi_rows[0].actions_row.isVisible())
        self.assertTrue(page.roi_rows[0].anchor_combo.isVisible())
        page._rename_roi(roi, "right")
        page.edit_rois_button.click()
        page.edit_rois_button.click()
        self.assertTrue(page.undo_roi_button.isEnabled())
        page.undo_roi_button.click()
        self.assertEqual(page.sop.rois[0].name, "left")
        self.assertEqual(page.sop.steps[0].conditions[0].roi, "left")
        self.assertTrue(page.roi_rows[0].actions_row.isVisible())
        page.redo_roi_button.click()
        self.assertEqual(page.sop.steps[0].conditions[0].roi, "right")
        page.set_production_running(True)
        self.assertFalse(page.undo_roi_button.isEnabled())
        self.assertFalse(page.roi_rows[0].name_edit.isEnabled())
        page.set_production_running(False)
        page.edit_rois_button.click()
        self.assertTrue(page.roi_rows[0].actions_row.isHidden())

    def test_clean_list_reorders_and_editing_preserves_names(self):
        page = EditorPage()
        self.addCleanup(page.deleteLater)
        self.addCleanup(page.close)
        steps = [Step("A"), Step("B"), Step("C")]
        page.set_sop(SOPDefinition(steps=steps.copy()))
        page.show()
        self.app.processEvents()
        for card in page.step_editors:
            for widget in (card.actions_row, card.body):
                self.assertTrue(widget.isHidden())
            self.assertTrue(card.edit_button.isVisible())
            self.assertTrue(card.step_name_edit.isVisible())
        page._drop_step(str(id(steps[0])), steps[2], True)
        self.assertEqual(page.sop.steps, [steps[1], steps[2], steps[0]])
        self.assertEqual([card.number_label.text() for card in page.step_editors], ["1", "2", "3"])
        QTest.keyClick(page.step_editors[0].drag_handle, Qt.Key.Key_Down, Qt.KeyboardModifier.AltModifier)
        self.assertEqual(page.sop.steps, [steps[2], steps[1], steps[0]])
        card = page.step_editors[0]
        card.step_name_edit.selectAll()
        QTest.keyClicks(card.step_name_edit, "renamed")
        page.edit_steps_button.click()
        self.assertTrue(card.body.isHidden())
        card.edit_button.click()
        self.assertTrue(card.body.isVisible())
        page.step_settings.close_button.click()
        page.edit_steps_button.click()
        self.assertTrue(card.body.isHidden())
        self.assertEqual(page.sop.steps[0].name, "renamed")

    def test_switch_retains_settings_and_roundtrips(self):
        page = EditorPage()
        self.addCleanup(page.deleteLater)
        self.addCleanup(page.close)
        sop = SOPDefinition(steps=[Step("A")], cycle=CycleSettings(auto_restart_sec=7))
        page.set_sop(sop)
        page.cycle_enabled_switch.click()
        self.assertFalse(sop.cycle.enabled)
        self.assertFalse(page.auto_restart_spin.isEnabled())
        self.assertTrue(page.sequence_check.isEnabled())
        restored = SOPDefinition.from_dict(sop.to_dict())
        self.assertFalse(restored.cycle.enabled)
        self.assertEqual(restored.cycle.auto_restart_sec, 7)
        page.set_sop(restored)
        self.assertFalse(page.cycle_enabled_switch.isChecked())
        page.cycle_enabled_switch.click()
        self.assertTrue(restored.cycle.enabled)
        self.assertEqual(page.auto_restart_spin.value(), 7)
        self.assertTrue(SOPDefinition.from_dict({}).cycle.enabled)

    def test_disabled_cycle_never_restarts_with_delay_or_reset_conditions(self):
        for conditions in ([], [Condition(label="part")]):
            with self.subTest(conditions=conditions):
                sop = SOPDefinition(steps=[Step("manual")], cycle=CycleSettings(
                    enabled=False, auto_restart_sec=0, reset_hold_sec=0,
                    reset_conditions=conditions))
                engine = SOPEngine(sop)
                engine.start(0)
                engine.confirm_manual_step(1)
                self.assertEqual(engine.phase, Phase.CYCLE_DONE)
                for at in (2, 10, 100):
                    self.assertEqual(engine.update(FrameResult(200, 100, at, [
                        Detection("part", .95, (10, 10, 30, 30))])), [])
                self.assertEqual(engine.cycle, 1)
                self.assertEqual(engine.phase, Phase.CYCLE_DONE)
