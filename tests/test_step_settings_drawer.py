"""One-step settings overlay interactions without devices or user data."""
from __future__ import annotations

from copy import deepcopy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QTabWidget
from shiboken6 import isValid

from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step
from sop_app.ui.editor_page import EditorPage


class StepSettingsDrawerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def settle(self):
        for _ in range(6):
            self.app.processEvents()

    def flush_deletes(self):
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settle()

    def dispose(self, page):
        if not isValid(page):
            return
        for timer in page.findChildren(QTimer):
            timer.stop()
        page.close()
        page.deleteLater()
        self.flush_deletes()

    def page(self, steps=None, *, height=780, rois=None):
        page = EditorPage()
        self.addCleanup(self.dispose, page)
        page._validation_watchdog.stop()
        page.resize(1489, height)
        page.set_sop(SOPDefinition(name="Drawer fixture", rois=rois or [], steps=steps if steps is not None else [
            Step("A", conditions=[Condition(label="part")]),
            Step("B", conditions=[Condition(label="other")]),
        ]))
        page.set_classes(["part", "other"])
        # This only enables the editor UI; no model, capture or recorder is opened.
        page.set_model_state("available")
        page.show()
        page.activateWindow()
        self.settle()
        return page

    def open(self, page, index):
        card = page.step_editors[index]
        page.page_scroll.ensureWidgetVisible(card.edit_button, 0, 0)
        self.settle()
        QTest.mouseClick(card.edit_button, Qt.MouseButton.LeftButton)
        self.settle()
        self.assertTrue(page.step_settings.isVisible())
        self.assertIs(page.step_settings.current_editor, card)
        return card

    def assert_closed(self, page):
        self.assertTrue(page.step_settings.isHidden())
        self.assertIsNone(page.step_settings.current_editor)

    def test_three_adds_create_only_three_editable_name_rows(self):
        page = self.page([])
        for _ in range(3):
            page.add_step_button.click()
            self.settle()
            self.assert_closed(page)
        self.assertEqual([step.name for step in page.sop.steps], ["工序 1", "工序 2", "工序 3"])
        self.assertEqual(len(page.step_editors), 3)
        for card, name in zip(page.step_editors, ("Take L_grip", "Install", "Check")):
            self.assertTrue(card.step_name_edit.isVisible())
            self.assertTrue(card.edit_button.isVisible())
            self.assertTrue(card.body.isHidden())
            self.assertTrue(card.actions_row.isHidden())
            card.step_name_edit.selectAll()
            QTest.keyClicks(card.step_name_edit, name)
        self.assertEqual([step.name for step in page.sop.steps], ["Take L_grip", "Install", "Check"])
        self.assert_closed(page)

    def test_edit_button_opens_only_its_named_step(self):
        page = self.page()
        first, second = page.step_editors
        self.assertTrue(first.edit_button.isVisible())
        self.assertTrue(second.edit_button.isVisible())
        self.open(page, 1)
        self.assertTrue(page.step_settings.scroll.isAncestorOf(second.body))
        self.assertTrue(page.step_settings.isAncestorOf(second.actions_row))
        self.assertTrue(second.condition_table.isVisible())
        self.assertTrue(first.body.isHidden())
        self.assertFalse(page.step_settings.isAncestorOf(first.body))
        second.hold_spin.setValue(2.4)
        self.assertEqual(page.sop.steps[1].hold_sec, 2.4)
        self.assertEqual(page.sop.steps[0].hold_sec, 1.)

    def test_switching_steps_restores_previous_body_and_actions(self):
        page = self.page()
        first, second = page.step_editors
        parents = {card: (card.body.parentWidget(), card.actions_row.parentWidget())
                   for card in (first, second)}
        self.open(page, 0)
        # The overlay is modal to the page; switching also uses the same guarded
        # entry point as the per-row edit signal.
        page._open_step_settings(second.step)
        self.settle()
        self.assertIs(page.step_settings.current_editor, second)
        self.assertIs(first.body.parentWidget(), parents[first][0])
        self.assertIs(first.actions_row.parentWidget(), parents[first][1])
        self.assertTrue(first.body.isHidden())
        self.assertFalse(page.step_settings.isAncestorOf(first.body))
        page.step_settings.close_button.click()
        self.settle()
        self.assert_closed(page)
        self.assertIs(second.body.parentWidget(), parents[second][0])
        self.assertIs(second.actions_row.parentWidget(), parents[second][1])
        self.assertTrue(second.body.isHidden())

    def test_close_and_reopen_keep_original_fields_and_live_edits(self):
        page = self.page()
        card = page.step_editors[1]
        fields = (card.body, card.actions_row, card.hold_spin, card.condition_table,
                  card.condition_table.field_widget(0, "min_count"))
        body_index = card.layout().indexOf(card.body)
        actions_index = card.header_row.layout().indexOf(card.actions_row)
        self.open(page, 1)
        card.instruction_edit.setPlainText("Install this part")
        card.hold_spin.setValue(3.2)
        card.ratio_spin.setValue(65)
        card.timeout_spin.setValue(12)
        card.condition_table.field_widget(0, "min_count").setValue(3)
        card.forbidden_table.add_button.click()
        card.forbidden_hold_spin.setValue(.4)
        self.assertEqual((card.step.instruction, card.step.hold_sec, card.step.ratio,
                          card.step.timeout_sec, card.step.conditions[0].min_count,
                          card.step.forbidden_hold_sec),
                         ("Install this part", 3.2, .65, 12, 3, .4))
        before_close = deepcopy(page.sop.to_dict())
        page.step_settings.close_button.click()
        self.settle()
        self.assert_closed(page)
        self.assertEqual(card.layout().indexOf(card.body), body_index)
        self.assertEqual(card.header_row.layout().indexOf(card.actions_row), actions_index)
        self.assertEqual(page.sop.to_dict(), before_close)
        self.open(page, 1)
        self.assertEqual(fields, (card.body, card.actions_row, card.hold_spin, card.condition_table,
                                 card.condition_table.field_widget(0, "min_count")))
        self.assertEqual(page.sop.to_dict(), before_close)

    def test_sections_share_one_scrollable_page_without_tabs(self):
        page = self.page()
        card = self.open(page, 0)
        drawer = page.step_settings
        self.assertFalse(drawer.findChildren(QTabWidget))
        for field in (card.instruction_edit, card.condition_table, card.hold_spin,
                      card.timeout_spin, card.forbidden_table, card.forbidden_hold_spin):
            self.assertTrue(drawer.scroll.isAncestorOf(field))
            self.assertFalse(field.isHidden())
            self.assertTrue(field.isVisibleTo(drawer.scroll.widget()))

    def test_removing_last_completion_condition_can_still_be_undone(self):
        original = Condition(label="part", min_count=3)
        page = self.page([Step("A", conditions=[original])])
        card = self.open(page, 0)
        card.condition_table.field_widget(0, "remove").click()
        self.assertEqual(card.step.conditions, [])
        self.assertEqual(card.completion_method.currentData(), "manual")
        self.assertTrue(card.condition_table.undo_delete_button.isEnabled())
        card.condition_table.undo_delete_button.click()
        self.assertEqual(card.step.conditions, [original])
        self.assertEqual(card.completion_method.currentData(), "auto")
        self.assertTrue(card.condition_table.add_button.isEnabled())

    def test_manual_mode_cache_follows_region_rename_undo_and_rebuild(self):
        region = ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)])
        page = self.page([Step("A", conditions=[Condition(label="part", roi="left")])], rois=[region])
        card = self.open(page, 0)
        card.completion_method.setCurrentIndex(card.completion_method.findData("manual"))
        self.assertEqual(card.step.conditions, [])
        page.step_settings.close_button.click()
        page._rename_roi(region, "right")
        page._undo_roi()
        self.assertEqual(card._automatic_conditions[0].roi, "left")
        page._redo_roi()
        self.assertEqual(card._automatic_conditions[0].roi, "right")
        page.add_step_button.click()
        self.settle()
        card = self.open(page, 0)
        card.completion_method.setCurrentIndex(card.completion_method.findData("auto"))
        self.assertEqual(card.step.conditions, [Condition(label="part", roi="right")])
        self.assertEqual(card.condition_table.field_widget(0, "roi").currentData(), "right")
        self.assertFalse(card.condition_table.field_widget(0, "error").isVisible())

    def test_deleted_condition_undo_follows_region_rename(self):
        region = ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)])
        page = self.page([Step("A", conditions=[Condition(label="part", roi="left")])], rois=[region])
        card = self.open(page, 0)
        card.condition_table.field_widget(0, "remove").click()
        page.step_settings.close_button.click()
        page._rename_roi(region, "right")
        page._undo_roi()
        page._redo_roi()
        self.open(page, 0)
        self.assertTrue(card.condition_table.undo_delete_button.isEnabled())
        card.condition_table.undo_delete_button.click()
        self.assertEqual(card.step.conditions[0].roi, "right")
        self.assertFalse(card.condition_table.field_widget(0, "error").isVisible())

    def test_region_delete_explains_retained_completion_reference(self):
        region = ROI("left", [(0, 0), (.5, 0), (.5, .5), (0, .5)])
        page = self.page([Step("A", conditions=[Condition(label="part", roi="left")])], rois=[region])
        card = self.open(page, 0)
        card.completion_method.setCurrentIndex(card.completion_method.findData("manual"))
        page.step_settings.close_button.click()
        with patch("sop_app.ui.editor_page.QMessageBox.information") as explanation, patch(
                "sop_app.ui.editor_page.QMessageBox.question") as confirmation:
            page._delete_roi(region)
        self.assertIn(region, page.sop.rois)
        explanation.assert_called_once()
        self.assertIn("保留的完成條件", explanation.call_args.args[2])
        confirmation.assert_not_called()

    def test_open_and_close_do_not_resize_image_or_three_columns(self):
        page = self.page()
        widgets = (page.video, page.settings_pane, page.monitor_pane, page.tools_pane)
        original = tuple(widget.geometry() for widget in widgets)
        self.open(page, 1)
        self.assertEqual(tuple(widget.geometry() for widget in widgets), original)
        self.assertTrue(page.isAncestorOf(page.step_settings))
        self.assertTrue(page.step_settings.panel.isVisible())
        page.step_settings.close_button.click()
        self.settle()
        self.assertEqual(tuple(widget.geometry() for widget in widgets), original)

    def test_resize_between_narrow_and_wide_keeps_settings_in_front(self):
        page = self.page()
        card = self.open(page, 0)
        drawer = page.step_settings
        for width, height in ((1000, 600), (1489, 780)):
            page.resize(width, height)
            self.settle()
            self.assertEqual(drawer.geometry(), page.rect())
            self.assertIs(drawer.current_editor, card)
            for point in (drawer.close_button.mapTo(page, drawer.close_button.rect().center()),
                          QPoint(page.width() - 12, page.height() // 2)):
                foreground = page.childAt(point)
                self.assertIsNotNone(foreground)
                self.assertTrue(drawer.isAncestorOf(foreground), foreground)

    def test_global_management_only_shows_row_actions(self):
        page = self.page()
        page.edit_steps_button.click()
        self.settle()
        self.assertTrue(page.edit_steps_button.isChecked())
        self.assert_closed(page)
        for card in page.step_editors:
            self.assertTrue(card.actions_row.isVisible())
            self.assertTrue(card.edit_button.isVisible())
            self.assertTrue(card.body.isHidden())
        card = self.open(page, 1)
        page.step_settings.close_button.click()
        self.settle()
        self.assertTrue(card.actions_row.isVisible())
        self.assertTrue(card.body.isHidden())
        page.edit_steps_button.click()
        self.assertTrue(all(card.actions_row.isHidden() and card.body.isHidden()
                            for card in page.step_editors))

    def test_structural_changes_release_old_drawer_widgets(self):
        for operation in ("rebuild", "move", "duplicate", "delete"):
            with self.subTest(operation=operation):
                page = self.page()
                old = self.open(page, 1)
                step, body, actions = old.step, old.body, old.actions_row
                old.hold_spin.setValue(3.7)
                if operation == "rebuild":
                    page._rebuild_steps()
                elif operation == "move":
                    old.up_button.click()
                elif operation == "duplicate":
                    old.duplicate_button.click()
                else:
                    with patch("sop_app.ui.editor_page.QMessageBox.question",
                               return_value=QMessageBox.StandardButton.Yes):
                        old.delete_button.click()
                self.settle()
                if operation == "delete":
                    self.assert_closed(page)
                else:
                    self.assertTrue(page.step_settings.isVisible())
                    self.assertIs(page.step_settings.current_editor.step, step)
                    self.assertIsNot(page.step_settings.current_editor, old)
                self.flush_deletes()
                self.assertFalse(isValid(old))
                self.assertFalse(isValid(body))
                self.assertFalse(isValid(actions))
                self.assertTrue(all(card.body.isHidden() for card in page.step_editors
                                    if card.step is not step))
                if operation == "delete":
                    self.assertFalse(any(candidate is step for candidate in page.sop.steps))
                else:
                    self.assertEqual(step.hold_sec, 3.7)
                    self.assertEqual(page.step_settings.current_editor.hold_spin.value(), 3.7)
                if operation == "duplicate":
                    clone = page.sop.steps[2]
                    self.assertIsNot(clone, step)
                    self.assertEqual((clone.name, clone.hold_sec), ("B（複製）", 3.7))
                self.dispose(page)

    def test_loading_new_sop_closes_old_settings_and_rejects_old_step(self):
        page = self.page()
        old = self.open(page, 1)
        old_step, old_body = old.step, old.body
        replacement = Step("Replacement", hold_sec=4.)
        page.set_sop(SOPDefinition(name="New SOP", steps=[replacement]))
        self.settle()
        self.assert_closed(page)
        self.flush_deletes()
        self.assertFalse(isValid(old_body))
        page._open_step_settings(old_step)
        self.settle()
        self.assert_closed(page)
        self.open(page, 0)
        self.assertIs(page.step_settings.current_editor.step, replacement)
        self.assertEqual(page.step_settings.current_editor.hold_spin.value(), 4.)

    def test_production_and_page_hide_close_settings_without_losing_data(self):
        page = self.page()
        card = self.open(page, 0)
        card.hold_spin.setValue(2.8)
        page.set_production_running(True)
        self.settle()
        self.assert_closed(page)
        self.assertFalse(card.edit_button.isEnabled())
        page._open_step_settings(card.step)
        self.settle()
        self.assert_closed(page)
        self.assertEqual(card.step.hold_sec, 2.8)
        page.set_production_running(False)
        self.open(page, 0)
        page.hide()
        self.settle()
        self.assert_closed(page)
        self.assertIs(card.body.parentWidget(), card)
        page.show()
        self.settle()
        self.assert_closed(page)
        self.open(page, 0)
        self.assertEqual(card.hold_spin.value(), 2.8)

    def test_wheel_over_reparented_spin_scrolls_drawer_without_editing(self):
        page = self.page([Step("Many conditions", hold_sec=2., conditions=[
            Condition(label="part", min_count=i + 1) for i in range(12)
        ])], height=620)
        card = self.open(page, 0)
        drawer_scroll = page.step_settings.scroll
        drawer_scroll.ensureWidgetVisible(card.hold_spin, 0, 0)
        self.settle()
        spin = card.hold_spin
        spin.lineEdit().setFocus()
        self.settle()
        bar = drawer_scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        angle = QPoint(0, -120 if bar.value() < bar.maximum() else 120)
        before_scroll = bar.value()
        before_outer = (page.page_scroll.verticalScrollBar().value(),
                        page.tools_scroll.verticalScrollBar().value())
        before_sop = deepcopy(page.sop.to_dict())
        changes = []
        page.modified.connect(lambda: changes.append(True))
        target = spin.lineEdit()
        position = QPointF(target.rect().center())
        event = QWheelEvent(position, QPointF(target.mapToGlobal(target.rect().center())),
                            QPoint(), angle, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        self.app.sendEvent(target, event)
        self.settle()
        self.assertEqual(spin.value(), 2.)
        self.assertEqual(page.sop.to_dict(), before_sop)
        self.assertEqual(changes, [])
        self.assertNotEqual(bar.value(), before_scroll)
        self.assertEqual((page.page_scroll.verticalScrollBar().value(),
                          page.tools_scroll.verticalScrollBar().value()), before_outer)
        self.assertTrue(target.hasFocus())

    def test_escape_from_field_closes_drawer_and_keeps_changes(self):
        page = self.page()
        card = self.open(page, 0)
        card.hold_spin.setValue(3.5)
        card.hold_spin.lineEdit().setFocus()
        QTest.keyClick(card.hold_spin.lineEdit(), Qt.Key.Key_Escape)
        self.settle()
        self.assert_closed(page)
        self.assertTrue(page.isVisible())
        self.assertEqual(card.step.hold_sec, 3.5)
        self.assertIs(card.body.parentWidget(), card)
        self.assertTrue(card.body.isHidden())

    def test_escape_commits_pending_numeric_text(self):
        page = self.page()
        card = self.open(page, 0)
        field = card.hold_spin.lineEdit()
        field.setFocus()
        field.setText("7.25")
        QTest.keyClick(field, Qt.Key.Key_Escape)
        self.settle()
        self.assert_closed(page)
        self.assertEqual(card.step.hold_sec, 7.25)
        self.open(page, 0)
        self.assertEqual(card.hold_spin.value(), 7.25)

    def test_keyboard_navigation_stays_in_active_settings(self):
        page = self.page()
        self.open(page, 0)
        drawer = page.step_settings
        drawer.close_button.setFocus()
        for key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            for _ in range(45):
                focused = self.app.focusWidget()
                self.assertIsNotNone(focused)
                self.assertTrue(drawer.isAncestorOf(focused))
                QTest.keyClick(focused, key)
                self.settle()
                self.assertTrue(drawer.isAncestorOf(self.app.focusWidget()))


if __name__ == "__main__":
    unittest.main()
