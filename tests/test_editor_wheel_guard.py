"""Real SOP-editor wheel interactions; no camera, model, files or records."""
from __future__ import annotations

from copy import deepcopy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from unittest.mock import patch
import warnings

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QScrollArea, QSpinBox, QStyle, QStyleOptionSpinBox
from shiboken6 import isValid

from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.editor_wheel_guard import EditorWheelGuard


class WheelCapture(QObject):
    def __init__(self, watched, consume=False):
        super().__init__(watched)
        self.events = []
        self.consume = consume
        watched.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel:
            self.events.append({"angle": event.angleDelta(), "pixel": event.pixelDelta(),
                "phase": event.phase(), "inverted": event.inverted(), "source": event.source(),
                "modifiers": event.modifiers(), "buttons": event.buttons(),
                "timestamp": event.timestamp(), "device": event.device(),
                "position": event.position(), "globalPosition": event.globalPosition(),
                "spontaneous": event.spontaneous()})
            if self.consume:
                event.accept()
                return True
        return False


class ObservedEditorWheelGuard(EditorWheelGuard):
    """Count real application dispatch while retaining the production behavior."""
    def __init__(self, root, target, visits):
        self.observed_target = target
        self.observed_visits = visits
        super().__init__(root)

    def eventFilter(self, watched, event):
        if watched is self.observed_target:
            self.observed_visits.append(event.type())
        return super().eventFilter(watched, event)


class EditorWheelGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def settle(self):
        for _ in range(8):
            self.app.processEvents()

    def setUp(self):
        rois = [ROI(f"R{i:02}", [(.1, .1), (.9, .1), (.9, .9), (.1, .9)]) for i in range(18)]
        steps = [Step(f"Step {i + 1}", hold_sec=2., ratio=.8,
                      conditions=[Condition(label="part", roi="R00", min_count=5, min_score=.8)],
                      forbidden=[Condition(label="other")]) for i in range(7)]
        self.page = EditorPage()
        # No video feed is part of this fixture. Its periodic validation refresh
        # would otherwise replace an injected log while testing text scrolling.
        self.page._validation_watchdog.stop()
        self.page.resize(1700, 740)
        self.page.set_sop(SOPDefinition(name="Wheel fixture", steps=steps, rois=rois))
        self.page.set_classes(["part", "other", *(f"object{i:02}" for i in range(60))])
        self.page.set_model_state("available")
        self.page.edit_rois_button.setChecked(True)
        self.page.show()
        self.page.activateWindow()
        self.settle()
        self.page._open_step_settings(steps[0])
        self.settle()
        self.modified = []
        self.page.modified.connect(lambda: self.modified.append(True))
        self.addCleanup(self.dispose)
        self.assertGreater(self.page.page_scroll.verticalScrollBar().maximum(), 0)
        self.assertGreater(self.page.tools_scroll.verticalScrollBar().maximum(), 0)
        self.assertGreater(self.page.step_settings.scroll.verticalScrollBar().maximum(), 0)

    def dispose(self, page=None):
        page = self.page if page is None else page
        popup = self.app.activePopupWidget()
        if popup:
            popup.close()
        for timer in page.findChildren(QTimer):
            timer.stop()
        page.close()
        page.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def nearest_scroll(self, widget):
        ancestor = widget.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QScrollArea):
                return ancestor
            ancestor = ancestor.parentWidget()
        self.fail(f"No containing scroll area for {type(widget).__name__}")

    def expose(self, widget, scroll=None):
        # Step details move from their list row into the same-window sidebar.
        # Always find their actual containing pane after such reparenting.
        actual = self.nearest_scroll(widget)
        if scroll is not None:
            self.assertIs(actual, scroll)
        actual.ensureWidgetVisible(widget, 0, 20)
        self.settle()
        return actual

    def open_step(self, editor):
        self.page._open_step_settings(editor.step)
        self.settle()

    def make_automatic(self, editor):
        editor.completion_method.setCurrentIndex(editor.completion_method.findData("auto"))
        self.settle()

    def wheel(self, target, *, angle=QPoint(0, -120), pixel=QPoint(),
              phase=Qt.ScrollPhase.NoScrollPhase, inverted=False,
              modifiers=Qt.KeyboardModifier.NoModifier,
              source=Qt.MouseEventSource.MouseEventNotSynthesized, timestamp=1234, offset=QPointF()):
        position = QPointF(target.rect().center()) + offset
        global_position = QPointF(target.mapToGlobal(target.rect().center())) + offset
        event = QWheelEvent(position, global_position, pixel, angle,
                            Qt.MouseButton.NoButton, modifiers, phase, inverted, source)
        event.setTimestamp(timestamp)
        self.app.sendEvent(target, event)
        self.settle()
        return event

    def assert_no_edit(self, target, **wheel_args):
        before = deepcopy(self.page.sop.to_dict())
        self.modified.clear()
        event = self.wheel(target, **wheel_args)
        self.assertEqual(self.page.sop.to_dict(), before)
        self.assertEqual(self.modified, [], "Scrolling must not mark the SOP as modified")
        return event

    def test_focused_unfocused_spin_and_inner_lineedit_never_change_values(self):
        first = self.page.step_editors[0]
        for spin in (first.hold_spin, first.condition_table.field_widget(0, "min_count")):
            for target in (spin, spin.lineEdit()):
                for focused in (False, True):
                    with self.subTest(kind=type(spin).__name__, inner=target is not spin, focused=focused):
                        self.expose(target, self.page.step_settings.scroll)
                        if focused:
                            target.setFocus()
                            self.settle()
                            self.assertTrue(target.hasFocus())
                        else:
                            self.page.name_edit.setFocus()
                            self.settle()
                            self.assertFalse(target.hasFocus())
                        value = spin.value()
                        self.assert_no_edit(target)
                        self.assertEqual(spin.value(), value)

    def test_wheel_does_not_clear_focus_or_commit_pending_numeric_text(self):
        spin = self.page.step_editors[0].hold_spin
        self.expose(spin, self.page.step_settings.scroll)
        spin.setKeyboardTracking(False)
        spin.lineEdit().setFocus()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), "3.7")
        self.assertEqual(spin.value(), 2.)
        text = spin.lineEdit().text()
        self.assert_no_edit(spin.lineEdit())
        self.assertTrue(spin.lineEdit().hasFocus())
        self.assertEqual(spin.lineEdit().text(), text)
        self.assertEqual(spin.value(), 2.)

    def assert_spontaneous_wheel_preserves_another_pending_field(self, editing, hovered):
        editing.setKeyboardTracking(False)
        scroll = self.expose(editing)
        editing.lineEdit().setFocus()
        editing.lineEdit().selectAll()
        QTest.keyClicks(editing.lineEdit(), "3.7")
        self.assertEqual(editing.value(), 2.)
        self.expose(hovered, scroll)
        self.assertTrue(editing.lineEdit().hasFocus())
        window = self.page.windowHandle()
        self.assertIsNotNone(window)
        window_capture = WheelCapture(window)
        viewport_capture = WheelCapture(scroll.viewport())
        position = hovered.mapTo(self.page, hovered.rect().center())
        viewport_position = hovered.mapTo(scroll.viewport(), hovered.rect().center())
        self.assertTrue(scroll.viewport().rect().contains(viewport_position))
        QTest.mouseMove(self.page, position)
        self.settle()
        self.assertTrue(editing.lineEdit().hasFocus())
        before = deepcopy(self.page.sop.to_dict())
        pending = editing.lineEdit().text()
        hovered_value = hovered.value()
        bar = scroll.verticalScrollBar()
        before_scroll = bar.value()
        angle = QPoint(0, -120 if before_scroll < bar.maximum() else 120)
        self.modified.clear()
        numeric_changes, editing_finished = [], []
        editing.valueChanged.connect(numeric_changes.append)
        editing.editingFinished.connect(lambda: editing_finished.append(True))
        # Unlike QApplication.sendEvent, the QWindow route exercises Qt's
        # spontaneous input handling, including wheel-focus before app filters.
        QTest.wheelEvent(window, QPointF(position), angle)
        self.settle()
        self.assertEqual(len(window_capture.events), 1)
        self.assertTrue(window_capture.events[0]["spontaneous"])
        self.assertEqual(len(viewport_capture.events), 1)
        self.assertNotEqual(bar.value(), before_scroll)
        self.assertTrue(editing.lineEdit().hasFocus(), "Wheel must not transfer focus to the hovered field")
        self.assertEqual(editing.lineEdit().text(), pending)
        self.assertEqual(editing.value(), 2., "A pending edit must not be committed by scrolling another field")
        self.assertEqual(hovered.value(), hovered_value)
        self.assertEqual(self.page.sop.to_dict(), before)
        self.assertEqual(self.modified, [])
        self.assertEqual(numeric_changes, [])
        self.assertEqual(editing_finished, [])

    def test_spontaneous_window_wheel_preserves_focus_and_another_fields_pending_text(self):
        step = self.page.step_editors[0]
        self.assert_spontaneous_wheel_preserves_another_pending_field(step.hold_spin, step.ratio_spin)

    def test_spontaneous_window_wheel_protects_a_late_rebuilt_condition_field(self):
        self.page.add_step_button.click()
        self.settle()
        added = self.page.step_editors[-1]
        self.open_step(added)
        self.make_automatic(added)
        added.hold_spin.setValue(2.)
        added.condition_table.add_button.click()
        self.settle()
        hovered = added.condition_table.field_widget(0, "min_count")
        self.assert_spontaneous_wheel_preserves_another_pending_field(added.hold_spin, hovered)

    def test_closed_and_editable_combos_keep_selection_even_when_focused(self):
        combo = self.page.step_editors[0].completion_mode
        for editable in (False, True):
            combo.setEditable(editable)
            targets = (combo, combo.lineEdit()) if editable else (combo,)
            for target in targets:
                for focused in (False, True):
                    with self.subTest(editable=editable, inner=target is not combo, focused=focused):
                        self.expose(target, self.page.step_settings.scroll)
                        (target if focused else self.page.name_edit).setFocus()
                        self.settle()
                        index, text = combo.currentIndex(), combo.currentText()
                        self.assert_no_edit(target)
                        self.assertEqual((combo.currentIndex(), combo.currentText()), (index, text))

    def test_new_steps_rebuilt_conditions_and_rebuilt_roi_rows_are_protected(self):
        self.page.add_step_button.click()
        self.settle()
        added = self.page.step_editors[-1]
        self.open_step(added)
        self.make_automatic(added)
        added.condition_table.add_button.click()
        self.settle()
        # Adding a condition rebuilt all controls, including their inner editors.
        for widget in (added.hold_spin, added.condition_table.field_widget(0, "min_score").lineEdit(),
                       added.condition_table.field_widget(0, "label")):
            self.expose(widget, self.page.step_settings.scroll)
            self.assert_no_edit(widget)
        self.page.step_settings.close()
        self.page._on_roi_drawn([(.2, .2), (.4, .2), (.4, .4), (.2, .4)])
        self.page.edit_rois_button.setChecked(True)
        self.settle()
        anchor = self.page.roi_rows[-1].anchor_combo
        self.expose(anchor, self.page.tools_scroll)
        self.assert_no_edit(anchor)

    def test_each_column_scrolls_once_at_the_pointer_despite_other_column_focus(self):
        sidebar, main, tools = self.page.step_settings.scroll, self.page.page_scroll, self.page.tools_scroll
        captures = {pane: WheelCapture(pane.viewport()) for pane in (sidebar, main, tools)}
        cases = ((self.page.step_editors[0].hold_spin, sidebar, self.page.validation_mode),
                 (self.page.roi_rows[0].anchor_combo, tools, self.page.auto_restart_spin),
                 (self.page.auto_restart_spin, main, self.page.validation_mode))
        for field, owned, opposite_focus in cases:
            with self.subTest(pane="sidebar" if owned is sidebar else "tools" if owned is tools else "main"):
                if owned is not sidebar:
                    self.page.step_settings.close()
                    self.settle()
                self.expose(field, owned)
                # An explicitly focused field in the opposite column must not
                # redirect a wheel that occurs under this pointer.
                opposite_focus.setFocus()
                self.settle()
                owned_bar = owned.verticalScrollBar()
                owned_bar.setValue(0)
                other_positions = {pane: pane.verticalScrollBar().value() for pane in captures if pane is not owned}
                self.wheel(owned.viewport())
                expected = owned_bar.value()
                self.assertGreater(expected, 0)
                owned_bar.setValue(0)
                for capture in captures.values():
                    capture.events.clear()
                self.assert_no_edit(field)
                self.assertEqual(owned_bar.value(), expected, "One wheel must equal one native viewport scroll")
                for pane, capture in captures.items():
                    if pane is owned:
                        self.assertEqual(len(capture.events), 1)
                    else:
                        self.assertEqual(capture.events, [])
                        self.assertEqual(pane.verticalScrollBar().value(), other_positions[pane])

    def test_touchpad_event_metadata_reaches_the_nearest_viewport_unchanged_once(self):
        spin = self.page.step_editors[0].hold_spin
        scroll = self.expose(spin, self.page.step_settings.scroll)
        capture = WheelCapture(scroll.viewport(), consume=True)
        cases = ((QPoint(0, -120), QPoint(0, -17), Qt.ScrollPhase.ScrollUpdate),
                 (QPoint(), QPoint(0, -17), Qt.ScrollPhase.ScrollUpdate),
                 (QPoint(), QPoint(), Qt.ScrollPhase.ScrollBegin),
                 (QPoint(), QPoint(), Qt.ScrollPhase.ScrollEnd))
        for angle, pixel, phase in cases:
            with self.subTest(angle=angle, pixel=pixel, phase=phase):
                capture.events.clear()
                original = self.assert_no_edit(spin.lineEdit(), angle=angle, pixel=pixel,
                                    phase=phase, inverted=True, modifiers=Qt.KeyboardModifier.AltModifier,
                                    source=Qt.MouseEventSource.MouseEventSynthesizedBySystem, timestamp=4321,
                                    offset=QPointF(.25, .75))
                self.assertEqual(len(capture.events), 1)
                event = capture.events[0]
                self.assertEqual(event["pixel"], pixel)
                self.assertEqual(event["angle"], angle)
                self.assertEqual(event["phase"], phase)
                self.assertTrue(event["inverted"])
                self.assertEqual(event["modifiers"], Qt.KeyboardModifier.AltModifier)
                self.assertEqual(event["source"], Qt.MouseEventSource.MouseEventSynthesizedBySystem)
                self.assertEqual(event["timestamp"], 4321)
                self.assertEqual(event["buttons"], Qt.MouseButton.NoButton)
                self.assertEqual(event["device"], original.device())
                self.assertEqual(event["globalPosition"], original.globalPosition())
                self.assertEqual(event["position"], original.globalPosition() - QPointF(
                    scroll.viewport().mapToGlobal(QPoint())))

    def test_top_and_bottom_limits_do_not_fall_back_to_editing_or_other_column(self):
        sidebar, main, tools = self.page.step_settings.scroll, self.page.page_scroll, self.page.tools_scroll
        for spin, scroll, at_bottom, angle in (
                (self.page.step_editors[0].hold_spin, sidebar, False, QPoint(0, 120)),
                (self.page.step_editors[0].forbidden_ratio_spin, sidebar, True, QPoint(0, -120)),
                (self.page.auto_restart_spin, main, True, QPoint(0, -120))):
            with self.subTest(bottom=at_bottom):
                if scroll is main:
                    self.page.step_settings.close()
                    self.settle()
                self.expose(spin, scroll)
                bar = scroll.verticalScrollBar()
                bar.setValue(bar.maximum() if at_bottom else 0)
                before = tuple(pane.verticalScrollBar().value() for pane in (sidebar, main, tools)), spin.value()
                self.assert_no_edit(spin, angle=angle)
                self.assertEqual((tuple(pane.verticalScrollBar().value() for pane in (sidebar, main, tools)), spin.value()), before)

    def test_keyboard_text_and_arrow_buttons_still_edit_the_real_sop(self):
        spin = self.page.step_editors[0].hold_spin
        self.expose(spin, self.page.step_settings.scroll)
        spin.setFocus()
        QTest.keyClick(spin, Qt.Key.Key_Up)
        self.assertAlmostEqual(self.page.sop.steps[0].hold_sec, 2.1)
        option = QStyleOptionSpinBox()
        spin.initStyleOption(option)
        up = spin.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option,
                                        QStyle.SubControl.SC_SpinBoxUp, spin)
        self.assertFalse(up.isEmpty())
        QTest.mouseClick(spin, Qt.MouseButton.LeftButton, pos=up.center())
        self.assertAlmostEqual(self.page.sop.steps[0].hold_sec, 2.2)
        spin.setKeyboardTracking(False)
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), "4.2")
        QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Return)
        self.assertAlmostEqual(self.page.sop.steps[0].hold_sec, 4.2)
        combo = self.page.step_editors[0].completion_mode
        combo.setFocus()
        QTest.keyClick(combo, Qt.Key.Key_Down)
        self.assertEqual(self.page.sop.steps[0].completion_mode, "any")
        self.assertTrue(self.modified)

    def test_open_combo_popup_scrolls_its_choices_without_scrolling_the_page(self):
        combo = self.page.step_editors[0].condition_table.field_widget(0, "label")
        self.expose(combo, self.page.step_settings.scroll)
        combo.setMaxVisibleItems(5)
        combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
        combo.showPopup()
        self.settle()
        view = combo.view()
        self.assertTrue(view.isVisible())
        self.assertGreater(view.verticalScrollBar().maximum(), 0)
        view.verticalScrollBar().setValue(0)
        panes = (self.page.step_settings.scroll, self.page.page_scroll, self.page.tools_scroll)
        outer = tuple(pane.verticalScrollBar().value() for pane in panes)
        self.assert_no_edit(view.viewport())
        self.assertGreater(view.verticalScrollBar().value(), 0)
        self.assertEqual(tuple(pane.verticalScrollBar().value() for pane in panes), outer)
        combo.hidePopup()

    def test_long_instruction_and_trial_log_keep_their_own_text_scrolling(self):
        instruction = self.page.step_editors[0].instruction_edit
        instruction.setPlainText("\n".join(f"Instruction line {i}" for i in range(60)))
        log = self.page.validation_details
        log.setPlainText("\n".join(f"Log entry {i}" for i in range(100)))
        self.settle()
        for text, outer in ((instruction, self.page.step_settings.scroll), (log, self.page.tools_scroll)):
            with self.subTest(text=type(text).__name__):
                if outer is self.page.tools_scroll:
                    self.page.step_settings.close()
                    self.settle()
                self.expose(text, outer)
                self.assertGreater(text.verticalScrollBar().maximum(), 0)
                text.verticalScrollBar().setValue(0)
                before_outer = outer.verticalScrollBar().value()
                self.assert_no_edit(text.viewport())
                self.assertGreater(text.verticalScrollBar().value(), 0)
                self.assertEqual(outer.verticalScrollBar().value(), before_outer)

    def test_unrelated_controls_outside_the_editor_keep_their_native_wheel_behavior(self):
        outside = QSpinBox()
        self.addCleanup(outside.close)
        outside.setRange(0, 10)
        outside.setValue(5)
        outside.show()
        self.settle()
        self.wheel(outside)
        self.assertEqual(outside.value(), 4)

        # QObject parentage alone must not extend the editor rule to a separate
        # window, such as an editor-owned device/configuration dialog.
        dialog = QDialog(self.page)
        self.addCleanup(dialog.close)
        inside_dialog = QSpinBox(dialog)
        inside_dialog.setRange(0, 10)
        inside_dialog.setValue(5)
        dialog.resize(200, 100)
        dialog.show()
        self.settle()
        self.wheel(inside_dialog)
        self.assertEqual(inside_dialog.value(), 4)

    def test_hidden_or_closed_retained_editor_does_not_receive_unrelated_app_events(self):
        outside = QSpinBox()
        self.addCleanup(outside.close)
        outside.show()
        self.settle()
        visits = []

        # PySide caches an eventFilter override when constructing its QObject;
        # install this observing subclass before construction, not by patching a
        # method on an existing object that Qt may already have cached.
        def install(root):
            root._wheel_guard = ObservedEditorWheelGuard(root, outside, visits)
            return root._wheel_guard

        with patch("sop_app.ui.editor_page.install_editor_wheel_guard", new=install):
            probe = EditorPage()
        self.addCleanup(lambda: self.dispose(probe) if isValid(probe) else None)
        probe.resize(1200, 700)
        probe.show()
        self.settle()

        def pulse():
            visits.clear()
            self.app.sendEvent(outside, QEvent(QEvent.Type.User))
            return len(visits)

        self.assertEqual(pulse(), 1, "Positive control: a visible editor has one active application filter")
        for hide in (probe.hide, probe.close):
            hide()
            self.settle()
            self.assertEqual(pulse(), 0, "Retaining a hidden/closed editor must add no application-event callbacks")
            probe.show()
            self.settle()
            self.assertEqual(pulse(), 1, "Showing again must restore one filter")
        destroyed = []
        probe.destroyed.connect(lambda: destroyed.append(True))
        probe.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertEqual(destroyed, [True])
        self.assertEqual(pulse(), 0)

    def test_reshown_editor_protects_controls_created_while_hidden(self):
        self.page.hide()
        self.settle()
        self.page.add_step_button.click()
        added = self.page.step_editors[-1]
        self.make_automatic(added)
        added.condition_table.add_button.click()
        self.settle()
        hovered = added.condition_table.field_widget(0, "min_count")
        for cycle in range(2):
            with self.subTest(show_cycle=cycle):
                self.page.show()
                self.page.activateWindow()
                # Offscreen Qt does not restore QApplication's active QWidget
                # consistently after hide/show. Establish that fixture state;
                # the wheel itself still enters through the spontaneous QWindow
                # path and must never move focus between numeric fields.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    self.app.setActiveWindow(self.page)
                self.settle()
                self.open_step(added)
                added.hold_spin.setValue(2.)
                self.assert_spontaneous_wheel_preserves_another_pending_field(added.hold_spin, hovered)
                if cycle == 0:
                    self.page.hide()
                    self.settle()


if __name__ == "__main__":
    unittest.main()
