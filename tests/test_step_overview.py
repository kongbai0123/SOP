"""Whole SOP entries wrap and pack into columns without becoming engine controls."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEvent, QRectF, Qt
from PySide6.QtGui import QFont, QFontDatabase, QHelpEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollBar, QWidget

from sop_app.engine import EngineSnapshot, Phase, StepRuntime, StepStatus
from sop_app.ui.step_overview import StepOverview


def snapshot(count, current=0, *, phase=Phase.RUNNING, now=None, cycle=1):
    steps = []
    for index in range(count):
        done = phase == Phase.CYCLE_DONE or (phase == Phase.RUNNING and index < current)
        active = phase == Phase.RUNNING and index == current
        steps.append(StepRuntime(status=StepStatus.DONE if done else StepStatus.ACTIVE if active else StepStatus.PENDING,
                                 started_at=float(index) if done or active else None,
                                 ended_at=float(index) + .5 if done else None))
    return EngineSnapshot(phase, cycle, current, float(current) + .25 if now is None else now,
                          True, 0., None, steps, [], [], 0.)


class StepOverviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cjk_font = Path("C:/Windows/Fonts/msjh.ttc")
        if cjk_font.exists():
            QFontDatabase.addApplicationFont(str(cjk_font))

    def overview(self, count=30, *, names=None, width=640, height=360):
        widget = StepOverview()
        self.addCleanup(widget.close)
        widget.setFont(QFont("Microsoft JhengHei", 10))
        widget.set_steps(names if names is not None else [f"工序 {index + 1}" for index in range(count)])
        widget.resize(width, height)
        widget.show()
        self.app.processEvents()
        return widget

    def list_height(self, widget, height):
        """Set the usable list frame height, independently of the display size."""
        footer = widget.height() - widget._grid.height()
        widget.resize(widget.width(), round(footer + 2 * widget._grid.MARGIN + height))
        self.app.processEvents()
        self.assertAlmostEqual(widget._grid.height() - 2 * widget._grid.MARGIN, height, delta=1.)

    def first_right(self, widget):
        return next(index for index in widget.visible_indices if widget.visible_slots[index][1] == 1)

    def assert_whole_entry_layout(self, widget, count):
        original_page, following = widget.page_index, widget.is_following
        pages = [widget.page_indices(page) for page in range(widget.page_count)]
        self.assertEqual(tuple(index for page in pages for index in page), tuple(range(count)))
        for page, indices in enumerate(pages):
            with self.subTest(page=page):
                self.assertTrue(indices)
                widget.show_page(page)
                self.assertEqual(widget.visible_indices, indices)
                bounds = QRectF(widget._grid.geometry()).adjusted(widget._grid.MARGIN, widget._grid.MARGIN,
                                                                 -widget._grid.MARGIN, -widget._grid.MARGIN)
                columns = {0: [], 1: []}
                for index in indices:
                    self.assertEqual(widget.page_for_index(index), page)
                    entry = widget.entry_rect(index)
                    self.assertFalse(entry.isEmpty())
                    self.assertTrue(bounds.contains(entry), f"entry {index} outside {bounds}: {entry}")
                    self.assertEqual(widget.card_rect(index), entry)
                    self.assertGreaterEqual(widget.entry_line_count(index), 1)
                    self.assertGreater(widget.entry_natural_height(index), 0.)
                    if not widget.entry_is_oversized(index):
                        self.assertAlmostEqual(entry.height(), widget.entry_natural_height(index), delta=1.)
                    column = widget.visible_slots[index][1]
                    self.assertIn(column, (0, 1))
                    columns[column].append(index)
                self.assertEqual(tuple(columns[0] + columns[1]), indices)
                for column, members in columns.items():
                    for row, index in enumerate(members):
                        self.assertEqual(widget.visible_slots[index], (row, column))
                        if row:
                            previous = widget.entry_rect(members[row - 1])
                            self.assertGreaterEqual(widget.entry_rect(index).top(), previous.bottom())
                for offset, index in enumerate(indices):
                    for other in indices[offset + 1:]:
                        self.assertFalse(widget.entry_rect(index).intersects(widget.entry_rect(other)))
                self.assertTrue(widget.entry_rect(-1).isEmpty())
                for index in range(count):
                    if index not in indices:
                        self.assertTrue(widget.entry_rect(index).isEmpty())
        if following:
            widget.follow_current()
        else:
            widget.show_page(original_page)

    def test_short_names_fill_actual_column_height_before_moving_right(self):
        widget = self.overview(8)
        self.list_height(widget, widget.entry_natural_height(0) * 10)
        self.assertEqual(widget.page_indices(0), tuple(range(8)))
        self.assertEqual(widget.visible_slots, {index: (index, 0) for index in range(8)})
        widget.set_steps([f"工序 {index + 1}" for index in range(21)])
        self.assertEqual(widget.page_indices(0), tuple(range(20)))
        self.assertEqual(widget.page_indices(1), (20,))
        self.assertEqual(widget.visible_slots[9], (9, 0))
        self.assertEqual(widget.visible_slots[10], (0, 1))
        self.assertEqual(widget.visible_slots[19], (9, 1))
        self.assert_whole_entry_layout(widget, 21)

    def test_right_column_start_changes_with_the_list_frame_height(self):
        widget = self.overview(40)
        unit = widget.entry_natural_height(0)
        self.list_height(widget, unit * 4 + unit / 3)
        lower_boundary = self.first_right(widget)
        lower_height = widget._grid.height()
        self.list_height(widget, unit * 7 + unit / 3)
        self.assertGreater(widget._grid.height(), lower_height)
        self.assertGreater(self.first_right(widget), lower_boundary)
        self.assertEqual(widget.entry_rect(0).top(), widget.entry_rect(self.first_right(widget)).top())
        self.assert_whole_entry_layout(widget, 40)

    def test_names_wrap_cjk_unbroken_english_and_explicit_newlines(self):
        names = ["鎖附", "第一行\n第二行", "請確認工件方向並鎖緊所有螺絲" * 8,
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 12]
        widget = self.overview(names=names, width=600, height=560)
        self.assertEqual(widget.entry_line_count(0), 1)
        self.assertEqual(widget.entry_line_count(1), 2)
        self.assertGreater(widget.entry_line_count(2), 2)
        self.assertGreater(widget.entry_line_count(3), 2)
        self.assertGreater(widget.entry_natural_height(1), widget.entry_natural_height(0))
        for name in names:
            self.assertIn(name, widget.accessibleDescription())
        self.assert_whole_entry_layout(widget, len(names))

    def test_entry_that_does_not_fit_remaining_space_moves_whole_to_next_column(self):
        widget = self.overview(names=["定位", "第一行\n第二行\n第三行", "鎖附"])
        short_height = widget.entry_natural_height(0)
        long_height = widget.entry_natural_height(1)
        self.list_height(widget, short_height + long_height - 1.)
        self.assertEqual(widget.visible_slots[0], (0, 0))
        self.assertEqual(widget.visible_slots[1], (0, 1))
        self.assertEqual(widget.entry_rect(0).top(), widget.entry_rect(1).top())
        self.assertAlmostEqual(widget.entry_rect(1).height(), long_height, delta=1.)
        self.assertGreater(long_height, widget._grid.height() - 2 * widget._grid.MARGIN - short_height)
        self.assert_whole_entry_layout(widget, 3)

    def test_eighth_or_ninth_long_name_is_never_split_across_columns(self):
        for long_index in (7, 8):
            with self.subTest(long_step=long_index + 1):
                names = [f"工序 {index + 1}" for index in range(12)]
                names[long_index] = "確認安裝方向\n完成第一顆螺絲\n完成第二顆螺絲"
                widget = self.overview(names=names)
                self.list_height(widget, widget.entry_natural_height(0) * 8)
                self.assertEqual(widget.page_for_index(long_index), 0)
                self.assertEqual(widget.visible_slots[long_index], (0, 1))
                self.assertEqual(widget.entry_rect(long_index).top(), widget.entry_rect(0).top())
                self.assertEqual(widget.entry_line_count(long_index), 3)
                self.assert_whole_entry_layout(widget, len(names))

    def test_variable_heights_produce_variable_page_counts_without_reordering(self):
        names = ["定位"] * 8 + ["第一行\n第二行\n第三行"] * 8 + ["鎖附"] * 8
        widget = self.overview(names=names)
        self.list_height(widget, widget.entry_natural_height(0) * 5)
        self.assertGreater(widget.page_count, 1)
        self.assertGreater(len({len(widget.page_indices(page)) for page in range(widget.page_count)}), 1)
        self.assert_whole_entry_layout(widget, len(names))

    def test_resize_narrower_or_taller_repacks_whole_entries(self):
        names = ["螺絲鎖附並確認工件安裝方向" * 4 for _ in range(30)]
        widget = self.overview(names=names, width=960, height=340)
        wide_height = widget.entry_natural_height(0)
        wide_pages = widget.page_count
        widget.resize(480, widget.height())
        self.app.processEvents()
        self.assertGreater(widget.entry_natural_height(0), wide_height)
        self.assertGreaterEqual(widget.page_count, wide_pages)
        narrow_pages = widget.page_count
        narrow_first_page = len(widget.page_indices(0))
        widget.resize(widget.width(), 620)
        self.app.processEvents()
        self.assertLessEqual(widget.page_count, narrow_pages)
        self.assertGreaterEqual(len(widget.page_indices(0)), narrow_first_page)
        self.assert_whole_entry_layout(widget, len(names))

    def test_empty_and_pagination_boundaries_never_show_invalid_indices(self):
        widget = self.overview(0)
        self.assertEqual((widget.page_index, widget.page_count, widget.visible_indices), (0, 0, ()))
        self.assertEqual(widget.page_label.text(), "0 / 0")
        self.assertEqual(widget.page_indices(-1), ())
        self.assertEqual(widget.page_indices(0), ())
        self.assertIsNone(widget.page_for_index(-1))
        self.assertIsNone(widget.page_for_index(0))
        self.assertFalse(widget.previous_button.isEnabled())
        self.assertFalse(widget.next_button.isEnabled())
        widget.set_snapshot(None)
        widget.show_page(500)
        self.assertEqual(widget.page_index, 0)
        widget.set_steps([f"工序 {index + 1}" for index in range(50)])
        widget.show_page(999)
        self.assertEqual(widget.page_index, widget.page_count - 1)
        self.assertEqual(widget.visible_indices, widget.page_indices(widget.page_count - 1))
        self.assertEqual(widget.page_indices(widget.page_count), ())
        self.assertIsNone(widget.page_for_index(50))
        widget.show_page(-99)
        self.assertEqual(widget.page_index, 0)
        self.assertEqual(widget.visible_indices, widget.page_indices(0))

    def test_last_item_stays_visible_until_done_and_next_step_becomes_current(self):
        widget = self.overview(60, height=240)
        first_page = widget.page_indices(0)
        last = first_page[-1]
        next_index = widget.page_indices(1)[0]
        self.assertEqual(widget.visible_slots[last][1], 1)
        widget.set_running(True)
        active = snapshot(60, last)
        original = deepcopy(active)
        widget.set_snapshot(active)
        self.assertEqual(widget.page_index, 0)
        widget.set_snapshot(snapshot(60, last, now=float(last) + 3.))
        self.assertEqual(widget.page_index, 0)
        completed = snapshot(60, last)
        completed.steps[last].status = StepStatus.DONE
        completed.steps[last].ended_at = float(last) + .5
        widget.set_snapshot(completed)
        self.assertEqual(widget.page_index, 0)
        next_state = snapshot(60, next_index)
        widget.set_snapshot(next_state)
        self.assertEqual(widget.page_index, 1)
        self.assertIn(next_index, widget.visible_indices)
        self.assertEqual(active, original)
        self.assertEqual(next_state.current_index, next_index)

    def test_unfinished_page_boundary_blocks_follow_until_done_or_skipped(self):
        for completed_status in (StepStatus.DONE, StepStatus.SKIPPED):
            with self.subTest(completed_status=completed_status):
                widget = self.overview(60, height=240)
                last = widget.page_indices(0)[-1]
                next_index = widget.page_indices(1)[0]
                widget.set_running(True)
                widget.set_snapshot(snapshot(60, last))
                next_state = snapshot(60, next_index)
                next_state.steps[last].status = StepStatus.ACTIVE
                widget.set_snapshot(next_state)
                self.assertEqual(widget.page_index, 0)
                completed = deepcopy(next_state)
                completed.steps[last].status = completed_status
                widget.set_snapshot(completed)
                self.assertEqual(widget.page_index, 1)
                self.assertIn(next_index, widget.visible_indices)

    def test_idle_updates_do_not_turn_pages_or_claim_interrupted_step_is_running(self):
        widget = self.overview(60, height=240)
        later_index = widget.page_indices(2)[0]
        widget.set_snapshot(snapshot(60, later_index, phase=Phase.IDLE))
        self.assertEqual(widget.page_index, 0)
        widget.set_running(True)
        widget.set_snapshot(snapshot(60, later_index))
        stopped_page = widget.page_index
        stopped = snapshot(60, later_index)
        stopped.phase = Phase.IDLE  # The engine retains an interrupted ACTIVE runtime.
        widget.set_running(False)
        widget.set_snapshot(stopped)
        self.assertEqual(widget.page_index, stopped_page)
        self.assertIn("已停止", widget.accessibleDescription())
        self.assertNotIn("執行中", widget.accessibleDescription())

    def test_manual_browsing_and_return_never_modify_the_snapshot(self):
        widget = self.overview(70, height=240)
        current = widget.page_indices(2)[0]
        state = snapshot(70, current)
        original = deepcopy(state)
        widget.set_running(True)
        widget.set_snapshot(state)
        self.assertEqual(widget.page_index, widget.page_for_index(current))
        widget.show_page(0)
        self.assertFalse(widget.is_following)
        self.assertFalse(widget.follow_button.isHidden())
        later = snapshot(70, widget.page_indices(3)[0])
        widget.set_snapshot(later)
        self.assertEqual(widget.page_index, 0)
        position = widget._grid.mapFrom(widget, widget.entry_rect(widget.visible_indices[0]).center().toPoint())
        QTest.mouseClick(widget._grid, Qt.MouseButton.LeftButton, pos=position)
        self.assertEqual(state, original)
        self.assertEqual(later.current_index, widget.page_indices(3)[0])
        widget.follow_button.click()
        self.assertTrue(widget.is_following)
        self.assertEqual(widget.page_index, widget.page_for_index(later.current_index))
        self.assertTrue(widget.follow_button.isHidden())

    def test_resize_keeps_followed_step_or_manual_anchor_visible(self):
        names = ["請確認安裝位置及鎖附方向" * (1 + index % 3) for index in range(60)]
        widget = self.overview(names=names)
        widget.set_running(True)
        widget.set_snapshot(snapshot(60, 35))
        widget.resize(480, 260)
        self.app.processEvents()
        self.assertIn(35, widget.visible_indices)
        widget.show_page(2)
        anchor = widget.visible_indices[0]
        widget.resize(760, 480)
        self.app.processEvents()
        self.assertFalse(widget.is_following)
        self.assertIn(anchor, widget.visible_indices)
        self.assertEqual(widget.page_index, widget.page_for_index(anchor))

    def test_cycle_done_keeps_last_page_and_next_cycle_restores_follow(self):
        widget = self.overview(60, height=240)
        widget.set_running(True)
        done = snapshot(60, 59, phase=Phase.CYCLE_DONE)
        widget.set_snapshot(done)
        self.assertEqual(widget.page_index, widget.page_count - 1)
        self.assertIn(59, widget.visible_indices)
        widget.set_snapshot(snapshot(60, 59, phase=Phase.CYCLE_DONE, now=100.))
        self.assertEqual(widget.page_index, widget.page_count - 1)
        widget.show_page(0)
        same_cycle = snapshot(60, 59, phase=Phase.CYCLE_DONE)
        widget.set_snapshot(same_cycle)
        self.assertEqual(widget.page_index, 0)
        self.assertFalse(widget.is_following)
        second = snapshot(60, 0, cycle=2)
        original = deepcopy(second)
        widget.set_snapshot(second)
        self.assertTrue(widget.is_following)
        self.assertEqual(widget.page_index, 0)
        self.assertEqual(second, original)
        widget.set_snapshot(snapshot(60, 59, phase=Phase.CYCLE_DONE, cycle=2))
        widget.show_page(1)
        widget.set_snapshot(snapshot(60, 0, cycle=2))
        self.assertTrue(widget.is_following)
        self.assertEqual(widget.page_index, 0)

    def test_new_sop_or_restart_resets_browsing_and_old_wrapping(self):
        widget = self.overview(60, height=240)
        current = widget.page_indices(2)[0]
        widget.set_snapshot(snapshot(60, current))
        widget.set_running(True)
        widget.show_page(0)
        widget.set_running(False)
        widget.set_running(True)
        self.assertTrue(widget.is_following)
        self.assertEqual(widget.page_index, widget.page_for_index(current))
        widget.show_page(1)
        widget.set_steps(["新的第一道", "新的第二道\n檢查方向"])
        self.assertTrue(widget.is_following)
        self.assertEqual(widget.page_index, 0)
        self.assertEqual(widget.page_count, 1)
        self.assertEqual(widget.page_indices(0), (0, 1))
        self.assertEqual(widget.entry_line_count(1), 2)
        self.assertIn("等待", widget.accessibleDescription())

    def test_states_duration_alarm_and_unchanged_render_cache(self):
        widget = self.overview(5)
        state = snapshot(5, 2)
        state.steps[0].manual = True
        state.steps[1].status = StepStatus.SKIPPED
        state.steps[1].skipped_alarmed = True
        widget.set_snapshot(state)
        for text in ("人工完成", "已跳過", "有警報", "執行中", "秒", "等待"):
            self.assertIn(text, widget.accessibleDescription())
        with patch.object(widget._grid, "update", wraps=widget._grid.update) as update:
            for _ in range(15):
                widget.set_snapshot(state)
            update.assert_not_called()
        children = len(widget.findChildren(QWidget))
        widget.set_steps([f"step {index}" for index in range(500)])
        self.assertEqual(len(widget.findChildren(QWidget)), children)

    def test_active_status_and_snapshot_ticks_do_not_change_layout(self):
        names = ["確認安裝方向並鎖緊固定螺絲" * 3 for _ in range(40)]
        widget = self.overview(names=names)
        geometry = [(widget.entry_line_count(index), widget.entry_natural_height(index), widget.page_for_index(index))
                    for index in range(len(names))]
        pages = tuple(widget.page_indices(page) for page in range(widget.page_count))
        with patch.object(widget, "_rebuild_layout", wraps=widget._rebuild_layout) as rebuild:
            for current in range(20):
                widget.set_snapshot(snapshot(len(names), current, now=float(current) + .9))
            rebuild.assert_not_called()
        self.assertEqual(pages, tuple(widget.page_indices(page) for page in range(widget.page_count)))
        self.assertEqual(geometry,
                         [(widget.entry_line_count(index), widget.entry_natural_height(index), widget.page_for_index(index))
                          for index in range(len(names))])

    def test_oversized_entries_scroll_locally_without_splitting_or_main_scroll(self):
        names = [(f"第 {index + 1} 道完整名稱\n" + "連續文字不得拆成另一頁\n" * 100) for index in range(3)]
        widget = self.overview(names=names, width=600, height=220)
        self.assertEqual(widget.page_count, 2)
        self.assertEqual(widget.page_indices(0), (0, 1))
        self.assertEqual(widget.page_indices(1), (2,))
        self.assertEqual(widget.visible_slots, {0: (0, 0), 1: (0, 1)})
        for index in (0, 1):
            self.assertTrue(widget.entry_is_oversized(index))
            self.assertGreater(widget.entry_natural_height(index), widget.entry_rect(index).height())
            scrollbar = widget.entry_scrollbar(index)
            self.assertIsInstance(scrollbar, QScrollBar)
            self.assertIs(scrollbar.parentWidget(), widget._grid)
            self.assertFalse(scrollbar.isHidden())
            self.assertGreater(scrollbar.maximum(), 0)
            self.assertAlmostEqual(scrollbar.pageStep(), widget.entry_rect(index).height(), delta=1.)
            self.assertIn(names[index], widget.accessibleDescription())
        self.assertIsNone(widget.entry_scrollbar(2))
        self.assertLessEqual(len(widget.findChildren(QScrollBar)), 2)
        self.assertTrue(all(scrollbar.parentWidget() is widget._grid for scrollbar in widget.findChildren(QScrollBar)))
        scrollbar = widget.entry_scrollbar(0)
        scrollbar.setValue(scrollbar.maximum())
        bottom = scrollbar.value()
        widget.set_snapshot(snapshot(3, 0))
        widget.set_snapshot(snapshot(3, 0, now=3.))
        self.assertEqual(widget.entry_scrollbar(0).value(), bottom)
        self.assert_whole_entry_layout(widget, 3)

    def test_long_names_keep_full_tooltip_and_accessibility_after_font_theme_changes(self):
        name = "非常長的工序名稱 <b> 保持純文字 " * 15
        widget = self.overview(names=[name, "另一道工序"], height=560)
        state = snapshot(2, 0)
        state.steps[0].violation_alarmed = True
        widget.set_snapshot(state)
        self.assertIn("清單", widget.accessibleName())
        self.assertNotIn("卡片", widget.accessibleName())
        self.assertIn(name, widget.accessibleDescription())
        position = widget._grid.mapFrom(widget, widget.entry_rect(0).center().toPoint())
        event = QHelpEvent(QEvent.Type.ToolTip, position, widget._grid.mapToGlobal(position))
        with patch("sop_app.ui.step_overview.QToolTip.showText") as tooltip:
            QApplication.sendEvent(widget._grid, event)
            self.assertIn(name, tooltip.call_args.args[1])
            self.assertIn("有警報", tooltip.call_args.args[1])
        before = widget.grab().toImage()
        natural_height = widget.entry_natural_height(0)
        widget.set_colors({"panel": "#17202a", "fg": "#f0f0f0", "muted": "#b0bec5", "ng": "#ff8a80"})
        self.assertNotEqual(before, widget.grab().toImage())
        self.assertIn(name, widget.accessibleDescription())
        larger = QFont(widget.font())
        larger.setPointSize(16)
        widget.setFont(larger)
        self.app.processEvents()
        self.assertGreater(widget.entry_natural_height(0), natural_height)
        self.assertGreater(widget.entry_line_count(0), 1)
        self.assert_whole_entry_layout(widget, 2)

    def test_navigation_buttons_are_keyboard_operable(self):
        widget = self.overview(60, height=240)
        widget.next_button.setFocus()
        QTest.keyClick(widget.next_button, Qt.Key.Key_Space)
        self.assertEqual(widget.page_index, 1)
        self.assertFalse(widget.is_following)
        widget.follow_button.setFocus()
        QTest.keyClick(widget.follow_button, Qt.Key.Key_Space)
        self.assertTrue(widget.is_following)
        self.assertEqual(widget.page_index, 0)


if __name__ == "__main__":
    unittest.main()
