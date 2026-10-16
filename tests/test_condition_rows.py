"""Stacked condition blocks stay complete, aligned and visible in a narrow column."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QRadioButton, QStyle, QStyleOptionComboBox,
                               QVBoxLayout, QWidget)

from sop_app.conditions import ConditionResult
from sop_app.sop_schema import Condition
from sop_app.ui.condition_rows import FIELDS, ConditionRows
from sop_app.ui.workbench_widgets import install_readability_guard, workbench_styles


def settle(app, rounds=4):
    for _ in range(rounds):
        app.processEvents()


class ConditionRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_rows(self, conditions=None, width=900):
        rows = ConditionRows()
        rows.set_options(["零件", "工具"], ["左側", "右側"])
        rows.set_conditions(conditions if conditions is not None else [Condition(label="零件", roi="左側")])
        rows.resize(width, rows.sizeHint().height())
        rows.show()
        settle(self.app)
        self.addCleanup(rows.close)
        return rows

    def test_all_parameters_are_visible_and_saved_without_expanding(self):
        rows = self.make_rows()
        self.assertFalse(rows.findChildren(QRadioButton))
        for name, _title in FIELDS:
            self.assertTrue(rows.field_widget(0, name).isVisible(), name)
            self.assertGreaterEqual(rows.field_widget(0, name).height(), 20, name)
        rows.field_widget(0, "label").setCurrentIndex(2)
        rows.field_widget(0, "roi").setCurrentIndex(2)
        rows.field_widget(0, "min_count").setValue(3)
        rows.field_widget(0, "min_score").setValue(.85)
        rows.field_widget(0, "roi_overlap").setValue(75)
        self.assertEqual(rows.conditions(), [Condition(label="工具", roi="右側", min_count=3,
                                                       min_score=.85, roi_overlap=.75)])

    def test_unused_fields_show_reason_in_place_and_keep_values(self):
        original = Condition(type="disappear", label="零件", min_count=4, roi_overlap=.75)
        rows = self.make_rows([original])
        self.assertFalse(rows.field_widget(0, "min_count").isVisible())
        self.assertTrue(rows.field_widget(0, "min_count_note").isVisible())
        self.assertIn("0", rows.field_widget(0, "min_count_note").text())
        self.assertFalse(rows.field_widget(0, "roi_overlap").isVisible())
        self.assertIn("不適用", rows.field_widget(0, "roi_overlap_note").text())
        self.assertEqual(rows.conditions(), [original])
        rows.field_widget(0, "type").setCurrentIndex(0)
        rows.field_widget(0, "roi").setCurrentIndex(1)
        self.assertTrue(rows.field_widget(0, "min_count").isVisible())
        self.assertTrue(rows.field_widget(0, "roi_overlap").isVisible())
        self.assertFalse(rows.field_widget(0, "roi_overlap_note").isVisible())
        self.assertEqual(rows.conditions()[0].min_count, 4)
        self.assertEqual(rows.conditions()[0].roi_overlap, .75)

    def test_option_refresh_preserves_fields_focus_values_and_reports_missing_choices(self):
        rows = self.make_rows()
        changes = QSignalSpy(rows.changed)
        count = rows.field_widget(0, "min_count")
        count.setFocus()
        settle(self.app)
        original = rows.conditions()
        rows.set_options(["另一種零件"], ["重新命名區域"])
        self.assertIs(rows.field_widget(0, "min_count"), count)
        self.assertTrue(count.hasFocus())
        self.assertEqual(rows.conditions(), original)
        self.assertEqual(rows.field_widget(0, "label").currentData(), "零件")
        self.assertEqual(rows.field_widget(0, "roi").currentData(), "左側")
        self.assertEqual(changes.count(), 0)
        self.assertTrue(rows.field_widget(0, "error").isVisible())
        self.assertIn("目前模型沒有", rows.field_widget(0, "error").text())
        self.assertIn("找不到區域", rows.field_widget(0, "error").text())
        rows.set_options(["零件"], ["左側"])
        self.assertFalse(rows.field_widget(0, "error").isVisible())

    def test_focus_highlights_region_and_reason_is_shown_only_when_not_met(self):
        rows = self.make_rows()
        selected = QSignalSpy(rows.region_selected)
        score = rows.field_widget(0, "min_score")
        score.setFocus()
        settle(self.app)
        self.assertGreater(selected.count(), 0)
        self.assertEqual(selected.at(selected.count() - 1), ["左側"])
        condition = rows.conditions()[0]
        rows.set_live_results([ConditionResult(condition, 0, False, error="找不到區域 左側", error_kind="config")])
        self.assertEqual(rows.field_widget(0, "live").text(), "本幀無法判斷")
        self.assertTrue(rows.field_widget(0, "detail").isVisible())
        self.assertIn("找不到區域 左側", rows.field_widget(0, "detail").text())
        rows.set_live_results([ConditionResult(condition, 0, False, detail="最高信心 0.45，門檻 0.50")])
        self.assertEqual(rows.field_widget(0, "live").text(), "本幀未成立")
        self.assertIn("最高信心", rows.field_widget(0, "detail").text())
        rows.set_live_results([ConditionResult(condition, 1, True, detail="通過 1 個")])
        self.assertEqual(rows.field_widget(0, "live").text(), "本幀成立")
        self.assertFalse(rows.field_widget(0, "detail").isVisible())
        rows.set_live_results(None)
        self.assertEqual(rows.field_widget(0, "live").text(), "本幀 —")
        self.assertFalse(rows.field_widget(0, "detail").isVisible())

    def test_remove_buttons_undo_original_order_and_new_context_clears_undo(self):
        values = [Condition(label="零件", min_score=.85), Condition(label="工具", roi_overlap=.75)]
        rows = self.make_rows(values)
        rows.field_widget(0, "remove").click()
        self.assertEqual(rows.conditions(), values[1:])
        rows.field_widget(0, "remove").click()
        self.assertEqual(rows.conditions(), [])
        rows.undo_delete_button.click()
        rows.undo_delete_button.click()
        self.assertEqual(rows.conditions(), values)
        rows.field_widget(0, "remove").click()
        rows.set_conditions([Condition(label="工具")])
        self.assertFalse(rows.undo_delete_button.isEnabled())
        rows._undo_delete()
        self.assertEqual(rows.conditions(), [Condition(label="工具")])

    def test_stacked_blocks_line_up_and_label_every_field_in_a_narrow_column(self):
        rows = self.make_rows([Condition(label="零件", roi="左側"), Condition(type="disappear", label="工具"),
                               Condition(label="工具", roi="右側")], width=360)
        install_readability_guard(rows)
        settle(self.app, 6)
        texts = {label.text() for label in rows.findChildren(QLabel)}
        for text in ("完成條件 1", "完成條件 3", "物件", "狀態", "區域", "數量", "信心", "區域內"):
            self.assertIn(text, texts)
        for field in ("label", "type", "roi", "min_score", "remove"):
            lefts = {rows.field_widget(row, field).mapTo(rows, QPoint(0, 0)).x() for row in range(3)}
            self.assertEqual(len(lefts), 1, (field, lefts))
        for row in range(3):
            for field in ("label", "type", "roi", "remove"):
                widget = rows.field_widget(row, field)
                self.assertLessEqual(widget.mapTo(rows, QPoint(widget.width(), 0)).x(), rows.width(), field)
        # Object/state, region/count and confidence/in-region ratio each share one line.
        y = lambda field: rows.field_widget(0, field).mapTo(rows, QPoint(0, 0)).y()
        self.assertEqual(y("label"), y("type"))
        self.assertEqual(y("roi"), y("min_count"))
        self.assertEqual(y("min_score"), y("roi_overlap"))
        self.assertLess(y("label"), y("roi"))
        self.assertLess(y("roi"), y("min_score"))

    def test_selected_names_fit_their_field_without_hover(self):
        name = "主要工件安裝位置辨識模型類別名稱"
        rows = self.make_rows([Condition(label=name)], width=900)
        rows.set_options([name, "工具"], [])
        install_readability_guard(rows)
        settle(self.app, 8)
        combo = rows.field_widget(0, "label")
        option = QStyleOptionComboBox()
        combo.initStyleOption(option)
        field = combo.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                             QStyle.SubControl.SC_ComboBoxEditField, combo)
        self.assertGreaterEqual(field.width(), combo.fontMetrics().horizontalAdvance(name))
        self.assertEqual(combo.toolTip(), name)

    def test_production_styles_keep_text_height_inside_row(self):
        original_font = QFont(self.app.font())
        self.addCleanup(self.app.setFont, original_font)
        if os.name == "nt":
            font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msjh.ttc")
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                self.app.setFont(QFont(families[0], 10))
        rows = self.make_rows([Condition(label="零件", roi="左側"), Condition(type="disappear", label="工具")],
                              width=1250)
        rows.setStyleSheet(workbench_styles(False))
        install_readability_guard(rows)
        for width in (1250, 900, 360, 900):
            rows.resize(width, rows.sizeHint().height())
            settle(self.app)
            for row in range(2):
                for field, _title in FIELDS:
                    control = rows.field_widget(row, field)
                    if not control.isVisible():
                        continue
                    self.assertGreaterEqual(control.height(), control.fontMetrics().height())
                    origin = control.mapTo(rows, QPoint(0, 0))
                    self.assertLessEqual(origin.x() + control.width(), max(width, rows.width()))


if __name__ == "__main__":
    unittest.main()
