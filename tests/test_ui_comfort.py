"""Readable native UI: spacing, feedback, wrapping and glyph-content bounds."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import subprocess
import sys
import unittest
from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QGroupBox, QPushButton, QStyle, QStyleOptionButton, QStyleOptionComboBox, QStyleOptionFrame, QStyleOptionGroupBox, QStyleOptionSpinBox

import test_workbench as fixtures
from sop_app.sop_schema import Condition, Step


class UIComfortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WorkbenchQtTests.setUpClass()
        cls.app = fixtures.WorkbenchQtTests.app

    make_editor = fixtures.WorkbenchQtTests.make_editor

    def test_control_text_area_is_taller_than_font_line(self):
        editor = self.make_editor()
        controls = [(editor.step_name_edit, "line"), (editor.completion_mode, "combo"),
                    (editor.validation_mode, "combo")]
        controls += [(editor.condition_table.field_widget(0, field), "spin" if field == "min_count" else "combo")
                     for field in ("label", "type", "roi", "min_count")]
        for widget, kind in controls:
            if kind == "line":
                option = QStyleOptionFrame()
                option.initFrom(widget)
                rect = widget.style().subElementRect(QStyle.SubElement.SE_LineEditContents, option, widget)
            elif kind == "combo":
                option = QStyleOptionComboBox()
                option.initFrom(widget)
                rect = widget.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                                     QStyle.SubControl.SC_ComboBoxEditField, widget)
            else:
                option = QStyleOptionSpinBox()
                option.initFrom(widget)
                option.frame = True
                rect = widget.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option,
                                                     QStyle.SubControl.SC_SpinBoxEditField, widget)
            self.assertGreaterEqual(rect.height(), widget.fontMetrics().height(), (kind, rect, widget.font()))
            self.assertGreaterEqual(rect.top(), 0)
            self.assertLessEqual(rect.bottom(), widget.height() - 1)

    def test_group_titles_fit_inside_widget_bounds(self):
        editor = self.make_editor()
        for group in editor.findChildren(QGroupBox):
            if not group.isVisible():
                continue
            option = QStyleOptionGroupBox()
            option.initFrom(group)
            option.text = group.title()
            rect = group.style().subControlRect(QStyle.ComplexControl.CC_GroupBox, option,
                                               QStyle.SubControl.SC_GroupBoxLabel, group)
            self.assertGreaterEqual(rect.height(), group.fontMetrics().height(), group.title())
            self.assertGreaterEqual(rect.top(), 0, group.title())
            self.assertLessEqual(rect.bottom(), group.height() - 1, group.title())

    def test_step_action_text_and_vector_icons_fit_completely(self):
        editor = self.make_editor()
        buttons = editor.step_actions.findChildren(QPushButton)
        self.assertEqual([button.text() for button in buttons], ["新增", "複製", "刪除", "上移", "下移"])
        for button in buttons:
            option = QStyleOptionButton()
            option.initFrom(button)
            rect = button.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, button)
            metrics = button.fontMetrics()
            self.assertGreaterEqual(rect.height(), metrics.height(), button.text())
            required_width = metrics.horizontalAdvance(button.text())
            if button.property("actionIcon"):
                self.assertFalse(button.icon().isNull(), button.text())
                pixmap = button.icon().pixmap(button.iconSize())
                image = pixmap.toImage()
                pixels = [(x, y) for y in range(image.height()) for x in range(image.width())
                          if image.pixelColor(x, y).alpha() > 0]
                self.assertTrue(pixels, button.text())
                # The SVG's complete strokes have transparent margins on every edge.
                self.assertGreater(min(x for x, y in pixels), 0, button.text())
                self.assertGreater(min(y for x, y in pixels), 0, button.text())
                self.assertLess(max(x for x, y in pixels), image.width() - 1, button.text())
                self.assertLess(max(y for x, y in pixels), image.height() - 1, button.text())
                self.assertGreaterEqual(rect.height(), button.iconSize().height(), button.text())
                required_width += button.iconSize().width() + 4
            self.assertGreaterEqual(rect.width(), required_width, button.text())
        artifact = os.environ.get("SOP_EDITOR_TEST_ARTIFACT_DIR")
        if artifact:
            directory = Path(artifact)
            directory.mkdir(parents=True, exist_ok=True)
            self.assertTrue(editor.step_actions.grab().save(str(directory / "step-actions-fixed.png")))

    def test_long_names_wrap_and_remain_available_in_full(self):
        name = "左側主要工件安裝與位置確認作業" * 2
        definition = fixtures.sop(Step(name, conditions=[Condition(label="grip")]))
        definition.name = "工序設定與試跑驗證介面字體完整顯示測試" * 2
        editor = self.make_editor(definition)
        self.app.processEvents()
        self.assertTrue(editor.workspace_title.wordWrap())
        self.assertTrue(editor.context_title.wordWrap())
        self.assertTrue(editor.step_list.wordWrap())
        self.assertIn(name, editor.step_list.item(0).toolTip())
        for label in (editor.workspace_title, editor.context_title):
            self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()), label.text())
        item = editor.step_list.item(0)
        rect = editor.step_list.visualItemRect(item)
        expected = editor.step_list.fontMetrics().boundingRect(
            QRect(0, 0, rect.width() - 20, 10000), Qt.TextFlag.TextWordWrap, item.text()).height()
        self.assertGreaterEqual(rect.height(), expected, (rect, expected))

    def test_routine_hints_are_quiet_but_action_feedback_is_visible(self):
        editor = self.make_editor()
        editor._return_live()
        self.assertFalse(editor.guide_label.isVisible())
        editor.guide_label.setText("找不到作業區域，請先建立或重新選擇。")
        self.assertTrue(editor.guide_label.isVisible())
        self.assertFalse(editor.mode_hint.isVisible())

    def test_description_can_expand_without_changing_its_value(self):
        editor = self.make_editor(fixtures.sop(Step("說明", instruction="請將工件放入指定位置。")))
        self.assertFalse(editor.instruction_edit.isVisible())
        editor.description_section.set_expanded(True)
        self.app.processEvents()
        self.assertTrue(editor.instruction_edit.isVisible())
        self.assertEqual(editor.instruction_edit.toPlainText(), "請將工件放入指定位置。")
        editor.description_section.set_expanded(False)
        self.assertEqual(editor.sop.steps[0].instruction, "請將工件放入指定位置。")

    def test_large_and_short_windows_use_appropriate_spacing(self):
        editor = self.make_editor()
        editor.resize(1300, 640)
        self.app.processEvents()
        self.assertTrue(editor._comfort_compact)
        compact = editor.center_panel.layout().spacing()
        editor.resize(1500, 900)
        self.app.processEvents()
        self.assertFalse(editor._comfort_compact)
        self.assertGreater(editor.center_panel.layout().spacing(), compact)

    def test_document_badge_reflects_unsaved_changes(self):
        editor = self.make_editor()
        editor.set_document_state(False, saved=True)
        self.assertIn("已儲存", editor.document_state.text())
        editor.condition_table.field_widget(0, "min_count").setValue(2)
        self.assertIn("未儲存變更", editor.document_state.text())

    def test_full_class_names_remain_accessible_when_combo_is_narrow(self):
        name = "主要工件安裝位置辨識模型類別名稱完整檢查"
        editor = self.make_editor(fixtures.sop(Step(conditions=[Condition(label=name)])))
        editor.set_classes([name, "工具"])
        field = editor.condition_table.field_widget(0, "label")
        self.assertEqual(field.currentData(), name)
        self.assertEqual(field.toolTip(), name)
        field.setCurrentIndex(field.findData("工具"))
        self.assertEqual(field.toolTip(), "工具")

    def test_zoomed_125_percent_has_no_vertical_text_clipping(self):
        script = r'''
import os,sys,tempfile
from pathlib import Path
from unittest.mock import Mock,patch
from PySide6.QtWidgets import QApplication,QPushButton,QStyle,QStyleOptionButton,QStyleOptionComboBox,QStyleOptionFrame
from PySide6.QtGui import QFont,QFontDatabase
from sop_app.ui.main_window import MainWindow
from sop_app.sop_schema import SOPDefinition,Step,Condition
app=QApplication([])
font_id=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msjh.ttc')
families=QFontDatabase.applicationFontFamilies(font_id)
if families:app.setFont(QFont(families[0],10))
QFontDatabase.addApplicationFont('C:/Windows/Fonts/seguisym.ttf')
with tempfile.TemporaryDirectory() as directory, \
     patch('sop_app.ui.main_window.RECORDS_DIR',Path(directory)), \
     patch.object(MainWindow,'_load_settings',return_value={}), \
     patch.object(MainWindow,'_save_settings'), \
     patch('sop_app.ui.main_window.VideoPipeline.start'), \
     patch('sop_app.ui.main_window.cv2.require',return_value=Mock()):
    window=MainWindow(defer_initial_sop=True)
    window._apply_sop(SOPDefinition(name='字體縮放檢查',steps=[Step('左側工件作業位置',conditions=[Condition(label='grip')])]),None)
    editor=window.editor;editor.set_classes(['grip']);window.tabs.setCurrentWidget(editor)
    window.resize(1366,768);window.show();app.processEvents()
    assert window.devicePixelRatioF()>1
    for widget in (editor.completion_mode,editor.validation_mode,editor.condition_table.field_widget(0,'label')):
        option=QStyleOptionComboBox();option.initFrom(widget)
        rect=widget.style().subControlRect(QStyle.ComplexControl.CC_ComboBox,option,QStyle.SubControl.SC_ComboBoxEditField,widget)
        assert rect.height()>=widget.fontMetrics().height(),(rect,widget.fontMetrics().height())
    widget=editor.step_name_edit;option=QStyleOptionFrame();option.initFrom(widget)
    rect=widget.style().subElementRect(QStyle.SubElement.SE_LineEditContents,option,widget)
    assert rect.height()>=widget.fontMetrics().height()
    for button in editor.step_actions.findChildren(QPushButton):
        option=QStyleOptionButton();option.initFrom(button)
        rect=button.style().subElementRect(QStyle.SubElement.SE_PushButtonContents,option,button)
        required=button.fontMetrics().horizontalAdvance(button.text())
        if button.property('actionIcon'):
            assert not button.icon().isNull()
            required+=button.iconSize().width()+4
        assert rect.width()>=required,(button.text(),rect,required)
        assert rect.height()>=button.fontMetrics().height()
    artifact=os.environ.get('SOP_EDITOR_TEST_ARTIFACT_DIR')
    if artifact:
        Path(artifact).mkdir(parents=True,exist_ok=True)
        window.grab().save(str(Path(artifact)/'workbench-fontscale-125.png'))
    window.dirty=False;window.close()
print('FONT_SCALE_125_OK')
'''
        environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_SCALE_FACTOR": "1.25", "PYTHONIOENCODING": "utf-8"}
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                   env=environment, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("FONT_SCALE_125_OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
