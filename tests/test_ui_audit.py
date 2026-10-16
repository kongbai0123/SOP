"""Whole-window typography audit, expanded settings, dialogs and scaling QA.

Images are optional and are always written outside the repository. No camera,
model, production run, user settings or existing record database is touched.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QLabel, QLineEdit,
                              QPlainTextEdit, QPushButton, QRadioButton, QStyle,
                              QStyleOptionButton, QStyleOptionComboBox, QStyleOptionFrame,
                              QStyleOptionSpinBox, QTabWidget, QToolButton, QWidget)

import test_workbench as fixtures
from sop_app.sop_schema import Condition, ROI, Step
from sop_app.ui.camera_dialog import CameraDialog
from sop_app.ui.main_window import MainWindow
from sop_app.ui.quick_setup_dialog import QuickPositionDialog
from sop_app.ui.records_page import _fill
from sop_app.ui.startup_window import StartupWindow
from sop_app.ui.video_widget import VideoWidget
from sop_app.overlay import _label_geometry


def settle(app):
    for _ in range(5):
        app.processEvents()


def failures(root):
    """Check actual text rectangles, not only the outer widget's nominal size."""
    errors = []
    for widget in root.findChildren(QWidget):
        if not widget.isVisible() or widget.window() != root.window():
            continue
        metrics = widget.fontMetrics()
        rect = None
        text_width = None
        text = ""
        if isinstance(widget, QPushButton):
            text = widget.text().replace("&", "")
            if not text:
                continue
            option = QStyleOptionButton(); option.initFrom(widget)
            rect = widget.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, widget)
            text_width = metrics.horizontalAdvance(text)
            if not widget.icon().isNull():
                text_width += widget.iconSize().width() + 4
        elif isinstance(widget, (QCheckBox, QRadioButton)):
            text = widget.text().replace("&", "")
            option = QStyleOptionButton(); option.initFrom(widget)
            kind = QStyle.SubElement.SE_CheckBoxContents if isinstance(widget, QCheckBox) else QStyle.SubElement.SE_RadioButtonContents
            rect = widget.style().subElementRect(kind, option, widget)
            text_width = metrics.horizontalAdvance(text)
        elif isinstance(widget, QToolButton) and widget.text():
            text = widget.text()
            if widget.width() < widget.sizeHint().width() or widget.height() < widget.sizeHint().height():
                errors.append((type(widget).__name__, text, "intrinsic size", widget.size(), widget.sizeHint()))
            continue
        elif isinstance(widget, QComboBox):
            text = widget.currentText()
            option = QStyleOptionComboBox(); option.initFrom(widget)
            rect = widget.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option, QStyle.SubControl.SC_ComboBoxEditField, widget)
            text_width = metrics.horizontalAdvance(text)
        elif isinstance(widget, QLineEdit):
            # Single-line text is intentionally scrollable; never clip a glyph vertically.
            text = widget.text() or widget.placeholderText()
            option = QStyleOptionFrame(); option.initFrom(widget)
            rect = widget.style().subElementRect(QStyle.SubElement.SE_LineEditContents, option, widget)
        elif isinstance(widget, QAbstractSpinBox):
            text = widget.text()
            option = QStyleOptionSpinBox(); option.initFrom(widget); option.frame = True
            rect = widget.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxEditField, widget)
            text_width = metrics.horizontalAdvance(text)
        elif isinstance(widget, QLabel) and widget.text() and widget.pixmap().isNull():
            text = widget.text()
            if widget.wordWrap():
                needed = widget.heightForWidth(widget.width())
                if needed > widget.height():
                    errors.append(("QLabel", text, "wrapped height", widget.height(), needed))
            elif widget.height() < widget.sizeHint().height() or widget.width() < widget.sizeHint().width():
                errors.append(("QLabel", text, "intrinsic size", widget.size(), widget.sizeHint()))
            continue
        elif isinstance(widget, QPlainTextEdit):
            if widget.viewport().height() < metrics.height():
                errors.append(("QPlainTextEdit", widget.placeholderText(), "viewport line height"))
            continue
        if rect is not None:
            if rect.height() < metrics.height():
                errors.append((type(widget).__name__, text, "text height", rect.height(), metrics.height()))
            if text_width is not None and rect.width() < text_width:
                errors.append((type(widget).__name__, text, "text + icon width", rect.width(), text_width))
    return errors


class WholeUIAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WorkbenchQtTests.setUpClass()
        cls.app = fixtures.WorkbenchQtTests.app

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for patcher in (patch("sop_app.ui.main_window.RECORDS_DIR", Path(self.directory.name)),
                        patch.object(MainWindow, "_load_settings", return_value={}),
                        patch.object(MainWindow, "_save_settings"),
                        patch("sop_app.ui.main_window.VideoPipeline.start"),
                        patch("sop_app.ui.main_window.cv2.require", return_value=Mock())):
            patcher.start(); self.addCleanup(patcher.stop)
        self.window = MainWindow(defer_initial_sop=True)
        self.addCleanup(self.close_window)
        definition = fixtures.sop(
            Step("拿取工具", conditions=[Condition(label="grip", roi="左側安裝區")],
                 forbidden=[Condition(label="tool")], instruction="確認零件方向，再將工件固定於治具。"),
            Step("左側安裝與位置確認", conditions=[Condition(label="grip")]),
            Step("右側安裝", conditions=[Condition(label="grip")]), Step("人工外觀確認"))
        definition.name = "工件安裝 SOP"
        definition.rois = [ROI("左側安裝區", [(.8, .8), (.98, .8), (.98, .98), (.8, .98)])]
        self.window._apply_sop(definition, None)
        self.editor = self.window.editor
        self.editor.set_classes(["grip", "tool"])
        self.editor.set_model_state("available")
        self.expected_size = tuple(map(int, os.environ.get("SOP_UI_AUDIT_SIZE", "1366x768").split("x")))
        self.window.resize(*self.expected_size)
        self.window.show()
        self.window.tabs.setCurrentWidget(self.editor)
        settle(self.app)
        self.editor.on_packet(fixtures.packet(0, "grip"))
        self.window.run_page.video.set_frame(np.zeros((360, 640, 3), np.uint8))

    def close_window(self):
        self.window.dirty = False
        self.window.close()
        self.window.deleteLater()
        settle(self.app)

    def check(self, root, name):
        settle(self.app)
        if root is self.window:
            self.assertEqual(self.window.size().toTuple(), self.expected_size, name)
        errors = failures(root)
        artifact = os.environ.get("SOP_UI_AUDIT_ARTIFACT_DIR")
        if artifact:
            directory = Path(artifact); directory.mkdir(parents=True, exist_ok=True)
            root.grab().save(str(directory / f"{name}.png"))
        self.assertEqual(errors, [], f"{name}: {errors}")

    def test_all_editor_workspaces_and_expanded_controls(self):
        """Collapsed summaries and every expanded field fit without clipping."""
        self.assertFalse(self.editor.findChildren(QTabWidget))
        self.check(self.window, "editor-collapsed")
        self.editor.expand_all_button.click()
        self.assertGreater(self.editor.page_scroll.verticalScrollBar().maximum(), 0)
        for index, step in enumerate(self.editor.step_editors):
            for control in (step.instruction_edit, step.hold_spin, step.forbidden_mode,
                            step.condition_table, step.forbidden_table):
                self.assertTrue(control.isVisible())
            self.editor.page_scroll.ensureWidgetVisible(step)
            self.check(self.window, "editor-step-" + str(index))
        self.editor.page_scroll.ensureWidgetVisible(self.editor.validation_box)
        self.check(self.window, "editor-validation")
        self.assertEqual(self.editor.page_scroll.horizontalScrollBar().maximum(), 0)
        self.editor.focus_button.click()
        self.check(self.window, "editor-large-image")
        self.editor.focus_button.click()

    def test_monitor_idle_running_alarm_and_long_names(self):
        page = self.window.run_page
        self.window.tabs.setCurrentWidget(page)
        self.check(self.window, "monitor-idle")
        page.set_running(True)
        page.details_button.setChecked(True)
        page.step_title.setText("工序 2：左側工件主要安裝位置與鎖附方向確認" * 2)
        page.instruction.setText("確認安裝方向與位置，將工具置入左側工作區域後再進行鎖附。" * 3)
        page.condition_label.setText("條件成立：左側工作區出現 grip，數量至少 1 個。")
        page._show_alarm("安裝順序不符，請確認左側位置與作業工序後重新操作。" * 2)
        self.check(self.window, "monitor-alarm")

    def test_records_filters_dates_and_full_table_values(self):
        page = self.window.records_page
        self.window.tabs.setCurrentWidget(page)
        page.use_dates.setChecked(True)
        # Exercise the current table geometry without relying on removed columns.
        for table in (page.cycles, page.steps, page.alarms, page.positioning):
            table.blockSignals(True)
            values = ["左側工作區安裝與位置確認" if column % 2 else "2026-10-16 10:20:00"
                      for column in range(table.columnCount())]
            _fill(table, [tuple(values)])
            table.blockSignals(False)
        self.check(self.window, "records")
        for table in (page.cycles, page.steps, page.alarms, page.positioning):
            self.assertFalse(table.horizontalHeader().stretchLastSection())
            self.assertEqual(table.textElideMode(), Qt.TextElideMode.ElideNone)

    def test_camera_both_tabs_and_pending_changes(self):
        dialog = CameraDialog(self.window)
        self.addCleanup(dialog.close)
        dialog.update_controls({"resolution": (1920, 1080), "fps": 30, "exposure": -6}, [])
        dialog.show()
        tabs = dialog.findChild(QTabWidget)
        for index in (0, 1):
            tabs.setCurrentIndex(index)
            dialog.rows["exposure"][2].setValue(-7)
            self.check(dialog, "camera-" + str(index))

    def test_position_dialog_and_startup_message(self):
        dialog = QuickPositionDialog("左側工件安裝位置", ["grip", "tool"], self.window)
        self.addCleanup(dialog.close)
        dialog.show()
        self.check(dialog, "position-dialog")
        startup = StartupWindow()
        self.addCleanup(startup.close)
        startup.message.setText("正在載入介面元件與工序設定，請稍候；模型將在主畫面開啟後繼續背景載入。")
        startup.show()
        self.check(startup, "startup")

    def test_image_tags_remain_inside_every_edge(self):
        image = QImage(320, 180, QImage.Format.Format_ARGB32)
        image.fill(QColor("black"))
        painter = QPainter(image)
        try:
            for origin in (QPointF(-20, -20), QPointF(315, 175), QPointF(315, 0)):
                box = VideoWidget._draw_tag(painter, origin, "左側主要工件安裝位置與方向確認" * 3,
                                            QColor("yellow"), QFont("Microsoft JhengHei", 10))
                self.assertTrue(QRectF(0, 0, 320, 180).contains(box), box)
        finally:
            painter.end()

    def test_long_model_status_does_not_force_window_off_screen(self):
        self.window.model_status.setText("模型：主要工件安裝辨識模型與版本資訊 " * 12)
        self.check(self.window, "long-model-status")

    def test_detection_class_labels_fit_image_edges_without_ellipsis(self):
        import cv2 as native_cv
        text = "main_object_installation_position_label_with_long_class_name 0.99"
        with patch("sop_app.overlay.cv2", native_cv):
            lines, sizes, scale, left, top, width, line_height = _label_geometry(text, 319, 179, 320, 180, .6, 2)
        self.assertEqual("".join(lines), text)
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(left + width, 319)
        self.assertLessEqual(top + line_height * len(lines), 179)

    def test_physical_1366_screen_at_125_and_150_percent(self):
        # Use a smaller logical window, not 1366 logical pixels enlarged to 1708px.
        script = """
import sys,unittest
sys.path.insert(0,'tests')
from test_ui_audit import WholeUIAuditTests
class ScaledAudit(WholeUIAuditTests):
    def setUp(self):
        super().setUp()
        factor=self.window.devicePixelRatioF()
        self.expected_size=(round(1366/factor),round(768/factor))
        self.window.resize(*self.expected_size)
suite=unittest.TestSuite(ScaledAudit(name) for name in (
    'test_all_editor_workspaces_and_expanded_controls',
    'test_monitor_idle_running_alarm_and_long_names',
    'test_records_filters_dates_and_full_table_values',
    'test_camera_both_tabs_and_pending_changes'))
result=unittest.TextTestRunner().run(suite)
sys.exit(not result.wasSuccessful())
"""
        for factor in ("1.25", "1.5"):
            environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_SCALE_FACTOR": factor,
                           "PYTHONIOENCODING": "utf-8"}
            if environment.get("SOP_UI_AUDIT_ARTIFACT_DIR"):
                environment["SOP_UI_AUDIT_ARTIFACT_DIR"] += "/scale-" + factor
            completed = subprocess.run([sys.executable, "-c", script], env=environment,
                                       capture_output=True, text=True, timeout=45)
            self.assertEqual(completed.returncode, 0, factor + completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
