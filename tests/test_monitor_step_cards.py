"""Monitor/card integration with real engine snapshots, never models or cameras."""
from contextlib import ExitStack
import copy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QPoint, QRect, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea

from sop_app.detection import Detection, FrameResult
from sop_app.engine import Phase, SOPEngine, StepStatus
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, CycleSettings, SOPDefinition, Step
from sop_app.ui.main_window import MainWindow
from sop_app.ui.run_page import RunPage


class MonitorStepCardsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        previous = cls.app.font()
        font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msjh.ttc")
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            cls.app.setFont(QFont(families[0], 10))
        cls.addClassCleanup(cls.app.setFont, previous)

    def setUp(self):
        self.beep = patch("sop_app.ui.run_page.QApplication.beep")
        self.beep.start()
        self.addCleanup(self.beep.stop)

    @staticmethod
    def dispose(widget):
        widget.close()
        for timer in widget.findChildren(QTimer):
            timer.stop()

    def settle(self):
        for _ in range(5):
            self.app.processEvents()

    @staticmethod
    def definition(count=30):
        return SOPDefinition(name="裝配流程", cycle=CycleSettings(enabled=False),
                             steps=[Step(f"工序 {index:02}") for index in range(1, count + 1)])

    @staticmethod
    def frame(at, labels=()):
        detections = [Detection(label, .99, (10, 10, 25, 25)) for label in labels]
        return FrameResult(200, 100, at, detections)

    def push(self, page, engine=None, at=0., labels=(), update=False):
        result = self.frame(at, labels)
        if engine is not None and update:
            page.on_events(engine.update(result))
        image = np.zeros((100, 200, 3), np.uint8)
        page.on_packet(FramePacket(image, image, result, engine.snapshot() if engine else None, 20., 4.))

    def page(self, definition):
        page = RunPage()
        self.addCleanup(self.dispose, page)
        page.set_sop(definition)
        page.resize(1100, 760)
        page.show()
        self.settle()
        return page

    def running(self, definition):
        page = self.page(definition)
        engine = SOPEngine(definition)
        engine.start(0.)
        page.set_running(True)
        clock = [0.]
        def confirm():
            clock[0] += .1
            page.on_events(engine.confirm_manual_step(clock[0]))
            self.push(page, engine, clock[0], update=True)
        def restart():
            clock[0] += 1.
            page.on_events(engine.restart_cycle(clock[0]))
            self.push(page, engine, clock[0])
        page.confirm_requested.connect(confirm)
        page.restart_requested.connect(restart)
        self.push(page, engine)
        self.settle()
        return page, engine, clock

    def test_empty_and_idle_sop_reset_overview_without_starting_engine(self):
        page = self.page(SOPDefinition(steps=[]))
        self.assertEqual(page.step_overview.page_count, 0)
        self.assertEqual(list(page.step_overview.visible_indices), [])
        self.push(page)
        page.set_sop(self.definition(12))
        self.settle()
        self.assertEqual(page.step_overview.page_index, 0)
        self.assertIn(0, page.step_overview.visible_indices)
        self.assertIn("待機", page.step_title.text())
        self.assertTrue(page.next_step_label.isVisible())
        self.assertFalse(page.confirm_button.isEnabled())

    def test_manual_confirmation_and_automatic_completion_update_fixed_current_and_next(self):
        definition = SOPDefinition(cycle=CycleSettings(enabled=False), steps=[
            Step("人工準備"), Step("自動定位", conditions=[Condition(label="part")], hold_sec=.2), Step("人工檢查")])
        page, engine, clock = self.running(definition)
        self.assertIn("人工準備", page.step_title.text())
        self.assertIn("自動定位", page.next_step_label.text())
        self.assertTrue(page.confirm_button.isEnabled())
        page.confirm_button.click()
        self.assertEqual(engine.index, 1)
        self.assertIn("自動定位", page.step_title.text())
        self.assertIn("人工檢查", page.next_step_label.text())
        self.assertFalse(page.confirm_button.isEnabled())
        for at in (.2, .3, .4, .5):
            self.push(page, engine, at, ("part",), update=True)
            if engine.index == 2:
                break
        self.assertEqual(engine.index, 2)
        self.assertIn("人工檢查", page.step_title.text())
        self.assertNotIn("自動定位", page.next_step_label.text())
        self.assertTrue(page.confirm_button.isEnabled())

    def test_follow_crosses_pages_and_manual_browsing_never_changes_execution(self):
        page, engine, _clock = self.running(self.definition())
        overview = page.step_overview
        self.assertGreater(overview.capacity, 0)
        self.assertGreater(overview.page_count, 1)
        target = overview.page_indices(1)[0]
        while engine.index < target:
            page.confirm_button.click()
        self.assertIn(engine.index, overview.visible_indices)
        self.assertEqual(overview.page_index, overview.page_for_index(engine.index))
        before = copy.deepcopy(engine.snapshot())
        overview.show_page(0)
        self.assertFalse(overview.is_following)
        self.assertEqual(engine.snapshot(), before)
        self.assertIn(engine.sop.steps[engine.index].name, page.step_title.text())
        self.assertIn(engine.sop.steps[engine.index + 1].name, page.next_step_label.text())
        page.confirm_button.click()
        self.assertEqual(overview.page_index, 0)
        self.assertIn(engine.sop.steps[engine.index].name, page.step_title.text())
        before = copy.deepcopy(engine.snapshot())
        overview.follow_current()
        self.assertTrue(overview.is_following)
        self.assertIn(engine.index, overview.visible_indices)
        self.assertEqual(engine.snapshot(), before)

    def test_page_stays_on_last_active_step_until_engine_completes_it(self):
        definition = self.definition(45)
        # Mix one-, two- and multi-line names without setting a fixed list size.
        for index, step in enumerate(definition.steps):
            if index % 5 == 2:
                step.name += "：確認工件方向、固定位置與接合狀態均符合要求"
            elif index % 5 == 3:
                step.name += "：安裝前先確認工件方向與全部定位孔，完成後檢查接合位置及零件標記" * 2
        page, engine, clock = self.running(definition)
        overview = page.step_overview
        first = overview.page_indices(0)
        self.assertTrue(first)
        self.assertGreater(overview.page_count, 1)
        boundary = first[-1]
        self.assertEqual(overview.visible_slots[boundary][1], 1)
        while engine.index < boundary:
            page.confirm_button.click()
        self.assertEqual(overview.page_index, 0)
        self.assertEqual(engine.snapshot().steps[boundary].status, StepStatus.ACTIVE)
        self.push(page, engine, clock[0] + .01, update=True)
        self.assertEqual(overview.page_index, 0)
        page.confirm_button.click()
        self.assertEqual(engine.snapshot().steps[boundary].status, StepStatus.DONE)
        self.assertEqual(engine.index, overview.page_indices(1)[0])
        self.assertEqual(overview.page_index, 1)
        self.assertIn(engine.sop.steps[engine.index].name, page.step_title.text())
        self.assertIn(engine.sop.steps[engine.index + 1].name, page.next_step_label.text())

    def test_skipping_last_entry_advances_page_while_retaining_ng_alarm(self):
        page, engine, clock = self.running(self.definition(45))
        overview = page.step_overview
        boundary = overview.page_indices(0)[-1]
        while engine.index < boundary:
            page.confirm_button.click()
        clock[0] += .1
        page.on_events(engine.skip_step(clock[0]))
        self.push(page, engine, clock[0])
        self.assertEqual(engine.snapshot().steps[boundary].status, StepStatus.SKIPPED)
        self.assertFalse(engine.cycle_ok)
        self.assertEqual(overview.page_index, 1)
        self.assertFalse(page.alarm_panel.isHidden())

    def test_alarm_ack_and_instruction_changes_preserve_running_page_geometry(self):
        page, engine, _clock = self.running(self.definition(45))
        self.settle()
        overview = page.step_overview
        height = overview._grid.height()
        pages = tuple(overview.page_indices(index) for index in range(overview.page_count))
        page._show_alarm("確認安裝方向及工件接合狀態" * 6)
        self.settle()
        self.assertEqual(overview._grid.height(), height)
        self.assertEqual(tuple(overview.page_indices(index) for index in range(overview.page_count)), pages)
        page.acknowledge_alarms()
        page._set_instruction("請確認安裝位置及標記方向" * 6)
        self.settle()
        self.assertEqual(overview._grid.height(), height)
        page._set_instruction("")
        self.settle()
        self.assertEqual(overview._grid.height(), height)
        self.assertEqual(overview.page_index, 0)
        self.assertEqual(engine.index, 0)

    def test_oversized_name_keeps_full_list_text_without_covering_camera_image(self):
        name = "確認安裝方向與全部定位孔的接合狀態" * 100 + "【說明結尾】"
        page, _engine, _clock = self.running(SOPDefinition(steps=[Step(name)]))
        overview = page.step_overview
        self.assertTrue(overview.entry_is_oversized(0))
        self.assertIn(name, overview.accessibleDescription())
        scrollbar = overview.entry_scrollbar(0)
        self.assertIsNotNone(scrollbar)
        self.assertGreater(scrollbar.maximum(), 0)
        scrollbar.setValue(scrollbar.maximum())
        image = page.video.grab().toImage()
        center = page.video._image_rect().center().toPoint()
        self.assertEqual(image.pixelColor(center).getRgb()[:3], (0, 0, 0))
        self.assertIn(name, page.video.toolTip())

    def test_cycle_completion_restart_and_new_sop_do_not_keep_old_navigation_or_names(self):
        page, engine, _clock = self.running(self.definition(2))
        page.confirm_button.click()
        page.confirm_button.click()
        self.assertEqual(engine.phase, Phase.CYCLE_DONE)
        self.assertIn("本輪完成", page.step_title.text())
        self.assertFalse(page.confirm_button.isEnabled())
        self.assertTrue(all(step.status == StepStatus.DONE for step in engine.snapshot().steps))
        with patch("sop_app.ui.run_page.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            page.restart_button.click()
        self.assertEqual((engine.phase, engine.cycle, engine.index), (Phase.RUNNING, 2, 0))
        self.assertIn("工序 01", page.step_title.text())
        self.assertIn("工序 02", page.next_step_label.text())
        page.set_running(False)
        page.set_sop(SOPDefinition(name="包裝", steps=[Step("包裝新工序")]))
        self.settle()
        self.assertEqual(page.step_overview.page_count, 1)
        self.assertEqual(page.step_overview.page_index, 0)
        self.assertEqual(list(page.step_overview.visible_indices), [0])
        self.assertNotIn("工序 02", page.next_step_label.text())
        self.assertNotIn("工序 02", page.step_overview.accessibleDescription())

    def test_alarm_status_and_navigation_do_not_complete_or_skip_current_step(self):
        definition = SOPDefinition(cycle=CycleSettings(enabled=False), steps=[
            Step("人工超時檢查", timeout_sec=.1), Step("後續檢查")])
        page, engine, clock = self.running(definition)
        self.push(page, engine, .2, update=True)
        self.assertFalse(engine.cycle_ok)
        self.assertTrue(engine.snapshot().steps[0].has_alarm)
        self.assertEqual((engine.index, engine.snapshot().steps[0].status), (0, StepStatus.ACTIVE))
        self.assertFalse(page.alarm_panel.isHidden())
        before = copy.deepcopy(engine.snapshot())
        page.step_overview.show_page(0)
        page.step_overview.follow_current()
        self.assertEqual(engine.snapshot(), before)
        self.assertIn("警報", page.step_overview.accessibleDescription())
        clock[0] = .2
        page.confirm_button.click()
        self.assertEqual(engine.index, 1)
        self.assertTrue(engine.snapshot().steps[0].has_alarm)
        self.assertEqual(engine.snapshot().steps[0].status, StepStatus.DONE)

    def window(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, path in (("RECORDS_DIR", root), ("PROJECT_ROOT", root),
                           ("SOPS_DIR", root / "sops"), ("APP_SETTINGS", root / "settings.json")):
            stack.enter_context(patch(f"sop_app.ui.main_window.{name}", path))
        stack.enter_context(patch.object(MainWindow, "_load_settings", return_value={}))
        stack.enter_context(patch.object(MainWindow, "_save_settings"))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.start"))
        stack.enter_context(patch("sop_app.ui.main_window.VideoPipeline.shutdown", return_value=True))
        window = MainWindow(defer_initial_sop=True)
        self.addCleanup(self.dispose, window)
        return window

    def test_primary_cards_current_and_next_stay_inside_monitor_at_supported_window_sizes(self):
        window = self.window()
        page = window.run_page
        definition = self.definition()
        page.set_sop(definition)
        engine = SOPEngine(definition)
        engine.start(0.)
        page.set_running(True)
        self.push(page, engine)
        window.show()
        for width, height in ((1366, 768), (1500, 900)):
            for expanded in (False, True):
                with self.subTest(width=width, height=height, auxiliary=expanded):
                    window.resize(width, height)
                    page.details_button.setChecked(expanded)
                    page.log_button.setChecked(expanded)
                    self.settle()
                    self.assertEqual(window.size().toTuple(), (width, height))
                    self.assertGreaterEqual(page.controls_scroll.height(), 32)
                    if not expanded:
                        for control in (page.operator_edit, page.confirm_button, page.skip_button, page.restart_button):
                            self.assertEqual(control.visibleRegion().boundingRect().height(), control.height(),
                                             (control.text() if hasattr(control, "text") else control, control.geometry()))
                    overview = page.step_overview
                    self.assertGreater(overview.capacity, 0)
                    self.assertTrue(overview.visible_indices)
                    for index in overview.visible_indices:
                        rect = overview.card_rect(index)
                        self.assertFalse(rect.isEmpty())
                        self.assertTrue(overview.rect().contains(rect.toAlignedRect()), (index, rect, overview.rect()))
                    for widget in (overview, page.step_title, page.next_step_label):
                        self.assertTrue(widget.isVisible())
                        bounds = QRect(widget.mapTo(window, QPoint()), widget.size())
                        self.assertTrue(window.rect().contains(bounds), (widget.objectName(), bounds, window.rect()))
                    ancestor = overview.parentWidget()
                    while ancestor is not None and ancestor is not page:
                        self.assertNotIsInstance(ancestor, QScrollArea)
                        ancestor = ancestor.parentWidget()


if __name__ == "__main__":
    unittest.main()
