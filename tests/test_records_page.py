"""Historical record binding and interruption summaries use temporary data only."""
from __future__ import annotations

import csv
import hashlib
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from PySide6.QtCore import QDate, QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage
from PySide6.QtWidgets import QApplication

from sop_app.recorder import _MIGRATIONS, connect
from sop_app.ui.record_summary import positioning_pauses, positioning_summary
from sop_app.ui.records_page import RecordsPage


class PositioningSummaryTests(unittest.TestCase):
    def test_repeated_loss_is_one_pause_and_source_time_measures_duration(self):
        events = [(1, 1, "position_lost", 10., "2026-10-16 10:00:00"),
                  (1, 1, "position_lost", 11., "2026-10-16 10:00:00"),
                  (1, 1, "position_recovered", 12.5, "2026-10-16 10:00:00")]
        pauses = positioning_pauses(events)
        self.assertEqual(len(pauses), 1)
        self.assertEqual(pauses[0].duration, 2.5)
        text, rows = positioning_summary(events, {1: "安裝"})
        self.assertIn("判定暫停 1 次", text)
        self.assertEqual(rows[0][:4], ("1 · 安裝", 1, "2.50", "已恢復"))

    def test_missing_source_time_never_infers_duration_from_write_timestamp(self):
        pauses = positioning_pauses([(1, 1, "position_lost", None, "2026-10-16 10:00:00"),
                                     (1, 1, "position_recovered", None, "2026-10-16 10:02:00")])
        self.assertTrue(pauses[0].recovered)
        self.assertIsNone(pauses[0].duration)

    def test_step_and_cycle_boundaries_never_pair_losses_with_other_recoveries(self):
        pauses = positioning_pauses([(1, 1, "position_lost", 1., "a"),
                                     (1, 2, "position_recovered", 2., "b"),
                                     (2, 1, "position_recovered", 3., "c"),
                                     (2, 2, "position_lost", 4., "d")])
        self.assertEqual(len(pauses), 4)
        self.assertEqual([(p.cycle_id, p.step_index) for p in pauses], [(1, 1), (1, 2), (2, 1), (2, 2)])
        self.assertFalse(pauses[0].recovered)
        self.assertEqual(pauses[1].started_at, "")
        self.assertFalse(pauses[-1].recovered)
        self.assertTrue(all(p.duration is None for p in pauses))

    def test_source_clock_reset_and_unknown_events_cannot_fake_a_recovery(self):
        pauses = positioning_pauses([(1, 1, "position_lost", 100., "a"),
                                     (1, 1, "unknown", 101., "b"),
                                     (1, 1, "position_recovered", 1., "c")])
        self.assertEqual(len(pauses), 2)
        self.assertFalse(pauses[0].recovered)
        self.assertFalse(pauses[1].started_at)
        text, _rows = positioning_summary([(1, 1, "position_lost", 1., "a")], {})
        self.assertIn("未記錄恢復 1 次", text)


class RecordsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def dispose(page):
        page.close()
        for timer in page.findChildren(QTimer):
            timer.stop()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = self.root / "records.db"
        with connect(self.db_path) as db:
            db.executemany("INSERT INTO cycles (id,sop_name,sop_version,model_version,started_at,duration_sec,"
                           "result,station,operator,work_order,serial_no) VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
                (1, "裝配 SOP", 7, "model-a", "2026-10-15 10:00:00", 8.5, "OK", "A站", "甲", "WO-1", "SN-1"),
                (2, "裝配 SOP", 8, "model-b", "2026-10-16 11:00:00", 9., "NG", "B站", "乙", "WO-2", "SN-2"),
                (3, "包裝 SOP", 1, "model-c", "2026-10-16 12:00:00", 10., "OK", "C站", "丙", "WO-3", "SN-3")])
            db.executemany("INSERT INTO step_records (cycle_id,step_index,step_name,status,manual,had_alarm,"
                           "duration_sec,finished_at,snapshot) VALUES (?,?,?,?,?,?,?,?,?)", [
                (1, 1, "安裝", "done", 0, 0, 8.5, "2026-10-15 10:00:08", "complete.png"),
                (2, 1, "安裝", "done", 1, 1, 9., "2026-10-16 11:00:09", "missing.png"),
                (2, 2, "確認", "skipped", 1, 1, 0., "2026-10-16 11:00:09", ""),
                (3, 1, "包裝", "done", 0, 0, 10., "2026-10-16 12:00:10", "")])
            db.execute("INSERT INTO alarms (cycle_id,step_index,step_name,kind,message,created_at,snapshot) "
                       "VALUES (2,1,'安裝','violation','違規：發現禁止物件','2026-10-16 11:00:05','alarm.png')")
            db.executemany("INSERT INTO positioning_events (cycle_id,step_index,kind,source_time,created_at) "
                           "VALUES (?,?,?,?,?)", [(2, 1, "position_lost", 1., "2026-10-16 11:00:01"),
                                                 (2, 1, "position_recovered", 2.5, "2026-10-16 11:00:02"),
                                                 (2, 1, "position_lost", 3., "2026-10-16 11:00:03")])
        db.close()
        snapshots = self.root / "snapshots"
        snapshots.mkdir()
        for name, size, color in (("complete.png", (200, 100), "green"), ("alarm.png", (100, 200), "red")):
            image = QImage(*size, QImage.Format.Format_RGB32)
            image.fill(QColor(color))
            self.assertTrue(image.save(str(snapshots / name)))
        self.page = RecordsPage(self.root)
        self.addCleanup(self.dispose, self.page)
        self.page.resize(1280, 800)
        self.page.refresh()

    def select_cycle(self, cycle_id):
        row = next(i for i in range(self.page.cycles.rowCount()) if
                   self.page.cycles.item(i, 0).text() == str(cycle_id))
        self.page.cycles.selectRow(row)

    def test_binding_simplifies_visible_columns_without_losing_metadata_or_evidence(self):
        self.select_cycle(2)
        self.assertEqual(self.page.cycles.columnCount(), 6)
        self.assertIn("SOP 版本 8", self.page.cycle_summary.text())
        self.assertIn("model-b", self.page.cycle_summary.text())
        # Traceability stays in the round row; the detail line adds only version and model.
        row = self.page.cycles.currentRow()
        self.assertIn("工單 WO-2", self.page.cycles.item(row, 5).text())
        self.assertNotIn("工單", self.page.cycle_summary.text())
        self.assertEqual(self.page.steps.item(0, 5).text(), "11:00:09")
        self.assertEqual(self.page.steps.item(0, 2).text(), "人工完成")
        self.assertEqual(self.page.steps.item(1, 2).text(), "人工跳過")
        self.assertEqual(self.page.steps.columnCount(), 6)
        self.assertEqual(self.page.steps.item(0, 0).data(Qt.ItemDataRole.UserRole)["snapshot"], "missing.png")
        self.assertEqual(self.page.alarms.item(0, 2).text(), "違規：發現禁止物件")
        self.assertEqual(self.page.positioning.rowCount(), 1)
        self.assertEqual(self.page.positioning.columnCount(), 4)
        self.assertEqual(self.page.positioning.item(0, 1).text(), "2")
        self.assertIn("未記錄恢復", self.page.positioning.item(0, 3).text())
        self.assertIn("判定暫停 2 次", self.page.positioning_summary.text())
        self.assertIn("累計 1.50", self.page.positioning_summary.text())
        self.assertEqual(self.page.alarms_title.text(), "警報 1 筆")
        self.assertIn("輪次 2", self.page.preview_caption.text())
        self.assertIn("警報／違規", self.page.preview_caption.text())
        self.assertFalse(self.page.preview.original.isNull())

    def test_switching_and_refresh_never_leave_an_old_image(self):
        self.select_cycle(1)
        self.assertEqual(self.page.preview.original.size().width(), 200)
        self.page.refresh()
        self.assertEqual(self.page._selected_cycle_id(), 1)
        self.assertIn("輪次 1", self.page.preview_caption.text())
        self.select_cycle(2)
        self.assertEqual(self.page.preview.original.size().height(), 200)
        self.page.steps.selectRow(0)
        self.assertTrue(self.page.preview.original.isNull())
        self.assertIn("找不到", self.page.preview.text())
        self.page.steps.selectRow(1)
        self.assertIn("未保存", self.page.preview.text())
        self.select_cycle(3)
        self.assertTrue(self.page.preview.original.isNull())
        self.assertIn("輪次 3", self.page.preview_caption.text())
        # A round without alarms or pauses says so in one line instead of an empty table.
        self.assertTrue(self.page.alarms.isHidden())
        self.assertEqual(self.page.alarms_empty.text(), "本輪無警報。")
        self.assertTrue(self.page.positioning.isHidden())
        self.assertIn("無定位暫停", self.page.positioning_summary.text())
        self.page.search.setText("不存在")
        self.page.refresh()
        self.assertEqual(self.page.cycles.rowCount(), 0)
        self.assertTrue(self.page.cycles.isHidden())
        self.assertFalse(self.page.cycles_empty.isHidden())
        self.assertIsNone(self.page._selected_cycle_id())
        self.assertEqual(self.page.steps.rowCount(), 0)
        self.assertTrue(self.page.preview.original.isNull())
        self.assertNotIn("輪次 3", self.page.preview_caption.text())

    def test_filters_and_csv_keep_all_step_fields_and_only_matching_cycles(self):
        self.page.search.setText("WO-2")
        self.page.result_filter.setCurrentIndex(self.page.result_filter.findData("NG"))
        self.assertEqual(self.page.cycles.rowCount(), 1)
        self.assertEqual(self.page._selected_cycle_id(), 2)
        target = self.root / "filtered.csv"
        self.assertEqual(self.page._export_records(target), 2)
        with target.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows[0]), 20)
        self.assertTrue(all(row[0] == "2" and row[6] == "8" and row[7] == "model-b" for row in rows[1:]))
        self.assertEqual(rows[1][-1], "違規：發現禁止物件")
        self.page.search.clear()
        self.page.result_filter.setCurrentIndex(0)
        self.page.date_from.setDate(QDate(2026, 10, 15))
        self.page.date_to.setDate(QDate(2026, 10, 15))
        self.page.use_dates.setChecked(True)
        self.assertEqual(self.page.cycles.rowCount(), 1)
        self.assertEqual(self.page._selected_cycle_id(), 1)

    def test_image_resizes_from_original_and_keeps_aspect_ratio(self):
        self.select_cycle(2)
        self.page.preview.resize(360, 180)
        self.page.preview._rescale()
        image = self.page.preview.pixmap()
        self.assertEqual((image.width(), image.height()), (90, 180))
        self.assertEqual((self.page.preview.original.width(), self.page.preview.original.height()), (100, 200))

    def test_opening_and_filtering_do_not_modify_database(self):
        before = hashlib.sha256(self.db_path.read_bytes()).digest()
        self.page.refresh()
        self.select_cycle(2)
        self.page._export_records(self.root / "readonly.csv")
        self.assertEqual(before, hashlib.sha256(self.db_path.read_bytes()).digest())

    def test_continuous_notifications_coalesce_without_postponing_refresh(self):
        self.page.show()
        self.app.processEvents()
        self.page._refresh_timer.setInterval(40)
        notifications = []
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: (notifications.append(1), self.page.mark_stale()))
        with patch.object(self.page, "_refresh_records", wraps=self.page._refresh_records) as refresh:
            timer.start()
            QTest.qWait(180)
            timer.stop()
            during = refresh.call_count
            QTest.qWait(80)
            self.assertGreaterEqual(during, 2, "ongoing writes must not starve the refresh timer")
            self.assertLess(refresh.call_count, len(notifications))
            self.assertFalse(self.page._stale)

    def test_manual_refresh_and_hide_cancel_scheduled_work(self):
        self.page.show()
        self.app.processEvents()
        self.page._refresh_timer.setInterval(40)
        with patch.object(self.page, "_refresh_records", wraps=self.page._refresh_records) as refresh:
            self.page.mark_stale()
            self.page.refresh()
            QTest.qWait(80)
            self.assertEqual(refresh.call_count, 1)
            self.page.mark_stale()
            self.page.hide()
            QTest.qWait(80)
            self.assertEqual(refresh.call_count, 1)
            self.assertTrue(self.page._stale)
            self.page.show()
            self.assertEqual(refresh.call_count, 2)
            self.assertFalse(self.page._refresh_timer.isActive())

    def test_appending_round_and_details_preserves_event_image_and_reuses_cells(self):
        with connect(self.db_path) as db:
            db.execute("INSERT INTO step_records(cycle_id,step_index,step_name,status,snapshot) "
                       "VALUES(1,2,'檢查','done','alarm.png')")
        db.close()
        self.page.refresh()
        self.select_cycle(1)
        self.page.steps.selectRow(1)
        caption = self.page.preview_caption.text()
        image_key = self.page.preview.original.cacheKey()
        item = self.page.cycles.item(self.page.cycles.currentRow(), 0)
        with connect(self.db_path) as db:
            db.execute("INSERT INTO cycles(id,sop_name,sop_version,started_at,result) "
                       "VALUES(4,'新輪次',1,'2026-10-16 13:00:00','RUNNING')")
            db.execute("INSERT INTO step_records(cycle_id,step_index,step_name,status,snapshot) "
                       "VALUES(1,3,'新明細','done','complete.png')")
        db.close()
        self.page.refresh()
        self.assertEqual(self.page._selected_cycle_id(), 1)
        self.assertEqual(self.page.steps.currentRow(), 1)
        self.assertEqual(self.page.preview_caption.text(), caption)
        self.assertEqual(self.page.preview.original.cacheKey(), image_key)
        self.assertIs(self.page.cycles.item(self.page.cycles.currentRow(), 0), item)
        self.assertEqual(self.page.steps.rowCount(), 3)

    def test_unchanged_detail_refresh_keeps_image_and_station_switch_clears_pending(self):
        self.select_cycle(1)
        with patch.object(self.page.preview, "show_image", wraps=self.page.preview.show_image) as show:
            self.page.refresh()
            self.assertFalse(show.called)
        self.page.show()
        self.app.processEvents()
        self.page.mark_stale()
        second = self.root / "second"
        second.mkdir()
        connect(second / "records.db").close()
        self.page.set_records_directory(second)
        self.assertFalse(self.page._refresh_timer.isActive())
        self.assertIsNone(self.page._selected_cycle_id())
        self.assertTrue(self.page.preview.original.isNull())
        self.assertEqual(self.page.db_path, second / "records.db")

    def test_main_window_shows_details_and_caption_without_outer_scroll_at_900_height(self):
        from sop_app.ui.main_window import MainWindow

        original_font = self.app.font()
        font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msjh.ttc")
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            self.app.setFont(QFont(families[0], 10))
        self.addCleanup(self.app.setFont, original_font)
        with patch("sop_app.ui.main_window.RECORDS_DIR", self.root), \
             patch.object(MainWindow, "_load_settings", return_value={}), \
             patch.object(MainWindow, "_save_settings"), \
             patch("sop_app.ui.main_window.VideoPipeline.start"), \
             patch("sop_app.ui.main_window.cv2.require", return_value=Mock()):
            window = MainWindow(defer_initial_sop=True)
            try:
                window.resize(1500, 900)
                page = window.records_page
                window.tabs.setCurrentWidget(page)
                window.show()
                page.refresh()
                page.cycles.selectRow(1)  # An alarm, positioning summary and image are all present.
                for _ in range(12):
                    self.app.processEvents()
                wrapper = page.parentWidget().parentWidget()
                viewport = wrapper.viewport()
                self.assertEqual(window.size().toTuple(), (1500, 900))
                self.assertEqual(wrapper.verticalScrollBar().maximum(), 0)
                for widget in (page.steps, page.alarms, page.positioning, page.preview_caption):
                    bottom = widget.mapTo(viewport, QPoint(0, widget.height())).y()
                    self.assertLessEqual(bottom, viewport.height())
                # Tables take the height of their rows instead of a large empty grid.
                for table in (page.cycles, page.steps):
                    rows = table.horizontalHeader().height() + table.rowCount() * table.rowHeight(0)
                    self.assertLessEqual(table.height(), rows + 8)
                self.assertAlmostEqual(page.preview.height() / page.preview.width(), 9 / 16, delta=.02)
                # Small screens keep the window size; anything taller scrolls.
                window.resize(911, 512)
                for _ in range(6):
                    self.app.processEvents()
                self.assertEqual(window.size().toTuple(), (911, 512))
                self.assertEqual(wrapper.horizontalScrollBar().maximum(), 0)
            finally:
                window.dirty = False
                window.close()

    def test_version_one_database_is_read_without_migration_and_can_export(self):
        old_root = self.root / "old"
        old_root.mkdir()
        old_path = old_root / "records.db"
        with sqlite3.connect(old_path) as db:
            db.executescript(_MIGRATIONS[1])
            db.execute("INSERT INTO cycles (id,sop_name,sop_version,model_version,started_at,result) "
                       "VALUES (1,'舊 SOP',1,'old-model','2025-01-01 10:00:00','NG')")
            db.execute("INSERT INTO step_records (cycle_id,step_index,step_name,status,manual) "
                       "VALUES (1,1,'安裝','done',0)")
            db.execute("INSERT INTO alarms (cycle_id,step_index,step_name,kind,message,created_at) "
                       "VALUES (1,1,'安裝','timeout','逾時','2025-01-01 10:00:00')")
        db.close()
        before = old_path.read_bytes()
        page = RecordsPage(old_root)
        self.addCleanup(page.close)
        page.refresh()
        self.assertEqual(page._selected_cycle_id(), 1)
        self.assertIn("old-model", page.cycle_summary.text())
        self.assertEqual(page.alarms.rowCount(), 1)
        self.assertIn("未保存", page.preview.text())
        self.assertEqual(page._export_records(old_root / "old.csv"), 1)
        self.assertEqual(before, old_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
