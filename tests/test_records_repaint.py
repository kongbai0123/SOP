"""Tab transitions must not turn inherited update suspension into permanent silence."""
from __future__ import annotations

import hashlib
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from sop_app.recorder import connect
from sop_app.ui.records_page import RecordTable, RecordsPage, _fill
from sop_app.ui.workbench_widgets import PageStack, application_styles


class PaintCounter(QObject):
    def __init__(self, page):
        super().__init__(page)
        self.counts = {}
        self.widgets = {}
        for name in ("cycles", "steps", "alarms", "positioning"):
            table = getattr(page, name)
            for part, widget in (("table", table), ("header", table.horizontalHeader().viewport()),
                                 ("viewport", table.viewport())):
                key = f"{name}.{part}"
                self.counts[key] = 0
                self.widgets[id(widget)] = key
                widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Paint:
            self.counts[self.widgets[id(watched)]] += 1
        return False


class RecordsRepaintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        font = Path("C:/Windows/Fonts/msjh.ttc")
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))

    def settle(self):
        for _ in range(12):
            self.app.processEvents()

    @staticmethod
    def flags(widget):
        return (widget.updatesEnabled(), widget.testAttribute(Qt.WidgetAttribute.WA_UpdatesDisabled),
                widget.testAttribute(Qt.WidgetAttribute.WA_ForceUpdatesDisabled))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sop-record-repaint-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db_path = self.root / "records.db"
        with connect(self.db_path) as db:
            db.executemany("INSERT INTO cycles(id,sop_name,sop_version,model_version,started_at,result,station,operator) "
                           "VALUES(?,?,?,?,?,?,?,?)",
                           [(index, "中文長名稱裝配與工件位置確認 SOP", 23, "M008", "2026-10-19 10:00:00",
                             "NG" if index == 38 else "OK", "ST001", "暫存作業員")
                            for index in range(1, 39)])
            db.executemany("INSERT INTO step_records(cycle_id,step_index,step_name,status,duration_sec,"
                           "finished_at,snapshot) VALUES(38,?,?,'done',?,'2026-10-19 10:00:05','evidence.png')",
                           [(index, f"工序 {index}：" + "確認安裝方向並鎖緊固定螺絲" * 4, float(index))
                            for index in range(1, 7)])
            db.executemany("INSERT INTO alarms(cycle_id,step_index,step_name,kind,message,created_at,snapshot) "
                           "VALUES(38,?,?,'timeout',?,'2026-10-19 10:00:03','evidence.png')",
                           [(index, "確認工作區", "等待條件成立，請確認長名稱工件與工具是否位於指定區域。" * 3)
                            for index in (1, 2)])
            db.executemany("INSERT INTO positioning_events(cycle_id,step_index,kind,source_time,created_at) "
                           "VALUES(38,?,?,?,'2026-10-19 10:00:04')",
                           [(1 + event % 6, kind, float(event * 10 + offset))
                            for event in range(11)
                            for offset, kind in ((0, "position_lost"), (2, "position_recovered"))])
        db.close()
        snapshots = self.root / "snapshots"
        snapshots.mkdir()
        self.evidence = snapshots / "evidence.png"
        image = QImage(640, 360, QImage.Format.Format_RGB32)
        image.fill(QColor("#546e7a"))
        self.assertTrue(image.save(str(self.evidence)))

        self.page = RecordsPage(self.root)
        self.page.setFont(QFont("Microsoft JhengHei", 10))
        self.stack = PageStack()
        self.editor = QLabel("上一頁工序編輯：模型與工序名稱不應殘留在作業紀錄表格內")
        self.stack.addTab(self.editor, "工序編輯")
        self.stack.addTab(self.page, "作業紀錄")
        self.stack.setStyleSheet(application_styles(False))
        self.stack.resize(1500, 900)
        self.counter = PaintCounter(self.page)
        self.addCleanup(self.dispose)
        self.stack.show()
        self.settle()

    def dispose(self):
        for timer in self.stack.findChildren(QTimer):
            timer.stop()
        self.stack.close()

    def assert_tables_can_paint(self):
        for name in ("cycles", "steps", "alarms", "positioning"):
            table = getattr(self.page, name)
            self.assertTrue(table.isVisible(), name)
            for widget in (table, table.horizontalHeader(), table.horizontalHeader().viewport(), table.viewport()):
                self.assertEqual(self.flags(widget), (True, False, False), (name, type(widget).__name__))

    def request_paint_and_assert(self):
        before = dict(self.counter.counts)
        for name in ("cycles", "steps", "alarms", "positioning"):
            table = getattr(self.page, name)
            table.update()
            table.horizontalHeader().viewport().update()
            table.viewport().update()
        self.settle()
        for key, count in before.items():
            self.assertGreater(self.counter.counts[key], count, key)

    def test_page_stack_show_refresh_restores_inherited_updates_and_paints(self):
        self.assertEqual(self.flags(self.page.cycles), (True, False, False))
        during = []
        original = self.page._refresh_records

        def observe_refresh():
            during.append((self.flags(self.page.cycles), self.flags(self.stack)))
            return original()

        with patch.object(self.page, "_refresh_records", side_effect=observe_refresh):
            self.stack.setCurrentWidget(self.page)
            self.settle()
        self.assertTrue(during, "showEvent must exercise the real initial refresh")
        self.assertEqual(during[0], ((False, True, False), (False, True, True)))
        self.assertEqual(self.page.cycles.rowCount(), 38)
        self.assertEqual(self.page.steps.rowCount(), 6)
        self.assertEqual(self.page.alarms.rowCount(), 2)
        self.assert_tables_can_paint()
        for key, count in self.counter.counts.items():
            self.assertGreater(count, 0, key)
        self.assertIn("確認安裝方向", self.page.steps.item(0, 1).text())
        self.assertFalse(self.page.preview.original.isNull())

    def test_fill_preserves_an_explicitly_disabled_table(self):
        table = RecordTable(2, 12, 1)
        self.addCleanup(table.close)
        table.setUpdatesEnabled(False)
        self.assertEqual(self.flags(table), (False, True, True))
        _fill(table, [(1, "明確停畫仍可更新資料模型")], keys=[1])
        self.assertEqual(table.item(0, 1).text(), "明確停畫仍可更新資料模型")
        self.assertEqual(self.flags(table), (False, True, True))
        self.assertFalse(table.viewport().updatesEnabled())
        table.setUpdatesEnabled(True)
        self.assertEqual(self.flags(table), (True, False, False))
        self.assertTrue(table.viewport().updatesEnabled())

    def test_fill_exception_restores_only_its_own_update_suspension(self):
        for state in ("enabled", "inherited", "explicit"):
            with self.subTest(initial_updates=state):
                parent = QWidget()
                self.addCleanup(parent.close)
                table = RecordTable(2, 12, 1)
                table.setParent(parent)
                if state == "inherited":
                    parent.setUpdatesEnabled(False)
                elif state == "explicit":
                    table.setUpdatesEnabled(False)
                before = self.flags(table)
                with patch.object(table, "fit_columns", side_effect=RuntimeError("synthetic layout failure")):
                    with self.assertRaisesRegex(RuntimeError, "synthetic layout failure"):
                        _fill(table, [(1, "例外不能永久停畫")], keys=[1])
                self.assertEqual(self.flags(table), before)
                if state == "inherited":
                    self.assertEqual(before, (False, True, False))
                    parent.setUpdatesEnabled(True)
                    self.assertEqual(self.flags(table), (True, False, False))
                    self.assertTrue(table.viewport().updatesEnabled())

    def test_refresh_tab_return_and_resize_keep_painting_without_changing_evidence(self):
        before_db = hashlib.sha256(self.db_path.read_bytes()).digest()
        before_evidence = hashlib.sha256(self.evidence.read_bytes()).digest()
        self.stack.setCurrentWidget(self.page)
        self.settle()
        self.page.refresh()
        self.assert_tables_can_paint()
        self.request_paint_and_assert()

        self.stack.setCurrentWidget(self.editor)
        self.page.mark_stale()
        self.settle()
        self.assertFalse(self.page._refresh_timer.isActive())
        self.stack.setCurrentWidget(self.page)
        self.settle()
        self.assertFalse(self.page._stale)
        self.assert_tables_can_paint()
        self.request_paint_and_assert()

        for size in ((1366, 768), (1920, 1006)):
            self.stack.resize(*size)
            self.settle()
            self.assert_tables_can_paint()
            self.request_paint_and_assert()
        self.assertEqual(hashlib.sha256(self.db_path.read_bytes()).digest(), before_db)
        self.assertEqual(hashlib.sha256(self.evidence.read_bytes()).digest(), before_evidence)


if __name__ == "__main__":
    unittest.main()
