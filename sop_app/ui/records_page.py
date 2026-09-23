"""作業紀錄頁：每輪結果、各工序耗時、警報與完成截圖。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..recorder import query

RESULT_COLORS = {"OK": "#c8e6c9", "NG": "#ffe0b2", "ABORTED": "#eceff1", "RUNNING": "#bbdefb"}
STATUS_TEXT = {"done": "完成", "skipped": "跳過"}


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _fill(table: QTableWidget, rows: list[tuple]):
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.setItem(r, c, QTableWidgetItem("" if value is None else str(value)))


class RecordsPage(QWidget):
    def __init__(self, records_dir: Path, parent=None):
        super().__init__(parent)
        self.records_dir = Path(records_dir)
        self.db_path = self.records_dir / "records.db"
        self._stale = True

        refresh_button = QPushButton("重新整理")
        refresh_button.clicked.connect(self.refresh)
        folder_button = QPushButton("開啟截圖資料夾")
        folder_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.records_dir / "snapshots"))))
        self.summary = QLabel("")
        toolbar = QHBoxLayout()
        toolbar.addWidget(refresh_button)
        toolbar.addWidget(folder_button)
        toolbar.addStretch()
        toolbar.addWidget(self.summary)

        self.cycles = _table(["編號", "SOP", "版本", "模型", "開始時間", "耗時(秒)", "結果"])
        self.cycles.itemSelectionChanged.connect(self._on_cycle_selected)
        self.steps = _table(["工序", "名稱", "狀態", "耗時(秒)", "手動", "警報", "完成時間", "截圖"])
        self.steps.itemSelectionChanged.connect(self._on_step_selected)
        self.alarms = _table(["時間", "工序", "類型", "訊息"])
        self.positioning = _table(["時間", "工序", "定位事件", "訊息"])
        self.preview = QLabel("選擇工序以預覽完成截圖")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 180)
        self.preview.setStyleSheet("background:#1b1f23; color:#8a949e")

        detail_left = QWidget()
        left_layout = QVBoxLayout(detail_left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("工序紀錄"))
        left_layout.addWidget(self.steps, 2)
        left_layout.addWidget(QLabel("警報"))
        left_layout.addWidget(self.alarms, 1)
        left_layout.addWidget(QLabel("定位紀錄"))
        left_layout.addWidget(self.positioning, 1)
        detail = QSplitter(Qt.Orientation.Horizontal)
        detail.addWidget(detail_left)
        detail.addWidget(self.preview)
        detail.setSizes([700, 500])

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.cycles)
        splitter.addWidget(detail)
        splitter.setSizes([300, 400])

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(splitter)

    def mark_stale(self):
        self._stale = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        if self._stale:
            self.refresh()

    def refresh(self):
        self._stale = False
        rows = query(self.db_path, "SELECT id, sop_name, sop_version, model_version, started_at, duration_sec, result "
                                   "FROM cycles ORDER BY id DESC LIMIT 500")
        self.cycles.blockSignals(True)
        _fill(self.cycles, rows)
        for r, row in enumerate(rows):
            self.cycles.item(r, 6).setBackground(QColor(RESULT_COLORS.get(row[6], "#ffffff")))
        self.cycles.blockSignals(False)
        stats = dict(query(self.db_path, "SELECT result, COUNT(*) FROM cycles GROUP BY result"))
        self.summary.setText(f"OK {stats.get('OK', 0)} · NG {stats.get('NG', 0)} · 中止 {stats.get('ABORTED', 0)}")
        self.steps.setRowCount(0)
        self.alarms.setRowCount(0)
        self.positioning.setRowCount(0)

    def _selected_cycle_id(self) -> int | None:
        row = self.cycles.currentRow()
        return int(self.cycles.item(row, 0).text()) if row >= 0 and self.cycles.item(row, 0) else None

    def _on_cycle_selected(self):
        cycle_id = self._selected_cycle_id()
        if cycle_id is None:
            return
        steps = query(self.db_path, "SELECT step_index, step_name, status, duration_sec, manual, had_alarm, "
                                    "finished_at, snapshot FROM step_records WHERE cycle_id=? ORDER BY id", (cycle_id,))
        _fill(self.steps, [(i, name, STATUS_TEXT.get(status, status), duration, "是" if manual else "",
                            "⚠" if alarm else "", finished, snapshot)
                           for i, name, status, duration, manual, alarm, finished, snapshot in steps])
        _fill(self.alarms, query(self.db_path, "SELECT created_at, step_index, kind, message FROM alarms "
                                               "WHERE cycle_id=? ORDER BY id", (cycle_id,)))

        self.positioning.setRowCount(0)
        if query(self.db_path, "SELECT name FROM sqlite_master WHERE type='table' AND name='positioning_events'"):
            rows = query(self.db_path, "SELECT created_at, step_index, kind, message FROM positioning_events "
                         "WHERE cycle_id=? ORDER BY id", (cycle_id,))
            _fill(self.positioning, [(time, step, "無法確認" if kind == "position_lost" else "恢復", message)
                                    for time, step, kind, message in rows])

    def _on_step_selected(self):
        row = self.steps.currentRow()
        item = self.steps.item(row, 7) if row >= 0 else None
        path = self.records_dir / "snapshots" / item.text() if item and item.text() else None
        if path is None or not path.exists():
            self.preview.setText("沒有截圖")
            return
        pixmap = QPixmap(str(path))
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
