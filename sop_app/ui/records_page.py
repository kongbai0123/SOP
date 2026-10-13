"""作業紀錄頁：每輪結果（可篩選、匯出）、各工序耗時、警報與當下截圖。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import QDate, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDateEdit, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..recorder import ALARM_TEXT, RESULT_TEXT, cycle_filter_sql, export_csv, has_column, query
from .workbench_widgets import install_readability_guard

RESULT_COLORS = {"OK": "#c8e6c9", "NG": "#ffe0b2", "ABORTED": "#eceff1", "RUNNING": "#bbdefb",
                 "INTERRUPTED": "#f8bbd0"}
STATUS_TEXT = {"done": "完成", "skipped": "跳過"}
PAGE_LIMIT = 1000


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(False)
    table.setTextElideMode(Qt.TextElideMode.ElideNone)
    return table


def _fill(table: QTableWidget, rows: list[tuple]):
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            text = "" if value is None else str(value)
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            table.setItem(r, c, item)
    table.resizeRowsToContents()


class RecordsPage(QWidget):
    def __init__(self, records_dir: Path, parent=None):
        super().__init__(parent)
        self.records_dir = Path(records_dir)
        self.db_path = self.records_dir / "records.db"
        self._stale = True

        refresh_button = QPushButton("重新整理")
        refresh_button.clicked.connect(self.refresh)
        export_button = QPushButton("匯出 CSV…")
        export_button.setToolTip("依目前篩選條件匯出逐工序明細，可直接以 Excel 開啟")
        export_button.clicked.connect(self._export)
        folder_button = QPushButton("開啟截圖資料夾")
        folder_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.records_dir / "snapshots"))))

        # 篩選：日期區間（預設不限）、結果、關鍵字（SOP／作業員／工單／序號／工站）
        self.use_dates = QCheckBox("日期")
        self.date_from = QDateEdit(QDate.currentDate().addDays(-7))
        self.date_to = QDateEdit(QDate.currentDate())
        for edit in (self.date_from, self.date_to):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
            edit.setEnabled(False)
            edit.dateChanged.connect(self.refresh)
        self.use_dates.toggled.connect(self._toggle_dates)
        self.result_filter = QComboBox()
        self.result_filter.addItem("全部結果", "")
        for key in ("OK", "NG", "ABORTED", "INTERRUPTED"):
            self.result_filter.addItem(RESULT_TEXT[key], key)
        self.result_filter.currentIndexChanged.connect(self.refresh)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜尋 SOP、作業員、工單、序號、工站")
        self.search.setClearButtonEnabled(True)
        self.search.returnPressed.connect(self.refresh)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.use_dates)
        toolbar.addWidget(self.date_from)
        toolbar.addWidget(QLabel("～"))
        toolbar.addWidget(self.date_to)
        toolbar.addWidget(self.result_filter)
        toolbar.addWidget(self.search, 1)
        summary_row = QHBoxLayout()
        summary_row.addWidget(self.summary, 1)
        for button in (refresh_button, export_button, folder_button):
            summary_row.addWidget(button)

        self.cycles = _table(["編號", "SOP", "版本", "模型", "開始時間", "耗時(秒)", "結果",
                              "工站", "作業員", "工單", "序號"])
        self.cycles.itemSelectionChanged.connect(self._on_cycle_selected)
        self.steps = _table(["工序", "名稱", "狀態", "耗時(秒)", "手動", "警報", "完成時間", "截圖"])
        self.steps.itemSelectionChanged.connect(lambda: self._preview_from(self.steps, 7))
        self.alarms = _table(["時間", "工序", "類型", "訊息", "確認", "截圖"])
        self.alarms.itemSelectionChanged.connect(lambda: self._preview_from(self.alarms, 5))
        self.positioning = _table(["時間", "工序", "定位事件", "訊息"])
        self.preview = QLabel("選擇工序或警報以預覽截圖")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setWordWrap(True)
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
        layout.addLayout(summary_row)
        layout.addWidget(splitter)
        install_readability_guard(self)

    def mark_stale(self):
        self._stale = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        if self._stale:
            self.refresh()

    def _toggle_dates(self, enabled: bool):
        self.date_from.setEnabled(enabled)
        self.date_to.setEnabled(enabled)
        self.refresh()

    def filters(self) -> dict:
        dates = self.use_dates.isChecked()
        return dict(date_from=self.date_from.date().toString("yyyy-MM-dd") if dates else "",
                    date_to=self.date_to.date().toString("yyyy-MM-dd") if dates else "",
                    result=self.result_filter.currentData() or "", text=self.search.text().strip())

    def refresh(self):
        self._stale = False
        extended = has_column(self.db_path, "cycles", "operator")
        where, params = cycle_filter_sql(**self.filters(), extended=extended)
        trace = "c.station, c.operator, c.work_order, c.serial_no" if extended else "'', '', '', ''"
        rows = query(self.db_path, f"SELECT c.id, c.sop_name, c.sop_version, c.model_version, c.started_at, "
                                   f"c.duration_sec, c.result, {trace} FROM cycles c{where} "
                                   f"ORDER BY c.id DESC LIMIT {PAGE_LIMIT}", tuple(params))
        self.cycles.blockSignals(True)
        _fill(self.cycles, [(*row[:6], RESULT_TEXT.get(row[6], row[6]), *row[7:]) for row in rows])
        for r, row in enumerate(rows):
            self.cycles.item(r, 6).setBackground(QColor(RESULT_COLORS.get(row[6], "#ffffff")))
            self.cycles.item(r, 6).setForeground(QColor("#1b1f23"))
        self.cycles.blockSignals(False)
        stats = dict(query(self.db_path, f"SELECT c.result, COUNT(*) FROM cycles c{where} GROUP BY c.result",
                           tuple(params)))
        judged = stats.get("OK", 0) + stats.get("NG", 0)
        rate = f" · 良率 {stats.get('OK', 0) / judged:.1%}" if judged else ""
        shown = f"（顯示最近 {PAGE_LIMIT} 筆）" if len(rows) >= PAGE_LIMIT else ""
        self.summary.setText(f"OK {stats.get('OK', 0)} · NG {stats.get('NG', 0)} · 中止 {stats.get('ABORTED', 0)}"
                             f" · 中斷 {stats.get('INTERRUPTED', 0)}{rate}{shown}")
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
                            "有警報" if alarm else "", finished, snapshot)
                           for i, name, status, duration, manual, alarm, finished, snapshot in steps])
        if has_column(self.db_path, "alarms", "snapshot"):
            alarms = query(self.db_path, "SELECT created_at, step_index, kind, message, "
                                         "acknowledged_at, acknowledged_by, snapshot FROM alarms "
                                         "WHERE cycle_id=? ORDER BY id", (cycle_id,))
            _fill(self.alarms, [(created, step, ALARM_TEXT.get(kind, kind), message,
                                 f"{acked} {by}".strip() if acked else "未確認", snapshot)
                                for created, step, kind, message, acked, by, snapshot in alarms])
        else:
            _fill(self.alarms, query(self.db_path, "SELECT created_at, step_index, kind, message FROM alarms "
                                                   "WHERE cycle_id=? ORDER BY id", (cycle_id,)))
        rows = query(self.db_path, "SELECT created_at, step_index, kind, message FROM positioning_events "
                                   "WHERE cycle_id=? ORDER BY id", (cycle_id,))
        _fill(self.positioning, [(time, step, "無法確認" if kind == "position_lost" else "恢復", message)
                                 for time, step, kind, message in rows])

    def _preview_from(self, table: QTableWidget, column: int):
        row = table.currentRow()
        item = table.item(row, column) if row >= 0 else None
        path = self.records_dir / "snapshots" / item.text() if item and item.text() else None
        if path is None or not path.exists():
            self.preview.setText("沒有截圖")
            return
        pixmap = QPixmap(str(path))
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))

    def _export(self):
        default = Path.home() / f"SOP紀錄_{date.today():%Y%m%d}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "匯出作業紀錄", str(default), "CSV 檔案 (*.csv)")
        if not path:
            return
        try:
            count = export_csv(self.db_path, Path(path), **self.filters())
        except Exception as exc:
            QMessageBox.warning(self, "匯出失敗", str(exc))
            return
        QMessageBox.information(self, "匯出完成", f"已匯出 {count} 列到\n{path}")
