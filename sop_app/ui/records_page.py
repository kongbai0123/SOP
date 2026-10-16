"""作業紀錄頁：每輪結果（可篩選、匯出）、各工序耗時、警報與當下截圖。"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from PySide6.QtCore import QDate, QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDateEdit, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
                               QSizePolicy, QTableWidgetItem, QVBoxLayout, QWidget)

from ..recorder import ALARM_TEXT, RESULT_TEXT, cycle_filter_sql, export_csv, has_column, query
from .compact_layout import set_state, text_label
from .record_summary import positioning_summary
from .workbench_widgets import install_readability_guard

RESULT_COLORS = {"OK": "#c8e6c9", "NG": "#ffe0b2", "ABORTED": "#eceff1", "RUNNING": "#bbdefb",
                 "INTERRUPTED": "#f8bbd0"}
PAGE_LIMIT = 1000


class RecordTable(QTableWidget):
    """Table sized to its rows; one flexible column absorbs spare width.

    Detail tables report the height of their rows (up to ``max_rows``) so a
    short list does not leave a large empty grid. Columns keep their full
    content width; only the flexible column grows to fill the viewport.
    """
    def __init__(self, columns: int, max_rows: int, flex_column: int, fill=False):
        super().__init__(0, columns)
        self.max_rows, self.flex_column, self.fill = max_rows, flex_column, fill
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding if fill else QSizePolicy.Policy.Maximum)

    def _row_height(self):
        return max(self.verticalHeader().defaultSectionSize(), self.fontMetrics().height() + 8)

    def _height_for(self, rows):
        height = self.horizontalHeader().sizeHint().height() + rows * self._row_height() + 2 * self.frameWidth()
        if self.horizontalScrollBar().maximum() > 0:
            height += self.horizontalScrollBar().sizeHint().height()
        return height

    def minimumSizeHint(self):
        return QSize(super().minimumSizeHint().width(), self._height_for(1))

    def sizeHint(self):
        rows = self.max_rows if self.fill else min(self.max_rows, max(1, self.rowCount()))
        return QSize(super().sizeHint().width(), self._height_for(rows))

    def fit_columns(self):
        header = self.horizontalHeader()
        self.resizeColumnsToContents()
        spare = self.viewport().width() - header.length()
        if spare > 0 and 0 <= self.flex_column < self.columnCount():
            header.resizeSection(self.flex_column, header.sectionSize(self.flex_column) + spare)
        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_columns()


def _table(headers: list[str], max_rows: int = 6, flex_column: int = 1, fill=False) -> RecordTable:
    table = RecordTable(len(headers), max_rows, flex_column, fill)
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(table.fontMetrics().height() + 8)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.horizontalHeader().setHighlightSections(False)
    table.horizontalHeader().setStretchLastSection(False)
    table.setTextElideMode(Qt.TextElideMode.ElideNone)
    table.setWordWrap(False)
    return table


def _fill(table: QTableWidget, rows: list[tuple]):
    table.clearContents()
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            text = "" if value is None else str(value)
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            table.setItem(r, c, item)
    if isinstance(table, RecordTable):
        table.fit_columns()
    else:
        table.resizeColumnsToContents()


def _short_time(value, day: str) -> str:
    """Events of the selected cycle show the time only; the date is in the cycle row."""
    text = "" if value is None else str(value)
    return text[len(day) + 1:] if day and text.startswith(day + " ") else text


class SnapshotPreview(QLabel):
    """16:9 evidence area; always scales from the original image.

    Without an image only a one-line message strip is shown, so a missing or
    unselected screenshot never leaves a large empty box.
    """
    MESSAGE_HEIGHT = 34
    def __init__(self):
        super().__init__()
        self.original = QPixmap()
        self.setObjectName("snapshotPreview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(240, 135)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setStyleSheet("background:#1b1f23; color:#c2cbd2")

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return max(135, round(width * 9 / 16)) if not self.original.isNull() else self.MESSAGE_HEIGHT

    def sizeHint(self):
        return QSize(480, 270 if not self.original.isNull() else self.MESSAGE_HEIGHT)

    def minimumSizeHint(self):
        return QSize(240, 135 if not self.original.isNull() else self.MESSAGE_HEIGHT)

    def _set_compact(self, compact):
        self.setMinimumHeight(self.MESSAGE_HEIGHT if compact else 135)
        self.updateGeometry()

    def show_image(self, pixmap: QPixmap):
        self.original = pixmap
        self._set_compact(pixmap.isNull())
        self._rescale()

    def show_message(self, text: str):
        self.original = QPixmap()
        self.clear()
        self.setText(text)
        self._set_compact(True)

    def _rescale(self):
        if not self.original.isNull():
            self.setPixmap(self.original.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def _column(*items, spacing=4) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for item in items:
        if isinstance(item, int):
            layout.addStretch(item)
        else:
            layout.addWidget(item)
    return widget


def _line(*items) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    for item in items:
        layout.addWidget(item)
    return layout


class RecordsPage(QWidget):
    def __init__(self, records_dir: Path, parent=None):
        super().__init__(parent)
        self.records_dir = Path(records_dir)
        self.db_path = self.records_dir / "records.db"
        self._stale = True
        self._cycle_details = {}

        refresh_button = QPushButton("重新整理")
        refresh_button.clicked.connect(self.refresh)
        export_button = QPushButton("匯出 CSV…")
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
        self.search.setMinimumWidth(140)
        self.search.returnPressed.connect(self.refresh)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        for widget in (self.use_dates, self.date_from, QLabel("～"), self.date_to, self.result_filter):
            toolbar.addWidget(widget)
        toolbar.addWidget(self.search, 1)
        for button in (refresh_button, export_button, folder_button):
            toolbar.addWidget(button)

        # Cycles: one row per round; version/model live in the selected-round line.
        self.summary = text_label("", "mutedLabel", wrap=False)
        # Tables take the height of their rows; spare height stays below them.
        self.cycles = _table(["輪次", "SOP", "開始時間", "耗時(秒)", "結果", "作業資訊"], 12, 5)
        self.cycles.itemSelectionChanged.connect(self._on_cycle_selected)
        self.cycles_empty = text_label("沒有符合條件的作業紀錄。", "mutedLabel")
        self.cycle_summary = text_label("", "sectionTitle")
        self.steps = _table(["工序", "名稱", "結果", "耗時(秒)", "警報", "完成時間"], 12, 1)
        self.steps.itemSelectionChanged.connect(lambda: self._preview_from(self.steps))
        self.steps_empty = text_label("", "mutedLabel")
        records = _column(self._header(_title("作業輪次"), self.summary), self.cycles, self.cycles_empty,
                          self.cycle_summary, self.steps, self.steps_empty, 1)
        records.layout().setSpacing(4)
        records.layout().insertSpacing(3, 6)

        # Evidence: the image for the selected step or alarm, then alarms and pauses.
        self.preview = SnapshotPreview()
        self.preview_caption = text_label("", "mutedLabel")
        self.alarms_title = _title("警報")
        self.alarms = _table(["時間", "工序", "警報原因", "確認"], 4, 2)
        self.alarms.itemSelectionChanged.connect(lambda: self._preview_from(self.alarms))
        self.alarms_empty = text_label("", "mutedLabel")
        self.positioning_summary = text_label("", "mutedLabel")
        self.positioning = _table(["工序", "暫停次數", "累計暫停(秒)", "恢復狀態"], 3, 0)
        evidence = _column(_title("歷史截圖"), self.preview, self.preview_caption, self.alarms_title, self.alarms,
                           self.alarms_empty, _title("定位暫停"), self.positioning_summary, self.positioning, 1)

        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(records, 3)
        body.addWidget(evidence, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        layout.addLayout(toolbar)
        layout.addLayout(body, 1)
        self._clear_details()
        install_readability_guard(self)

    @staticmethod
    def _header(title, detail) -> QWidget:
        widget = QWidget()
        layout = _line(title, detail)
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

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
        selected_id = self._selected_cycle_id()
        self._clear_details()
        extended = has_column(self.db_path, "cycles", "operator")
        where, params = cycle_filter_sql(**self.filters(), extended=extended)
        trace = "c.station, c.operator, c.work_order, c.serial_no" if extended else "'', '', '', ''"
        rows = query(self.db_path, f"SELECT c.id, c.sop_name, c.sop_version, c.model_version, c.started_at, "
                                   f"c.duration_sec, c.result, {trace} FROM cycles c{where} "
                                   f"ORDER BY c.id DESC LIMIT {PAGE_LIMIT}", tuple(params))
        self._cycle_details = {row[0]: row for row in rows}
        self.cycles.blockSignals(True)
        self.cycles.clearSelection()
        _fill(self.cycles, [(row[0], row[1], row[4], row[5], RESULT_TEXT.get(row[6], row[6]),
                             self._trace_text(row[7:])) for row in rows])
        for r, row in enumerate(rows):
            self.cycles.item(r, 4).setBackground(QColor(RESULT_COLORS.get(row[6], "#ffffff")))
            self.cycles.item(r, 4).setForeground(QColor("#1b1f23"))
        self.cycles.blockSignals(False)
        self.cycles.setVisible(bool(rows))
        self.cycles_empty.setVisible(not rows)
        stats = dict(query(self.db_path, f"SELECT c.result, COUNT(*) FROM cycles c{where} GROUP BY c.result",
                           tuple(params)))
        judged = stats.get("OK", 0) + stats.get("NG", 0)
        rate = f" · 良率 {stats.get('OK', 0) / judged:.1%}" if judged else ""
        shown = f"（顯示最近 {PAGE_LIMIT} 筆）" if len(rows) >= PAGE_LIMIT else ""
        self.summary.setText(f"OK {stats.get('OK', 0)} · NG {stats.get('NG', 0)} · 中止 {stats.get('ABORTED', 0)}"
                             f" · 中斷 {stats.get('INTERRUPTED', 0)}{rate}{shown}")
        if rows:
            index = next((i for i, row in enumerate(rows) if row[0] == selected_id), 0)
            self.cycles.selectRow(index)

    @staticmethod
    def _trace_text(values):
        return " · ".join(f"{label} {value}" for label, value in zip(("工站", "作業員", "工單", "序號"), values)
                          if value)

    @staticmethod
    def _alarm_reason(kind, message):
        label = ALARM_TEXT.get(kind, kind)
        if message and message.lstrip().startswith((f"{label}：", f"{label}:")):
            return message
        return f"{label}：{message}" if message else label

    def _clear_preview(self, text="選擇工序或警報以預覽歷史截圖"):
        self.preview.show_message(text)
        self.preview_caption.setText("")
        self.preview_caption.hide()

    def _show_table(self, table, empty, rows, empty_text):
        table.setVisible(bool(rows))
        empty.setText(empty_text)
        empty.setVisible(not rows)

    def _clear_details(self):
        for table in (self.steps, self.alarms, self.positioning):
            table.blockSignals(True)
            table.clearSelection()
            table.setRowCount(0)
            table.blockSignals(False)
        self.cycle_summary.setText("輪次明細")
        self._show_table(self.steps, self.steps_empty, [], "選擇一輪作業，查看工序結果與歷史截圖。")
        self.alarms_title.setText("警報")
        self._show_table(self.alarms, self.alarms_empty, [], "—")
        self.positioning_summary.setText("—")
        self.positioning.hide()
        self._clear_preview()

    def _selected_cycle_id(self) -> int | None:
        selected = self.cycles.selectionModel().selectedRows()
        row = selected[0].row() if selected else -1
        return int(self.cycles.item(row, 0).text()) if row >= 0 and self.cycles.item(row, 0) else None

    def _on_cycle_selected(self):
        self._clear_details()
        cycle_id = self._selected_cycle_id()
        if cycle_id is None:
            return
        cycle = self._cycle_details.get(cycle_id)
        day = str(cycle[4] or "")[:10] if cycle else ""
        if cycle:
            self.cycle_summary.setText(
                f"輪次 {cycle_id} 明細 · SOP 版本 {cycle[2] if cycle[2] is not None else '未記錄'}"
                f" · 模型 {cycle[3] or '未記錄'}")
        steps = query(self.db_path, "SELECT step_index, step_name, status, duration_sec, manual, had_alarm, "
                                    "finished_at, snapshot FROM step_records WHERE cycle_id=? ORDER BY id", (cycle_id,))
        _fill(self.steps, [(i, name, "人工跳過" if status == "skipped" else
                            ("人工完成" if manual else "自動完成") if status == "done" else status,
                            duration, "有警報" if alarm else "無", _short_time(finished, day))
                           for i, name, status, duration, manual, alarm, finished, snapshot in steps])
        for row, (step, name, status, _duration, manual, _alarm, finished, snapshot) in enumerate(steps):
            event = "人工跳過" if status == "skipped" else "人工完成" if manual else "工序完成"
            self.steps.item(row, 0).setData(Qt.ItemDataRole.UserRole,
                {"snapshot": snapshot, "caption": f"輪次 {cycle_id} · 工序 {step} {name} · {event} · {finished or '時間未記錄'}"})
            if status == "skipped":
                self.steps.item(row, 2).setForeground(QColor("#b3261e"))
        self._show_table(self.steps, self.steps_empty, steps, "本輪沒有工序紀錄。")
        optional = [name if has_column(self.db_path, "alarms", name) else "''"
                    for name in ("acknowledged_at", "acknowledged_by", "snapshot")]
        alarms = query(self.db_path, "SELECT created_at, step_index, step_name, kind, message, "
                       + ", ".join(optional) + " FROM alarms WHERE cycle_id=? ORDER BY id", (cycle_id,))
        _fill(self.alarms, [(_short_time(created, day), f"{step} · {name}" if name else step,
                             self._alarm_reason(kind, message),
                             f"{_short_time(acked, day)} {by}".strip() if acked else "未確認")
                            for created, step, name, kind, message, acked, by, snapshot in alarms])
        for row, (created, step, name, kind, _message, _acked, _by, snapshot) in enumerate(alarms):
            self.alarms.item(row, 0).setData(Qt.ItemDataRole.UserRole,
                {"snapshot": snapshot, "caption": f"輪次 {cycle_id} · 工序 {step} {name} · 警報／{ALARM_TEXT.get(kind, kind)} · {created or '時間未記錄'}"})
        self.alarms_title.setText(f"警報 {len(alarms)} 筆" if alarms else "警報")
        self._show_table(self.alarms, self.alarms_empty, alarms, "本輪無警報。")
        source = "source_time" if has_column(self.db_path, "positioning_events", "source_time") else "NULL"
        events = query(self.db_path, f"SELECT cycle_id, step_index, kind, {source}, created_at "
                                      "FROM positioning_events WHERE cycle_id=? ORDER BY id", (cycle_id,))
        text, summaries = positioning_summary(events, {row[0]: row[1] for row in steps})
        self.positioning_summary.setText(text)
        set_state(self.positioning_summary, "warning" if any("未記錄恢復" in row[3] for row in summaries) else "neutral")
        _fill(self.positioning, summaries)
        self.positioning.setVisible(bool(summaries))
        # Show the most useful available evidence immediately, then let either
        # table select its own event without retaining the other table's selection.
        evidence = self.alarms if cycle and cycle[6] == "NG" and alarms else self.steps
        if evidence.rowCount():
            evidence.selectRow(0)
        else:
            self._clear_preview("本輪沒有可預覽的事件")

    def _preview_from(self, table: QTableWidget):
        row = table.currentRow()
        if not table.selectedItems() or row < 0:
            return
        other = self.alarms if table is self.steps else self.steps
        other.blockSignals(True)
        other.clearSelection()
        other.blockSignals(False)
        item = table.item(row, 0)
        data = item.data(Qt.ItemDataRole.UserRole) if item else None
        self._clear_preview()
        if not data:
            return
        self.preview_caption.setText(data["caption"])
        self.preview_caption.show()
        snapshot = data.get("snapshot")
        if not snapshot:
            self.preview.show_message("此事件未保存截圖")
            return
        root = (self.records_dir / "snapshots").resolve()
        path = (root / snapshot).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            self.preview.show_message("找不到此事件的歷史截圖，紀錄仍可查閱")
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.preview.show_message("歷史截圖無法讀取，紀錄仍可查閱")
            return
        self.preview.show_image(pixmap)

    def _export_records(self, target: Path) -> int:
        if not self.db_path.exists() or has_column(self.db_path, "cycles", "operator"):
            return export_csv(self.db_path, target, **self.filters())
        # Version-1 databases remain read-only. Export the same columns, leaving
        # later traceability fields blank instead of migrating a historical DB.
        where, params = cycle_filter_sql(**self.filters(), extended=False)
        rows = query(self.db_path, f"""SELECT c.id, c.started_at, c.ended_at, c.duration_sec, c.result,
            c.sop_name, c.sop_version, c.model_version, '', '', '', '', s.step_index, s.step_name,
            s.status, s.duration_sec, s.manual, s.had_alarm, s.finished_at,
            (SELECT GROUP_CONCAT(a.message, '；') FROM alarms a
             WHERE a.cycle_id=c.id AND a.step_index=s.step_index)
            FROM cycles c LEFT JOIN step_records s ON s.cycle_id=c.id {where}
            ORDER BY c.id, s.id""", tuple(params))
        headers = ["輪次編號", "開始時間", "結束時間", "本輪耗時(秒)", "結果", "SOP", "SOP版本", "模型版本",
                   "工站", "作業員", "工單", "序號", "工序", "工序名稱", "工序狀態", "工序耗時(秒)", "手動",
                   "警報", "完成時間", "警報內容"]
        with target.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for row in rows:
                row = list(row)
                row[4] = RESULT_TEXT.get(row[4], row[4])
                writer.writerow(["" if value is None else value for value in row])
        return len(rows)

    def _export(self):
        default = Path.home() / f"SOP紀錄_{date.today():%Y%m%d}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "匯出作業紀錄", str(default), "CSV 檔案 (*.csv)")
        if not path:
            return
        try:
            count = self._export_records(Path(path))
        except Exception as exc:
            QMessageBox.warning(self, "匯出失敗", str(exc))
            return
        QMessageBox.information(self, "匯出完成", f"已匯出 {count} 列到\n{path}")
