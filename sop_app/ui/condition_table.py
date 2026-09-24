"""條件編輯表格：類型 / 物件 / 區域 / 數量 / 信心度 / 區域比例與即時結果。"""
from __future__ import annotations

import copy
from typing import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..conditions import ConditionResult
from ..sop_schema import CONDITION_TYPES, Condition

WHOLE_FRAME = "（整個畫面）"


class ConditionTable(QWidget):
    changed = Signal()

    def __init__(self, parent=None, rows_visible: int = 3):
        super().__init__(parent)
        self._classes: list[str] = []
        self._roi_names: list[str] = []
        self._conditions: list[Condition] = []

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["類型", "物件", "區域", "數量≥", "信心度≥", "物件在區域≥", "即時"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        for column, mode in enumerate([QHeaderView.ResizeMode.ResizeToContents, QHeaderView.ResizeMode.Stretch,
                                       QHeaderView.ResizeMode.Stretch, QHeaderView.ResizeMode.ResizeToContents,
                                       QHeaderView.ResizeMode.ResizeToContents,
                                       QHeaderView.ResizeMode.ResizeToContents,
                                       QHeaderView.ResizeMode.ResizeToContents]):
            header.setSectionResizeMode(column, mode)
        self.table.setMinimumHeight(34 + 34 * rows_visible)

        add_button = QPushButton("＋ 新增條件")
        remove_button = QPushButton("－ 刪除選取")
        add_button.clicked.connect(self._add)
        remove_button.clicked.connect(self._remove)
        buttons = QHBoxLayout()
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

    # ---- 外部介面 --------------------------------------------------------------
    def set_options(self, classes: Sequence[str], roi_names: Sequence[str]):
        self._classes, self._roi_names = list(classes), list(roi_names)
        self._rebuild()

    def set_conditions(self, conditions: Sequence[Condition]):
        self._conditions = [copy.copy(c) for c in conditions]
        self._rebuild()

    def conditions(self) -> list[Condition]:
        return [copy.copy(c) for c in self._conditions]

    def set_live_results(self, results: Sequence[ConditionResult] | None):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 6)
            if results is None or row >= len(results):
                item.setText("—")
                item.setBackground(QColor(0, 0, 0, 0))
                continue
            result = results[row]
            item.setText(f"{'✓' if result.met else '✗'} {result.count}")
            item.setBackground(QColor("#c8e6c9") if result.met else QColor("#ffcdd2"))
            item.setToolTip(result.error or result.detail or f"目前偵測到 {result.count} 個")

    # ---- 內部 -----------------------------------------------------------------
    def _rebuild(self):
        self.table.setRowCount(len(self._conditions))
        for row, cond in enumerate(self._conditions):
            type_box = QComboBox()
            for key, text in CONDITION_TYPES.items():
                type_box.addItem(text, key)
            type_box.setCurrentIndex(max(0, type_box.findData(cond.type)))
            type_box.currentIndexChanged.connect(lambda _i, r=row, w=type_box: self._set(r, "type", w.currentData()))

            label_box = QComboBox()
            for name in self._classes:
                label_box.addItem(name, name)
            if cond.label and cond.label not in self._classes:
                label_box.addItem(f"{cond.label}（模型無此類別）", cond.label)
            label_box.setCurrentIndex(max(0, label_box.findData(cond.label)))
            if not cond.label and self._classes:
                cond.label = self._classes[0]
            label_box.currentIndexChanged.connect(lambda _i, r=row, w=label_box: self._set(r, "label", w.currentData()))

            roi_box = QComboBox()
            roi_box.addItem(WHOLE_FRAME, "")
            for name in self._roi_names:
                roi_box.addItem(name, name)
            if cond.roi and cond.roi not in self._roi_names:
                roi_box.addItem(f"{cond.roi}（不存在）", cond.roi)
            roi_box.setCurrentIndex(max(0, roi_box.findData(cond.roi)))
            roi_box.currentIndexChanged.connect(lambda _i, r=row, w=roi_box: self._set(r, "roi", w.currentData()))

            count_box = QSpinBox()
            count_box.setRange(1, 99)
            count_box.setValue(cond.min_count)
            count_box.setEnabled(cond.type == "appear")
            count_box.valueChanged.connect(lambda v, r=row: self._set(r, "min_count", v))

            score_box = QDoubleSpinBox()
            score_box.setRange(0.05, 1.0)
            score_box.setSingleStep(0.05)
            score_box.setDecimals(2)
            score_box.setValue(cond.min_score)
            score_box.valueChanged.connect(lambda v, r=row: self._set(r, "min_score", round(v, 2)))

            overlap_box = QSpinBox()
            overlap_box.setRange(1, 100)
            overlap_box.setSuffix(" %")
            overlap_box.setValue(round(cond.roi_overlap * 100))
            overlap_box.setToolTip("辨識物件本身至少有多少比例必須位於指定區域內")
            overlap_box.valueChanged.connect(lambda v, r=row: self._set(r, "roi_overlap", v / 100))

            for column, widget in enumerate([type_box, label_box, roi_box, count_box, score_box, overlap_box]):
                widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                self.table.setCellWidget(row, column, widget)
            live = QTableWidgetItem("—")
            live.setFlags(Qt.ItemFlag.ItemIsEnabled)
            live.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 6, live)

    def _set(self, row: int, field: str, value):
        if row >= len(self._conditions):
            return
        setattr(self._conditions[row], field, value)
        if field == "type":
            self.table.cellWidget(row, 3).setEnabled(value == "appear")
        self.changed.emit()

    def _add(self):
        self._conditions.append(Condition(label=self._classes[0] if self._classes else ""))
        self._rebuild()
        self.table.selectRow(len(self._conditions) - 1)
        self.changed.emit()

    def _remove(self):
        if not self._conditions:
            return
        row = self.table.currentRow()
        del self._conditions[row if 0 <= row < len(self._conditions) else -1]
        self._rebuild()
        self.changed.emit()
