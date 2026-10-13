"""Condition cards with the existing Condition schema and bounded deletion undo."""
from __future__ import annotations

import copy
from typing import Sequence

from PySide6.QtCore import QEvent, Signal
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QLayout, QPushButton, QRadioButton,
                               QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from ..conditions import ConditionResult
from ..sop_schema import CONDITION_TYPES, Condition
from .condition_table import WHOLE_FRAME
from .workbench_widgets import CollapsibleSection


class ConditionCards(QWidget):
    changed = Signal()
    region_selected = Signal(str)

    def __init__(self, parent=None, rows_visible=3):
        super().__init__(parent)
        self._classes: list[str] = []
        self._roi_names: list[str] = []
        self._conditions: list[Condition] = []
        self._deleted: list[tuple[int, Condition]] = []
        self._selected: int | None = None
        self._fields: list[dict] = []
        self._cards: list[QFrame] = []
        self._advanced: list[bool] = []
        self._watched: dict[QWidget, int] = {}
        self._group = QButtonGroup(self)
        self._group.idClicked.connect(self.select_condition)
        self.cards_layout = QVBoxLayout()
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(10)
        self.add_button = QPushButton("新增條件")
        self.add_button.setProperty("actionIcon", "plus")
        self.remove_button = QPushButton("刪除選取")
        self.undo_delete_button = QPushButton("復原刪除")
        self.undo_delete_button.setToolTip("僅復原目前條件表的刪除；切換工序或重新載入條件後清除。")
        self.add_button.clicked.connect(self._add)
        self.remove_button.clicked.connect(self._remove)
        self.undo_delete_button.clicked.connect(self._undo_delete)
        actions = QHBoxLayout()
        for button in (self.add_button, self.remove_button, self.undo_delete_button):
            actions.addWidget(button)
        actions.addStretch()
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.cards_layout)
        layout.addLayout(actions)
        self._update_buttons()

    def set_options(self, classes: Sequence[str], roi_names: Sequence[str]):
        self._classes, self._roi_names = list(classes), list(roi_names)
        self._rebuild()

    def set_conditions(self, conditions: Sequence[Condition]):
        self._conditions = [copy.copy(c) for c in conditions]
        self._deleted.clear()
        self._selected = None
        self._advanced = []
        self._rebuild()

    def conditions(self) -> list[Condition]:
        return [copy.copy(c) for c in self._conditions]

    def field_widget(self, row: int, field: str):
        return self._fields[row][field]

    def select_condition(self, row: int | None):
        self._selected = row if row is not None and 0 <= row < len(self._conditions) else None
        for index, card in enumerate(self._cards):
            card.setProperty("selected", index == self._selected)
            card.style().unpolish(card)
            card.style().polish(card)
        self._group.setExclusive(False)
        for button in self._group.buttons():
            button.setChecked(self._group.id(button) == self._selected)
        self._group.setExclusive(True)
        self._update_buttons()
        if self._selected is not None:
            self.region_selected.emit(self._conditions[self._selected].roi)

    def eventFilter(self, watched, event):
        if watched in self._watched and event.type() in (QEvent.Type.FocusIn, QEvent.Type.MouseButtonPress):
            self.select_condition(self._watched[watched])
        return super().eventFilter(watched, event)

    def set_live_results(self, results: Sequence[ConditionResult] | None):
        for row, fields in enumerate(self._fields):
            label = fields["live"]
            if results is None or row >= len(results):
                label.setText("本幀：—")
                label.setToolTip("")
            else:
                result = results[row]
                label.setText("本幀：無法判斷" if result.error else f"本幀：{'成立' if result.met else '未成立'} {result.count}")
                label.setToolTip(result.error or result.detail)

    def _rebuild(self):
        for button in self._group.buttons():
            self._group.removeButton(button)
        while self.cards_layout.count():
            widget = self.cards_layout.takeAt(0).widget()
            if widget:
                widget.hide()
                widget.deleteLater()
        self._fields, self._cards, self._watched = [], [], {}
        self._advanced = (self._advanced + [False] * len(self._conditions))[:len(self._conditions)]
        for row, cond in enumerate(self._conditions):
            card = QFrame()
            card.setObjectName("conditionCard")
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            layout = QVBoxLayout(card)
            layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)
            header = QHBoxLayout()
            choose = QRadioButton(f"條件 {row + 1}")
            self._group.addButton(choose, row)
            live = QLabel("本幀：—")
            live.setObjectName("mutedLabel")
            remove = QPushButton("移除")
            remove.setObjectName("dangerButton")
            remove.clicked.connect(lambda _checked=False, r=row: self._remove_row(r))
            header.addWidget(choose)
            header.addStretch()
            header.addWidget(live)
            header.addWidget(remove)
            layout.addLayout(header)
            fields = {"live": live}
            choices = {"type": list(CONDITION_TYPES.items()), "label": [("", "（請選擇物件）")] + [(s, s) for s in self._classes],
                       "roi": [("", WHOLE_FRAME)] + [(s, s) for s in self._roi_names]}
            for field, options in choices.items():
                box = QComboBox()
                box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                box.setMinimumContentsLength(6)
                for value, text in options:
                    box.addItem(text, value)
                value = getattr(cond, field)
                if box.findData(value) < 0:
                    suffix = "模型無此類別" if field == "label" else "不存在" if field == "roi" else "無效"
                    box.addItem(f"{value}（{suffix}）", value)
                box.setCurrentIndex(box.findData(value))
                box.setToolTip(box.currentText())
                box.currentIndexChanged.connect(lambda _index, r=row, f=field, w=box: self._set(r, f, w.currentData()))
                fields[field] = box
            count = QSpinBox()
            count.setRange(1, 99)
            count.setValue(cond.min_count)
            count.setEnabled(cond.type == "appear")
            count.valueChanged.connect(lambda value, r=row: self._set(r, "min_count", value))
            fields["min_count"] = count
            grid = QGridLayout()
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(6)
            for column, (field, title) in enumerate((("label", "辨識物件"), ("type", "物件狀態"),
                                                    ("roi", "綁定區域"), ("min_count", "最低數量"))):
                # Direct grid rows avoid nested QFormLayout height-for-width
                # collapsing input controls after a hidden-page rebuild.
                label = QLabel(title)
                label.setObjectName("fieldLabel")
                grid.addWidget(label, (column // 2) * 2, column % 2)
                grid.addWidget(fields[field], (column // 2) * 2 + 1, column % 2)
            layout.addLayout(grid)
            advanced = QWidget()
            advanced_form = QFormLayout(advanced)
            advanced_form.setContentsMargins(0, 0, 0, 0)
            score = QDoubleSpinBox()
            score.setRange(0.05, 1)
            score.setSingleStep(.05)
            score.setDecimals(2)
            score.setValue(cond.min_score)
            score.valueChanged.connect(lambda value, r=row: self._set(r, "min_score", round(value, 2)))
            overlap = QSpinBox()
            overlap.setRange(1, 100)
            overlap.setSuffix(" %")
            overlap.setValue(round(cond.roi_overlap * 100))
            overlap.valueChanged.connect(lambda value, r=row: self._set(r, "roi_overlap", value / 100))
            advanced_form.addRow("最低信心度", score)
            advanced_form.addRow("最低區域比例", overlap)
            fields.update(min_score=score, roi_overlap=overlap)
            section = CollapsibleSection("進階條件參數", advanced, self._advanced[row])
            section.expanded_changed.connect(lambda value, r=row: self._advanced.__setitem__(r, value))
            layout.addWidget(section)
            fields["advanced"] = section
            self._cards.append(card)
            self._fields.append(fields)
            for field in ("type", "label", "roi", "min_count", "min_score", "roi_overlap"):
                fields[field].setAccessibleName(f"條件 {row + 1} {field}")
                fields[field].installEventFilter(self)
                self._watched[fields[field]] = row
            self.cards_layout.addWidget(card)
            # Cards are also rebuilt while the inspector's page is hidden. Explicit
            # visibility keeps nested form controls in the layout on the next show.
            for field in ("type", "label", "roi", "min_count"):
                fields[field].setVisible(True)
            layout.invalidate()
            card.setMinimumHeight(layout.minimumSize().height())
        self.cards_layout.invalidate()
        self.updateGeometry()
        self.select_condition(self._selected)

    def _set(self, row: int, field: str, value):
        if not 0 <= row < len(self._conditions):
            return
        setattr(self._conditions[row], field, value)
        widget = self._fields[row].get(field)
        if isinstance(widget, QComboBox):
            widget.setToolTip(widget.currentText())
        if field == "type":
            self._fields[row]["min_count"].setEnabled(value == "appear")
        self.select_condition(row)
        self.changed.emit()

    def _update_buttons(self):
        self.remove_button.setEnabled(self._selected is not None)
        self.undo_delete_button.setEnabled(bool(self._deleted))

    def _add(self):
        self._conditions.append(Condition(label=self._classes[0] if self._classes else ""))
        self._rebuild()
        self.select_condition(len(self._conditions) - 1)
        self.changed.emit()

    def _remove(self):
        if self._selected is not None:
            self._remove_row(self._selected)

    def _remove_row(self, row: int):
        if not 0 <= row < len(self._conditions):
            return
        self._deleted.append((row, copy.copy(self._conditions.pop(row))))
        if row < len(self._advanced):
            self._advanced.pop(row)
        self._selected = None
        self._rebuild()
        self.changed.emit()

    def _undo_delete(self):
        if not self._deleted:
            return
        row, condition = self._deleted.pop()
        row = min(row, len(self._conditions))
        self._conditions.insert(row, condition)
        self._advanced.insert(row, False)
        self._selected = row
        self._rebuild()
        self.changed.emit()
