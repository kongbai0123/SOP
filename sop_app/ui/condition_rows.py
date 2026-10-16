"""Conditions as stacked blocks for a narrow settings column.

Each condition is one small aligned grid: object and state, region and count,
confidence and in-region ratio, then this frame's result and any reason.
"""
from __future__ import annotations

import copy
from typing import Sequence

from PySide6.QtCore import QEvent, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..conditions import ConditionResult
from ..sop_schema import CONDITION_TYPES, Condition
from .compact_layout import CompactDoubleSpinBox, CompactSpinBox, set_state, text_label

WHOLE_FRAME = "（整個畫面）"
FIELDS = (("label", "辨識物件"), ("type", "物件狀態"), ("roi", "判定區域"),
          ("min_count", "最低數量"), ("min_score", "最低信心度"), ("roi_overlap", "物件在區域內至少"))
ROLE_MET_STATE = {"completion": "ok", "reset": "ok", "violation": "ng"}
ROLE_TITLE = {"completion": "完成條件", "violation": "違規條件", "reset": "重置條件"}


def _field_label(text):
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return label


class ConditionRows(QWidget):
    changed = Signal()
    region_selected = Signal(str)

    def __init__(self, role="completion", add_text="新增條件", empty_text="未設定條件", parent=None):
        super().__init__(parent)
        self.role = role
        self._classes: list[str] = []
        self._roi_names: list[str] = []
        self._conditions: list[Condition] = []
        self._deleted: list[tuple[int, Condition]] = []
        self._selected: int | None = None
        self._fields: list[dict] = []
        self._rows: list[QWidget] = []
        self._watched: dict[QWidget, int] = {}
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(4)
        self.empty_label = text_label(empty_text, "mutedLabel")
        self.add_button = QPushButton(add_text)
        self.undo_delete_button = QPushButton("復原刪除")
        self.add_button.clicked.connect(self._add)
        self.undo_delete_button.clicked.connect(self._undo_delete)
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)
        actions.addWidget(self.add_button)
        actions.addWidget(self.undo_delete_button)
        actions.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(self.rows_layout)
        layout.addWidget(self.empty_label)
        layout.addLayout(actions)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._update_buttons()

    # ---- data ---------------------------------------------------------------
    def set_options(self, classes: Sequence[str], roi_names: Sequence[str]):
        self._classes = list(dict.fromkeys(classes))
        self._roi_names = list(dict.fromkeys(roi_names))
        # Refresh choices in place: model/region changes must not discard a saved
        # selection, emit a spurious edit, or replace the field holding focus.
        for row in range(len(self._conditions)):
            self._populate_choices(row)
            self._update_state(row)

    def set_conditions(self, conditions: Sequence[Condition]):
        self._conditions = [copy.copy(c) for c in conditions]
        self._deleted.clear()
        self._selected = None
        self._rebuild()

    def conditions(self) -> list[Condition]:
        return [copy.copy(c) for c in self._conditions]

    def field_widget(self, row: int, field: str):
        return self._fields[row][field]

    def select_condition(self, row: int | None):
        self._selected = row if row is not None and 0 <= row < len(self._conditions) else None
        for index, widget in enumerate(self._rows):
            selected = index == self._selected
            if widget.property("selected") != selected:
                widget.setProperty("selected", selected)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
        if self._selected is not None:
            self.region_selected.emit(self._conditions[self._selected].roi)

    def set_live_results(self, results: Sequence[ConditionResult] | None):
        for row, fields in enumerate(self._fields):
            live, detail = fields["live"], fields["detail"]
            if results is None or row >= len(results):
                live.setText("本幀 —")
                set_state(live, "neutral")
                detail.setText("")
            else:
                result = results[row]
                live.setText("本幀無法判斷" if result.error else "本幀成立" if result.met else "本幀未成立")
                set_state(live, "warning" if result.error else ROLE_MET_STATE[self.role] if result.met else "neutral")
                # The reason is shown only when it explains a result that is not met.
                detail.setText("" if result.met and not result.error else result.error or result.detail)
            detail.setVisible(bool(detail.text()))

    # ---- events -------------------------------------------------------------
    def eventFilter(self, watched, event):
        if watched in self._watched and event.type() in (QEvent.Type.FocusIn, QEvent.Type.MouseButtonPress):
            self.select_condition(self._watched[watched])
        return super().eventFilter(watched, event)

    # ---- blocks -------------------------------------------------------------
    def _populate_choices(self, row):
        cond, fields = self._conditions[row], self._fields[row]
        choices = {"type": list(CONDITION_TYPES.items()),
                   "label": [("", "（請選擇物件）")] + [(s, s) for s in self._classes],
                   "roi": [("", WHOLE_FRAME)] + [(s, s) for s in self._roi_names]}
        for field, options in choices.items():
            box = fields[field]
            blocker = QSignalBlocker(box)
            box.clear()
            for value, text in options:
                box.addItem(text, value)
            value = getattr(cond, field)
            if box.findData(value) < 0:
                box.addItem(value, value)
            box.setCurrentIndex(box.findData(value))
            box.setToolTip(box.currentText())
            del blocker

    def _choice(self, row, field):
        box = QComboBox()
        # The readability guard keeps every choice wide enough for its full text;
        # the settings column widens instead of cutting a long name.
        box.setMinimumWidth(0)
        box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        box.currentIndexChanged.connect(lambda _index, r=row, f=field, w=box: self._set(r, f, w.currentData()))
        return box

    @staticmethod
    def _with_note(field, note):
        """An input and, when it does not apply, the reason shown in its place."""
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(field)
        layout.addWidget(note)
        return holder

    def _rebuild(self):
        while self.rows_layout.count():
            widget = self.rows_layout.takeAt(0).widget()
            if widget:
                widget.hide()
                widget.deleteLater()
        self._fields, self._rows, self._watched = [], [], {}
        for row, cond in enumerate(self._conditions):
            fields = {"type": self._choice(row, "type"), "label": self._choice(row, "label"),
                      "roi": self._choice(row, "roi")}
            count = CompactSpinBox()
            count.setRange(1, 99)
            count.setValue(cond.min_count)
            count.valueChanged.connect(lambda value, r=row: self._set(r, "min_count", value))
            score = CompactDoubleSpinBox()
            score.setRange(0, 1)
            score.setSingleStep(.05)
            score.setDecimals(2)
            score.setValue(cond.min_score)
            score.valueChanged.connect(lambda value, r=row: self._set(r, "min_score", round(value, 2)))
            overlap = CompactSpinBox()
            overlap.setRange(1, 100)
            overlap.setSuffix(" %")
            overlap.setValue(round(cond.roi_overlap * 100))
            overlap.valueChanged.connect(lambda value, r=row: self._set(r, "roi_overlap", value / 100))
            remove = QPushButton("移除")
            remove.setObjectName("dangerButton")
            remove.clicked.connect(lambda _checked=False, r=row: self._remove_row(r))
            fields.update(min_count=count, min_score=score, roi_overlap=overlap, remove=remove)
            for field, title in FIELDS:
                fields[field].setAccessibleName(f"{ROLE_TITLE[self.role]} {row + 1} {title}")
                for widget in (fields[field], *fields[field].findChildren(QLineEdit)):
                    widget.installEventFilter(self)
                    self._watched[widget] = row
            # A field that does not apply shows why in place of the input; the
            # stored value is kept for when the condition changes back.
            fields["min_count_note"] = text_label("需為 0", "mutedLabel", wrap=False)
            fields["roi_overlap_note"] = text_label("不適用", "mutedLabel", wrap=False)
            for note in (fields["min_count_note"], fields["roi_overlap_note"]):
                note.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                note.setMinimumHeight(count.sizeHint().height())
            fields["live"] = text_label("本幀 —", "livePreview", wrap=False)
            fields["detail"] = text_label("", "livePreview")
            fields["error"] = text_label("", "warningLabel")
            fields["detail"].hide()

            block = QFrame()
            block.setObjectName("conditionBlock")
            grid = QGridLayout(block)
            grid.setContentsMargins(6, 4, 6, 4)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(4)
            title = QLabel(f"{ROLE_TITLE[self.role]} {row + 1}")
            title.setObjectName("conditionTitle")
            head = QHBoxLayout()
            head.setContentsMargins(0, 0, 0, 0)
            head.addWidget(title)
            head.addWidget(fields["live"], 1)
            head.addWidget(remove)
            grid.addLayout(head, 0, 0, 1, 4)
            for line, (left, left_field, right, right_field) in enumerate((
                    ("物件", fields["label"], "狀態", fields["type"]),
                    ("區域", fields["roi"], "數量", self._with_note(count, fields["min_count_note"])),
                    ("信心", score, "區域內", self._with_note(overlap, fields["roi_overlap_note"]))), 1):
                grid.addWidget(_field_label(left), line, 0)
                grid.addWidget(left_field, line, 1)
                grid.addWidget(_field_label(right), line, 2)
                grid.addWidget(right_field, line, 3)
            grid.addWidget(fields["detail"], 4, 0, 1, 4)
            grid.addWidget(fields["error"], 5, 0, 1, 4)
            grid.setColumnStretch(1, 1)
            self._rows.append(block)
            self._fields.append(fields)
            self.rows_layout.addWidget(block)
            # Show at once so the surrounding page keeps its height and scroll position.
            block.setVisible(True)
            self._populate_choices(row)
            self._update_state(row)
        self._update_buttons()
        self.select_condition(self._selected)
        self.updateGeometry()

    def _update_state(self, row):
        cond, fields = self._conditions[row], self._fields[row]
        disappear, whole_frame = cond.type == "disappear", not cond.roi
        fields["min_count"].setVisible(not disappear)
        fields["min_count_note"].setVisible(disappear)
        fields["roi_overlap"].setVisible(not whole_frame)
        fields["roi_overlap_note"].setVisible(whole_frame)
        errors = []
        if cond.type not in CONDITION_TYPES:
            errors.append(f"未知的物件狀態「{cond.type}」")
        if not cond.label:
            errors.append("尚未選擇辨識物件")
        elif cond.label not in self._classes:
            errors.append(f"目前模型沒有「{cond.label}」類別，請確認模型或重新選擇物件")
        if cond.roi and cond.roi not in self._roi_names:
            errors.append(f"找不到區域「{cond.roi}」，請重新選擇或建立區域")
        if cond.min_count < 1:
            errors.append("最低數量需至少為 1")
        if not 0 <= cond.min_score <= 1:
            errors.append("最低信心度需介於 0～1")
        if not 0 < cond.roi_overlap <= 1:
            errors.append("物件在區域內比例需介於 1%～100%")
        fields["error"].setText("；".join(errors))
        fields["error"].setVisible(bool(errors))

    def _set(self, row: int, field: str, value):
        if not 0 <= row < len(self._conditions):
            return
        setattr(self._conditions[row], field, value)
        widget = self._fields[row].get(field)
        if isinstance(widget, QComboBox):
            widget.setToolTip(widget.currentText())
        self._update_state(row)
        self.select_condition(row)
        self.changed.emit()

    def _update_buttons(self):
        self.undo_delete_button.setEnabled(bool(self._deleted))
        self.empty_label.setVisible(not self._conditions and bool(self.empty_label.text()))

    def _add(self):
        self._conditions.append(Condition(label=self._classes[0] if self._classes else ""))
        self._rebuild()
        self.select_condition(len(self._conditions) - 1)
        self.changed.emit()

    def _remove_row(self, row: int):
        if not 0 <= row < len(self._conditions):
            return
        self._deleted.append((row, copy.copy(self._conditions.pop(row))))
        self._selected = None
        self._rebuild()
        self.changed.emit()

    def _undo_delete(self):
        if not self._deleted:
            return
        row, condition = self._deleted.pop()
        row = min(row, len(self._conditions))
        self._conditions.insert(row, condition)
        self._selected = row
        self._rebuild()
        self.changed.emit()
