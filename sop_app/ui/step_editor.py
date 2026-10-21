"""A reorderable name row with settings opened in a single-step sidebar."""
from __future__ import annotations

import copy

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout, QWidget)

from ..engine import violation_window
from ..sop_schema import Condition, Step
from .compact_layout import (AutoHeightTextEdit, CompactDoubleSpinBox, CompactSpinBox, FlowLayout,
                             set_state, text_label)
from .condition_rows import ConditionRows


def field_pair(title: str, field: QWidget) -> QWidget:
    """A visible label kept on the same line as its control when rows wrap."""
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    label = QLabel(title)
    label.setObjectName("fieldLabel")
    label.setBuddy(field)
    layout.addWidget(label)
    layout.addWidget(field)
    return widget


def flow_row(*widgets, spacing=6) -> QWidget:
    container = QWidget()
    layout = FlowLayout(container, spacing=spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return container


def group_label(text: str, role: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("groupLabel")
    label.setProperty("role", role)
    return label


def parameter_grid(pairs) -> QWidget:
    """Label/field pairs two per line, so every step lines its values up the same way."""
    widget = QWidget()
    grid = QGridLayout(widget)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(4)
    for index, (title, field) in enumerate(pairs):
        label = QLabel(title)
        label.setObjectName("fieldLabel")
        label.setBuddy(field)
        row, column = divmod(index, 2)
        grid.addWidget(label, row, column * 2)
        grid.addWidget(field, row, column * 2 + 1)
    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(3, 1)
    return widget


def section(title: QLabel, *widgets) -> QWidget:
    widget = QFrame()
    widget.setObjectName("panel")
    widget.setProperty("stepSettingsSection", True)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    layout.addWidget(title)
    for item in widgets:
        layout.addWidget(item)
    return widget


class StepDragHandle(QLabel):
    MIME = "application/x-sop-step"

    def __init__(self, owner):
        super().__init__("≡", owner)
        self.owner = owner
        self.start = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("拖曳調整工序順序；聚焦後按 Alt＋↑／↓ 也可移動")
        self.setAccessibleName("拖曳調整工序順序")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self.start is not None and event.buttons() & Qt.MouseButton.LeftButton
                and (event.position().toPoint() - self.start).manhattanLength() >= QApplication.startDragDistance()):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData(self.MIME, str(id(self.owner.step)).encode("ascii"))
            drag.setMimeData(mime)
            self.start = None
            drag.exec(Qt.DropAction.MoveAction)

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.AltModifier and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self.owner.move_requested.emit(-1 if event.key() == Qt.Key.Key_Up else 1)
            event.accept()
        else:
            super().keyPressEvent(event)


class StepEditor(QFrame):
    changed = Signal()
    edit_requested = Signal()
    region_selected = Signal(str)
    duplicate_requested = Signal()
    delete_requested = Signal()
    move_requested = Signal(int)
    trial_requested = Signal()
    expanded_changed = Signal(bool)
    reorder_requested = Signal(str, bool)

    def __init__(self, step: Step, classes, roi_names, parent=None):
        super().__init__(parent)
        self.setObjectName("stepBlock")
        self.setAcceptDrops(True)
        self._editing = False
        self._details_open = False
        self.step = step
        self._loading = True
        self._automatic_conditions = copy.deepcopy(step.conditions)
        self.number_label = QLabel("1")
        self.number_label.setObjectName("stepNumber")
        self.drag_handle = StepDragHandle(self)
        self.step_name_edit = QLineEdit(step.name)
        self.step_name_edit.setPlaceholderText("工序名稱")
        self.step_name_edit.setAccessibleName("工序名稱")
        self.instruction_edit = AutoHeightTextEdit(step.instruction)
        self.instruction_edit.setPlaceholderText("給作業員的說明（選填）")
        self.completion_method = QComboBox()
        self.completion_method.setAccessibleName("工序完成方式")
        self.completion_method.addItem("自動判定", "auto")
        self.completion_method.addItem("人工確認", "manual")
        self.completion_method.setCurrentIndex(0 if step.conditions else 1)
        self.completion_mode = self.mode_choice(step.completion_mode)
        self.forbidden_mode = self.mode_choice(step.forbidden_mode)
        self.hold_spin = self.seconds(step.hold_sec, 60)
        self.ratio_spin = self.percent(round(step.ratio * 100), 1)
        self.timeout_spin = self.seconds(step.timeout_sec, 3600, special="不限時")
        self.min_duration_spin = self.seconds(step.min_duration_sec, 3600, special="不檢查")
        self.forbidden_hold_spin = self.seconds(-0.1 if step.forbidden_hold_sec is None else step.forbidden_hold_sec,
                                                60, minimum=-0.1, special="沿用完成")
        self.forbidden_ratio_spin = self.percent(0 if step.forbidden_ratio is None else round(step.forbidden_ratio * 100), 0)
        self.forbidden_ratio_spin.setSpecialValueText("沿用完成")
        self.trial_notice_interval_spin = self.seconds(step.trial_notice_interval_sec,
                                                       max(3600, step.trial_notice_interval_sec), special="不限制")
        self.trial_notice_interval_spin.setToolTip("同一工序的後續違規提示寫入試跑紀錄的最小間隔。首次警報立即記錄；畫面警示與區域標紅始終即時更新，不受此間隔延遲。0 秒仍只在違規狀態改變時記錄。")
        self.forbidden_hold_spin.setToolTip("違規判定的滑動觀察秒數；這段時間內的成立影格比例達標，且最新一幀成立才警報。0 秒為當幀判定；沿用完成會使用上方的完成觀察時間。")
        self.forbidden_ratio_spin.setToolTip("觀察時間內，違規條件組合成立的影格數占總影格數的比例，用來容許短暫漏檢。這不是模型信心度或物件落在區域內的比例；沿用完成會使用上方的完成成立比例。")
        self.condition_table = ConditionRows("completion", "新增完成條件", "尚未設定完成條件")
        self.forbidden_table = ConditionRows("violation", "新增違規條件", "未設定違規條件")
        for table, conditions in ((self.condition_table, step.conditions), (self.forbidden_table, step.forbidden)):
            table.set_options(classes, roi_names)
            table.set_conditions(conditions)
            table.changed.connect(self._save)
            table.region_selected.connect(self.region_selected)
        self.summary_label = text_label("", "warningLabel")
        self.live_label = text_label("單幀預覽：等待影像", "livePreview")
        self.manual_label = text_label()
        self.expanded = False
        self.edit_button = QPushButton("編輯")
        self.edit_button.setAccessibleName("編輯此工序")
        self.edit_button.clicked.connect(self.edit_requested)
        # Keep the public alias for callers that previously used expand_button.
        self.expand_button = self.edit_button

        self.up_button, self.down_button = QPushButton("上移"), QPushButton("下移")
        self.duplicate_button, self.delete_button = QPushButton("複製"), QPushButton("刪除")
        self.delete_button.setObjectName("dangerButton")
        self.trial_button = QPushButton("試跑")
        for button, text in ((self.up_button, "上移工序"), (self.down_button, "下移工序"),
                             (self.duplicate_button, "複製工序"), (self.delete_button, "刪除工序"),
                             (self.trial_button, "試跑此工序")):
            button.setAccessibleName(text)
        self.up_button.clicked.connect(lambda: self.move_requested.emit(-1))
        self.down_button.clicked.connect(lambda: self.move_requested.emit(1))
        self.duplicate_button.clicked.connect(self.duplicate_requested)
        self.delete_button.clicked.connect(self.delete_requested)
        self.trial_button.clicked.connect(self.trial_requested)

        self.header_row = QWidget()
        self.header_row.setObjectName("stepHeader")
        self.header_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        name_line = QHBoxLayout()
        name_line.setContentsMargins(0, 0, 0, 0)
        name_line.setSpacing(6)
        name_line.addWidget(self.drag_handle)
        name_line.addWidget(self.number_label)
        name_line.addWidget(self.step_name_edit, 1)
        name_line.addWidget(self.edit_button)
        self.actions_row = QWidget()
        actions = QHBoxLayout(self.actions_row)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)
        for button in (self.up_button, self.down_button, self.duplicate_button, self.delete_button):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self.trial_button)
        header = QVBoxLayout(self.header_row)
        header.setContentsMargins(6, 5, 6, 5)
        header.setSpacing(4)
        header.addLayout(name_line)
        header.addWidget(self.actions_row)

        self.description_row = field_pair("說明", self.instruction_edit)
        self.manual_row = self.manual_label
        self.basic_row = section(group_label("基本資料", "basic"),
                                 field_pair("完成方式", self.completion_method),
                                 self.manual_row, self.description_row)
        self.completion_row = section(group_label("完成條件", "completion"),
                                      field_pair("條件組合", self.completion_mode), self.condition_table)
        self.timing_note = text_label("", "mutedLabel")
        self.timing_row = section(group_label("時間與限制", "timing"), parameter_grid((
            ("觀察時間", self.hold_spin), ("成立比例", self.ratio_spin),
            ("超時", self.timeout_spin), ("最短工時", self.min_duration_spin))), self.timing_note)
        self.forbidden_note = text_label("", "mutedLabel")
        self.forbidden_row = section(group_label("違規檢查（選填）", "violation"), parameter_grid((
            ("判定", self.forbidden_mode), ("觀察", self.forbidden_hold_spin),
            ("成立比例", self.forbidden_ratio_spin), ("訊息間隔", self.trial_notice_interval_spin))), self.forbidden_note, self.forbidden_table)
        self.status_row = QWidget()
        status_layout = QVBoxLayout(self.status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(1)
        status_layout.addWidget(self.summary_label)
        status_layout.addWidget(self.live_label)

        body = self.body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)
        for widget in (self.basic_row, self.completion_row, self.timing_row, self.forbidden_row, self.status_row):
            body_layout.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(self.header_row)
        layout.addWidget(body)
        self.step_name_edit.textEdited.connect(self._save)
        self.instruction_edit.textChanged.connect(self._save)
        for spin in (self.hold_spin, self.ratio_spin, self.timeout_spin, self.min_duration_spin,
                     self.forbidden_hold_spin, self.forbidden_ratio_spin, self.trial_notice_interval_spin):
            spin.valueChanged.connect(self._save)
        for box in (self.completion_mode, self.forbidden_mode):
            box.currentIndexChanged.connect(self._save)
        self.completion_method.currentIndexChanged.connect(self._on_completion_method)
        self._loading = False
        self.update_summary()
        self.set_details_open(False)
        self.set_editing(False)

    def set_editing(self, editing, individual=False):
        """Show list-management actions without expanding any settings."""
        self._editing = bool(editing)
        self.actions_row.setVisible(self._editing or self._details_open)
        self.edit_button.show()

    def set_details_open(self, opened):
        """The page calls this after moving this body to/from its sidebar."""
        self._details_open = self.expanded = bool(opened)
        self.body.setVisible(self._details_open)
        self.actions_row.setVisible(self._editing or self._details_open)
        self.edit_button.show()

    def dragEnterEvent(self, event):
        source = event.source()
        if (isinstance(source, StepDragHandle) and source.owner.parent() is self.parent()
                and source.owner is not self and event.mimeData().hasFormat(StepDragHandle.MIME)):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        source = event.source()
        if isinstance(source, StepDragHandle) and source.owner.parent() is self.parent() and source.owner is not self:
            token = bytes(event.mimeData().data(StepDragHandle.MIME)).decode("ascii")
            after = event.position().y() >= self.height() / 2
            event.acceptProposedAction()
            self.reorder_requested.emit(token, after)

    @staticmethod
    def mode_choice(value):
        widget = QComboBox()
        widget.addItem("全部成立", "all")
        widget.addItem("任一成立", "any")
        widget.setCurrentIndex(max(0, widget.findData(value)))
        return widget

    @staticmethod
    def seconds(value, maximum, minimum=0, special=""):
        widget = CompactDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(.1)
        widget.setSuffix(" 秒")
        if special:
            widget.setSpecialValueText(special)
        widget.setValue(value)
        return widget

    @staticmethod
    def percent(value, minimum):
        widget = CompactSpinBox()
        widget.setRange(minimum, 100)
        widget.setSuffix(" %")
        widget.setValue(value)
        return widget

    def _on_completion_method(self, *_args):
        if self._loading:
            return
        conditions = self.condition_table.conditions()
        if self.completion_method.currentData() == "manual":
            if conditions:
                self._automatic_conditions = copy.deepcopy(conditions)
            self.condition_table.set_conditions([])
        elif not conditions:
            # An incomplete automatic condition remains explicit and can be
            # validated; merely opening a manual step never creates one.
            restored = self._automatic_conditions or [Condition()]
            self.condition_table.set_conditions(restored)
        self._save()

    def _save(self, *_args):
        if self._loading:
            return
        step = self.step
        step.name = self.step_name_edit.text()
        step.instruction = self.instruction_edit.toPlainText()
        step.completion_mode = self.completion_mode.currentData()
        step.forbidden_mode = self.forbidden_mode.currentData()
        step.hold_sec = round(self.hold_spin.value(), 2)
        step.ratio = self.ratio_spin.value() / 100
        step.timeout_sec = self.timeout_spin.value()
        step.min_duration_sec = self.min_duration_spin.value()
        step.forbidden_hold_sec = None if self.forbidden_hold_spin.value() < 0 else round(self.forbidden_hold_spin.value(), 2)
        step.forbidden_ratio = None if self.forbidden_ratio_spin.value() == 0 else self.forbidden_ratio_spin.value() / 100
        step.trial_notice_interval_sec = round(self.trial_notice_interval_spin.value(), 2)
        step.conditions = self.condition_table.conditions()
        step.forbidden = self.forbidden_table.conditions()
        if step.conditions:
            self._automatic_conditions = copy.deepcopy(step.conditions)
        self.update_summary()
        self.changed.emit()

    def set_expanded(self, expanded, notify=False):
        """Legacy requests must not reveal settings inside the name list."""
        if not expanded:
            self.set_details_open(False)
        elif notify:
            self.edit_requested.emit()
        if notify:
            self.expanded_changed.emit(self.expanded)

    @staticmethod
    def _seconds(value):
        return f"{value:g} 秒"

    def rules_text(self):
        """Every rule as text, so collapsed steps can still be compared."""
        step = self.step
        mode = {"all": "全部成立", "any": "任一成立"}
        lines = [f"說明：{step.instruction.strip()}"] if step.instruction.strip() else []
        if step.conditions:
            parts = [f"完成（{mode.get(step.completion_mode, step.completion_mode)}）："
                     + "、".join(condition.describe() for condition in step.conditions),
                     f"觀察 {self._seconds(step.hold_sec)}、成立比例 {step.ratio:.0%}"]
        else:
            parts = ["完成：人工確認"]
        if step.timeout_sec > 0:
            parts.append(f"超時 {self._seconds(step.timeout_sec)}")
        if step.conditions and step.min_duration_sec > 0:
            parts.append(f"最短工時 {self._seconds(step.min_duration_sec)}")
        lines.append(" · ".join(parts))
        if step.forbidden:
            hold, ratio = violation_window(step)
            lines.append(f"違規（{mode.get(step.forbidden_mode, step.forbidden_mode)}）："
                         + "、".join(condition.describe() for condition in step.forbidden)
                         + f"，觀察 {self._seconds(hold)}、成立比例 {ratio:.0%}")
        else:
            lines.append("違規：未設定")
        return "\n".join(lines)

    def update_summary(self):
        manual = not self.step.conditions
        blocked = self.completion_method.blockSignals(True)
        self.completion_method.setCurrentIndex(1 if manual else 0)
        self.completion_method.blockSignals(blocked)
        self.manual_label.setText("人工確認：執行時由作業員按「人工確認完成」；仍可設定超時與違規檢查。" if manual else
                                  "自動判定：完成條件成立後，仍需達到觀察時間與成立比例。")
        set_state(self.manual_label, "warning" if manual else "neutral")
        self.completion_mode.setEnabled(not manual)
        # Removing the last automatic condition must leave its undo available.
        # The explicit manual mode only disables adding new completion rules.
        self.condition_table.setEnabled(True)
        self.condition_table.add_button.setEnabled(not manual)
        self.condition_table.empty_label.setText("人工確認不需設定完成條件；如需辨識物件，請先選擇「自動判定」。" if manual else
                                                 "尚未設定完成條件")
        inherited_hold = bool(self.step.forbidden) and self.step.forbidden_hold_sec is None
        inherited_ratio = bool(self.step.forbidden) and self.step.forbidden_ratio is None
        self.hold_spin.setEnabled(not manual or inherited_hold)
        self.ratio_spin.setEnabled(not manual or inherited_ratio)
        self.min_duration_spin.setEnabled(not manual)
        self.timing_note.setText(
            "人工確認不使用完成觀察時間、成立比例或最短工時。違規設為「沿用完成」時，仍會使用上方對應數值。" if manual else
            "觀察時間與成立比例用來確認完成條件持續成立；超時 0 秒為不限時，最短工時 0 秒為不檢查。")
        for widget in (self.forbidden_mode, self.forbidden_hold_spin, self.forbidden_ratio_spin,
                       self.trial_notice_interval_spin):
            widget.setEnabled(bool(self.step.forbidden))
        hold, ratio = violation_window(self.step)
        self.forbidden_note.setText(
            f"目前違規觀察 {self._seconds(hold)}、成立比例 {ratio:.0%}。"
            f"最近觀察窗內，至少 {ratio:.0%} 的判定影格符合違規組合，且最新一幀成立才警報。"
            "設為「沿用完成」時，使用「時間與限制」中的數值。"
            f"試跑訊息間隔 {self.step.trial_notice_interval_sec:g} 秒，只限制重複紀錄；畫面警示即時更新。"
            if self.step.forbidden else "選填。人工確認與自動判定工序皆可加入違規條件。")
        self._update_rules_tooltip()

    def _update_rules_tooltip(self):
        title = f"工序 {self.number_label.text()} {self.step.name}".rstrip()
        self.header_row.setToolTip(f"{title}\n{self.rules_text()}")

    def set_issues(self, issues: list[str]):
        self.summary_label.setText("待修正：" + "；".join(issues) if issues else "")
        self.summary_label.setVisible(bool(issues))

    def set_preview(self, text: str, state: str):
        self.live_label.setText(text)
        set_state(self.live_label, state)

    def reference_conditions(self):
        """Keep retained automatic rules and undo entries bound to ROI names."""
        yield from self._automatic_conditions
        yield from self.condition_table.reference_conditions()
        yield from self.forbidden_table.reference_conditions()

    def set_options(self, classes, roi_names):
        for table, conditions in ((self.condition_table, self.step.conditions), (self.forbidden_table, self.step.forbidden)):
            table.set_options(classes, roi_names)
            if table.conditions() != conditions:
                table.set_conditions(conditions)
        self.update_summary()

    def set_position(self, index, count):
        self.number_label.setText(str(index + 1))
        self._update_rules_tooltip()
        self.number_label.setAccessibleName(f"工序 {index + 1}")
        self.up_button.setEnabled(index > 0)
        self.down_button.setEnabled(index < count - 1)
