"""工序編輯頁：自由增減工序、設定完成／違規條件、在即時畫面上框選區域並即時測試。"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..conditions import evaluate_conditions
from ..paths import PROJECT_ROOT, relativize
from ..pipeline import FramePacket
from ..sop_schema import ROI, SOPDefinition, Step
from ..tracking import WorkpieceLocator, capture_workpiece, decode_reference, display_rois
from .condition_table import ConditionTable
from .video_widget import VideoWidget


class EditorPage(QWidget):
    modified = Signal()
    model_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sop = SOPDefinition()
        self.classes: list[str] = []
        self._packet: FramePacket | None = None
        self._loading = False
        self._locator = None
        self._preview_result = None
        self._capture_target = False
        self._reference_frame = None
        self._build()

    # ---- 版面 ------------------------------------------------------------------
    def _build(self):
        # 左：SOP 與工序清單
        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._on_sop_name)
        self.model_label = QLabel("（未選擇）")
        self.model_label.setWordWrap(True)
        model_button = QPushButton("選擇模型…")
        model_button.clicked.connect(self._choose_model)
        self.version_label = QLabel("1")
        sop_form = QFormLayout()
        sop_form.addRow("名稱", self.name_edit)
        sop_form.addRow("模型", self.model_label)
        sop_form.addRow("", model_button)
        sop_form.addRow("版本", self.version_label)
        sop_box = QGroupBox("SOP")
        sop_box.setLayout(sop_form)

        self.step_list = QListWidget()
        self.step_list.currentRowChanged.connect(self._on_step_selected)
        step_buttons = QGridLayout()
        for index, (text, handler) in enumerate([("＋ 新增", self._add_step), ("複製", self._duplicate_step),
                                                 ("刪除", self._delete_step), ("▲ 上移", lambda: self._move_step(-1)),
                                                 ("▼ 下移", lambda: self._move_step(1))]):
            button = QPushButton(text)
            button.clicked.connect(handler)
            step_buttons.addWidget(button, index // 3, index % 3)
        steps_layout = QVBoxLayout()
        steps_layout.addWidget(self.step_list)
        steps_layout.addLayout(step_buttons)
        steps_box = QGroupBox("工序清單（由上而下依序執行）")
        steps_box.setLayout(steps_layout)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(sop_box)
        left_layout.addWidget(steps_box, 1)

        # 中：畫面與框選
        self.video = VideoWidget()
        self.video.roi_drawn.connect(self._on_roi_drawn)
        self.draw_button = QPushButton("▭ 框選偵測區域")
        self.draw_button.setCheckable(True)
        self.draw_button.toggled.connect(self.video.set_draw_mode)
        self.freeze_check = QCheckBox("凍結畫面")
        self.freeze_check.toggled.connect(self._on_freeze)
        self.target_button = QPushButton("框選參考工件")
        self.target_button.clicked.connect(self._begin_target)
        self.reference_button = QPushButton("在參考影像標記作業區")
        self.reference_button.setCheckable(True)
        self.reference_button.setEnabled(False)
        self.reference_button.toggled.connect(self._show_reference)
        self.tracking_label = QLabel("尚未設定參考工件")
        self.tracking_label.setWordWrap(True)
        self.live_label = QLabel("即時測試：等待影像")
        self.live_label.setTextFormat(Qt.TextFormat.RichText)
        video_bar = QHBoxLayout()
        video_bar.addWidget(self.draw_button)
        video_bar.addWidget(self.freeze_check)
        video_bar.addStretch()
        video_bar.addWidget(self.live_label)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(self.video, 1)
        center_layout.addLayout(video_bar)
        target_bar = QHBoxLayout()
        target_bar.addWidget(self.target_button)
        target_bar.addWidget(self.reference_button)
        center_layout.addLayout(target_bar)
        center_layout.addWidget(self.tracking_label)

        # 右：分頁設定
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_step_tab(), "工序內容")
        self.tabs.addTab(self._build_roi_tab(), "偵測區域")
        self.tabs.addTab(self._build_cycle_tab(), "循環設定")
        self.tabs.currentChanged.connect(lambda _i: self._refresh_video_rois())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        for widget, stretch in ((left, 0), (center, 1), (self.tabs, 0)):
            splitter.addWidget(widget)
            splitter.setStretchFactor(splitter.count() - 1, stretch)
        splitter.setSizes([260, 700, 520])
        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

    def _build_step_tab(self) -> QWidget:
        self.step_name_edit = QLineEdit()
        self.step_name_edit.textEdited.connect(self._on_step_form)
        self.instruction_edit = QPlainTextEdit()
        self.instruction_edit.setPlaceholderText("顯示給作業員看的說明")
        self.instruction_edit.setMaximumHeight(70)
        self.instruction_edit.textChanged.connect(self._on_step_form)
        self.hold_spin = QDoubleSpinBox()
        self.hold_spin.setRange(0.0, 60.0)
        self.hold_spin.setSingleStep(0.1)
        self.hold_spin.setSuffix(" 秒")
        self.hold_spin.valueChanged.connect(self._on_step_form)
        self.ratio_spin = QSpinBox()
        self.ratio_spin.setRange(10, 100)
        self.ratio_spin.setSuffix(" %")
        self.ratio_spin.valueChanged.connect(self._on_step_form)
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0, 3600)
        self.timeout_spin.setSuffix(" 秒")
        self.timeout_spin.setSpecialValueText("不限時")
        self.timeout_spin.valueChanged.connect(self._on_step_form)

        form = QFormLayout()
        form.addRow("工序名稱", self.step_name_edit)
        form.addRow("作業說明", self.instruction_edit)
        form.addRow("條件需持續", self.hold_spin)
        form.addRow("持續期間成立比例", self.ratio_spin)
        form.addRow("超時警報", self.timeout_spin)

        self.condition_table = ConditionTable()
        self.condition_table.changed.connect(self._on_step_form)
        self.forbidden_table = ConditionTable(rows_visible=2)
        self.forbidden_table.changed.connect(self._on_step_form)

        hint = QLabel("「條件需持續」與「成立比例」用來過濾辨識閃爍：例如 1 秒內有 80% 的畫面成立才算完成。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#607d8b")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self._section("完成條件（全部成立 → 此工序完成）"))
        layout.addWidget(self.condition_table)
        layout.addWidget(self._section("違規條件（選填；全部成立 → 發出警報，例如偵測到跳步）"))
        layout.addWidget(self.forbidden_table)
        layout.addStretch()
        self.step_content = content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _build_roi_tab(self) -> QWidget:
        self.roi_list = QListWidget()
        self.roi_list.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                      | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.roi_list.itemChanged.connect(self._on_roi_renamed)
        self.roi_list.currentRowChanged.connect(lambda _r: self._refresh_video_rois())
        delete_button = QPushButton("刪除選取區域")
        delete_button.clicked.connect(self._delete_roi)
        hint = QLabel("1. 按畫面下方「框選偵測區域」\n2. 在畫面上拖曳框出範圍並命名\n3. 雙擊清單項目可重新命名\n\n"
                      "條件指定區域後，只有落在區域內的物件才會被計算，可排除桌邊雜物造成的誤判。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#607d8b")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.roi_list)
        layout.addWidget(delete_button)
        layout.addWidget(hint)
        layout.addStretch()
        return widget

    def _build_cycle_tab(self) -> QWidget:
        self.auto_restart_spin = QDoubleSpinBox()
        self.auto_restart_spin.setRange(0, 600)
        self.auto_restart_spin.setSuffix(" 秒")
        self.auto_restart_spin.valueChanged.connect(self._on_cycle_form)
        self.reset_hold_spin = QDoubleSpinBox()
        self.reset_hold_spin.setRange(0, 60)
        self.reset_hold_spin.setSingleStep(0.5)
        self.reset_hold_spin.setSuffix(" 秒")
        self.reset_hold_spin.valueChanged.connect(self._on_cycle_form)
        self.reset_table = ConditionTable(rows_visible=2)
        self.reset_table.changed.connect(self._on_cycle_form)
        form = QFormLayout()
        form.addRow("無重置條件時，幾秒後開始下一輪", self.auto_restart_spin)
        form.addRow("重置條件需持續", self.reset_hold_spin)
        hint = QLabel("最後一道工序完成後，若有設定重置條件（例如工件「消失」），需等條件成立才開始下一輪，"
                      "避免上一件工件還在畫面上就被誤判為下一輪的第一道工序完成。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#607d8b")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addLayout(form)
        layout.addWidget(self._section("重置條件（選填，全部成立 → 開始下一輪）"))
        layout.addWidget(self.reset_table)
        layout.addWidget(hint)
        layout.addStretch()
        return widget

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight:bold; margin-top:8px")
        return label

    # ---- 外部介面 --------------------------------------------------------------
    def set_sop(self, sop: SOPDefinition):
        self.sop = sop
        self._loading = True
        self.name_edit.setText(sop.name)
        self.version_label.setText(str(sop.version))
        self._set_model_label(sop.model_path)
        self._capture_target = False
        self.reference_button.blockSignals(True)
        self.reference_button.setChecked(False)
        self.reference_button.blockSignals(False)
        self.freeze_check.setEnabled(True)
        self._locator = None
        self._reference_frame = None
        self._preview_result = None
        self._packet = None
        self.freeze_check.setChecked(False)
        self.draw_button.setChecked(False)
        self.draw_button.setText("▭ 框選偵測區域")
        if sop.workpiece:
            try:
                self._reference_frame = decode_reference(sop.workpiece)
                self._locator = WorkpieceLocator(sop.workpiece)
                self.tracking_label.setText("已設定參考工件；可標記作業區或觀察即時定位")
            except ValueError as exc:
                self.tracking_label.setText(str(exc))
        else:
            self.tracking_label.setText("尚未設定參考工件")
        self.reference_button.setEnabled(self._locator is not None)
        self.auto_restart_spin.setValue(sop.cycle.auto_restart_sec)
        self.reset_hold_spin.setValue(sop.cycle.reset_hold_sec)
        self._loading = False
        self._refresh_options()
        self._refresh_roi_list()
        self._refresh_step_list(0 if sop.steps else -1)

    def set_classes(self, classes):
        self.classes = list(classes)
        self._refresh_options()

    def on_packet(self, packet: FramePacket):
        if self.reference_button.isChecked():
            return
        if not self.freeze_check.isChecked() or self._packet is None:
            self._packet = packet
            self._preview_result = replace(packet.result, workpiece_transform=None)
            if self._locator:
                matrix, status = self._locator.locate(packet.frame)
                self._preview_result = replace(packet.result, workpiece_transform=matrix, tracking_status=status)
                self.tracking_label.setText(status)
            self.video.set_frame(packet.display)
            self._refresh_video_rois()
        self._update_live()

    def clear_video(self, text: str):
        self._packet = None
        self._preview_result = None
        if self._locator:
            self._locator.reset()
        self.video.clear_frame(text)
        self._refresh_video_rois()

    # ---- 刷新 ------------------------------------------------------------------
    def _current_step(self) -> Step | None:
        row = self.step_list.currentRow()
        return self.sop.steps[row] if 0 <= row < len(self.sop.steps) else None

    def _refresh_options(self):
        names = [roi.name for roi in self.sop.rois]
        self._loading = True
        for table in (self.condition_table, self.forbidden_table, self.reset_table):
            table.set_options(self.classes, names)
        self.reset_table.set_conditions(self.sop.cycle.reset_conditions)
        step = self._current_step()
        if step is not None:
            self.condition_table.set_conditions(step.conditions)
            self.forbidden_table.set_conditions(step.forbidden)
        self._loading = False

    def _refresh_step_list(self, select: int):
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for index, step in enumerate(self.sop.steps, 1):
            self.step_list.addItem(QListWidgetItem(self._step_text(index, step)))
        self.step_list.blockSignals(False)
        select = min(select, len(self.sop.steps) - 1)
        self.step_list.setCurrentRow(select)
        self._on_step_selected(select)

    @staticmethod
    def _step_text(index: int, step: Step) -> str:
        return f"{index}. {step.name}" + ("   ⚠ 無條件" if not step.conditions else "")

    def _refresh_roi_list(self):
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for roi in self.sop.rois:
            item = QListWidgetItem(roi.name)
            item.setToolTip("跟隨工件的作業區域" if roi.anchor == "workpiece" else "固定畫面區域")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.roi_list.addItem(item)
        self.roi_list.blockSignals(False)
        self._refresh_video_rois()

    def _refresh_video_rois(self):
        selected = None
        if self.tabs.currentIndex() == 1 and 0 <= self.roi_list.currentRow() < len(self.sop.rois):
            selected = self.sop.rois[self.roi_list.currentRow()].name
        step = self._current_step()
        used = {c.roi for c in (step.conditions + step.forbidden) if c.roi} if step else set()
        if self.reference_button.isChecked():
            rois = [r for r in self.sop.rois if r.anchor == "workpiece"]
            if self.sop.workpiece:
                rois = [ROI("參考工件", self.sop.workpiece.points)] + rois
        else:
            matrix = self._preview_result.workpiece_transform if self._preview_result else None
            rois = display_rois(self.sop.rois, matrix)
        self.video.set_rois(rois, used, selected)

    def _update_live(self):
        packet, step = self._packet, self._current_step()
        if packet is None or self._preview_result is None or self.reference_button.isChecked():
            self.condition_table.set_live_results(None)
            self.forbidden_table.set_live_results(None)
            self.reset_table.set_live_results(None)
            self.live_label.setText("參考影像：標記作業區域，返回即時畫面後測試" if self.reference_button.isChecked()
                                    else "即時測試：等待影像")
            return
        rois = self.sop.roi_map()
        _, reset_results = evaluate_conditions(self.sop.cycle.reset_conditions, self._preview_result, rois)
        self.reset_table.set_live_results(reset_results)
        if step is None:
            self.live_label.setText("即時測試：請選擇工序")
            return
        met, results = evaluate_conditions(step.conditions, self._preview_result, rois)
        forbidden_met, forbidden_results = evaluate_conditions(step.forbidden, self._preview_result, rois)
        self.condition_table.set_live_results(results)
        self.forbidden_table.set_live_results(forbidden_results)
        if not step.conditions:
            text = "<span style='color:#f57c00'>此工序無條件（需手動確認）</span>"
        elif met:
            text = "<span style='color:#2e7d32'><b>✓ 完成條件成立</b></span>"
        else:
            text = "<span style='color:#c62828'><b>✗ 完成條件未成立</b></span>"
        if forbidden_met:
            text += "　<span style='color:#c62828'><b>⚠ 違規條件成立</b></span>"
        self.live_label.setText(f"即時測試：{text}")

    def _mark_modified(self):
        if not self._loading:
            self.modified.emit()

    # ---- SOP 與模型 ------------------------------------------------------------
    def _set_model_label(self, path: str | None):
        """畫面顯示檔名，完整路徑放在提示中，避免 ZIP 路徑擠壓表單。"""
        if not path:
            self.model_label.setText("（未選擇）")
            self.model_label.setToolTip("")
            return
        self.model_label.setText(Path(path).name)
        self.model_label.setToolTip(str(path))

    def _on_sop_name(self, text: str):
        self.sop.name = text
        self._mark_modified()

    def _choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇模型", str(PROJECT_ROOT),
                                              "模型包 (*.zip);;模型設定 (model.json)")
        if path:
            self.sop.model_path = relativize(path)
            self._set_model_label(self.sop.model_path)
            self.model_selected.emit(path)
            self._mark_modified()

    # ---- 工序 ------------------------------------------------------------------
    def _on_step_selected(self, row: int):
        step = self._current_step() if row >= 0 else None
        self.step_content.setEnabled(step is not None)
        self._loading = True
        if step is not None:
            self.step_name_edit.setText(step.name)
            self.instruction_edit.setPlainText(step.instruction)
            self.hold_spin.setValue(step.hold_sec)
            self.ratio_spin.setValue(round(step.ratio * 100))
            self.timeout_spin.setValue(step.timeout_sec)
            self.condition_table.set_conditions(step.conditions)
            self.forbidden_table.set_conditions(step.forbidden)
        else:
            self.condition_table.set_conditions([])
            self.forbidden_table.set_conditions([])
        self._loading = False
        self._refresh_video_rois()
        self._update_live()

    def _on_step_form(self, *_args):
        step = self._current_step()
        if self._loading or step is None:
            return
        step.name = self.step_name_edit.text()
        step.instruction = self.instruction_edit.toPlainText()
        step.hold_sec = round(self.hold_spin.value(), 2)
        step.ratio = self.ratio_spin.value() / 100
        step.timeout_sec = self.timeout_spin.value()
        step.conditions = self.condition_table.conditions()
        step.forbidden = self.forbidden_table.conditions()
        row = self.step_list.currentRow()
        self.step_list.item(row).setText(self._step_text(row + 1, step))
        self._refresh_video_rois()
        self._update_live()
        self._mark_modified()

    def _add_step(self):
        row = self.step_list.currentRow()
        insert_at = row + 1 if row >= 0 else len(self.sop.steps)
        self.sop.steps.insert(insert_at, Step(name=f"工序 {len(self.sop.steps) + 1}"))
        self._refresh_step_list(insert_at)
        self.step_name_edit.setFocus()
        self.step_name_edit.selectAll()
        self._mark_modified()

    def _duplicate_step(self):
        step = self._current_step()
        if step is None:
            return
        clone = copy.deepcopy(step)
        clone.name += "（複製）"
        row = self.step_list.currentRow() + 1
        self.sop.steps.insert(row, clone)
        self._refresh_step_list(row)
        self._mark_modified()

    def _delete_step(self):
        step = self._current_step()
        if step is None:
            return
        if QMessageBox.question(self, "刪除工序", f"確定刪除「{step.name}」？") != QMessageBox.StandardButton.Yes:
            return
        row = self.step_list.currentRow()
        del self.sop.steps[row]
        self._refresh_step_list(max(0, row - 1) if self.sop.steps else -1)
        self._mark_modified()

    def _move_step(self, delta: int):
        row = self.step_list.currentRow()
        target = row + delta
        if row < 0 or not 0 <= target < len(self.sop.steps):
            return
        steps = self.sop.steps
        steps[row], steps[target] = steps[target], steps[row]
        self._refresh_step_list(target)
        self._mark_modified()

    # ---- 循環設定 --------------------------------------------------------------
    def _on_cycle_form(self, *_args):
        if self._loading:
            return
        self.sop.cycle.auto_restart_sec = self.auto_restart_spin.value()
        self.sop.cycle.reset_hold_sec = self.reset_hold_spin.value()
        self.sop.cycle.reset_conditions = self.reset_table.conditions()
        self._update_live()
        self._mark_modified()

    # ---- 偵測區域 --------------------------------------------------------------
    def _on_freeze(self, frozen: bool):
        if not frozen:
            self._packet = None
            self._preview_result = None
            self._capture_target = False

    def _on_roi_drawn(self, points):
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        if self._capture_target:
            self._capture_target = False
            try:
                target = capture_workpiece(self._packet.frame, points)
            except (ValueError, AttributeError) as exc:
                QMessageBox.warning(self, "無法設定參考工件", str(exc))
                return
            self.sop.workpiece = target
            self._locator = WorkpieceLocator(target)
            self._reference_frame = decode_reference(target)
            self.reference_button.setEnabled(True)
            self.reference_button.setChecked(True)
            self._mark_modified()
            return
        existing = {roi.name for roi in self.sop.rois}
        default = next(f"區域{i}" for i in range(1, 1000) if f"區域{i}" not in existing)
        name, ok = QInputDialog.getText(self, "新增偵測區域", "區域名稱：", text=default)
        name = name.strip()
        if not ok or not name:
            return
        if name in existing:
            QMessageBox.warning(self, "名稱重複", f"已經有名為「{name}」的區域")
            return
        anchor = "workpiece" if self.reference_button.isChecked() else "fixed"
        if anchor == "workpiece":
            bounds = np.asarray(self.sop.workpiece.points)
            if (np.asarray(points) < bounds.min(axis=0)).any() or (np.asarray(points) > bounds.max(axis=0)).any():
                QMessageBox.warning(self, "作業區域", "請在參考工件範圍內標記作業位置")
                return
        self.sop.rois.append(ROI(name, [tuple(p) for p in points], anchor))
        self._refresh_roi_list()
        self._refresh_options()
        self.tabs.setCurrentIndex(1)
        self.roi_list.setCurrentRow(len(self.sop.rois) - 1)
        self._mark_modified()

    def _begin_target(self):
        if any(r.anchor == "workpiece" for r in self.sop.rois):
            QMessageBox.information(self, "已有作業區域", "重新設定參考工件會改變座標基準。請先刪除跟隨工件的區域，再重新框選。")
            return
        if self._packet is None:
            QMessageBox.information(self, "等待影像", "請先連線影像來源並顯示即時畫面")
            return
        packet = self._packet
        self.reference_button.setChecked(False)
        self._packet = packet
        self.freeze_check.setChecked(True)
        self.video.set_frame(self._packet.frame)
        self._capture_target = True
        self.draw_button.setChecked(True)
        self.tracking_label.setText("畫面已凍結：框選待作業工件，盡量排除背景與手部")

    def _show_reference(self, enabled):
        self._capture_target = False
        self.draw_button.setChecked(False)
        if enabled and self._reference_frame is not None:
            self.video.set_frame(self._reference_frame)
            self.freeze_check.setEnabled(False)
            self.draw_button.setText("框選跟隨工件的作業區")
            self.tracking_label.setText("參考影像：框選每道工序的作業區，再於工序條件的「區域」選取它")
        else:
            self.freeze_check.setEnabled(True)
            self.freeze_check.setChecked(False)
            self.draw_button.setText("▭ 框選偵測區域")
            if self._packet:
                self.video.set_frame(self._packet.display)
            else:
                self.video.clear_frame("等待即時影像")
        self._refresh_video_rois()
        self._update_live()

    def _all_conditions(self):
        for step in self.sop.steps:
            yield from step.conditions
            yield from step.forbidden
        yield from self.sop.cycle.reset_conditions

    def _on_roi_renamed(self, item: QListWidgetItem):
        row = self.roi_list.row(item)
        old, new = self.sop.rois[row].name, item.text().strip()
        if new == old:
            return
        if not new or any(roi.name == new for roi in self.sop.rois):
            QMessageBox.warning(self, "無法重新命名", "名稱不可空白或重複")
            self._refresh_roi_list()
            return
        self.sop.rois[row].name = new
        for cond in self._all_conditions():
            if cond.roi == old:
                cond.roi = new
        self._refresh_roi_list()
        self._refresh_options()
        self._mark_modified()

    def _delete_roi(self):
        row = self.roi_list.currentRow()
        if not 0 <= row < len(self.sop.rois):
            return
        name = self.sop.rois[row].name
        users = [c for c in self._all_conditions() if c.roi == name]
        if users and self.sop.rois[row].anchor == "workpiece":
            QMessageBox.information(self, "區域仍在使用", "請先修改或移除引用此作業區域的工序／循環條件，再刪除區域。")
            return
        message = f"確定刪除區域「{name}」？"
        if users:
            message += f"\n有 {len(users)} 個條件使用此區域，刪除後會改為「整個畫面」。"
        if QMessageBox.question(self, "刪除偵測區域", message) != QMessageBox.StandardButton.Yes:
            return
        for cond in users:
            cond.roi = ""
        del self.sop.rois[row]
        self._refresh_roi_list()
        self._refresh_options()
        self._mark_modified()
