"""工序編輯頁：自由增減工序、設定完成／違規條件、在即時畫面上框選區域並即時測試。"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..conditions import evaluate_conditions
from ..paths import PROJECT_ROOT, relativize
from ..pipeline import FramePacket
from ..sop_schema import Condition, ROI, SOPDefinition, Step
from ..tracking import WorkpieceLocator, capture_workpiece, decode_reference, display_rois
from .condition_table import ConditionTable
from .quick_setup_dialog import QuickPositionDialog
from .video_widget import VideoWidget


class EditorPage(QWidget):
    modified = Signal()
    model_selected = Signal(str)
    save_requested = Signal()

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
        self._roi_undo = []
        self._roi_redo = []
        self._quick_phase = ""
        self._editor_mode = "overview"
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
        self.sop_box = QGroupBox("專案資料")
        self.sop_box.setLayout(sop_form)

        self.step_list = QListWidget()
        self.step_list.currentRowChanged.connect(self._on_step_selected)
        self.step_list.itemDoubleClicked.connect(lambda _item: self._set_editor_mode("steps"))
        self.step_list.setAlternatingRowColors(True)
        self.step_list.setMinimumHeight(260)
        self.step_list.setStyleSheet(
            "QListWidget{font-size:14px; border:1px solid #cfd8dc; background:white;}"
            "QListWidget::item{padding:9px 7px; border-bottom:1px solid #edf1f3;}"
            "QListWidget::item:selected{background:#1565c0; color:white; font-weight:bold;}"
        )
        step_buttons = QGridLayout()
        for index, (text, handler) in enumerate([("＋ 新增", self._add_step), ("複製", self._duplicate_step),
                                                 ("刪除", self._delete_step), ("▲ 上移", lambda: self._move_step(-1)),
                                                 ("▼ 下移", lambda: self._move_step(1))]):
            button = QPushButton(text)
            button.clicked.connect(handler)
            step_buttons.addWidget(button, index // 3, index % 3)
        steps_layout = QVBoxLayout()
        steps_layout.addWidget(self.step_list)
        self.step_actions = QWidget()
        self.step_actions.setLayout(step_buttons)
        steps_layout.addWidget(self.step_actions)
        self.edit_step_button = QPushButton("編輯這道工序")
        self.edit_step_button.clicked.connect(self._open_step_advanced)
        steps_layout.addWidget(self.edit_step_button)
        self.steps_box = QGroupBox("工序流程 · 由上而下執行")
        self.steps_box.setStyleSheet("QGroupBox{font-weight:bold;}")
        self.steps_box.setLayout(steps_layout)

        left = QWidget()
        self.left_panel = left
        left.setMinimumWidth(240)
        left.setMaximumWidth(330)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.steps_box, 1)
        left_layout.addWidget(self.sop_box)

        # 中：畫面與框選
        self.video = VideoWidget()
        self.video.roi_drawn.connect(self._on_roi_drawn)
        self.video.roi_selected.connect(self._select_roi)
        self.video.roi_edited.connect(self._edit_roi_points)
        self.video.roi_rename_requested.connect(self._rename_selected_roi)
        self.draw_button = QPushButton("▭ 框選偵測區域")
        self.draw_button.setCheckable(True)
        self.draw_button.toggled.connect(self.video.set_draw_mode)
        self.freeze_check = QCheckBox("凍結畫面")
        self.freeze_check.toggled.connect(self._on_freeze)
        self.target_button = QPushButton("① 設定位置跟隨")
        self.target_button.clicked.connect(self._begin_target)
        self.reference_button = QPushButton("② 編輯安裝位置")
        self.reference_button.setCheckable(True)
        self.reference_button.setEnabled(False)
        self.reference_button.toggled.connect(self._show_reference)
        self.live_button = QPushButton("③ 返回即時驗證")
        self.live_button.clicked.connect(self._return_live)
        self.help_button = QPushButton("操作說明")
        self.help_button.clicked.connect(self._show_tracking_help)
        self.quick_button = QPushButton("開始引導設定")
        self.quick_button.setMinimumHeight(42)
        self.quick_button.setStyleSheet("font-weight:bold; background:#1565c0; color:white; padding:9px; border-radius:4px")
        self.quick_button.clicked.connect(self._start_quick_setup)
        self.advanced_button = QPushButton("顯示進階設定")
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self._toggle_advanced)
        self.advanced_button.hide()
        self.quick_next_button = QPushButton("完成位置設定 → 即時驗證")
        self.quick_next_button.clicked.connect(self._finish_quick_positions)
        self.quick_save_button = QPushButton("✓ 完成並儲存 SOP")
        self.quick_save_button.setStyleSheet("font-weight:bold; background:#2e7d32; color:white; padding:7px")
        self.quick_save_button.clicked.connect(self._save_quick_setup)
        self.quick_restart_button = QPushButton("重新框選車把手")
        self.quick_restart_button.clicked.connect(lambda: self._clear_workpiece(restart=True))
        self.quick_exit_button = QPushButton("離開引導")
        self.quick_exit_button.clicked.connect(self._exit_quick_setup)
        self.quick_steps = []
        quick_steps = QHBoxLayout()
        for text in ("① 車把手", "② 安裝位置", "③ 即時驗證"):
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(30)
            self.quick_steps.append(label)
            quick_steps.addWidget(label)
        quick_actions = QHBoxLayout()
        quick_actions.addWidget(self.quick_next_button)
        quick_actions.addWidget(self.quick_save_button)
        quick_actions.addWidget(self.quick_restart_button)
        quick_actions.addWidget(self.quick_exit_button)
        self.quick_panel = QWidget()
        quick_layout = QVBoxLayout(self.quick_panel)
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_layout.addLayout(quick_steps)
        quick_layout.addLayout(quick_actions)
        self.quick_panel.hide()
        self.guide_label = QLabel("按「開始引導設定」，依序完成車把手、安裝位置與即時驗證。")
        self.guide_label.setWordWrap(True)
        self.guide_label.setStyleSheet("padding:10px; background:#eaf3fc; color:#163f67; border-radius:4px")
        self.setup_summary = QLabel()
        self.setup_summary.setTextFormat(Qt.TextFormat.RichText)
        self.setup_summary.setWordWrap(True)
        self.setup_summary.setStyleSheet(
            "padding:10px; background:#f7f9fb; color:#263238; border:1px solid #dbe3e8; border-radius:5px")
        self.tracking_label = QLabel("尚未設定位置跟隨")
        self.tracking_label.setWordWrap(True)
        self.live_label = QLabel("即時測試：等待影像")
        self.live_label.setTextFormat(Qt.TextFormat.RichText)
        video_bar = QHBoxLayout()
        video_bar.addWidget(self.draw_button)
        video_bar.addWidget(self.freeze_check)
        video_bar.addStretch()
        self.manual_tools = QWidget()
        manual_layout = QVBoxLayout(self.manual_tools)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.addLayout(video_bar)
        target_bar = QHBoxLayout()
        target_bar.addWidget(self.target_button)
        target_bar.addWidget(self.reference_button)
        self.clear_target_button = QPushButton("清除位置跟隨")
        self.clear_target_button.clicked.connect(self._clear_workpiece)
        target_bar.addWidget(self.clear_target_button)
        manual_layout.addLayout(target_bar)
        navigation = QHBoxLayout()
        navigation.addWidget(self.live_button)
        manual_layout.addLayout(navigation)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        title = QLabel("設定作業流程")
        title.setStyleSheet("font-size:18px; font-weight:bold; color:#1f2933")
        subtitle = QLabel("一般設定使用引導即可；只有需要調整條件、位置細節或循環規則時才展開進階設定。")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#607d8b")
        center_layout.addWidget(title)
        center_layout.addWidget(subtitle)
        center_layout.addWidget(self.setup_summary)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons = {}
        mode_row = QHBoxLayout()
        for key, text in (("overview", "總覽"), ("follow", "位置跟隨"),
                          ("positions", "安裝位置"), ("steps", "工序條件"), ("cycle", "循環設定")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setMinimumHeight(36)
            button.setStyleSheet(
                "QPushButton{padding:7px; border:1px solid #cfd8dc; background:#f5f7f8; color:#37474f;}"
                "QPushButton:checked{background:#1565c0; color:white; border-color:#1565c0; font-weight:bold;}")
            button.clicked.connect(lambda _checked=False, value=key: self._set_editor_mode(value))
            self.mode_group.addButton(button)
            self.mode_buttons[key] = button
            mode_row.addWidget(button)
        self.mode_buttons["overview"].setChecked(True)
        center_layout.addLayout(mode_row)

        self.mode_hint = QLabel()
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setStyleSheet("padding:8px; background:#f7f9fb; color:#455a64; border-radius:4px")
        center_layout.addWidget(self.mode_hint)
        center_layout.addWidget(self.video, 1)
        primary_actions = QHBoxLayout()
        primary_actions.addWidget(self.quick_button, 1)
        primary_actions.addWidget(self.help_button)
        center_layout.addLayout(primary_actions)
        center_layout.addWidget(self.quick_panel)
        center_layout.addWidget(self.guide_label)
        center_layout.addWidget(self.manual_tools)
        center_layout.addWidget(self.live_label)
        center_layout.addWidget(self.tracking_label)

        # 右：分頁設定
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_step_tab(), "進階｜工序")
        self.tabs.addTab(self._build_roi_tab(), "進階｜安裝位置")
        self.tabs.addTab(self._build_cycle_tab(), "進階｜循環")
        self.tabs.currentChanged.connect(lambda _i: self._refresh_video_rois())
        self.tabs.tabBar().hide()
        self.tabs.hide()
        self.manual_tools.hide()
        self.step_actions.hide()

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        for widget, stretch in ((left, 0), (center, 1), (self.tabs, 0)):
            self.splitter.addWidget(widget)
            self.splitter.setStretchFactor(self.splitter.count() - 1, stretch)
        self.splitter.setSizes([280, 1000, 0])
        layout = QHBoxLayout(self)
        layout.addWidget(self.splitter)
        self._set_editor_mode("overview")

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
        self.completion_mode = QComboBox()
        self.completion_mode.addItem("全部位置成立（順序不限，需同時滿足）", "all")
        self.completion_mode.addItem("任一位置成立即可", "any")
        self.completion_mode.currentIndexChanged.connect(self._on_step_form)
        form.addRow("完成方式", self.completion_mode)

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
        layout.addWidget(self._section("完成條件（依上方完成方式判定；工序仍由上而下依序）"))
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
        hint = QLabel("1. 按畫面下方「框選偵測區域」\n2. 在畫面上拖曳框出範圍並命名\n3. 雙擊清單項目可重新命名\n\n"
                      "條件指定區域後，只有落在區域內的物件才會被計算，可排除桌邊雜物造成的誤判。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#607d8b")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.roi_list)
        primary_actions = QHBoxLayout()
        for label, handler in (("＋ 新增位置", self._add_position),
                               ("編輯選取位置", self._begin_roi_edit),
                               ("刪除", self._delete_roi)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            primary_actions.addWidget(button)
        layout.addLayout(primary_actions)

        self.roi_more_button = QPushButton("更多位置工具…")
        self.roi_more_button.setCheckable(True)
        self.roi_more_button.toggled.connect(self._toggle_roi_more)
        layout.addWidget(self.roi_more_button)
        self.roi_extra_actions = QWidget()
        actions = QGridLayout(self.roi_extra_actions)
        actions.setContentsMargins(0, 0, 0, 0)
        for i, (label, handler) in enumerate([
                ("複製位置", self._duplicate_roi), ("重新命名", self._rename_selected_roi),
                ("復原修改", self._undo_roi), ("重做修改", self._redo_roi),
                ("套用到工序…", self._apply_roi), ("改為跟隨本體", self._anchor_selected_roi),
                ("完成編輯／返回即時", self._return_live)]):
            button = QPushButton(label)
            button.clicked.connect(handler)
            actions.addWidget(button, i // 2, i % 2)
        self.roi_extra_actions.hide()
        layout.addWidget(self.roi_extra_actions)
        self.roi_status = QLabel("選取位置後，可編輯、複製或套用到目前工序。")
        self.roi_status.setWordWrap(True)
        layout.addWidget(self.roi_status)
        hint.setText("可建立多個安裝位置。編輯時拖曳框內移動、拖動四角縮放，Esc 取消當次拖曳。\n"
                     "雙擊清單名稱可改名。跟隨工件的位置在參考影像編輯；固定位置在凍結畫面編輯。\n"
                     "依序安裝請為各位置建立獨立工序；順序不限請把多個位置加入同一工序。")
        layout.addWidget(hint)
        layout.addStretch()
        return widget

    def _toggle_roi_more(self, visible: bool):
        self.roi_extra_actions.setVisible(visible)
        self.roi_more_button.setText("收起更多工具" if visible else "更多位置工具…")

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

    def _toggle_advanced(self, visible: bool):
        """保留舊呼叫介面；新版以獨立任務頁取代整包進階面板。"""
        self._set_editor_mode("steps" if visible else "overview")

    def _set_editor_mode(self, mode: str):
        if mode not in self.mode_buttons:
            mode = "overview"
        self._editor_mode = mode
        self.mode_buttons[mode].setChecked(True)
        config = {
            "overview": (True, False, False, None,
                         "先查看設定完成度；需要修改時，選擇一項功能。"),
            "follow": (True, False, True, None,
                       "只設定工作區域如何跟隨車把手；完成後可直接切換到安裝位置。"),
            "positions": (True, True, True, 1,
                          "新增、移動、重新命名或刪除安裝位置；此頁不會修改工序條件。"),
            "steps": (True, True, False, 0,
                      "選擇一道工序後，只編輯該工序的物件、區域與判定門檻。"),
            "cycle": (True, True, False, 2,
                      "只設定一輪完成後何時開始下一輪。"),
        }[mode]
        show_left, show_tabs, show_manual, tab_index, hint = config
        self.left_panel.setVisible(show_left)
        # 工序流程是所有設定工作的核心，專案資料只在總覽出現，避免壓縮清單。
        self.steps_box.setVisible(True)
        self.sop_box.setVisible(mode == "overview")
        self.tabs.setVisible(show_tabs)
        self.manual_tools.setVisible(show_manual)
        self.step_actions.setVisible(mode == "steps")
        self.edit_step_button.setVisible(mode != "steps")
        if tab_index is not None:
            self.tabs.setCurrentIndex(tab_index)
        self.mode_hint.setText(hint)
        self.quick_button.setVisible(mode == "overview" and not self._quick_phase)
        self.help_button.setVisible(mode == "overview")
        self.draw_button.setVisible(mode == "positions")
        self.freeze_check.setVisible(mode == "positions")
        self.target_button.setVisible(mode == "follow")
        self.clear_target_button.setVisible(mode == "follow")
        self.reference_button.setVisible(mode == "positions")
        self.live_button.setVisible(mode in ("follow", "positions"))
        self.live_label.setVisible(mode in ("positions", "steps") or bool(self._quick_phase))
        self.tracking_label.setVisible(mode in ("follow", "positions") or bool(self._quick_phase))
        if show_tabs:
            self.splitter.setSizes([260, 720, 520])
        else:
            self.splitter.setSizes([280, 1200, 0])
        self._refresh_video_rois()
        self._update_live()

    def _open_step_advanced(self):
        self._set_editor_mode("steps")

    def _refresh_setup_summary(self):
        reference_ok = self.sop.workpiece is not None
        positions = len([roi for roi in self.sop.rois if roi.anchor == "workpiece"])
        configured = len([step for step in self.sop.steps if step.conditions])
        model_ok = bool(self.sop.model_path)
        mark = lambda ready: "<span style='color:#2e7d32'>✓</span>" if ready else "<span style='color:#90a4ae'>○</span>"
        self.setup_summary.setText(
            f"{mark(model_ok)} 辨識模型　　{mark(reference_ok)} 位置跟隨　　"
            f"{mark(positions > 0)} 安裝位置 <b>{positions}</b>　　"
            f"{mark(configured > 0)} 已設定工序 <b>{configured}/{len(self.sop.steps)}</b>")
        self.quick_button.setText("繼續設定安裝位置" if reference_ok else "開始引導設定")
        self.clear_target_button.setEnabled(reference_ok)

    # ---- 外部介面 --------------------------------------------------------------
    def set_sop(self, sop: SOPDefinition):
        self._roi_undo.clear()
        self._roi_redo.clear()
        self.roi_more_button.setChecked(False)
        self._set_quick_phase("")
        self._return_live()
        self._set_editor_mode("overview")
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
                self.tracking_label.setText("已設定位置跟隨；可標記安裝位置或觀察即時定位")
            except ValueError as exc:
                self.tracking_label.setText(str(exc))
        else:
            self.tracking_label.setText("尚未設定位置跟隨")
        self.reference_button.setEnabled(self._locator is not None)
        self.auto_restart_spin.setValue(sop.cycle.auto_restart_sec)
        self.reset_hold_spin.setValue(sop.cycle.reset_hold_sec)
        self._loading = False
        self._refresh_options()
        self._refresh_roi_list()
        self._refresh_step_list(0 if sop.steps else -1)
        self._refresh_setup_summary()

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
            clean_modes = {"overview", "follow", "positions", "cycle"}
            self.video.set_frame(packet.frame if self._editor_mode in clean_modes else packet.display)
            self._refresh_video_rois()
        self._update_live()

    def clear_video(self, text: str):
        self._return_live()
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
        self._roi_undo.clear()
        self._roi_redo.clear()
        count = len(self.sop.steps)
        self.steps_box.setTitle(f"工序流程 · {count} 道 · 由上而下執行")
        self.step_list.setToolTip("單擊選取工序；雙擊直接編輯工序條件")
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for index, step in enumerate(self.sop.steps, 1):
            self.step_list.addItem(QListWidgetItem(self._step_text(index, step)))
        self.step_list.blockSignals(False)
        select = min(select, len(self.sop.steps) - 1)
        self.step_list.setCurrentRow(select)
        self._on_step_selected(select)
        self._refresh_setup_summary()

    @staticmethod
    def _step_text(index: int, step: Step) -> str:
        return f"{index}. {step.name}" + ("   ⚠ 無條件" if not step.conditions else "")

    def _refresh_roi_list(self):
        selected = self.roi_list.currentRow()
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for roi in self.sop.rois:
            item = QListWidgetItem(roi.name)
            mode = "跟隨本體" if roi.anchor == "workpiece" else "固定畫面（不會跟隨）"
            item.setToolTip(mode)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.roi_list.addItem(item)
        self.roi_list.blockSignals(False)
        self.roi_list.setCurrentRow(min(selected, len(self.sop.rois) - 1))
        self._refresh_video_rois()
        self._refresh_setup_summary()

    def _refresh_video_rois(self):
        selected = None
        if self._editor_mode == "positions" and 0 <= self.roi_list.currentRow() < len(self.sop.rois):
            selected = self.sop.rois[self.roi_list.currentRow()].name
        step = self._current_step()
        used = {c.roi for c in (step.conditions + step.forbidden) if c.roi} if step else set()
        if self._editor_mode == "cycle":
            used = {c.roi for c in self.sop.cycle.reset_conditions if c.roi}
        if self.reference_button.isChecked():
            rois = [r for r in self.sop.rois if r.anchor == "workpiece"]
        else:
            matrix = self._preview_result.workpiece_transform if self._preview_result else None
            rois = display_rois(self.sop.rois, matrix)
        if self._editor_mode == "overview" and self._quick_phase != "verify":
            rois = []
        elif self._editor_mode in ("steps", "cycle"):
            rois = [roi for roi in rois if roi.name in used]
        elif self._editor_mode == "follow":
            rois = []
        self.video.set_rois(rois, used, selected)
        editable = [r.name for r in self.sop.rois
                    if (r.anchor == "workpiece" and self.reference_button.isChecked())
                    or (r.anchor == "fixed" and self.freeze_check.isChecked() and not self.reference_button.isChecked())]
        self.video.set_editable(editable if not self._capture_target else [])

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
        advanced = self._editor_mode in ("steps", "cycle")
        if advanced:
            _, reset_results = evaluate_conditions(self.sop.cycle.reset_conditions, self._preview_result, rois)
            self.reset_table.set_live_results(reset_results)
        if step is None:
            self.live_label.setText("即時測試：請選擇工序")
            return
        met, results = evaluate_conditions(step.conditions, self._preview_result, rois, step.completion_mode)
        results_by_roi: dict[str, list] = {}
        for condition, result in zip(step.conditions, results):
            if condition.roi:
                results_by_roi.setdefault(condition.roi, []).append(result)
        statuses = []
        for roi in self.sop.rois:
            checks = results_by_roi.get(roi.name, [])
            status = "未套用目前工序" if not checks else ("定位／辨識無法確認" if any(c.error for c in checks)
                     else "✓ 條件成立" if all(c.met for c in checks) else "尚未到位")
            mode = "跟隨本體" if roi.anchor == "workpiece" else "固定畫面"
            statuses.append(f"{roi.name}［{mode}］：{status}")
        self.roi_status.setText("\n".join(statuses) or "尚未新增安裝位置")
        forbidden_met = False
        if advanced:
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
            self._refresh_setup_summary()
            self.model_selected.emit(path)
            self._mark_modified()

    # ---- 工序 ------------------------------------------------------------------
    def _on_step_selected(self, row: int):
        step = self._current_step() if row >= 0 else None
        self.edit_step_button.setEnabled(step is not None)
        self.step_content.setEnabled(step is not None)
        self._loading = True
        if step is not None:
            self.step_name_edit.setText(step.name)
            self.instruction_edit.setPlainText(step.instruction)
            self.hold_spin.setValue(step.hold_sec)
            self.ratio_spin.setValue(round(step.ratio * 100))
            self.timeout_spin.setValue(step.timeout_sec)
            self.completion_mode.setCurrentIndex(max(0, self.completion_mode.findData(step.completion_mode)))
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
        step.completion_mode = self.completion_mode.currentData()
        self._roi_undo.clear()
        self._roi_redo.clear()
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
        self._roi_undo.clear()
        self._roi_redo.clear()
        self.sop.cycle.auto_restart_sec = self.auto_restart_spin.value()
        self.sop.cycle.reset_hold_sec = self.reset_hold_spin.value()
        self.sop.cycle.reset_conditions = self.reset_table.conditions()
        self._update_live()
        self._mark_modified()

    # ---- 車把手快速設定 --------------------------------------------------------
    def _set_quick_phase(self, phase: str):
        self._quick_phase = phase
        active = {"target": 0, "positions": 1, "verify": 2}.get(phase, -1)
        self.quick_panel.setVisible(active >= 0)
        self.quick_button.setVisible(active < 0 and self._editor_mode == "overview")
        self.quick_next_button.setVisible(phase == "positions")
        self.quick_save_button.setVisible(phase == "verify")
        self.quick_restart_button.setVisible(active >= 1 and self.sop.workpiece is not None)
        self.quick_exit_button.setVisible(active >= 0)
        for index, label in enumerate(self.quick_steps):
            if index < active:
                style = "background:#e8f5e9; color:#2e7d32; font-weight:bold; padding:5px; border-radius:4px"
            elif index == active:
                style = "background:#1565c0; color:white; font-weight:bold; padding:5px; border-radius:4px"
            else:
                style = "background:#eceff1; color:#78909c; padding:5px; border-radius:4px"
            label.setStyleSheet(style)

    def _start_quick_setup(self):
        if self._packet is None and self._reference_frame is None:
            QMessageBox.information(self, "等待影像", "請先連線攝影機，讓完整車把手清楚入鏡。")
            return
        if self._locator is None:
            self._set_quick_phase("target")
            self._set_editor_mode("follow")
            self._begin_target()
            self.guide_label.setText("快速設定 1/3｜拖曳框住完整車把手。請包含左右端與中央特徵，排除手部及背景。")
        else:
            self._set_quick_phase("positions")
            self._set_editor_mode("positions")
            self.reference_button.setChecked(True)
            self.tabs.setCurrentIndex(1)
            self.draw_button.setChecked(True)
            self.guide_label.setText("快速設定 2/3｜使用已保存的車把手。拖曳第一個安裝位置，放開後配對物件。")

    def _finish_quick_positions(self):
        positions = [r for r in self.sop.rois if r.anchor == "workpiece"]
        if not positions:
            self.guide_label.setText("請至少建立一個跟隨車把手的安裝位置，再進入驗證。")
            return
        self._return_live()
        self._set_quick_phase("verify")
        self._set_editor_mode("overview")
        self.guide_label.setText(
            f"快速設定 3/3｜已建立 {len(positions)} 個安裝位置。移動車把手，確認框線跟隨且即時條件正確，再儲存。")

    def _save_quick_setup(self):
        self.save_requested.emit()
        self._set_quick_phase("")
        self._set_editor_mode("overview")
        self.guide_label.setText("快速設定完成。日常使用可直接開始作業；定位狀態會持續提示。")

    def _exit_quick_setup(self):
        self._return_live()
        self._set_quick_phase("")
        self._set_editor_mode("overview")
        self.guide_label.setText("已離開快速引導；目前變更仍保留，可繼續使用進階設定或儲存 SOP。")

    # ---- 偵測區域 --------------------------------------------------------------
    def _selected_roi(self):
        row = self.roi_list.currentRow()
        return self.sop.rois[row] if 0 <= row < len(self.sop.rois) else None

    def _select_roi(self, name):
        self._set_editor_mode("positions")
        for i, roi in enumerate(self.sop.rois):
            if roi.name == name:
                self.roi_list.setCurrentRow(i)
                break

    def _roi_snapshot(self):
        return (copy.deepcopy(self.sop.rois), [(c, c.roi) for c in self._all_conditions()])

    def _remember_roi(self):
        self._roi_undo.append(self._roi_snapshot())
        self._roi_undo = self._roi_undo[-50:]
        self._roi_redo.clear()

    def _restore_roi(self, source, destination):
        if not source:
            self.guide_label.setText("沒有可復原／重做的區域修改。")
            return
        destination.append(self._roi_snapshot())
        self.sop.rois, refs = source.pop()
        for cond, name in refs:
            cond.roi = name
        self._refresh_roi_list()
        self._refresh_options()
        self._update_live()
        self._mark_modified()

    def _undo_roi(self):
        self._restore_roi(self._roi_undo, self._roi_redo)

    def _redo_roi(self):
        self._restore_roi(self._roi_redo, self._roi_undo)

    def _begin_roi_edit(self):
        roi = self._selected_roi()
        if roi is None:
            self.guide_label.setText("請先從位置清單選取要編輯的位置。")
            return False
        if roi.anchor == "workpiece":
            if self._reference_frame is None:
                self.guide_label.setText("請先設定有效的參考工件。")
                return False
            self.reference_button.setChecked(True)
        else:
            if self.reference_button.isChecked():
                self._return_live()
            if self._packet is None:
                self.guide_label.setText("請等待即時影像，再按編輯位置。")
                return False
            self.freeze_check.setChecked(True)
        self.draw_button.setChecked(False)
        self._set_editor_mode("positions")
        self._refresh_video_rois()
        self.guide_label.setText("編輯中｜畫面已暫停。拖曳框內移動、四角縮放；Esc 取消拖曳。完成後按返回即時。")
        return True

    def _add_position(self):
        if self._reference_frame is not None:
            self.reference_button.setChecked(True)
        elif self._packet is not None:
            self.freeze_check.setChecked(True)
        else:
            self.guide_label.setText("請先連線影像，再新增位置。")
            return
        self._refresh_video_rois()
        self.draw_button.setChecked(True)
        self.guide_label.setText("新增位置｜畫面已暫停。拖曳框選並命名，可重複按「新增位置」建立多處。")

    def _edit_roi_points(self, name, points):
        roi = next((r for r in self.sop.rois if r.name == name), None)
        if roi is None:
            return
        self._remember_roi()
        roi.points = [tuple(p) for p in points]
        self._refresh_video_rois()
        self._mark_modified()

    def _duplicate_roi(self):
        roi = self._selected_roi()
        if roi is None or not self._begin_roi_edit():
            return
        self._remember_roi()
        names = {r.name for r in self.sop.rois}
        name = next(f"{roi.name}（副本{i}）" for i in range(1, 10000) if f"{roi.name}（副本{i}）" not in names)
        self.sop.rois.append(ROI(name, list(roi.points), roi.anchor))
        self._refresh_roi_list()
        self._refresh_options()
        self.roi_list.setCurrentRow(len(self.sop.rois) - 1)
        self.guide_label.setText("已複製位置（目前與原框重疊）。請拖曳選取框到另一安裝位置，再重新命名。")
        self._mark_modified()

    def _rename_selected_roi(self, _name=None):
        if self._selected_roi() is not None:
            self.roi_list.editItem(self.roi_list.currentItem())

    def _apply_roi(self):
        roi, step = self._selected_roi(), self._current_step()
        if roi is None or step is None:
            self.guide_label.setText("請先選取安裝位置與要套用的工序。")
            return
        if not self.classes:
            self.guide_label.setText("請先載入模型，才能選擇要偵測的物件。")
            return
        label, ok = QInputDialog.getItem(self, "套用位置到目前工序", f"在「{roi.name}」偵測哪個物件？", self.classes, 0, False)
        if not ok:
            return
        if not any(c.roi == roi.name and c.label == label and c.type == "appear" for c in step.conditions):
            step.conditions.append(Condition(label=label, roi=roi.name))
            self._roi_undo.clear()
            self._roi_redo.clear()
            self._refresh_options()
            self._mark_modified()
        self._set_editor_mode("steps")
        self._update_live()
        self.guide_label.setText(f"已新增「{roi.name} 出現 {label}」完成條件；可在右側調整數量、信心度及完成方式。")

    def _anchor_selected_roi(self):
        roi = self._selected_roi()
        if roi is None:
            self.guide_label.setText("請先選取要跟隨本體的位置。")
            return
        if roi.anchor == "workpiece":
            self.guide_label.setText("此位置已設定為跟隨本體。")
            return
        matrix = self._preview_result.workpiece_transform if self._preview_result else None
        if self.reference_button.isChecked() or self.freeze_check.isChecked() or matrix is None:
            self.guide_label.setText("請先設定參考工件、返回即時畫面，待本體定位穩定後再轉換。")
            return
        homogeneous = np.vstack([matrix, [0,0,1]])
        try:
            inverse = np.linalg.inv(homogeneous)[:2]
        except np.linalg.LinAlgError:
            self.guide_label.setText("目前定位無效，請等待重新定位。")
            return
        points = np.asarray(roi.points) @ inverse[:,:2].T + inverse[:,2]
        if not np.isfinite(points).all():
            self.guide_label.setText("目前位置座標無效，請等待重新定位後再轉換。")
            return
        self._remember_roi()
        roi.anchor = "workpiece"
        roi.points = [tuple(p) for p in points]
        self._refresh_roi_list()
        self._update_live()
        self.guide_label.setText("已改為跟隨本體；移動本體時請確認位置框持續對準安裝位置。")
        self._mark_modified()

    def _on_freeze(self, frozen: bool):
        if not frozen:
            self.video.set_editable([])
            self._packet = None
            self._preview_result = None
            self._capture_target = False
            self.draw_button.setChecked(False)
            self.video.set_draw_mode(False)
            self.target_button.setText("① 設定位置跟隨")
            self.guide_label.setText("即時畫面：設定車把手 → 標記安裝位置 → 返回即時驗證")

    def _on_roi_drawn(self, points):
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        if self._capture_target:
            try:
                target = capture_workpiece(self._packet.frame, points)
                locator = WorkpieceLocator(target)
                reference = decode_reference(target)
            except (ValueError, AttributeError) as exc:
                QMessageBox.warning(self, "無法設定參考工件", str(exc))
                self._capture_target = True
                self.draw_button.setChecked(True)
                self.video.set_draw_mode(True)
                self.guide_label.setText("① 框選未成功：可直接重新拖曳，或按「返回即時測試」取消。")
                return
            self._capture_target = False
            self.sop.workpiece = target
            self._roi_undo.clear()
            self._roi_redo.clear()
            self._locator = locator
            self._reference_frame = reference
            self.reference_button.setEnabled(True)
            self.reference_button.setChecked(True)
            # 成功後直接進入下一次框選，避免停在靜態畫面卻沒有可拖曳的狀態。
            self.draw_button.setChecked(True)
            self.video.set_draw_mode(True)
            if not self._quick_phase:
                self._set_editor_mode("positions")
            if self._quick_phase == "target":
                self._set_quick_phase("positions")
                self._set_editor_mode("positions")
                self.guide_label.setText("快速設定 2/3｜車把手設定完成。拖曳安裝位置，放開後選擇到位物件。")
            self._mark_modified()
            return
        existing = {roi.name for roi in self.sop.rois}
        if self._quick_phase == "positions":
            default = next(f"安裝位置 {i}" for i in range(1, 1000) if f"安裝位置 {i}" not in existing)
            name, label, ok = QuickPositionDialog.get_values(self, default, self.classes)
        else:
            default = next(f"區域{i}" for i in range(1, 1000) if f"區域{i}" not in existing)
            name, ok = QInputDialog.getText(self, "新增偵測區域", "區域名稱：", text=default)
            label = ""
        name = name.strip()
        if not ok or not name:
            if self._quick_phase == "positions":
                self.draw_button.setChecked(True)
            return
        if name in existing:
            QMessageBox.warning(self, "名稱重複", f"已經有名為「{name}」的區域")
            if self._quick_phase == "positions":
                self.draw_button.setChecked(True)
            return
        anchor = "workpiece" if self.reference_button.isChecked() else "fixed"
        self._remember_roi()
        self.sop.rois.append(ROI(name, [tuple(p) for p in points], anchor))
        self._refresh_roi_list()
        self._refresh_options()
        self.tabs.setCurrentIndex(1)
        self.roi_list.setCurrentRow(len(self.sop.rois) - 1)
        if self._quick_phase == "positions" and label:
            self.sop.steps.append(Step(name=f"安裝{name}", instruction=f"將 {label} 安裝到「{name}」",
                                       conditions=[Condition(label=label, roi=name)]))
            self._refresh_step_list(len(self.sop.steps) - 1)
            self.tabs.setCurrentIndex(1)
        if anchor == "workpiece":
            if self._quick_phase == "positions":
                paired = f"，已建立 {label} 到位流程" if label else "，尚未配對物件"
                self.guide_label.setText(f"已新增「{name}」{paired}。可繼續拖曳下一位置，完成後按下方驗證按鈕。")
                self.draw_button.setChecked(True)
            else:
                self.guide_label.setText(f"已新增「{name}」。可拖曳調整、繼續新增位置，或按「套用到目前工序」。")
        self._mark_modified()

    def _clear_workpiece(self, restart: bool = False):
        """清除定位錨點與所有依附資料，避免留下無法使用的相對座標。"""
        if self.sop.workpiece is None:
            self.guide_label.setText("目前沒有位置跟隨設定。")
            return
        dependent_names = {roi.name for roi in self.sop.rois if roi.anchor == "workpiece"}
        affected = sum(c.roi in dependent_names for c in self._all_conditions())
        details = []
        if dependent_names:
            details.append(f"{len(dependent_names)} 個跟隨主工件的安裝位置")
        if affected:
            details.append(f"{affected} 個引用這些位置的條件")
        impact = "，並一併移除" + "與".join(details) if details else ""
        if QMessageBox.question(
                self, "清除位置跟隨設定",
                f"確定清除位置跟隨設定{impact}？\n\n固定畫面的區域與工序會保留。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return

        packet = self._packet
        self._return_live()
        self.sop.workpiece = None
        self.sop.rois = [roi for roi in self.sop.rois if roi.anchor != "workpiece"]
        for step in self.sop.steps:
            step.conditions = [c for c in step.conditions if c.roi not in dependent_names]
            step.forbidden = [c for c in step.forbidden if c.roi not in dependent_names]
        self.sop.cycle.reset_conditions = [
            c for c in self.sop.cycle.reset_conditions if c.roi not in dependent_names]
        self._locator = None
        self._reference_frame = None
        self._preview_result = None
        self.reference_button.setEnabled(False)
        self._set_quick_phase("")
        self._refresh_roi_list()
        self._refresh_options()
        self._refresh_step_list(self.step_list.currentRow())
        self.tracking_label.setText("尚未設定位置跟隨")
        self.guide_label.setText("主工件已清除。可重新開始引導設定。")
        self._mark_modified()

        if restart and packet is not None:
            self._packet = packet
            self._set_quick_phase("target")
            self._set_editor_mode("follow")
            self._begin_target()
            self.guide_label.setText("重新設定 1/3｜請拖曳框住完整車把手。")

    def _begin_target(self):
        if self._capture_target:
            self._return_live()
            return
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
        self.target_button.setText("取消設定車把手")
        self.draw_button.setChecked(True)
        self.video.set_draw_mode(True)
        self.guide_label.setText("① 畫面已暫停供框選，並非當機。請按住滑鼠左鍵框出工件；成功後會進入步驟②。")
        self.tracking_label.setText("畫面已凍結：框選待作業工件，盡量排除背景與手部")

    def _show_reference(self, enabled):
        self._capture_target = False
        self.target_button.setText("① 設定位置跟隨")
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        if enabled and self._reference_frame is not None:
            self.roi_status.setText("參考影像編輯中；返回即時後顯示各位置判定結果。")
            self.video.set_frame(self._reference_frame)
            self.freeze_check.setEnabled(False)
            self.draw_button.setText("＋ 框選車把手安裝位置")
            self.guide_label.setText("② 目前是車把手參考影像。請拖曳安裝位置；按「③ 返回即時驗證」恢復影像。")
            self.draw_button.setChecked(True)
            self.tracking_label.setText("參考影像：框選每道工序的作業區，再於工序條件的「區域」選取它")
        else:
            self.freeze_check.setEnabled(True)
            self.freeze_check.setChecked(False)
            self.draw_button.setText("▭ 框選偵測區域")
            self.guide_label.setText("③ 即時測試：移動工件，確認作業區是否跟隨；再查看工序條件的「即時」結果。")
            if self._packet:
                self.video.set_frame(self._packet.display)
            else:
                self.video.clear_frame("等待即時影像")
        self._refresh_video_rois()
        self._update_live()

    def _return_live(self):
        self.video.set_editable([])
        self.reference_button.setChecked(False)
        self.freeze_check.setEnabled(True)
        self.freeze_check.setChecked(False)
        self._capture_target = False
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        self.draw_button.setText("▭ 框選偵測區域")
        self.target_button.setText("① 設定位置跟隨")
        self.guide_label.setText("即時畫面：設定車把手 → 標記安裝位置 → 返回即時驗證")

    def _show_tracking_help(self):
        QMessageBox.information(self, "車把手快速設定",
            "一般設定請按「開始引導設定」：\n\n"
            "① 位置跟隨\n"
            "讓完整車把手清楚入鏡，框住左右端與中央特徵。畫面暫停是正常狀態。\n\n"
            "② 安裝位置\n"
            "在車把手參考影像拖曳安裝位置，輸入名稱並選擇到位物件。系統會自動建立流程條件，"
            "可連續加入左側、右側與中間位置。\n\n"
            "③ 即時驗證\n"
            "移動車把手，確認所有安裝位置同步跟隨，再按完成並儲存。\n\n"
            "跟隨基準只在內部用於計算，不會顯示成黃色框。安裝位置可畫在畫面任何位置，"
            "不必限制於先前框選的跟隨基準內。若要重設，按「重新框選車把手」。\n\n"
            "日常作業\n"
            "開啟 SOP 後可直接開始；定位狀態會持續提示。位置尚未確認時，跟隨位置的條件會暫停累積。\n\n"
            "需要自訂持續時間、違規條件或循環規則時，再使用右側的進階設定。")

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
        self._remember_roi()
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
        if users:
            affected = [f"{i+1}. {s.name}" for i, s in enumerate(self.sop.steps)
                        if any(c.roi == name for c in s.conditions + s.forbidden)]
            if any(c.roi == name for c in self.sop.cycle.reset_conditions):
                affected.append("循環重置條件")
            QMessageBox.information(self, "區域仍在使用", "請先修改以下條件的區域，再刪除：\n" + "\n".join(affected))
            return
        message = f"確定刪除區域「{name}」？"
        if users:
            message += f"\n有 {len(users)} 個條件使用此區域，刪除後會改為「整個畫面」。"
        if QMessageBox.question(self, "刪除偵測區域", message) != QMessageBox.StandardButton.Yes:
            return
        for cond in users:
            cond.roi = ""
        self._remember_roi()
        del self.sop.rois[row]
        self._refresh_roi_list()
        self._refresh_options()
        self._mark_modified()
