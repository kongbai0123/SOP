"""工序編輯頁：自由增減工序、設定完成／違規條件、在即時畫面上框選區域並即時測試。"""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from html import escape
from pathlib import Path
import numpy as np

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
                               QSizePolicy, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..conditions import evaluate_conditions
from ..paths import PROJECT_ROOT, relativize, resolve
from ..pipeline import FramePacket
from ..sop_schema import CONDITION_TYPES, Condition, ROI, SOPDefinition, Step
from ..detection import FrameResult
from ..tracking import WorkpieceLocator, capture_workpiece, decode_reference, display_rois, workpiece_key
from .condition_cards import ConditionCards
from .editor_validation import EditorValidation
from .quick_setup_dialog import QuickPositionDialog
from .video_widget import VideoWidget
from .workbench_widgets import (ActionHint, CollapsibleSection, apply_action_icons,
                               install_readability_guard, workbench_styles)


class EditorPage(QWidget):
    modified = Signal()
    model_selected = Signal(str)
    save_requested = Signal()
    workpiece_changed = Signal(object)   # Workpiece | None；背景管線據此更新待機定位

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sop = SOPDefinition()
        self.classes: list[str] = []
        self._packet: FramePacket | None = None
        self._loading = False
        self._locator = None             # 只用來確認參考設定有效；逐幀定位在背景管線
        self._workpiece_key = ""
        self._preview_result = None
        self._capture_target = False
        self._reference_frame = None
        self._roi_undo = []
        self._roi_redo = []
        self._quick_phase = ""
        self._quick_save_pending = False
        self._model_state = "unselected"
        self._model_detail = ""
        self._editor_mode = "steps"
        self._refreshing_steps = False
        self._highlighted_condition_roi = None
        self._validation = EditorValidation()
        self._packet_arrived_at = None
        self._production_running = False
        self._theme_dark = None
        self._comfort_compact = None
        self._build()
        self._update_comfort_density()
        if QApplication.instance() is not None:
            QApplication.instance().paletteChanged.connect(self._apply_workbench_theme)
        self._apply_workbench_theme()
        install_readability_guard(self)

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
        self.step_list.setWordWrap(True)
        self.step_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.step_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.step_list.setMinimumHeight(180)
        step_buttons = QGridLayout()
        for index, (text, handler) in enumerate([("新增", self._add_step), ("複製", self._duplicate_step),
                                                 ("刪除", self._delete_step), ("上移", lambda: self._move_step(-1)),
                                                 ("下移", lambda: self._move_step(1))]):
            button = QPushButton(text)
            if index in (0, 3, 4):
                button.setProperty("actionIcon", {0: "plus", 3: "chevron-up", 4: "chevron-down"}[index])
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
        left.setMinimumWidth(190)
        left.setMaximumWidth(290)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.steps_box, 1)

        # 中：畫面與框選
        self.video = VideoWidget()
        self.video.roi_drawn.connect(self._on_roi_drawn)
        self.video.roi_selected.connect(self._select_roi)
        self.video.roi_edited.connect(self._edit_roi_points)
        self.video.roi_rename_requested.connect(self._rename_selected_roi)
        self.draw_button = QPushButton("框選偵測區域")
        self.draw_button.setProperty("actionIcon", "frame")
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
        self.live_button = QPushButton("返回即時")
        self.live_button.setToolTip("離開凍結／參考影像，回到即時預覽與驗證")
        self.live_button.clicked.connect(self._return_live)
        self.help_button = QPushButton("操作說明")
        self.help_button.clicked.connect(self._show_tracking_help)
        self.quick_button = QPushButton("開始引導設定")
        self.quick_button.clicked.connect(self._start_quick_setup)
        self.advanced_button = QPushButton("顯示進階設定")
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self._toggle_advanced)
        self.advanced_button.hide()
        self.quick_next_button = QPushButton("完成位置設定 → 即時驗證")
        self.quick_next_button.clicked.connect(self._finish_quick_positions)
        self.quick_save_button = QPushButton("完成並儲存 SOP")
        self.quick_save_button.setObjectName("primaryButton")
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
        quick_actions = QGridLayout()
        for index, button in enumerate((self.quick_next_button, self.quick_save_button,
                                        self.quick_restart_button, self.quick_exit_button)):
            quick_actions.addWidget(button, index // 2, index % 2)
        self.quick_panel = QWidget()
        quick_layout = QVBoxLayout(self.quick_panel)
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_layout.addLayout(quick_steps)
        quick_layout.addLayout(quick_actions)
        self.quick_panel.hide()
        self.guide_label = ActionHint("按「開始引導設定」，依序完成車把手、安裝位置與即時驗證。")
        self.guide_label.setWordWrap(True)
        self.guide_label.setObjectName("guideLabel")
        self.setup_summary = QLabel()
        self.setup_summary.setTextFormat(Qt.TextFormat.RichText)
        self.setup_summary.setWordWrap(True)
        self.setup_summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setup_summary.setObjectName("setupSummary")
        self.tracking_label = QLabel("尚未設定位置跟隨")
        self.tracking_label.setWordWrap(True)
        self.tracking_label.setObjectName("mutedLabel")
        self.live_label = QLabel("即時測試：等待影像")
        self.live_label.setTextFormat(Qt.TextFormat.RichText)
        self.live_label.setWordWrap(True)
        self.live_label.setObjectName("livePreview")
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
        center = QWidget()
        self.center_panel = center
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(5, 0, 5, 0)
        center_layout.setSpacing(8)
        self.context_title = QLabel("工序編輯")
        self.context_title.setTextFormat(Qt.TextFormat.PlainText)
        self.context_title.setWordWrap(True)
        self.context_title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.context_title.setObjectName("contextTitle")
        center_layout.addWidget(self.context_title)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons = {}
        mode_row = QHBoxLayout()
        for key, text in (("overview", "作業準備"), ("steps", "工序編輯"), ("cycle", "循環設定")):
            button = QPushButton(text)
            button.setObjectName("workbenchNav")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, value=key: self._set_editor_mode(value))
            self.mode_group.addButton(button)
            self.mode_buttons[key] = button
            mode_row.addWidget(button)
        self.mode_buttons["steps"].setChecked(True)
        mode_row.addStretch()
        mode_row.addWidget(self.quick_button)
        mode_row.addWidget(self.help_button)

        self.mode_hint = QLabel()
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.mode_hint.setObjectName("mutedLabel")
        center_layout.addWidget(self.video, 1)
        image_actions = QHBoxLayout()
        for key, text in (("follow", "主要工件"), ("positions", "區域設定")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, value=key: self._set_editor_mode(value))
            self.mode_buttons[key] = button
            image_actions.addWidget(button)
        image_actions.addWidget(self.live_button)
        image_actions.addStretch()
        center_layout.addLayout(image_actions)
        center_layout.addWidget(self.guide_label)
        manual_layout.addWidget(self.tracking_label)
        self.validation_box = self._build_validation_area()
        center_layout.addWidget(self.validation_box)

        # 右：分頁設定
        self.tabs = QTabWidget()
        self.tabs.setObjectName("workbenchInspector")
        self.tabs.addTab(self._build_step_tab(), "進階｜工序")
        self.tabs.addTab(self._build_roi_tab(), "進階｜安裝位置")
        self.tabs.addTab(self._build_cycle_tab(), "進階｜循環")
        self.tabs.addTab(self._build_prepare_tab(), "作業準備")
        self.tabs.currentChanged.connect(lambda _i: self._refresh_video_rois())
        self.tabs.tabBar().hide()
        self.tabs.setMinimumWidth(360)
        self.manual_tools.hide()

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(10)
        for widget, stretch in ((left, 0), (center, 1), (self.tabs, 0)):
            self.splitter.addWidget(widget)
            self.splitter.setStretchFactor(self.splitter.count() - 1, stretch)
        self.splitter.setSizes([230, 740, 410])
        self.setObjectName("workbenchEditor")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.workspace_title = QLabel("SOP · 設定作業流程")
        self.workspace_title.setTextFormat(Qt.TextFormat.PlainText)
        self.workspace_title.setWordWrap(True)
        self.workspace_title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.workspace_title.setObjectName("workspaceTitle")
        header.addWidget(self.workspace_title, 1)
        self.document_state = QLabel("草稿 · v1")
        self.document_state.setObjectName("documentState")
        self.document_state.setTextFormat(Qt.TextFormat.PlainText)
        header.addWidget(self.document_state)
        self.save_button = QPushButton("儲存草稿")
        self.save_button.setObjectName("primaryButton")
        self.save_button.clicked.connect(lambda _checked=False: self.save_requested.emit())
        header.addWidget(self.save_button)
        layout.addLayout(header)
        layout.addWidget(self.setup_summary)
        layout.addLayout(mode_row)
        layout.addWidget(self.mode_hint)
        self.mode_hint.hide()  # Navigation tooltips carry this repeated explanation.
        self.mode_hint.setFixedHeight(0)
        layout.addWidget(self.splitter, 1)
        self._set_editor_mode("steps")

    def _build_prepare_tab(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        self.prepare_layout = layout
        layout.addWidget(self.quick_panel)
        layout.addWidget(self.manual_tools)
        layout.addWidget(self.sop_box)
        hint = QLabel("影像來源仍由主視窗上方工具列連線。主要工件與區域由影像下方入口設定；不改變既有定位演算法。")
        hint.setWordWrap(True)
        hint.setObjectName("mutedLabel")
        layout.addWidget(hint)
        layout.addStretch()
        return self._scroll(content)

    @staticmethod
    def _scroll(content):
        content.setObjectName("inspectorContent")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _build_validation_area(self):
        box = QGroupBox("即時預覽與試跑")
        box.setObjectName("validationBox")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        layout.addWidget(self.live_label)
        self.validation_mode = QComboBox()
        self.validation_mode.addItem("單步驗證", "single")
        self.validation_mode.addItem("完整流程", "flow")
        self.validation_mode.currentIndexChanged.connect(self._validation_scope_changed)
        self.validation_start = QPushButton("開始")
        self.validation_start.setObjectName("primaryButton")
        self.validation_start.clicked.connect(self._start_validation)
        self.validation_stop = QPushButton("停止")
        self.validation_stop.clicked.connect(self._stop_validation)
        self.validation_restart = QPushButton("重跑")
        self.validation_restart.clicked.connect(self._start_validation)
        self.validation_confirm = QPushButton("人工確認")
        self.validation_confirm.clicked.connect(self._confirm_validation)
        tools = QGridLayout()
        for column, widget in enumerate((self.validation_mode, self.validation_start,
                                          self.validation_stop, self.validation_restart)):
            tools.addWidget(widget, 0, column, Qt.AlignmentFlag.AlignLeft)
        tools.setColumnStretch(4, 1)
        tools.addWidget(self.validation_confirm, 1, 2, 1, 2)
        layout.addLayout(tools)
        self.validation_status = QLabel("尚未開始試跑")
        self.validation_status.setWordWrap(True)
        self.validation_status.setTextFormat(Qt.TextFormat.PlainText)
        self.validation_status.setObjectName("validationStatus")
        self.validation_target = QLabel("尚未擷取測試設定")
        self.validation_target.setTextFormat(Qt.TextFormat.PlainText)
        self.validation_target.setWordWrap(True)
        self.validation_target.setObjectName("mutedLabel")
        layout.addWidget(self.validation_status)
        layout.addWidget(self.validation_target)
        metrics = QHBoxLayout()
        self.validation_count, self.validation_elapsed, self.validation_ratio = QLabel(), QLabel(), QLabel()
        for label in (self.validation_count, self.validation_elapsed, self.validation_ratio):
            label.setObjectName("mutedLabel")
            label.setWordWrap(True)
            metrics.addWidget(label, 1)
        layout.addLayout(metrics)
        self.validation_details = QPlainTextEdit()
        self.validation_details.setReadOnly(True)
        self.validation_details.setMinimumHeight(40)
        self.validation_details.setMaximumHeight(90)
        self.validation_details_section = CollapsibleSection("詳細結果與事件（試跑不寫正式紀錄）", self.validation_details)
        layout.addWidget(self.validation_details_section)
        self._validation_watchdog = QTimer(self)
        self._validation_watchdog.setInterval(500)
        self._validation_watchdog.timeout.connect(self._check_validation_feed)
        self._validation_watchdog.start()
        self._update_validation_view()
        return box

    def set_document_state(self, dirty: bool, saved: bool = False):
        text = "未儲存變更" if dirty else "已儲存" if saved else "草稿"
        self.document_state.setText(f"{text} · v{self.sop.version}")

    def _apply_workbench_theme(self, _palette=None):
        # Read the application palette, not the window color overridden by our QSS.
        palette = QApplication.instance().palette() if QApplication.instance() else self.palette()
        dark = palette.window().color().lightness() < 128
        if self._theme_dark != dark:
            self._theme_dark = dark
            self.setStyleSheet(workbench_styles(dark))
            apply_action_icons(self, dark)
            self._fit_step_actions()
            if hasattr(self, "workspace_title"):
                self._refresh_setup_summary()
                self._update_live()

    def _fit_step_actions(self):
        # An explicit 190px splitter minimum previously allowed the action
        # grid to shrink below its real size hint, cutting icon/text strokes.
        action_width = max(190, self.left_panel.minimumSizeHint().width())
        self.left_panel.setMaximumWidth(max(290, action_width))
        self.left_panel.setMinimumWidth(action_width)

    def _update_comfort_density(self):
        if not hasattr(self, "validation_details") or self.layout() is None:
            return
        compact = self.height() < 710
        if compact != self._comfort_compact:
            self._comfort_compact = compact
            self.center_panel.layout().setSpacing(4 if compact else 8)
            self.layout().setContentsMargins(14, 10 if compact else 12, 14, 10 if compact else 12)
            self.validation_details.setMinimumHeight(36 if compact else 48)
            self.validation_details.setMaximumHeight(72 if compact else 112)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_step_actions()
        self._update_comfort_density()

    def showEvent(self, event):
        super().showEvent(event)
        # Final font/style metrics become available when the page is polished.
        self._fit_step_actions()

    def _highlight_condition_region(self, name):
        self._highlighted_condition_roi = name or None
        self._refresh_video_rois()

    def _update_step_rule_summary(self):
        step = self._current_step()
        if step is None:
            self.step_rule_summary.setText("請新增或選取工序。")
        else:
            rules = "；".join(condition.describe() for condition in step.conditions) or "人工確認"
            extra = f" · 最短工時 {step.min_duration_sec:g} 秒" if step.min_duration_sec > 0 else ""
            self.step_rule_summary.setText(f"判定摘要：{rules}\n時間窗 {step.hold_sec:g} 秒 · "
                                           f"成立比例至少 {step.ratio:.0%}{extra}")
            if self._editor_mode == "steps":
                self.context_title.setText(f"工序 {self.step_list.currentRow()+1} · {step.name}")

    def _update_cycle_summary(self):
        uses_conditions = bool(self.sop.cycle.reset_conditions)
        self.auto_restart_spin.setEnabled(not uses_conditions)
        self.reset_hold_spin.setEnabled(uses_conditions)
        self.cycle_rule_summary.setText(
            f"有重置條件：以 {self.sop.cycle.reset_hold_sec:g} 秒觀察窗、{self.sop.cycle.reset_ratio:.0%} "
            f"影格成立比例判定；延遲秒數不生效。"
            if uses_conditions else f"無重置條件：本輪完成後延遲 {self.sop.cycle.auto_restart_sec:g} 秒開始下一輪。")
        self.reset_ratio_spin.setEnabled(uses_conditions)

    def _validation_blocker(self):
        if self._production_running:
            return "請先停止正式監控，再開始編輯器試跑。"
        if self._current_step() is None:
            return "請先選取工序。"
        if self.freeze_check.isChecked() or self.reference_button.isChecked():
            return "目前為凍結／參考影像；請返回即時畫面。"
        if self._packet is None or self._packet_arrived_at is None or time.monotonic() - self._packet_arrived_at > 3:
            return "等待即時影像；請先連線影像來源。"
        steps = self.sop.steps if self.validation_mode.currentData() == "flow" else [self._current_step()]
        needs_model = any(step.conditions or step.forbidden for step in steps)
        if self.validation_mode.currentData() == "flow":
            needs_model |= bool(self.sop.cycle.reset_conditions)
        if needs_model and (self._model_state != "available" or not self._preview_result.detection_valid):
            return "模型尚未就緒，無法開始自動條件試跑。"
        return ""

    def _start_validation(self, _checked=False):
        blocker = self._validation_blocker()
        if blocker:
            self._validation.message = blocker
            self._update_validation_view()
            return
        candidate = copy.deepcopy(self.sop)
        if self.validation_mode.currentData() == "single":
            candidate.steps = [candidate.steps[self.step_list.currentRow()]]
            candidate.cycle.reset_conditions = []  # Single-step trials do not exercise cycle reset.
        try:
            errors, _warnings = candidate.validate(self.classes or None)
            if errors:
                raise ValueError("；".join(errors))
            self._validation.start(self.sop, self.validation_mode.currentData(),
                                   self.step_list.currentRow(), self._preview_result.timestamp)
        except Exception as exc:
            self._validation.invalidate(f"無法開始試跑：{exc}")
            self._validation.message = f"無法開始試跑：{exc}"
        self._update_validation_view()

    def _stop_validation(self, _checked=False):
        self._validation.stop()
        self._update_validation_view()

    def _confirm_validation(self):
        self._validation.confirm()
        self._update_validation_view()

    def _invalidate_validation(self, reason="設定已變更，請重新試跑"):
        self._validation.invalidate(reason)
        if hasattr(self, "validation_status"):
            self._update_validation_view()

    def _validation_scope_changed(self, _index):
        self._invalidate_validation("驗證範圍已變更，請重新試跑")

    def set_production_running(self, running):
        self._production_running = running
        if running:
            self._invalidate_validation("正式監控已啟動；請停止監控後重新試跑")
        self._update_validation_view()

    def _check_validation_feed(self):
        if self._validation.running and (self._packet_arrived_at is None or time.monotonic() - self._packet_arrived_at > 3):
            self._invalidate_validation("影像停止更新，請重新試跑")
        self._update_validation_view()

    def _update_validation_view(self):
        if not hasattr(self, "validation_status"):
            return
        trial = self._validation
        blocker = self._validation_blocker()
        self.validation_status.setText(trial.message if trial.engine is not None or trial.message.startswith("無法") else blocker or trial.message)
        self.validation_status.setToolTip(blocker)
        state = "warning" if trial.stale or blocker else "running" if trial.running else "neutral"
        if self.validation_status.property("state") != state:
            self.validation_status.setProperty("state", state)
            self.validation_status.style().unpolish(self.validation_status)
            self.validation_status.style().polish(self.validation_status)
        metrics = trial.metrics()
        self.validation_target.setText(metrics["target"])
        self.validation_count.setText("條件 " + metrics["count"])
        self.validation_elapsed.setText("觀察 " + metrics["elapsed"])
        self.validation_ratio.setText("比例 " + metrics["ratio"].replace("≥", "至少"))
        self.validation_start.setEnabled(not trial.running and not blocker)
        self.validation_restart.setEnabled(not blocker)
        self.validation_stop.setEnabled(trial.running)
        manual = trial.running and trial.engine is not None and trial.engine.phase.value == "running" and not trial.engine.sop.steps[trial.engine.index].conditions
        self.validation_confirm.setVisible(manual)
        lines = ["僅使用目前設定快照與現有 SOPEngine；不寫正式作業紀錄。", trial.message]
        lines.extend(event.message for event in trial.events)
        detail = "\n".join(lines)
        if self.validation_details.toPlainText() != detail:
            scrollbar = self.validation_details.verticalScrollBar()
            position, at_end = scrollbar.value(), scrollbar.value() == scrollbar.maximum()
            self.validation_details.setPlainText(detail)
            scrollbar.setValue(scrollbar.maximum() if at_end else position)

    def hideEvent(self, event):
        if self._validation.running:
            self._validation.stop("已離開編輯器，試跑已停止")
        super().hideEvent(event)

    def closeEvent(self, event):
        self._validation.stop()
        self._validation_watchdog.stop()
        super().closeEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange) and hasattr(self, "_theme_dark"):
            self._apply_workbench_theme()

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
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0, 3600)
        self.min_duration_spin.setSingleStep(0.5)
        self.min_duration_spin.setSuffix(" 秒")
        self.min_duration_spin.setSpecialValueText("不檢查")
        self.min_duration_spin.setToolTip("自動完成的時間短於此值時發出「工時過短」警報並判定 NG")
        self.min_duration_spin.valueChanged.connect(self._on_step_form)
        self.forbidden_mode = QComboBox()
        self.forbidden_mode.addItem("全部違規條件成立", "all")
        self.forbidden_mode.addItem("任一違規條件成立", "any")
        self.forbidden_mode.currentIndexChanged.connect(self._on_step_form)
        self.forbidden_hold_spin = QDoubleSpinBox()
        self.forbidden_hold_spin.setRange(-0.1, 60.0)
        self.forbidden_hold_spin.setSingleStep(0.1)
        self.forbidden_hold_spin.setSuffix(" 秒")
        self.forbidden_hold_spin.setSpecialValueText("沿用完成條件")
        self.forbidden_hold_spin.valueChanged.connect(self._on_step_form)
        self.forbidden_ratio_spin = QSpinBox()
        self.forbidden_ratio_spin.setRange(0, 100)
        self.forbidden_ratio_spin.setSuffix(" %")
        self.forbidden_ratio_spin.setSpecialValueText("沿用完成條件")
        self.forbidden_ratio_spin.valueChanged.connect(self._on_step_form)

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow("工序名稱", self.step_name_edit)
        self.completion_mode = QComboBox()
        self.completion_mode.addItem("全部條件成立", "all")
        self.completion_mode.addItem("任一條件成立", "any")
        self.completion_mode.currentIndexChanged.connect(self._on_step_form)
        form.addRow("完成方式", self.completion_mode)
        description = QWidget()
        description_layout = QVBoxLayout(description)
        description_layout.setContentsMargins(0, 0, 0, 0)
        description_layout.addWidget(self.instruction_edit)
        self.description_section = CollapsibleSection("作業說明（選填）", description)
        form.addRow(self.description_section)

        self.condition_table = ConditionCards()
        self.condition_table.changed.connect(self._on_step_form)
        self.condition_table.region_selected.connect(self._highlight_condition_region)
        self.forbidden_table = ConditionCards(rows_visible=2)
        self.forbidden_table.changed.connect(self._on_step_form)
        self.forbidden_table.region_selected.connect(self._highlight_condition_region)

        hint = QLabel("「條件需持續」與「成立比例」用來過濾辨識閃爍：例如 1 秒內有 80% 的畫面成立才算完成。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#607d8b")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 0, 8, 8)
        layout.setSpacing(12)
        data_box = QGroupBox("工序資料")
        data_box.setLayout(form)
        layout.addWidget(data_box)
        conditions_box = QGroupBox("完成條件")
        conditions_layout = QVBoxLayout(conditions_box)
        conditions_layout.addWidget(self.condition_table)
        timing = QWidget()
        timing_form = QFormLayout(timing)
        timing_form.addRow("觀察時間", self.hold_spin)
        timing_form.addRow("成立比例", self.ratio_spin)
        timing_form.addRow("超時提醒", self.timeout_spin)
        timing_form.addRow("最短工時", self.min_duration_spin)
        timing_form.addRow(hint)
        self.step_advanced = CollapsibleSection("進階判定設定", timing)
        conditions_layout.addWidget(self.step_advanced)
        layout.addWidget(conditions_box)
        forbidden_content = QWidget()
        forbidden_layout = QVBoxLayout(forbidden_content)
        forbidden_layout.setContentsMargins(0, 0, 0, 0)
        forbidden_form = QFormLayout()
        forbidden_form.addRow("違規判定", self.forbidden_mode)
        forbidden_form.addRow("違規觀察時間", self.forbidden_hold_spin)
        forbidden_form.addRow("違規成立比例", self.forbidden_ratio_spin)
        forbidden_layout.addWidget(self.forbidden_table)
        forbidden_layout.addLayout(forbidden_form)
        self.forbidden_section = CollapsibleSection("違規條件（選填）", forbidden_content)
        layout.addWidget(self.forbidden_section)
        self.step_rule_summary = QLabel()
        self.step_rule_summary.setWordWrap(True)
        self.step_rule_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.step_rule_summary.setObjectName("mutedLabel")
        layout.addWidget(self.step_rule_summary)
        layout.addStretch()
        self.step_content = content
        return self._scroll(content)

    def _build_roi_tab(self) -> QWidget:
        self.roi_list = QListWidget()
        self.roi_list.setWordWrap(True)
        self.roi_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.roi_list.setResizeMode(QListWidget.ResizeMode.Adjust)
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
        self.roi_settings_layout = layout
        layout.addWidget(self.roi_list)
        primary_actions = QHBoxLayout()
        for label, handler in (("新增位置", self._add_position),
                               ("編輯選取位置", self._begin_roi_edit),
                               ("刪除", self._delete_roi)):
            button = QPushButton(label)
            if label == "新增位置":
                button.setProperty("actionIcon", "plus")
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
        return self._scroll(widget)

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
        self.reset_ratio_spin = QSpinBox()
        self.reset_ratio_spin.setRange(10, 100)
        self.reset_ratio_spin.setSuffix(" %")
        self.reset_ratio_spin.valueChanged.connect(self._on_cycle_form)
        self.sequence_check = QCheckBox("偵測跳步")
        self.sequence_check.setToolTip("只檢查含「出現」條件的後續工序。若後續工序的條件\n"
                                       "會被目前工序的完成狀態同時滿足，請勿啟用，以免誤報。")
        self.sequence_check.toggled.connect(self._on_cycle_form)
        self.reset_table = ConditionCards(rows_visible=2)
        self.reset_table.changed.connect(self._on_cycle_form)
        form = QFormLayout()
        form.addRow("無重置條件：延遲重啟", self.auto_restart_spin)
        form.addRow("重置條件觀察時間", self.reset_hold_spin)
        form.addRow("重置條件成立比例", self.reset_ratio_spin)
        form.addRow(self.sequence_check)
        sequence_hint = QLabel("後續工序的條件提前成立時，發出「順序錯誤」警報。")
        sequence_hint.setWordWrap(True)
        sequence_hint.setObjectName("mutedLabel")
        form.addRow(sequence_hint)
        hint = QLabel("最後一道工序完成後，若有設定重置條件（例如工件「消失」），需等條件成立才開始下一輪，"
                      "避免上一件工件還在畫面上就被誤判為下一輪的第一道工序完成。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#607d8b")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.cycle_rule_summary = QLabel("沿用原引擎：有重置條件才等條件，否則使用延遲秒數。")
        self.cycle_rule_summary.setWordWrap(True)
        self.cycle_rule_summary.setObjectName("mutedLabel")
        layout.addWidget(self.cycle_rule_summary)
        layout.addLayout(form)
        layout.addWidget(self._section("重置條件（選填，全部成立 → 開始下一輪）"))
        layout.addWidget(self.reset_table)
        layout.addWidget(hint)
        layout.addStretch()
        return self._scroll(widget)

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("font-weight:bold; margin-top:8px")
        return label

    def _toggle_advanced(self, visible: bool):
        """保留舊呼叫介面；新版以獨立任務頁取代整包進階面板。"""
        self._set_editor_mode("steps" if visible else "overview")

    def _set_editor_mode(self, mode: str):
        if mode not in self.mode_buttons:
            mode = "overview"
        self._editor_mode = mode
        primary = "overview" if mode in ("follow", "positions") else mode
        self.mode_buttons[primary].setChecked(True)
        for key in ("follow", "positions"):
            self.mode_buttons[key].setChecked(key == mode)
        config = {
            "overview": (True, True, False, 3,
                         "影像由上方工具列連線；模型、主要工件與區域在此準備。"),
            "follow": (True, True, True, 3,
                       "設定主要工件定位；完成後切換區域設定。"),
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
        self.sop_box.setVisible(mode in ("overview", "follow"))
        self.tabs.setVisible(show_tabs)
        tools_layout = self.roi_settings_layout if mode == "positions" else self.prepare_layout
        if mode in ("overview", "follow", "positions"):
            tools_layout.insertWidget(0, self.quick_panel)
            tools_layout.insertWidget(1, self.manual_tools)
        self.quick_panel.setVisible(bool(self._quick_phase))
        self.manual_tools.setVisible(show_manual)
        self.step_actions.setVisible(True)
        self.edit_step_button.hide()
        if tab_index is not None:
            self.tabs.setCurrentIndex(tab_index)
        self.mode_hint.setText(hint)
        self.mode_buttons[primary].setToolTip(hint)
        self.quick_button.setVisible(mode == "overview" and not self._quick_phase)
        self.help_button.setVisible(True)
        self.draw_button.setVisible(mode == "positions")
        self.freeze_check.setVisible(mode == "positions")
        self.target_button.setVisible(mode == "follow")
        self.clear_target_button.setVisible(mode == "follow")
        self.reference_button.setVisible(mode == "positions")
        self.live_button.setVisible(True)
        self.live_label.setVisible(True)
        self.tracking_label.setVisible(mode in ("follow", "positions") or bool(self._quick_phase))
        # All workspaces share the same panes; do not overwrite user-adjusted widths.
        step = self._current_step()
        self.context_title.setText(f"工序 {self.step_list.currentRow()+1} · {step.name}" if mode == "steps" and step else
                                   {"overview": "作業準備", "follow": "主要工件", "positions": "區域設定", "cycle": "循環設定"}.get(mode, "工序編輯"))
        self._refresh_video_rois()
        self._update_live()

    def _open_step_advanced(self):
        self._set_editor_mode("steps")

    def _refresh_setup_summary(self):
        reference_ok = self.sop.workpiece is not None
        positions = len([roi for roi in self.sop.rois if roi.anchor == "workpiece"])
        invalid = sum(bool(self._step_issues(step)) for step in self.sop.steps)
        manual = sum(not step.conditions and not self._step_issues(step) for step in self.sop.steps)
        automated = len(self.sop.steps) - invalid - manual
        configured = len(self.sop.steps) - invalid
        model_ok = self._model_state == "available"
        model_text = {"unselected": "未選擇", "selected": "已選路徑，尚未載入", "loading": "載入中",
                      "available": "可用", "failed": "失敗"}[self._model_state]
        positive = "#9cdfaf" if self._theme_dark else "#2e7d32"
        mark = lambda ready: f"<span style='color:{positive}'>就緒</span>" if ready else "<span style='color:#90a4ae'>未設定</span>"
        self.setup_summary.setText(
            f"{mark(model_ok)} 辨識模型：{model_text}　　{mark(reference_ok)} 位置跟隨　　"
            f"{mark(positions > 0)} 安裝位置 <b>{positions}</b>　　"
            f"{mark(bool(self.sop.steps) and invalid == 0)} 已設定工序 <b>{configured}/{len(self.sop.steps)}</b>"
            f"（自動 {automated} · 人工確認 {manual} · 待修正 {invalid}）")
        self.setup_summary.setToolTip(self._model_detail)
        self.quick_button.setText("繼續設定安裝位置" if reference_ok else "開始引導設定")
        self.clear_target_button.setEnabled(reference_ok)

    # ---- 外部介面 --------------------------------------------------------------
    def set_sop(self, sop: SOPDefinition):
        self._validation = EditorValidation()
        self._packet_arrived_at = None
        self._roi_undo.clear()
        self._roi_redo.clear()
        self.roi_more_button.setChecked(False)
        self._set_quick_phase("")
        self._return_live()
        self._set_editor_mode("steps")
        self.sop = sop
        self._quick_save_pending = False
        self.quick_save_button.setEnabled(True)
        self._model_state = "selected" if sop.model_path else "unselected"
        self._model_detail = ""
        if sop.model_path and not resolve(sop.model_path).exists():
            self._model_state, self._model_detail = "failed", "模型路徑不存在"
        self._loading = True
        self.name_edit.setText(sop.name)
        self.workspace_title.setText(f"SOP · {sop.name}")
        self.version_label.setText(str(sop.version))
        self._set_model_label(sop.model_path)
        self._capture_target = False
        self.reference_button.blockSignals(True)
        self.reference_button.setChecked(False)
        self.reference_button.blockSignals(False)
        self.freeze_check.setEnabled(True)
        self._locator = None
        self._workpiece_key = workpiece_key(sop.workpiece)
        self._reference_frame = None
        self._preview_result = None
        self._packet = None
        self.video.clear_frame("SOP 已更新，等待即時影像")
        self.freeze_check.setChecked(False)
        self.draw_button.setChecked(False)
        self.draw_button.setText("框選偵測區域")
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
        self.reset_ratio_spin.setValue(round(sop.cycle.reset_ratio * 100))
        self.sequence_check.setChecked(sop.cycle.detect_sequence)
        self._loading = False
        self._refresh_options()
        self._refresh_roi_list()
        self._refresh_step_list(0 if sop.steps else -1)
        self._refresh_setup_summary()
        self.set_document_state(False)
        self._update_cycle_summary()
        self._update_validation_view()
        self.workpiece_changed.emit(sop.workpiece)

    def set_classes(self, classes):
        self.classes = list(classes)
        self._refresh_options()
        self._refresh_step_summaries()

    def set_model_state(self, state: str, detail: str = ""):
        if state not in ("unselected", "selected", "loading", "available", "failed"):
            raise ValueError(f"Unknown model state: {state}")
        self._model_state, self._model_detail = state, detail
        self._set_model_label(self.sop.model_path)
        self._refresh_setup_summary()
        if state != "available":
            self._invalidate_validation("模型狀態已變更，請重新試跑")
        self._update_validation_view()

    def on_packet(self, packet: FramePacket):
        if self.reference_button.isChecked():
            return
        if not self.freeze_check.isChecked() or self._packet is None:
            self._packet_arrived_at = time.monotonic()
            self._packet = packet
            self._preview_result = self._tracked(packet.result)
            if self._locator:
                self.tracking_label.setText(self._preview_result.tracking_status)
            clean_modes = {"overview", "follow", "positions", "cycle"}
            self.video.set_frame(packet.frame if self._editor_mode in clean_modes else packet.display)
            self._refresh_video_rois()
            if self._validation.running:
                try:
                    # 依序餵入背景處理過的每一幀，取樣與正式監控一致，不受介面更新速度影響。
                    for frame_result in packet.all_results():
                        self._validation.update(self._tracked(frame_result))
                        if not self._validation.running:
                            break
                except Exception as exc:
                    self._invalidate_validation(f"試跑發生錯誤：{exc}")
        self._update_live()
        self._update_validation_view()

    def _tracked(self, result: FrameResult) -> FrameResult:
        """只採用與目前參考工件相符的背景定位結果，避免套用舊設定的位置。"""
        if self.sop.workpiece is None or self._locator is None:
            return replace(result, workpiece_transform=None)
        if result.workpiece_key != self._workpiece_key:
            return replace(result, workpiece_transform=None, tracking_status="定位設定同步中…")
        return result

    def clear_video(self, text: str):
        self._invalidate_validation("影像來源已變更或中斷，請重新試跑")
        self._return_live()
        self._packet = None
        self._preview_result = None
        self._packet_arrived_at = None
        self.video.clear_frame(text)
        self._refresh_video_rois()
        self._update_live()
        self._update_validation_view()

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
        self._refreshing_steps = True
        self._roi_undo.clear()
        self._roi_redo.clear()
        count = len(self.sop.steps)
        self.steps_box.setTitle(f"工序流程 · {count} 道")
        self.step_list.setToolTip("由上而下執行；單擊工序即可編輯，試跑目標與編輯選取互不影響")
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for index, step in enumerate(self.sop.steps, 1):
            item = QListWidgetItem(self._step_text(index, step))
            item.setToolTip(self._step_text(index, step))
            self.step_list.addItem(item)
        self.step_list.blockSignals(False)
        select = min(select, len(self.sop.steps) - 1)
        self.step_list.setCurrentRow(select)
        self._on_step_selected(select)
        self._refreshing_steps = False
        self._refresh_setup_summary()

    def _step_issues(self, step: Step) -> list[str]:
        issues = []
        if step.completion_mode not in ("all", "any"):
            issues.append("完成方式無效")
        if not 0 < step.ratio <= 1:
            issues.append("成立比例無效")
        rois = self.sop.roi_map()
        for condition in step.conditions + step.forbidden:
            if condition.type not in CONDITION_TYPES:
                issues.append("條件類型無效")
            if not condition.label:
                issues.append("未選物件")
            elif self.classes and condition.label not in self.classes:
                issues.append(f"模型無類別 {condition.label}")
            if condition.roi and condition.roi not in rois:
                issues.append(f"區域不存在：{condition.roi}")
            elif condition.roi:
                roi = rois[condition.roi]
                if roi.anchor == "workpiece" and self.sop.workpiece is None:
                    issues.append(f"區域缺少參考工件：{roi.name}")
                elif roi.anchor not in ("fixed", "workpiece"):
                    issues.append(f"區域定位基準無效：{roi.name}")
        return list(dict.fromkeys(issues))

    def _step_text(self, index: int, step: Step) -> str:
        issues = self._step_issues(step)
        status = "⚠ " + "；".join(issues) if issues else (
            "人工確認（未設自動條件）" if not step.conditions else f"自動 · {len(step.conditions)} 條件")
        return f"{index}. {step.name}\n{status}"

    def _refresh_step_summaries(self):
        # Update existing items, preserving selection and ROI history.
        for index, step in enumerate(self.sop.steps):
            item = self.step_list.item(index)
            if item is not None:
                item.setText(self._step_text(index + 1, step))
                item.setToolTip(self._step_text(index + 1, step))
        self._refresh_setup_summary()

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
        self._refresh_step_summaries()

    def _refresh_video_rois(self):
        selected = self._highlighted_condition_roi if self._editor_mode == "steps" else None
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
                     else "條件成立" if all(c.met for c in checks) else "尚未到位")
            mode = "跟隨本體" if roi.anchor == "workpiece" else "固定畫面"
            statuses.append(f"{roi.name}［{mode}］：{status}")
        self.roi_status.setText("\n".join(statuses) or "尚未新增安裝位置")
        forbidden_met = False
        if advanced:
            forbidden_met, forbidden_results = evaluate_conditions(step.forbidden, self._preview_result, rois)
            self.condition_table.set_live_results(results)
            self.forbidden_table.set_live_results(forbidden_results)
        positive = "#9cdfaf" if self._theme_dark else "#2e7d32"
        negative = "#ff9f9f" if self._theme_dark else "#c62828"
        warning = "#ffd099" if self._theme_dark else "#f57c00"
        if not step.conditions:
            text = f"<span style='color:{warning}'>此工序無條件（需手動確認）</span>"
        elif met:
            text = f"<span style='color:{positive}'><b>本幀成立，尚未代表工序完成</b></span>"
        else:
            text = f"<span style='color:{negative}'><b>本幀條件未成立</b></span>"
        errors = list(dict.fromkeys(result.error for result in results if result.error))
        if errors:
            text += "　" + escape("；".join(errors))
        if forbidden_met:
            text += f"　<span style='color:{negative}'><b>本幀違規條件成立</b></span>"
        timing = f"時間窗 {step.hold_sec:g} 秒 · 成立比例至少 {step.ratio:.0%}；此預覽未累積時間。"
        self.live_label.setText(f"單幀預覽：{text}<br>{timing}" if step.conditions else f"單幀預覽：{text}")
        self.live_label.setWordWrap(True)

    def _mark_modified(self):
        if not self._loading:
            self.set_document_state(True)
            self._invalidate_validation()
            self.modified.emit()

    # ---- SOP 與模型 ------------------------------------------------------------
    def _set_model_label(self, path: str | None):
        """畫面顯示檔名，完整路徑放在提示中，避免 ZIP 路徑擠壓表單。"""
        if not path:
            self.model_label.setText("（未選擇）")
            self.model_label.setToolTip("")
            return
        self.model_label.setText(Path(path).name)
        self.model_label.setToolTip(str(path) + (f"\n{self._model_detail}" if self._model_detail else ""))

    def _on_sop_name(self, text: str):
        self.sop.name = text
        self.workspace_title.setText(f"SOP · {text}")
        self._mark_modified()

    def _choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇模型", str(PROJECT_ROOT),
                                              "模型包 (*.zip);;模型設定 (model.json)")
        if path:
            self.sop.model_path = relativize(path)
            self.set_model_state("selected")
            self.model_selected.emit(path)
            self._mark_modified()

    # ---- 工序 ------------------------------------------------------------------
    def _on_step_selected(self, row: int):
        if row >= 0 and not self._loading and not self._refreshing_steps and not self._quick_phase:
            self._set_editor_mode("steps")
        self._highlighted_condition_roi = None
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
            self.min_duration_spin.setValue(step.min_duration_sec)
            self.completion_mode.setCurrentIndex(max(0, self.completion_mode.findData(step.completion_mode)))
            self.forbidden_mode.setCurrentIndex(max(0, self.forbidden_mode.findData(step.forbidden_mode)))
            self.forbidden_hold_spin.setValue(-0.1 if step.forbidden_hold_sec is None else step.forbidden_hold_sec)
            self.forbidden_ratio_spin.setValue(0 if step.forbidden_ratio is None else round(step.forbidden_ratio * 100))
            self.condition_table.set_conditions(step.conditions)
            self.forbidden_table.set_conditions(step.forbidden)
        else:
            self.condition_table.set_conditions([])
            self.forbidden_table.set_conditions([])
        self._loading = False
        self._refresh_video_rois()
        self._update_live()
        self._update_step_rule_summary()
        self._update_validation_view()

    def _on_step_form(self, *_args):
        step = self._current_step()
        if self._loading or step is None:
            return
        step.name = self.step_name_edit.text()
        step.instruction = self.instruction_edit.toPlainText()
        step.hold_sec = round(self.hold_spin.value(), 2)
        step.ratio = self.ratio_spin.value() / 100
        step.timeout_sec = self.timeout_spin.value()
        step.min_duration_sec = self.min_duration_spin.value()
        step.completion_mode = self.completion_mode.currentData()
        step.forbidden_mode = self.forbidden_mode.currentData()
        hold = self.forbidden_hold_spin.value()
        step.forbidden_hold_sec = None if hold < 0 else round(hold, 2)
        ratio = self.forbidden_ratio_spin.value()
        step.forbidden_ratio = None if ratio <= 0 else ratio / 100
        self._roi_undo.clear()
        self._roi_redo.clear()
        step.conditions = self.condition_table.conditions()
        step.forbidden = self.forbidden_table.conditions()
        self._refresh_step_summaries()
        self._refresh_video_rois()
        self._update_live()
        self._update_step_rule_summary()
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
        self.sop.cycle.reset_ratio = self.reset_ratio_spin.value() / 100
        self.sop.cycle.detect_sequence = self.sequence_check.isChecked()
        self.sop.cycle.reset_conditions = self.reset_table.conditions()
        self._update_live()
        self._update_cycle_summary()
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
        if self._quick_save_pending:
            return
        self._quick_save_pending = True
        self.quick_save_button.setEnabled(False)
        self.save_requested.emit()

    def on_save_result(self, result: str):
        if not self._quick_save_pending:
            return
        self._quick_save_pending = False
        self.quick_save_button.setEnabled(True)
        if result != "saved":
            message = "已取消儲存" if result == "cancelled" else "儲存失敗"
            self.guide_label.setText(f"{message}；引導與編輯內容仍保留，請確認後重新儲存。")
            return
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
        self._refresh_step_summaries()
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
        if frozen:
            self._invalidate_validation("畫面已凍結，請返回即時後重新試跑")
        if not frozen:
            self.video.set_editable([])
            self._packet = None
            self._preview_result = None
            self._packet_arrived_at = None
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
            self._workpiece_key = workpiece_key(target)
            self.workpiece_changed.emit(target)
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
        self._workpiece_key = ""
        self.workpiece_changed.emit(None)
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
        if enabled:
            self._invalidate_validation("已切換參考影像，請返回即時後重新試跑")
        self._capture_target = False
        self.target_button.setText("① 設定位置跟隨")
        self.draw_button.setChecked(False)
        self.video.set_draw_mode(False)
        if enabled and self._reference_frame is not None:
            self.roi_status.setText("參考影像編輯中；返回即時後顯示各位置判定結果。")
            self.video.set_frame(self._reference_frame)
            self.freeze_check.setEnabled(False)
            self.draw_button.setText("框選車把手安裝位置")
            self.guide_label.setText("② 目前是車把手參考影像。請拖曳安裝位置；按「③ 返回即時驗證」恢復影像。")
            self.draw_button.setChecked(True)
            self.tracking_label.setText("參考影像：框選每道工序的作業區，再於工序條件的「區域」選取它")
        else:
            self.freeze_check.setEnabled(True)
            self.freeze_check.setChecked(False)
            self.draw_button.setText("框選偵測區域")
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
        self.draw_button.setText("框選偵測區域")
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
