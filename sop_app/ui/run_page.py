"""執行監控頁：給作業員看的即時畫面、目前工序、條件狀態、警報。"""
from __future__ import annotations

import copy
from datetime import datetime

import numpy as np

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSplitter,
                               QScrollArea, QSizePolicy, QStyle, QVBoxLayout, QWidget)

from ..conditions import combine_results
from ..engine import EngineEvent, EngineSnapshot, Phase
from ..pipeline import FramePacket
from ..overlay import draw_detections
from ..sop_schema import SOPDefinition
from ..tracking import WorkpieceLocator, display_rois, workpiece_key
from ..tracking_debug import TrackingDebug
from .video_widget import VideoWidget
from .compact_layout import StatusText
from .step_overview import StepOverview
from .workbench_widgets import install_readability_guard, workbench_palette


def run_palette(dark: bool) -> dict:
    """監控頁顏色：沿用工作台色票，另加狀態色；深色模式改用較亮的狀態色以維持對比。"""
    colors = workbench_palette(dark)
    if dark:
        colors.update(ok="#81c995", active="#8ab4f8", warn="#fdd663", ng="#f28b82", pending="#9aa0a6",
                      alarm_bg="#5c1a1a", alarm_fg="#ffdad6", alarm_flash="#8c1d18", start="#2e7d32", stop="#c62828")
    else:
        colors.update(ok="#2e7d32", active="#1565c0", warn="#ef6c00", ng="#c62828", pending="#78909c",
                      alarm_bg="#ffebee", alarm_fg="#8e0000", alarm_flash="#ffcdd2", start="#2e7d32", stop="#c62828")
    return colors


class _MonitorControlsScroll(QScrollArea):
    """Use the controls' natural height, but allow a short screen to shrink it."""

    def sizeHint(self):
        content = self.widget()
        height = content.sizeHint().height() if content is not None else 155
        return QSize(380, min(155, max(32, height)))

    def minimumSizeHint(self):
        return QSize(0, 32)


class RunPage(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    confirm_requested = Signal()       # 只用於無自動條件的人工工序
    skip_requested = Signal()          # 明確跳過（判 NG），需二次確認
    restart_requested = Signal()
    alarm_ack_requested = Signal()
    context_changed = Signal(dict)     # 作業員／工單／序號

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("ownsPageScroll", True)
        self.sop: SOPDefinition | None = None
        self._running = False
        self._transform = None
        self._last_transform = None
        self._tracking_misses = 0
        self._workpiece_key = ""
        self._locator_error = ""
        self._snapshot = None
        self._last_packet = None
        self._tracking_status = "等待影像定位"
        self._stats = {"OK": 0, "NG": 0, "中止": 0}
        self._debug = TrackingDebug()
        self._today: dict = {}
        self._pending_alarms: list[str] = []
        self._flash_on = False
        self._dark = None
        self._colors = run_palette(False)
        self._start_appearance = None
        self._build()
        self._apply_theme()
        install_readability_guard(self)

    def _build(self):
        self.video = VideoWidget()

        self.step_title = StatusText("目前：待機中")
        self.step_title.setObjectName("currentStepTitle")
        self.step_title.setAccessibleName("目前工序")
        self.step_title.setFont(QFont(self.font().family(), 15, QFont.Weight.Bold))
        self.next_step_label = StatusText("下一道：—")
        self.next_step_label.setObjectName("nextStepTitle")
        self.next_step_label.setAccessibleName("下一道工序")
        self.instruction_preview = StatusText()
        self.instruction = QLabel("")
        self.instruction.setTextFormat(Qt.TextFormat.PlainText)
        self.instruction.setWordWrap(True)
        self.instruction.setObjectName("instructionLabel")
        self.tracking_label = QLabel("")
        self.tracking_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setMinimumHeight(22)
        self.progress.setRange(0, 100)
        self.progress.setFormat("條件累積 %p%")
        self.condition_label = QLabel("")
        self.condition_label.setTextFormat(Qt.TextFormat.RichText)
        self.condition_label.setWordWrap(True)

        self.step_overview = StepOverview()
        self.cycle_label = StatusText()
        self.stats_label = StatusText()
        self.log = QListWidget()
        self.log.setWordWrap(True)
        self.log.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.log.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.log.setMinimumHeight(70)
        self.details_button = QPushButton("顯示判定詳情")
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self._toggle_details)
        self.details_panel = QWidget()
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(0, 0, 0, 0)
        self.details_panel.hide()
        self.log_button = QPushButton("顯示事件紀錄")
        self.log_button.setCheckable(True)
        self.log_button.toggled.connect(self._toggle_log)
        self.log_panel = QWidget()
        log_layout = QVBoxLayout(self.log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(self.log)
        self.log_panel.hide()

        self.start_button = QPushButton("開始作業")
        self.start_button.setMinimumHeight(36)
        self.start_button.setToolTip("可直接開始作業；位置尚未確認時，跟隨位置的條件會暫停累積。")
        self.start_button.clicked.connect(self._on_start_clicked)
        self.display_mode = QComboBox()
        self.display_mode.addItem("逐步｜目前流程", "step")
        self.display_mode.addItem("全開｜所有結果", "all")
        self.display_mode.setToolTip(
            "逐步：只輸出目前流程的完成與違規條件物件及位置\n"
            "全開：輸出模型所有辨識結果與所有位置\n"
            "兩種模式都不會改變背景判定")
        self.display_mode.currentIndexChanged.connect(self._on_display_mode_changed)
        self.display_summary = QLabel("逐步｜等待 SOP 與影像")
        self.display_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.display_summary.setWordWrap(True)
        self.display_summary.setObjectName("displaySummary")

        # 人工確認與跳過分開：確認只對人工工序有效；跳過必定判 NG，需再次確認。
        self.confirm_button = QPushButton("人工確認完成")
        self.confirm_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.confirm_button.setMinimumHeight(32)
        self.confirm_button.setToolTip("只有「未設自動條件」的人工工序可以確認完成")
        self.confirm_button.clicked.connect(lambda: self.confirm_requested.emit())
        self.skip_button = QPushButton("跳過本工序（判 NG）")
        self.skip_button.setToolTip("跳過後本輪判定為 NG，並留下警報紀錄")
        self.skip_button.clicked.connect(self._on_skip_clicked)
        self.restart_button = QPushButton("重新開始本輪")
        self.restart_button.clicked.connect(self._on_restart_clicked)
        buttons = QGridLayout()
        buttons.addWidget(self.confirm_button, 0, 0, 1, 2)
        buttons.addWidget(self.skip_button, 1, 0)
        buttons.addWidget(self.restart_button, 1, 1)

        # 追溯資訊：序號可用條碼槍掃描，按 Enter 即套用到下一輪。
        self.operator_edit = QLineEdit()
        self.operator_edit.setPlaceholderText("作業員")
        self.work_order_edit = QLineEdit()
        self.work_order_edit.setPlaceholderText("工單（選填）")
        self.serial_edit = QLineEdit()
        self.serial_edit.setPlaceholderText("序號（可掃條碼）")
        self.serial_edit.setToolTip("工件序號，可用條碼槍掃描；按 Enter 套用到下一輪")
        for edit in (self.operator_edit, self.work_order_edit, self.serial_edit):
            edit.editingFinished.connect(self._emit_context)
        trace = QHBoxLayout()
        trace.setContentsMargins(0, 0, 0, 0)
        trace.addWidget(self.operator_edit, 2)
        trace.addWidget(self.work_order_edit, 2)
        trace.addWidget(self.serial_edit, 3)

        # 警報需人工確認；未確認前持續閃爍，避免只響一聲就被忽略。
        self.alarm_panel = QFrame()
        self.alarm_panel.setObjectName("alarmPanel")
        self.alarm_text = StatusText()
        self.alarm_text.setAccessibleName("未確認警報，完整內容可查看提示或事件紀錄")
        self.alarm_text.setFont(QFont(self.font().family(), 13, QFont.Weight.Bold))
        self.alarm_ack_button = QPushButton("確認警報")
        self.alarm_ack_button.setMinimumHeight(28)
        self.alarm_ack_button.clicked.connect(self.acknowledge_alarms)
        alarm_layout = QHBoxLayout(self.alarm_panel)
        alarm_layout.setContentsMargins(6, 2, 6, 2)
        alarm_layout.addWidget(self.alarm_text, 1)
        alarm_layout.addWidget(self.alarm_ack_button)
        self.alarm_panel.hide()
        self.alarm_slot = QWidget()
        self.alarm_slot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        alarm_slot_layout = QVBoxLayout(self.alarm_slot)
        alarm_slot_layout.setContentsMargins(0, 0, 0, 0)
        alarm_slot_layout.addWidget(self.alarm_panel)
        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(500)
        self._flash_timer.timeout.connect(self._flash_alarm)

        panel = QWidget()
        self.monitor_panel = panel
        panel.setMinimumWidth(380)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)
        layout.addWidget(self.start_button)
        layout.addWidget(self.alarm_slot)
        layout.addWidget(self.step_title)
        layout.addWidget(self.next_step_label)
        layout.addWidget(self.progress)
        details_layout.addWidget(self.instruction)
        details_layout.addWidget(self.condition_label)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(3)
        controls_layout.addLayout(trace)
        controls_layout.addWidget(self.instruction_preview)
        controls_layout.addLayout(buttons)
        controls_layout.addWidget(self.cycle_label)
        controls_layout.addWidget(self.stats_label)
        # On a short/high-DPI screen, only the secondary controls scroll. The
        # current/next headings and paged cards keep their own fixed area.
        self.controls_scroll = _MonitorControlsScroll()
        self.controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumHeight(32)
        self.controls_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setWidget(controls)
        self.controls_scroll.setMaximumHeight(155)
        layout.addWidget(self.controls_scroll)
        details_layout.addWidget(self.tracking_label)
        layout.addWidget(self._section("工序進度"))
        layout.addWidget(self.step_overview, 1)
        display_row = QHBoxLayout()
        display_row.addWidget(QLabel("畫面顯示"))
        display_row.addWidget(self.display_mode, 1)
        details_layout.addLayout(display_row)
        details_layout.addWidget(self.display_summary)
        toggles = QHBoxLayout()
        toggles.addWidget(self.details_button)
        toggles.addWidget(self.log_button)
        layout.addLayout(toggles)

        for label in panel.findChildren(QLabel):
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.monitor_scroll = QScrollArea()
        self.monitor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.monitor_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.monitor_scroll.setWidgetResizable(True)
        self.monitor_scroll.setMinimumHeight(0)
        self.monitor_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.monitor_scroll.setMaximumHeight(220)
        auxiliary = QWidget()
        auxiliary_layout = QVBoxLayout(auxiliary)
        auxiliary_layout.setContentsMargins(6, 4, 6, 4)
        auxiliary_layout.addWidget(self.details_panel)
        auxiliary_layout.addWidget(self.log_panel)
        self.monitor_scroll.setWidget(auxiliary)
        self.monitor_scroll.hide()

        self.monitor_splitter = QSplitter(Qt.Orientation.Vertical)
        self.monitor_splitter.setChildrenCollapsible(False)
        self.monitor_splitter.addWidget(panel)
        self.monitor_splitter.addWidget(self.monitor_scroll)
        self.monitor_splitter.setStretchFactor(0, 1)
        self.monitor_splitter.setStretchFactor(1, 0)
        self.monitor_splitter.setSizes([600, 140])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(self.monitor_splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1000, 420])
        root = QHBoxLayout(self)
        root.addWidget(splitter)
        self.set_running(False)
        self._update_stats()

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def _toggle_details(self, visible):
        self.details_panel.setVisible(visible)
        self.details_button.setText("隱藏判定詳情" if visible else "顯示判定詳情")
        self._toggle_auxiliary()

    def _toggle_log(self, visible):
        self.log_panel.setVisible(visible)
        self.log_button.setText("隱藏事件紀錄" if visible else "顯示事件紀錄")
        self._toggle_auxiliary()

    def _toggle_auxiliary(self):
        visible = self.details_button.isChecked() or self.log_button.isChecked()
        self.monitor_scroll.setVisible(visible)
        if visible:
            self.monitor_splitter.setSizes([max(1, self.monitor_splitter.height() - 140), 140])

    # ---- 主題 ------------------------------------------------------------------
    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_theme()

    def _apply_theme(self):
        app = QApplication.instance()
        dark = app.palette().window().color().lightness() < 128 if app else False
        if dark == self._dark:
            return
        self._dark = dark
        self._colors = c = run_palette(dark)
        self.step_overview.set_colors(c)
        self.alarm_text.setStyleSheet(f"color:{c['alarm_fg']};")
        self.setStyleSheet(f"""
            QWidget#currentStepTitle {{ color:{c['active']}; font-size:16px; font-weight:bold; }}
            QWidget#nextStepTitle {{ color:{c['muted']}; font-size:13px; }}
            QLabel#instructionLabel {{ color:{c['fg']}; font-size:15px; }}
            QLabel#displaySummary {{ padding:6px; background:{c['soft']}; color:{c['fg']}; border-radius:4px; }}
            QLabel#sectionLabel {{ font-weight:bold; margin-top:6px; color:{c['fg']}; }}
            QFrame#alarmPanel {{ background:{c['alarm_bg']}; border:2px solid {c['ng']}; border-radius:6px; }}
            QFrame#alarmPanel[flash="true"] {{ background:{c['alarm_flash']}; }}
            QFrame#alarmPanel QLabel {{ color:{c['alarm_fg']}; background:transparent; }}
        """)
        self._update_start_availability()
        if self.sop is not None and self._snapshot is not None and self._running:
            self._render_steps(self._snapshot)

    # ---- 追溯與警報 ------------------------------------------------------------
    def context(self) -> dict:
        return {"operator": self.operator_edit.text().strip(),
                "work_order": self.work_order_edit.text().strip(),
                "serial_no": self.serial_edit.text().strip()}

    def set_context(self, context: dict):
        self.operator_edit.setText(context.get("operator", ""))
        self.work_order_edit.setText(context.get("work_order", ""))
        self.serial_edit.setText(context.get("serial_no", ""))

    def _emit_context(self):
        self.context_changed.emit(self.context())

    def _show_alarm(self, message: str):
        self._pending_alarms.append(message)
        count = len(self._pending_alarms)
        suffix = f"\n（共 {count} 筆未確認警報）" if count > 1 else ""
        self.alarm_text.setText(f"警報：{message}{suffix}")
        self.alarm_panel.show()
        self._sync_alarm_slot()
        if not self._flash_timer.isActive():
            self._flash_timer.start()

    def _flash_alarm(self):
        self._flash_on = not self._flash_on
        self.alarm_panel.setProperty("flash", "true" if self._flash_on else "false")
        self.alarm_panel.style().unpolish(self.alarm_panel)
        self.alarm_panel.style().polish(self.alarm_panel)

    def acknowledge_alarms(self):
        if not self._pending_alarms:
            return
        self._pending_alarms.clear()
        self._flash_timer.stop()
        self._flash_on = False
        self.alarm_panel.setProperty("flash", "false")
        self.alarm_panel.hide()
        self._sync_alarm_slot()
        self.alarm_ack_requested.emit()

    def _sync_alarm_slot(self):
        # A normal alarm/ack must not repaginate an active SOP before its last
        # visible step finishes. Reserve the same strip throughout the run.
        self.alarm_panel.ensurePolished()
        height = self.alarm_panel.sizeHint().height() if self._running or self._pending_alarms else 0
        self.alarm_slot.setMinimumHeight(max(0, height))

    def set_today_stats(self, stats: dict):
        self._today = dict(stats)
        self._update_stats()

    def reset_station(self):
        self.set_running(False)
        self._stats = {"OK": 0, "NG": 0, "中止": 0}
        self._today = {}
        self._pending_alarms.clear()
        self._flash_timer.stop()
        self._flash_on = False
        self.alarm_panel.hide()
        self._sync_alarm_slot()
        self.clear_video("已切換工站，請連線影像來源")
        self._update_stats()

    # ---- 外部介面 --------------------------------------------------------------
    def set_sop(self, sop: SOPDefinition):
        """作業進行中不更換，避免畫面與背景引擎使用不同版本的 SOP。"""
        if self._running:
            return
        self.sop = copy.deepcopy(sop)
        self._transform = None
        self._last_transform = None
        self._tracking_misses = 0
        self._snapshot = None
        self._last_packet = None
        self._tracking_status = "等待影像定位"
        self._locator_error = ""
        self._workpiece_key = workpiece_key(self.sop.workpiece)
        if self.sop.workpiece:
            try:
                WorkpieceLocator(self.sop.workpiece)   # 只檢查設定；逐幀定位由背景管線執行
            except ValueError as exc:
                self._locator_error = str(exc)
        self.step_overview.set_steps([step.name for step in self.sop.steps])
        self.video.set_rois(display_rois(self.sop.rois, self._transform))
        self._show_idle()
        self._update_start_availability()

    def set_running(self, running: bool):
        if running != self._running:
            self._transform = None
            self._tracking_misses = 0
        self._running = running
        self._sync_alarm_slot()
        self.step_overview.set_running(running)
        self.confirm_button.setEnabled(False)
        self.skip_button.setEnabled(running)
        self.restart_button.setEnabled(running)
        self.operator_edit.setEnabled(not running)
        self.work_order_edit.setEnabled(not running)
        self._update_start_availability()
        if not running:
            self._show_idle()

    def on_packet(self, packet: FramePacket):
        self._last_packet = packet
        if self.sop is None or self.sop.workpiece is None or self._locator_error:
            self._transform = None
            status = self._locator_error or "尚未設定參考工件"
        elif packet.result.workpiece_key == self._workpiece_key:
            # 待機與正式作業都使用背景管線同一幀的定位結果；介面不另外計算。
            self._transform = packet.result.workpiece_transform
            status = packet.result.tracking_status
        else:
            self._transform = None
            status = "定位設定同步中…"
        if self._transform is not None:
            self._last_transform = self._transform.copy()
            self._tracking_misses = 0
        elif self.sop and self.sop.workpiece:
            self._tracking_misses += 1
        self._update_start_availability()
        self._tracking_status = status
        self._debug.capture(packet, self._transform, status, running=self._running)
        snapshot = packet.snapshot
        self._snapshot = snapshot
        self.video.set_frame(self._monitor_frame(packet, snapshot))
        self._refresh_regions(status, snapshot)
        if self.sop is None or snapshot is None or snapshot.phase == Phase.IDLE or not self._running:
            return
        self._render_steps(snapshot)
        if snapshot.phase == Phase.RUNNING:
            self._render_running(snapshot)
        else:
            self._render_cycle_done(snapshot)

    def clear_video(self, text: str):
        self._transform = None
        self._last_transform = None
        self._tracking_misses = 0
        self.video.clear_frame(text)
        self.video.set_rois([])
        self.tracking_label.setText(text)
        self._tracking_status = text
        self._update_start_availability()

    def _update_start_availability(self):
        color = self._colors["stop" if self._running else "start"]
        appearance = (self._running, color)
        if appearance == self._start_appearance:
            return
        self._start_appearance = appearance
        self.start_button.setEnabled(True)
        self.start_button.setText("停止作業" if self._running else "開始作業")
        self.start_button.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaStop if self._running else QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.setStyleSheet(f"background:{color}; color:white; font-size:16px; font-weight:bold;"
                                        "border-radius:6px")

    @staticmethod
    def _step_sets(step):
        conditions = step.conditions + step.forbidden
        return {c.roi for c in conditions if c.roi}, {c.label for c in conditions if c.label}

    def _step_context(self, snapshot=None):
        if not self.sop:
            return "等待 SOP", set(), set()
        if snapshot and snapshot.phase == Phase.CYCLE_DONE:
            if not self.sop.cycle.enabled:
                return "本輪完成，已停用自動開始下一輪", set(), set()
            conditions = self.sop.cycle.reset_conditions
            return ("循環重置" if conditions else "本輪完成，等待下一輪",
                    {c.roi for c in conditions if c.roi}, {c.label for c in conditions if c.label})
        index = snapshot.current_index if snapshot and snapshot.phase == Phase.RUNNING else 0
        if not 0 <= index < len(self.sop.steps):
            return "尚未設定流程", set(), set()
        rois, labels = self._step_sets(self.sop.steps[index])
        prefix = "目前" if snapshot and snapshot.phase == Phase.RUNNING else "待機預覽"
        return f"{prefix}流程 {index + 1}：{self.sop.steps[index].name}", rois, labels

    def _monitor_frame(self, packet, snapshot):
        mode = self.display_mode.currentData()
        if mode == "all" or not packet.classes:
            return packet.preview_display
        _context, _rois, labels = self._step_context(snapshot)
        detections = [d for d in packet.result.detections if d.label in labels]
        return draw_detections(packet.frame, detections, packet.classes, packet.min_score, packet.show_masks)

    def set_regions_visible(self, visible):
        self.video.set_regions_visible(visible)
        self._refresh_regions(self._tracking_status, self._snapshot)

    def _refresh_regions(self, status="等待影像定位", snapshot=None):
        if self.sop is None:
            self.video.set_rois([])
            self.tracking_label.setText("")
            return
        context, current, labels = self._step_context(snapshot)
        selected = self.sop.rois if self.display_mode.currentData() == "all" else [r for r in self.sop.rois if r.name in current]
        tracking_lost = (self.sop.workpiece is not None and self._transform is None
                         and (self._last_transform is None or self._tracking_misses >= 2))
        display_transform = self._transform
        if display_transform is None and self._last_transform is not None:
            display_transform = self._last_transform
        if display_transform is None and self.sop.workpiece is not None:
            display_transform = np.array([[1., 0., 0.], [0., 1., 0.]])
        visible = display_rois(selected, display_transform)
        stale = {roi.name for roi in selected if tracking_lost and roi.anchor == "workpiece"}
        self.video.set_rois(visible, current if self.display_mode.currentData() == "step" else set(), stale=stale)
        hidden = len(selected) - len(visible)
        mode_text = self.display_mode.currentText()
        self.tracking_label.setToolTip(status)
        if self.display_mode.currentData() == "all":
            packet = self._last_packet
            detected = sum(d.score >= packet.min_score for d in packet.result.detections) if packet else 0
            classes = len(packet.classes) if packet else 0
            self.display_summary.setText(
                f"全開｜模型全部輸出｜目前偵測 {detected} 個物件（{classes} 類）｜位置 {len(visible)}/{len(selected)}")
        else:
            label_text = "、".join(sorted(labels)) or "無指定物件"
            roi_text = "、".join(r.name for r in selected) or "整個畫面／無指定位置"
            self.display_summary.setText(f"逐步｜{context}\n物件：{label_text}｜位置：{roi_text}")
        if not selected:
            self.tracking_label.setText("目前工序不使用工作區域" if self.display_mode.currentData() == "step"
                                        else "全開｜目前沒有工作區域")
        elif stale and not hidden:
            self.tracking_label.setText(
                f"{mode_text}｜區域 {len(visible)}/{len(selected)}｜位置跟隨暫停｜顯示最後位置，條件停止累積")
        elif hidden:
            self.tracking_label.setText(
                f"{mode_text}｜區域 {len(visible)}/{len(selected)}｜{hidden} 個區域超出畫面｜條件停止累積")
        elif self.sop.workpiece:
            state = "位置跟隨正常" if self._transform is not None or self._last_transform is not None else "位置跟隨準備中"
            self.tracking_label.setText(f"{mode_text}｜區域 {len(visible)}/{len(selected)}｜{state}")
        else:
            self.tracking_label.setText(f"{mode_text}｜已顯示 {len(visible)} 個固定區域" if visible else
                                        f"{mode_text}｜目前工序未指定區域")
        if not self.video.regions_visible:
            detail = self.tracking_label.text().replace("已顯示", "已設定").replace("顯示最後位置", "保留最後位置")
            self.tracking_label.setText(f"區域框線已隱藏｜{detail}")

    def _on_display_mode_changed(self, _index):
        self._refresh_regions(self._tracking_status, self._snapshot)
        if self._last_packet is not None:
            self.video.set_frame(self._monitor_frame(self._last_packet, self._snapshot))

    def on_events(self, events: list[EngineEvent]):
        for event in events:
            if event.kind == "step_started":
                continue
            c = self._colors
            color = {"alarm": c["ng"], "cycle_aborted": c["warn"]}.get(event.kind, c["fg"])
            if event.kind == "cycle_completed":
                color = c["ok"] if event.data.get("result") == "OK" else c["warn"]
                self._stats[event.data.get("result", "NG")] += 1
            elif event.kind == "cycle_aborted":
                self._stats["中止"] += 1
            item = QListWidgetItem(f"{datetime.now():%H:%M:%S}  {event.message}")
            item.setForeground(QColor(color))
            item.setToolTip(event.message)
            self.log.insertItem(0, item)
            if event.kind == "alarm":
                QApplication.beep()
                self._show_alarm(event.message)
        while self.log.count() > 300:
            self.log.takeItem(self.log.count() - 1)
        self._update_stats()

    # ---- 繪製 ------------------------------------------------------------------
    def _set_instruction(self, text):
        self.instruction.setText(text)
        self.instruction_preview.setText(text)
        # Keep its line even when empty so advancing to another instruction
        # cannot resize the list and move its page boundary mid-operation.
        self.instruction_preview.setVisible(True)

    def _set_next_step(self, index, suffix=""):
        if self.sop and 0 <= index < len(self.sop.steps):
            self.next_step_label.setText(f"下一道：{index + 1}. {self.sop.steps[index].name}{suffix}")
        else:
            self.next_step_label.setText("下一道：無（本輪最後一道）" if self.sop and self.sop.steps else "下一道：尚未設定工序")

    def _show_idle(self):
        self.video.set_banner("待機中 · 按「開始作業」開始監控", self._colors["pending"], single_line=True)
        self.step_title.setText("目前：待機中")
        self._set_next_step(0)
        self._set_instruction(f"SOP：{self.sop.name}（共 {len(self.sop.steps)} 道工序）" if self.sop else "")
        self.step_overview.set_snapshot(None)
        self.progress.setValue(0)
        self.condition_label.setText("")
        self.cycle_label.setText("")
        self._refresh_regions()

    def _render_steps(self, snapshot: EngineSnapshot):
        self.step_overview.set_snapshot(snapshot)
        elapsed = snapshot.now - snapshot.cycle_started_at if snapshot.cycle_started_at is not None else 0
        state = "OK" if snapshot.cycle_ok else "NG（有警報或跳過）"
        self.cycle_label.setText(f"第 {snapshot.cycle} 輪 · 已進行 {elapsed:.0f} 秒 · 目前判定 {state}")

    def _render_running(self, snapshot: EngineSnapshot):
        index = snapshot.current_index
        step, runtime = self.sop.steps[index], snapshot.steps[index]
        title = f"工序 {index + 1} / {len(self.sop.steps)}：{step.name}"
        self.step_title.setText(f"目前 {index + 1} / {len(self.sop.steps)}：{step.name}")
        self._set_next_step(index + 1)
        self._set_instruction(step.instruction)
        self.progress.setFormat("條件累積 %p%")
        self.progress.setValue(round(snapshot.progress * 100))
        self.confirm_button.setEnabled(not step.conditions)

        c = self._colors
        if runtime.violation_alarmed or runtime.sequence_alarmed or runtime.too_fast_alarmed:
            banner_color = c["stop"]
        elif runtime.timeout_alarmed:
            banner_color = "#ef6c00"
        else:
            banner_color = "#1565c0"
        self.video.set_banner(f"進行中｜{title}", banner_color, single_line=True)

        lines = []
        if not step.conditions:
            lines.append(f"<span style='color:{c['warn']}'>此工序沒有自動條件，完成後請按「人工確認完成」</span>")
        for result in snapshot.results:
            if result.error:
                lines.append(f"<span style='color:{c['warn']}'>暫停判定：{result.error}</span>")
            else:
                mark, color = ("成立", c["ok"]) if result.met else ("未成立", c["ng"])
                lines.append(f"<span style='color:{color}'>{mark} {result.condition.describe()}"
                             f"　（{result.detail or f'目前 {result.count}'}）</span>")
        for result in snapshot.forbidden_results:
            if result.met:
                lines.append(f"<span style='color:{c['ng']}'>違規：{result.condition.describe()}</span>")
        if step.timeout_sec > 0:
            remaining = step.timeout_sec - (snapshot.now - runtime.started_at)
            lines.append(f"<span style='color:{c['muted']}'>剩餘時間 {max(0, remaining):.0f} 秒</span>")
        if combine_results(snapshot.results, step.completion_mode)[1]:
            self.progress.setFormat("判定暫停")
        self.condition_label.setText("<br>".join(lines))

    def _render_cycle_done(self, snapshot: EngineSnapshot):
        ok = snapshot.cycle_ok
        result = "OK" if ok else "NG"
        self.confirm_button.setEnabled(False)
        self.video.set_banner(f"第 {snapshot.cycle} 輪完成：{result}",
                              self._colors["start"] if ok else "#ef6c00", single_line=True)
        self.step_title.setText(f"本輪完成：{result}")
        cycle = self.sop.cycle
        if not cycle.enabled:
            self.next_step_label.setText("下一道：無（自動開始下一輪已停用）")
            self._set_instruction("本輪已完成；自動開始下一輪已停用")
            self.progress.setFormat("")
            self.progress.setValue(0)
            self.condition_label.setText("")
        elif cycle.reset_conditions:
            self._set_next_step(0, "（下一輪）")
            self._set_instruction("等待重置條件成立後開始下一輪")
            self.progress.setFormat("重置條件 %p%")
            self.progress.setValue(round(snapshot.progress * 100))
            lines = [f"{'成立' if r.met else '未成立'} {r.condition.describe()}" for r in snapshot.results]
            self.condition_label.setText("<br>".join(lines))
        else:
            self._set_next_step(0, "（下一輪）")
            remaining = cycle.auto_restart_sec - (snapshot.now - snapshot.cycle_ended_at)
            self._set_instruction(f"{max(0, remaining):.0f} 秒後開始下一輪")
            self.progress.setFormat("")
            self.progress.setValue(0)
            self.condition_label.setText("")

    def _update_stats(self):
        total = sum(self._stats.values())
        text = (f"本次開啟以來：共 {total} 輪 · OK {self._stats['OK']} · "
                f"NG {self._stats['NG']} · 中止 {self._stats['中止']}")
        if self._today:
            today_total = sum(self._today.values())
            ok = self._today.get("OK", 0)
            rate = f" · 良率 {ok / today_total:.0%}" if today_total else ""
            text += (f"\n今日（全部紀錄）：共 {today_total} 輪 · OK {ok} · NG {self._today.get('NG', 0)}"
                     f"{rate}")
        self.stats_label.setText(text)

    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text, QMessageBox.StandardButton.Yes
                                    | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def _on_start_clicked(self):
        if not self._running:
            self._emit_context()
            self.start_requested.emit()
        elif self._confirm("停止作業", "停止後本輪將記為「中止」。確定要停止作業嗎？"):
            self.stop_requested.emit()

    def _on_skip_clicked(self):
        step = ""
        if self.sop and self._snapshot and self._snapshot.phase == Phase.RUNNING:
            step = f"「{self.sop.steps[self._snapshot.current_index].name}」"
        if self._confirm("跳過工序", f"確定跳過目前工序{step}？\n跳過後本輪判定為 NG，並留下警報紀錄。"):
            self.skip_requested.emit()

    def _on_restart_clicked(self):
        if self._confirm("重新開始本輪", "目前這一輪會記為「中止」並重新開始。確定嗎？"):
            self.restart_requested.emit()
