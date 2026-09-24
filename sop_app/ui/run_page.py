"""執行監控頁：給作業員看的即時畫面、目前工序、條件狀態、警報。"""
from __future__ import annotations

import copy
from datetime import datetime

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..engine import EngineEvent, EngineSnapshot, Phase, StepStatus
from ..pipeline import FramePacket
from ..overlay import draw_detections
from ..sop_schema import SOPDefinition
from ..tracking import WorkpieceLocator, display_rois
from .video_widget import VideoWidget

STATUS_COLORS = {StepStatus.PENDING: "#78909c", StepStatus.ACTIVE: "#1565c0",
                 StepStatus.DONE: "#2e7d32", StepStatus.SKIPPED: "#ef6c00"}


class RunPage(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    confirm_requested = Signal()
    restart_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sop: SOPDefinition | None = None
        self._running = False
        self._transform = None
        self._last_transform = None
        self._tracking_misses = 0
        self._preview_locator = None
        self._locator_error = ""
        self._snapshot = None
        self._last_packet = None
        self._tracking_status = "等待影像定位"
        self._stats = {"OK": 0, "NG": 0, "中止": 0}
        self._build()

    def _build(self):
        self.video = VideoWidget()

        self.step_title = QLabel("待機中")
        self.step_title.setFont(QFont(self.font().family(), 18, QFont.Weight.Bold))
        self.step_title.setWordWrap(True)
        self.instruction = QLabel("")
        self.instruction.setWordWrap(True)
        self.instruction.setStyleSheet("color:#455a64; font-size:14px")
        self.tracking_label = QLabel("")
        self.tracking_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFormat("條件累積 %p%")
        self.condition_label = QLabel("")
        self.condition_label.setTextFormat(Qt.TextFormat.RichText)
        self.condition_label.setWordWrap(True)

        self.step_list = QListWidget()
        self.step_list.setFont(QFont(self.font().family(), 11))
        self.cycle_label = QLabel("")
        self.stats_label = QLabel("")
        self.log = QListWidget()
        self.log.setWordWrap(True)

        self.start_button = QPushButton("▶ 開始作業")
        self.start_button.setMinimumHeight(40)
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
        self.display_summary.setWordWrap(True)
        self.display_summary.setStyleSheet("padding:6px; background:#eef4f8; color:#29434e; border-radius:3px")
        self.confirm_button = QPushButton("手動確認／跳過本工序")
        self.confirm_button.clicked.connect(lambda: self.confirm_requested.emit())
        self.restart_button = QPushButton("重新開始本輪")
        self.restart_button.clicked.connect(lambda: self.restart_requested.emit())
        buttons = QHBoxLayout()
        buttons.addWidget(self.confirm_button)
        buttons.addWidget(self.restart_button)

        panel = QWidget()
        panel.setMinimumWidth(360)
        layout = QVBoxLayout(panel)
        layout.addWidget(self.start_button)
        display_row = QHBoxLayout()
        display_row.addWidget(QLabel("畫面顯示"))
        display_row.addWidget(self.display_mode, 1)
        layout.addLayout(display_row)
        layout.addWidget(self.display_summary)
        layout.addLayout(buttons)
        layout.addWidget(self.step_title)
        layout.addWidget(self.instruction)
        layout.addWidget(self.tracking_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.condition_label)
        layout.addWidget(self._section("工序進度"))
        layout.addWidget(self.step_list, 2)
        layout.addWidget(self.cycle_label)
        layout.addWidget(self.stats_label)
        layout.addWidget(self._section("事件紀錄"))
        layout.addWidget(self.log, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1000, 420])
        root = QHBoxLayout(self)
        root.addWidget(splitter)
        self.set_running(False)
        self._update_stats()

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight:bold; margin-top:6px")
        return label

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
        self._preview_locator = None
        self._locator_error = ""
        if self.sop.workpiece:
            try:
                self._preview_locator = WorkpieceLocator(self.sop.workpiece)
            except ValueError as exc:
                self._locator_error = str(exc)
        self.step_list.clear()
        for index, step in enumerate(self.sop.steps, 1):
            self.step_list.addItem(QListWidgetItem(f"○  {index}. {step.name}"))
        self.video.set_rois(display_rois(self.sop.rois, self._transform))
        self._show_idle()
        self._update_start_availability()

    def set_running(self, running: bool):
        if running != self._running:
            self._transform = None
            self._tracking_misses = 0
            if self._preview_locator:
                self._preview_locator.reset()
        self._running = running
        self.start_button.setText("■ 停止作業" if running else "▶ 開始作業")
        self.start_button.setStyleSheet(
            f"background:{'#c62828' if running else '#2e7d32'}; color:white; font-size:15px; font-weight:bold")
        self.confirm_button.setEnabled(running)
        self.restart_button.setEnabled(running)
        self._update_start_availability()
        if not running:
            self._show_idle()

    def on_packet(self, packet: FramePacket):
        self._last_packet = packet
        if self._running:
            # 正式作業只用引擎同一幀的定位結果，預覽不參與完成判定。
            self._transform = packet.result.workpiece_transform
            status = packet.result.tracking_status
        elif self._preview_locator:
            self._transform, status = self._preview_locator.locate(packet.frame)
        else:
            self._transform = None
            status = self._locator_error or "尚未設定參考工件"
        if self._transform is not None:
            self._last_transform = self._transform.copy()
            self._tracking_misses = 0
        elif self.sop and self.sop.workpiece:
            self._tracking_misses += 1
        self._update_start_availability()
        self._tracking_status = status
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
        if self._preview_locator:
            self._preview_locator.reset()
        self.video.clear_frame(text)
        self.video.set_rois([])
        self.tracking_label.setText(text)
        self._tracking_status = text
        self._update_start_availability()

    def _update_start_availability(self):
        if self._running:
            self.start_button.setEnabled(True)
            self.start_button.setText("■ 停止作業")
            self.start_button.setStyleSheet("background:#c62828; color:white; font-size:15px; font-weight:bold")
        else:
            self.start_button.setEnabled(True)
            self.start_button.setText("▶ 開始作業")
            self.start_button.setStyleSheet("background:#2e7d32; color:white; font-size:15px; font-weight:bold")

    @staticmethod
    def _step_sets(step):
        conditions = step.conditions + step.forbidden
        return {c.roi for c in conditions if c.roi}, {c.label for c in conditions if c.label}

    def _step_context(self, snapshot=None):
        if not self.sop:
            return "等待 SOP", set(), set()
        if snapshot and snapshot.phase == Phase.CYCLE_DONE:
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
            return packet.display
        _context, _rois, labels = self._step_context(snapshot)
        detections = [d for d in packet.result.detections if d.label in labels]
        return draw_detections(packet.frame, detections, packet.classes, packet.min_score, packet.show_masks)

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

    def _on_display_mode_changed(self, _index):
        self._refresh_regions(self._tracking_status, self._snapshot)
        if self._last_packet is not None:
            self.video.set_frame(self._monitor_frame(self._last_packet, self._snapshot))

    def on_events(self, events: list[EngineEvent]):
        for event in events:
            if event.kind == "step_started":
                continue
            color = {"alarm": "#c62828", "cycle_aborted": "#ef6c00"}.get(event.kind, "#263238")
            if event.kind == "cycle_completed":
                color = "#2e7d32" if event.data.get("result") == "OK" else "#ef6c00"
                self._stats[event.data.get("result", "NG")] += 1
            elif event.kind == "cycle_aborted":
                self._stats["中止"] += 1
            item = QListWidgetItem(f"{datetime.now():%H:%M:%S}  {event.message}")
            item.setForeground(QColor(color))
            item.setToolTip(event.message)
            self.log.insertItem(0, item)
            if event.kind == "alarm":
                QApplication.beep()
        while self.log.count() > 300:
            self.log.takeItem(self.log.count() - 1)
        self._update_stats()

    # ---- 繪製 ------------------------------------------------------------------
    def _show_idle(self):
        self.video.set_banner("待機中 · 按「開始作業」開始監控", "#546e7a")
        self.step_title.setText("待機中")
        self.instruction.setText(f"SOP：{self.sop.name}（共 {len(self.sop.steps)} 道工序）" if self.sop else "")
        self.progress.setValue(0)
        self.condition_label.setText("")
        self.cycle_label.setText("")
        self._refresh_regions()

    def _render_steps(self, snapshot: EngineSnapshot):
        for index, (step, runtime) in enumerate(zip(self.sop.steps, snapshot.steps)):
            if runtime.status == StepStatus.ACTIVE:
                text = f"▶  {index + 1}. {step.name}   {snapshot.now - runtime.started_at:.1f}s"
            elif runtime.status == StepStatus.DONE:
                manual = "（手動）" if runtime.manual else ""
                text = f"✔  {index + 1}. {step.name}   {runtime.ended_at - runtime.started_at:.1f}s{manual}"
            elif runtime.status == StepStatus.SKIPPED:
                text = f"⏭  {index + 1}. {step.name}   已跳過"
            else:
                text = f"○  {index + 1}. {step.name}"
            if runtime.has_alarm:
                text += "   ⚠"
            item = self.step_list.item(index)
            if item is not None and item.text() != text:
                item.setText(text)
                item.setForeground(QColor("#c62828" if runtime.has_alarm else STATUS_COLORS[runtime.status]))

        elapsed = snapshot.now - snapshot.cycle_started_at if snapshot.cycle_started_at is not None else 0
        state = "OK" if snapshot.cycle_ok else "NG（有警報或跳過）"
        self.cycle_label.setText(f"第 {snapshot.cycle} 輪 · 已進行 {elapsed:.0f} 秒 · 目前判定 {state}")

    def _render_running(self, snapshot: EngineSnapshot):
        index = snapshot.current_index
        step, runtime = self.sop.steps[index], snapshot.steps[index]
        title = f"工序 {index + 1} / {len(self.sop.steps)}：{step.name}"
        self.step_title.setText(title)
        self.instruction.setText(step.instruction)
        self.progress.setFormat("條件累積 %p%")
        self.progress.setValue(round(snapshot.progress * 100))

        if runtime.violation_alarmed:
            banner_color = "#c62828"
        elif runtime.timeout_alarmed:
            banner_color = "#ef6c00"
        else:
            banner_color = "#1565c0"
        self.video.set_banner(f"▶ {title}", banner_color)

        lines = []
        if not step.conditions:
            lines.append("<span style='color:#ef6c00'>此工序沒有自動條件，完成後請按「手動確認」</span>")
        for result in snapshot.results:
            if result.error:
                lines.append(f"<span style='color:#ef6c00'>⏸ 暫停判定：{result.error}</span>")
            else:
                mark, color = ("✓", "#2e7d32") if result.met else ("✗", "#c62828")
                lines.append(f"<span style='color:{color}'>{mark} {result.condition.describe()}"
                             f"　（{result.detail or f'目前 {result.count}'}）</span>")
        for result in snapshot.forbidden_results:
            if result.met:
                lines.append(f"<span style='color:#c62828'>⚠ 違規：{result.condition.describe()}</span>")
        if step.timeout_sec > 0:
            remaining = step.timeout_sec - (snapshot.now - runtime.started_at)
            lines.append(f"<span style='color:#607d8b'>剩餘時間 {max(0, remaining):.0f} 秒</span>")
        if any(result.error for result in snapshot.results):
            self.progress.setFormat("判定暫停")
        self.condition_label.setText("<br>".join(lines))
        self._refresh_regions(self._tracking_status, snapshot)

    def _render_cycle_done(self, snapshot: EngineSnapshot):
        ok = snapshot.cycle_ok
        result = "OK" if ok else "NG"
        self.video.set_banner(f"{'✅' if ok else '⚠'} 第 {snapshot.cycle} 輪完成：{result}",
                              "#2e7d32" if ok else "#ef6c00")
        self.step_title.setText(f"本輪完成：{result}")
        cycle = self.sop.cycle
        if cycle.reset_conditions:
            self.instruction.setText("等待重置條件成立後開始下一輪")
            self.progress.setFormat("重置條件 %p%")
            self.progress.setValue(round(snapshot.progress * 100))
            lines = [f"{'✓' if r.met else '✗'} {r.condition.describe()}" for r in snapshot.results]
            self.condition_label.setText("<br>".join(lines))
        else:
            remaining = cycle.auto_restart_sec - (snapshot.now - snapshot.cycle_ended_at)
            self.instruction.setText(f"{max(0, remaining):.0f} 秒後開始下一輪")
            self.progress.setFormat("")
            self.progress.setValue(0)
            self.condition_label.setText("")
        self._refresh_regions(self._tracking_status, snapshot)

    def _update_stats(self):
        total = sum(self._stats.values())
        self.stats_label.setText(f"本次開啟以來：共 {total} 輪 · OK {self._stats['OK']} · "
                                 f"NG {self._stats['NG']} · 中止 {self._stats['中止']}")

    def _on_start_clicked(self):
        (self.stop_requested if self._running else self.start_requested).emit()
