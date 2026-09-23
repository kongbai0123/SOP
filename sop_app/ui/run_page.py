"""執行監控頁：給作業員看的即時畫面、目前工序、條件狀態、警報。"""
from __future__ import annotations

import copy
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..engine import EngineEvent, EngineSnapshot, Phase, StepStatus
from ..pipeline import FramePacket
from ..sop_schema import SOPDefinition
from ..tracking import display_rois
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
        self.start_button.clicked.connect(self._on_start_clicked)
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
        self.step_list.clear()
        for index, step in enumerate(self.sop.steps, 1):
            self.step_list.addItem(QListWidgetItem(f"○  {index}. {step.name}"))
        self.video.set_rois(display_rois(self.sop.rois, self._transform))
        self._show_idle()

    def set_running(self, running: bool):
        self._running = running
        self.start_button.setText("■ 停止作業" if running else "▶ 開始作業")
        self.start_button.setStyleSheet(
            f"background:{'#c62828' if running else '#2e7d32'}; color:white; font-size:15px; font-weight:bold")
        self.confirm_button.setEnabled(running)
        self.restart_button.setEnabled(running)
        if not running:
            self._show_idle()

    def on_packet(self, packet: FramePacket):
        self.video.set_frame(packet.display)
        self._transform = packet.result.workpiece_transform
        self.tracking_label.setText(packet.result.tracking_status if self.sop and self.sop.workpiece else "")
        snapshot = packet.snapshot
        if self.sop is None or snapshot is None or snapshot.phase == Phase.IDLE or not self._running:
            return
        self._render_steps(snapshot)
        if snapshot.phase == Phase.RUNNING:
            self._render_running(snapshot)
        else:
            self._render_cycle_done(snapshot)

    def clear_video(self, text: str):
        self.video.clear_frame(text)

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
        self.tracking_label.setText("")
        if self.sop:
            self.video.set_rois(display_rois(self.sop.rois, None))

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
            mark, color = ("✓", "#2e7d32") if result.met else ("✗", "#c62828")
            lines.append(f"<span style='color:{color}'>{mark} {result.condition.describe()}"
                         f"　（{result.error or f'目前 {result.count}'}）</span>")
        for result in snapshot.forbidden_results:
            if result.met:
                lines.append(f"<span style='color:#c62828'>⚠ 違規：{result.condition.describe()}</span>")
        if step.timeout_sec > 0:
            remaining = step.timeout_sec - (snapshot.now - runtime.started_at)
            lines.append(f"<span style='color:#607d8b'>剩餘時間 {max(0, remaining):.0f} 秒</span>")
        self.condition_label.setText("<br>".join(lines))
        self.video.set_rois(display_rois(self.sop.rois, self._transform), {c.roi for c in step.conditions if c.roi})

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
        self.video.set_rois(display_rois(self.sop.rois, self._transform))

    def _update_stats(self):
        total = sum(self._stats.values())
        self.stats_label.setText(f"本次開啟以來：共 {total} 輪 · OK {self._stats['OK']} · "
                                 f"NG {self._stats['NG']} · 中止 {self._stats['中止']}")

    def _on_start_clicked(self):
        (self.stop_requested if self._running else self.start_requested).emit()
