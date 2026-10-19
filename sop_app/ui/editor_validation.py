"""Read-only editor trial adapter; decisions come exclusively from SOPEngine.

No camera, model, recorder or pipeline commands are issued here. Display metrics
read the engine's existing window; no second completion algorithm is introduced.
"""
from __future__ import annotations

import copy
from collections import deque

from ..conditions import evaluate_conditions
from ..detection import FrameResult
from ..engine import Phase, SOPEngine
from ..sop_schema import SOPDefinition


class EditorValidation:
    def __init__(self):
        self.engine: SOPEngine | None = None
        self.running = False
        self.stale = False
        self.completed = False
        self.mode = "single"
        self.original_indices: list[int] = []
        self.events = deque(maxlen=60)
        self.message = "尚未開始試跑"
        self.last_timestamp: float | None = None
        self.last_frame: FrameResult | None = None

    def start(self, sop: SOPDefinition, mode: str, selected: int, now: float):
        if mode not in ("single", "flow"):
            raise ValueError("未知的驗證範圍")
        if not sop.steps or not 0 <= selected < len(sop.steps):
            raise ValueError("請先選擇有效工序")
        snapshot = copy.deepcopy(sop)
        self.original_indices = [selected] if mode == "single" else list(range(len(sop.steps)))
        if mode == "single":
            snapshot.steps = [snapshot.steps[selected]]
        self.engine = SOPEngine(snapshot)
        self.mode, self.running, self.stale, self.completed = mode, True, False, False
        self.last_timestamp, self.last_frame = now, None
        self.events.clear()
        self.events.extend(self.engine.start(now))
        self.message = "試跑中（不寫入正式作業紀錄）"

    def update(self, frame: FrameResult):
        if not self.running or self.engine is None:
            return
        if self.last_timestamp is not None and frame.timestamp < self.last_timestamp:
            self.invalidate("影像時間已重置，請重新試跑")
            return
        if self.last_timestamp is not None and frame.timestamp == self.last_timestamp:
            return  # The same/frozen frame must not accumulate another sample.
        self.last_timestamp = frame.timestamp
        self.last_frame = frame
        events = self.engine.update(frame)
        self._accept_events(events)
        if self.running:
            if self.engine.phase == Phase.CYCLE_DONE:
                self.message = ("本輪完成，等待重置／下一輪（沿用原引擎規則）" if self.engine.sop.cycle.enabled
                                else "本輪完成，已停用自動開始下一輪")
            elif any(result.error for result in self.engine.snapshot().results):
                self.message = "辨識／定位無法確認，條件停止累積"
            elif not self.engine.sop.steps[self.engine.index].conditions:
                self.message = "等待人工確認"
            else:
                self.message = "試跑中（不寫入正式作業紀錄）"

    def _accept_events(self, events):
        self.events.extend(events)
        if self.mode == "single" and any(event.kind == "cycle_completed" for event in events):
            self.running, self.completed = False, True
            result = "OK" if self.engine.cycle_ok else "NG"
            self.message = f"單步驗證完成：{result}（非正式作業紀錄）"
        elif any(event.kind == "cycle_completed" for event in events):
            self.message = ("本輪完成，等待重置／下一輪（沿用原引擎規則）" if self.engine.sop.cycle.enabled
                                else "本輪完成，已停用自動開始下一輪")

    def confirm(self):
        if not self.running or self.engine is None or self.engine.phase != Phase.RUNNING:
            return
        if self.engine.sop.steps[self.engine.index].conditions:
            return  # Editor confirmation never skips an automatic step.
        self._accept_events(self.engine.confirm_step(self.last_timestamp or 0.0))

    def stop(self, reason="已停止試跑"):
        if self.running and self.engine is not None:
            self.events.extend(self.engine.stop(self.last_timestamp or 0.0))
        self.running = False
        if not self.stale:
            self.message = reason

    def invalidate(self, reason="設定已變更，請重新試跑"):
        if self.engine is None:
            if self.message.startswith("無法開始"):
                self.message = "尚未開始試跑"
            return
        self.stop()
        self.stale, self.completed, self.message = True, False, reason

    def metrics(self) -> dict[str, str]:
        blank = dict(target="尚未擷取測試設定", count="—", elapsed="—", ratio="—")
        if self.engine is None:
            return blank
        engine = self.engine
        step = engine.sop.steps[engine.index]
        target = f"第 {engine.cycle} 輪 · 工序 {self.original_indices[engine.index] + 1}：{step.name}"
        if self.stale:
            return {**blank, "target": target + "（設定已過期）"}
        now = self.last_timestamp or 0.0
        if self.completed:
            _, results = evaluate_conditions(step.conditions, self.last_frame, engine.rois, step.completion_mode) if self.last_frame else (False, [])
            return dict(target=target, count=f"{sum(r.met for r in results)}/{len(results)}" if results else "人工確認",
                        elapsed="已完成", ratio=f"達標 ≥ {step.ratio:.0%}" if step.conditions else "不套用")
        if engine.phase == Phase.CYCLE_DONE and not engine.sop.cycle.enabled:
            return dict(target=f"第 {engine.cycle} 輪已完成 · 自動開始下一輪已停用",
                        count="不套用", elapsed="已完成", ratio="不套用")
        if engine.phase == Phase.CYCLE_DONE and not engine.sop.cycle.reset_conditions:
            delay = engine.sop.cycle.auto_restart_sec
            return dict(target=f"第 {engine.cycle} 輪已完成 · 等待下一輪", count="不套用",
                        elapsed=f"{max(0, now - engine.cycle_ended_at):.1f}/{delay:g} 秒", ratio="不套用")
        snapshot = engine.snapshot()
        window = engine._window  # Read-only diagnostics from the actual engine window.
        results = snapshot.results
        invalid = any(result.error for result in results)
        count = "無法判斷" if invalid else f"{sum(r.met for r in results)}/{len(results)}" if results else "—"
        if engine.phase == Phase.CYCLE_DONE:
            target = f"第 {engine.cycle} 輪已完成 · 驗證重置條件"
        elif not step.conditions:
            return dict(target=target, count="人工確認", elapsed="不套用", ratio="不套用")
        observed = min(window.span, max(0.0, now - window._started))
        return dict(target=target, count=count,
                    elapsed="—" if invalid else f"{observed:.1f}/{window.span:g} 秒",
                    ratio="—" if invalid or not results else f"{window._true_ratio():.0%} / ≥ {window.ratio:.0%}")
