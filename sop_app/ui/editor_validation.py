"""Read-only editor trial adapter; decisions come exclusively from SOPEngine.

No camera, model, recorder or pipeline commands are issued here. Display metrics
read the engine's existing window; no second completion algorithm is introduced.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import replace

from ..conditions import evaluate_conditions
from ..detection import FrameResult
from ..engine import EngineEvent, Phase, SOPEngine
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
        self._last_alarm: EngineEvent | None = None
        self._last_display_alarm: EngineEvent | None = None
        self._last_generic_alarm: EngineEvent | None = None
        self._notice_signature: tuple | None = None
        self._notice_scope: tuple[int, int] | None = None
        self._notice_recorded_at: float | None = None
        self._pending_notice: EngineEvent | None = None
        self._stopped = False

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
        self._clear_alert()
        self._stopped = False
        self._accept_events(self.engine.start(now))
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
        self._accept_events(events, frame)
        self._update_violation_notice(events)
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

    def _accept_events(self, events, frame: FrameResult | None = None):
        for event in events:
            if event.kind == "cycle_started":
                self._clear_alert()
            elif event.kind == "step_started":
                self._reset_notice_log()
            elif event.kind == "alarm":
                if event.data.get("alarm") == "violation":
                    event = replace(event, data={**event.data,
                                                 "regions": self._alarm_regions(event, frame)})
                    self._notice_scope = (event.cycle, event.step_index)
                    self._notice_recorded_at = event.time
                else:
                    self._last_generic_alarm = event
                self._last_alarm = event
                self._last_display_alarm = event
            self.events.append(event)
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
        events = self.engine.confirm_step(self.last_timestamp or 0.0)
        self._accept_events(events, self.last_frame)
        self._update_violation_notice(events)

    def stop(self, reason="已停止試跑"):
        if self.running and self.engine is not None:
            self.events.extend(self.engine.stop(self.last_timestamp or 0.0))
        self.running = False
        self._stopped = True
        self._notice_signature = None
        self._pending_notice = None
        if not self.stale:
            self.message = reason

    def invalidate(self, reason="設定已變更，請重新試跑"):
        if self.engine is None:
            if self.message.startswith("無法開始"):
                self.message = "尚未開始試跑"
            return
        self.stop()
        self.stale, self.completed, self.message = True, False, reason
        self._clear_alert()

    def _clear_alert(self):
        self._last_alarm = None
        self._last_display_alarm = None
        self._last_generic_alarm = None
        self._reset_notice_log()

    def _reset_notice_log(self):
        self._notice_signature = None
        self._notice_scope = None
        self._notice_recorded_at = None
        self._pending_notice = None

    def _alarm_regions(self, event: EngineEvent, frame: FrameResult | None) -> tuple[str, ...]:
        """Keep alarm metadata even if the same update completed/advanced a step."""
        if self.engine is None or frame is None:
            return ()
        snapshot = self.engine.snapshot()
        if snapshot.phase == Phase.RUNNING and snapshot.current_index == event.step_index:
            results = snapshot.forbidden_results
        else:
            # The engine clears the old results on completion. This is only
            # metadata for an alarm already emitted by that engine, not another
            # decision or time window.
            step = self.engine.sop.steps[event.step_index]
            _, results = evaluate_conditions(step.forbidden, frame, self.engine.rois, step.forbidden_mode)
        return tuple(dict.fromkeys(result.condition.roi for result in results
                                   if result.met and not result.error and result.condition.roi))

    def _current_violation(self):
        if not self.running or self.stale or self.engine is None or self.engine.phase != Phase.RUNNING:
            return None
        engine = self.engine
        snapshot = engine.snapshot()
        results = snapshot.forbidden_results
        # Use the actual engine window. In any mode C2 -> C3 shares the same
        # group window; neither the old alarm latch nor a second per-region
        # timer should hide the current offending condition.
        if not results or not engine._forbidden_window.satisfied(snapshot.now):
            return None
        matched = [(index, result) for index, result in enumerate(results) if result.met and not result.error]
        if not matched:
            return None
        step = engine.sop.steps[snapshot.current_index]
        descriptions = "、".join(result.condition.describe() for _index, result in matched)
        regions = tuple(dict.fromkeys(result.condition.roi for _index, result in matched if result.condition.roi))
        signature = (snapshot.cycle, snapshot.current_index, tuple(index for index, _result in matched))
        original = self.original_indices[snapshot.current_index] + 1
        text = f"違規警示：工序 {original}「{step.name}」—{descriptions}（本輪已NG）"
        return text, regions, signature

    def _update_violation_notice(self, events):
        current = self._current_violation()
        if current is None:
            self._notice_signature = None
            self._pending_notice = None
            return
        text, regions, signature = current
        scope = (self.engine.cycle, self.engine.index)
        if scope != self._notice_scope:
            self._reset_notice_log()
            self._notice_scope = scope
        now = self.last_timestamp or 0.0
        detail = text.removeprefix("違規警示：").removesuffix("（本輪已NG）")
        notice = EngineEvent("trial_notice", now, self.engine.cycle, self.engine.index,
                             self.engine.sop.steps[self.engine.index].name,
                             "試跑提示（非正式）：" + detail,
                             {"notice": "violation", "regions": regions, "summary": detail})
        # Display/history follows every current frame, including changes that
        # have not been written to the log because its interval has not elapsed.
        self._last_display_alarm = notice
        first_alarm = any(event.kind == "alarm" and event.data.get("alarm") == "violation"
                          and event.cycle == self.engine.cycle and event.step_index == self.engine.index
                          for event in events)
        if first_alarm:
            self._pending_notice = None
        else:
            if signature != self._notice_signature or self._pending_notice is not None:
                self._pending_notice = notice  # Replace pending changes with the latest current condition.
            interval = self.engine.sop.steps[self.engine.index].trial_notice_interval_sec
            # Only log writes are rate limited, using source-frame time. The
            # actual engine alarm and the live display remain immediate.
            if (self._pending_notice is not None and
                    (self._notice_recorded_at is None or now - self._notice_recorded_at >= interval)):
                self.events.append(notice)
                self._last_alarm = notice
                self._notice_recorded_at = now
                self._pending_notice = None
        self._notice_signature = signature

    def alert(self) -> dict[str, str | tuple[str, ...] | bool]:
        """Current engine-confirmed warning, or clearly labelled display history.

        Reading this method never creates events. Only active violations return
        regions, so stopped/stale trials cannot leave live warning outlines.
        """
        blank = dict(text="", regions=(), active=False)
        if self.engine is None or self.stale:
            return blank
        current = self._current_violation()
        if current is not None:
            text, regions, _signature = current
            if self._last_generic_alarm is not None:
                text += "；另有警報：" + self._last_generic_alarm.message
            return dict(text=text, regions=regions, active=True)
        snapshot = self.engine.snapshot()
        unavailable = any(result.error for result in snapshot.results + snapshot.forbidden_results)
        last_display = self._last_display_alarm or self._last_alarm
        if last_display is None:
            return {**blank, "text": "辨識／位置無法確認，暫停相關條件判斷" if unavailable and self.running else ""}
        prefix = "已停止試跑；" if self._stopped else "單步驗證完成；" if self.completed else ""
        summary = last_display.data.get("summary", last_display.message)
        text = f"{prefix}本輪已NG；最後警報：{summary}"
        # A continuously displayed violation must not hide a later timeout or
        # sequence/too-fast alarm, even when its own notice is throttled.
        if self._last_generic_alarm is not None and self._last_generic_alarm is not last_display:
            text += "；另有警報：" + self._last_generic_alarm.message
        if unavailable and self.running:
            text += "；辨識／位置無法確認，暫停相關條件判斷"
        return dict(text=text, regions=(), active=False)

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
