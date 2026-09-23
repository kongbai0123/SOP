"""SOP 判定引擎：嚴格依序，只評估目前工序；以滑動時間窗抗辨識閃爍。

純邏輯、不依賴 Qt 或模型，時間由呼叫端傳入，因此可以用假資料做單元測試。
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass, field, replace
from enum import Enum

from .conditions import ConditionResult, evaluate_conditions
from .detection import FrameResult
from .sop_schema import SOPDefinition


class Phase(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    CYCLE_DONE = "cycle_done"


class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    SKIPPED = "skipped"


@dataclass
class EngineEvent:
    kind: str      # cycle_started / step_started / step_completed / alarm / cycle_completed / cycle_aborted
    time: float
    cycle: int
    step_index: int = -1
    step_name: str = ""
    message: str = ""
    data: dict = field(default_factory=dict)


class SlidingWindow:
    """最近 span 秒內成立的幀比例 ≥ ratio，且最新一幀成立，才算滿足。"""

    def __init__(self, span: float = 1.0, ratio: float = 0.8):
        self.span, self.ratio = span, ratio
        self._samples: deque[tuple[float, bool]] = deque()
        self._started = 0.0

    def reset(self, now: float, span: float | None = None, ratio: float | None = None):
        if span is not None:
            self.span = max(0.0, span)
        if ratio is not None:
            self.ratio = min(max(ratio, 0.0), 1.0)
        self._samples.clear()
        self._started = now

    def push(self, now: float, value: bool):
        self._samples.append((now, bool(value)))
        cutoff = now - self.span
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _true_ratio(self) -> float:
        return sum(v for _, v in self._samples) / len(self._samples) if self._samples else 0.0

    def satisfied(self, now: float) -> bool:
        if not self._samples or not self._samples[-1][1]:
            return False
        if now - self._started < self.span:        # 觀察時間還不夠
            return False
        return self._true_ratio() >= self.ratio

    def progress(self, now: float) -> float:
        if not self._samples or not self._samples[-1][1]:
            return 0.0
        coverage = 1.0 if self.span <= 0 else min(1.0, (now - self._started) / self.span)
        fill = 1.0 if self.ratio <= 0 else min(1.0, self._true_ratio() / self.ratio)
        return coverage * fill


@dataclass
class StepRuntime:
    status: StepStatus = StepStatus.PENDING
    started_at: float | None = None
    ended_at: float | None = None
    manual: bool = False
    timeout_alarmed: bool = False
    violation_alarmed: bool = False

    @property
    def has_alarm(self) -> bool:
        return self.timeout_alarmed or self.violation_alarmed


@dataclass
class EngineSnapshot:
    phase: Phase
    cycle: int
    current_index: int
    now: float
    cycle_ok: bool
    cycle_started_at: float | None
    cycle_ended_at: float | None
    steps: list[StepRuntime]
    results: list[ConditionResult]
    forbidden_results: list[ConditionResult]
    progress: float


class SOPEngine:
    def __init__(self, sop: SOPDefinition):
        self.sop = copy.deepcopy(sop)
        self.rois = self.sop.roi_map()
        self.phase = Phase.IDLE
        self.cycle = 0
        self.index = 0
        self.cycle_ok = True
        self.cycle_started_at: float | None = None
        self.cycle_ended_at: float | None = None
        self.runtimes = [StepRuntime() for _ in self.sop.steps]
        self._window = SlidingWindow()
        self._forbidden_window = SlidingWindow()
        self._results: list[ConditionResult] = []
        self._forbidden_results: list[ConditionResult] = []
        self._now = 0.0
        self._position_blocked = False
        self._completion_invalid = False
        self._forbidden_invalid = False

    # ---- 控制指令 -------------------------------------------------------------
    def start(self, now: float) -> list[EngineEvent]:
        if not self.sop.steps:
            raise ValueError("SOP 沒有任何工序")
        self._now = now
        return self._begin_cycle(now)

    def stop(self, now: float) -> list[EngineEvent]:
        events = self._abort(now, "停止作業")
        self.phase = Phase.IDLE
        return events

    def restart_cycle(self, now: float) -> list[EngineEvent]:
        if self.phase == Phase.IDLE:
            return []
        return self._abort(now, "重新開始本輪") + self._begin_cycle(now)

    def confirm_step(self, now: float) -> list[EngineEvent]:
        """手動確認／跳過目前工序。有自動條件的工序被跳過時，本輪判定為 NG。"""
        if self.phase != Phase.RUNNING:
            return []
        step = self.sop.steps[self.index]
        if step.conditions:
            self.cycle_ok = False
            return self._finish_step(now, StepStatus.SKIPPED, manual=True)
        return self._finish_step(now, StepStatus.DONE, manual=True)

    # ---- 每幀更新 -------------------------------------------------------------
    def update(self, frame: FrameResult) -> list[EngineEvent]:
        now = self._now = frame.timestamp
        events: list[EngineEvent] = []

        if self.phase == Phase.RUNNING:
            step, runtime = self.sop.steps[self.index], self.runtimes[self.index]
            met, self._results = evaluate_conditions(step.conditions, frame, self.rois)
            invalid = any(r.error for r in self._results)
            if invalid or self._completion_invalid:
                self._window.reset(now)
            self._completion_invalid = invalid
            self._window.push(now, met)

            if step.forbidden:
                forbidden_met, self._forbidden_results = evaluate_conditions(step.forbidden, frame, self.rois)
                invalid = any(r.error for r in self._forbidden_results)
                if invalid or self._forbidden_invalid:
                    self._forbidden_window.reset(now)
                self._forbidden_invalid = invalid
                self._forbidden_window.push(now, forbidden_met)
                if not runtime.violation_alarmed and self._forbidden_window.satisfied(now):
                    runtime.violation_alarmed = True
                    self.cycle_ok = False
                    detail = "、".join(r.condition.describe() for r in self._forbidden_results)
                    events.append(self._event("alarm", now, f"違規：進行「{step.name}」時偵測到 {detail}",
                                              alarm="violation"))

            blocked = any(r.error for r in self._results + self._forbidden_results)
            if blocked != self._position_blocked:
                self._position_blocked = blocked
                events.append(self._event("position_lost" if blocked else "position_recovered", now,
                                          "作業位置或辨識無法確認，相關條件停止累積" if blocked
                                          else "作業位置與辨識恢復，重新累積條件"))

            if step.timeout_sec > 0 and not runtime.timeout_alarmed and now - runtime.started_at >= step.timeout_sec:
                runtime.timeout_alarmed = True
                self.cycle_ok = False
                events.append(self._event("alarm", now, f"超時：「{step.name}」超過 {step.timeout_sec:g} 秒未完成",
                                          alarm="timeout"))

            if step.conditions and self._window.satisfied(now):
                events += self._finish_step(now, StepStatus.DONE)

        elif self.phase == Phase.CYCLE_DONE:
            cycle = self.sop.cycle
            if cycle.reset_conditions:
                met, self._results = evaluate_conditions(cycle.reset_conditions, frame, self.rois)
                invalid = any(r.error for r in self._results)
                if invalid or self._completion_invalid:
                    self._window.reset(now)
                self._completion_invalid = invalid
                self._window.push(now, met)
                if self._window.satisfied(now):
                    events += self._begin_cycle(now)
            elif now - self.cycle_ended_at >= cycle.auto_restart_sec:
                events += self._begin_cycle(now)

        return events

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(
            phase=self.phase, cycle=self.cycle, current_index=self.index, now=self._now,
            cycle_ok=self.cycle_ok, cycle_started_at=self.cycle_started_at, cycle_ended_at=self.cycle_ended_at,
            steps=[replace(r) for r in self.runtimes], results=list(self._results),
            forbidden_results=list(self._forbidden_results),
            progress=self._window.progress(self._now) if self.phase != Phase.IDLE else 0.0,
        )

    # ---- 內部 -----------------------------------------------------------------
    def _event(self, kind: str, now: float, message: str = "", index: int | None = None, **data) -> EngineEvent:
        index = self.index if index is None else index
        name = self.sop.steps[index].name if 0 <= index < len(self.sop.steps) else ""
        return EngineEvent(kind, now, self.cycle, index, name, message, data)

    def _begin_cycle(self, now: float) -> list[EngineEvent]:
        self._position_blocked = False
        self.cycle += 1
        self.phase = Phase.RUNNING
        self.index = 0
        self.cycle_ok = True
        self.cycle_started_at, self.cycle_ended_at = now, None
        self.runtimes = [StepRuntime() for _ in self.sop.steps]
        return [self._event("cycle_started", now, f"第 {self.cycle} 輪開始", index=-1)] + self._activate(0, now)

    def _activate(self, index: int, now: float) -> list[EngineEvent]:
        self._completion_invalid = self._forbidden_invalid = False
        step = self.sop.steps[index]
        self.index = index
        self.runtimes[index].status = StepStatus.ACTIVE
        self.runtimes[index].started_at = now
        self._window.reset(now, step.hold_sec, step.ratio)
        self._forbidden_window.reset(now, step.hold_sec, step.ratio)
        self._results, self._forbidden_results = [], []
        return [self._event("step_started", now, f"開始：{step.name}")]

    def _finish_step(self, now: float, status: StepStatus, manual: bool = False) -> list[EngineEvent]:
        runtime = self.runtimes[self.index]
        runtime.status, runtime.ended_at, runtime.manual = status, now, manual
        duration = now - runtime.started_at
        verb = "跳過" if status == StepStatus.SKIPPED else ("手動確認" if manual else "完成")
        events = [self._event("step_completed", now, f"{verb}：{self.sop.steps[self.index].name}（{duration:.1f}s）",
                              status=status.value, manual=manual, duration=duration, alarm=runtime.has_alarm)]
        if self.index + 1 < len(self.sop.steps):
            return events + self._activate(self.index + 1, now)

        self.phase = Phase.CYCLE_DONE
        self.cycle_ended_at = now
        cycle = self.sop.cycle
        self._window.reset(now, cycle.reset_hold_sec, 0.8)
        self._results, self._forbidden_results = [], []
        result = "OK" if self.cycle_ok else "NG"
        return events + [self._event("cycle_completed", now, f"第 {self.cycle} 輪完成：{result}", index=-1,
                                     result=result, duration=now - self.cycle_started_at)]

    def _abort(self, now: float, reason: str) -> list[EngineEvent]:
        if self.phase != Phase.RUNNING:
            return []
        return [self._event("cycle_aborted", now, f"第 {self.cycle} 輪中止：{reason}", index=-1,
                            duration=now - self.cycle_started_at)]
