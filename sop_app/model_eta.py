"""Local estimates from successful loads, measured in the inference process."""
from __future__ import annotations

import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import re
import sys
import time


def model_key(path):
    model = Path(path).resolve()
    stat = model.stat()
    runtime = []
    for package in ("torch", "ultralytics", "numpy", "opencv-python-headless"):
        try:
            runtime.append(version(package))
        except PackageNotFoundError:
            runtime.append("unknown")
    return hashlib.sha256(repr((str(model), stat.st_size, stat.st_mtime_ns,
                                sys.version, runtime)).encode()).hexdigest()


def stage_key(stage):
    # Runtime import reports include a measured duration in their display text.
    return re.sub(r"（[^）]*秒）", "", str(stage))


def duration(seconds):
    seconds = max(1, math.ceil(seconds))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes} 分 {seconds} 秒" if minutes else f"{seconds} 秒"


class ModelLoadProgress:
    """Count finished work stages, rather than pretending time is work done."""

    STAGES = ("載入 AI 執行環境", "讀取模型資訊", "載入推論套件", "選擇運算裝置",
              "讀取與驗證模型權重", "建立辨識模型", "預熱模型，準備首次辨識")

    def __init__(self):
        self.stages = self.STAGES
        self.completed = 0
        self.failed = False

    def observe(self, event):
        stage = stage_key(event)
        self.stages = self.STAGES[1:] if event.mode == "reload" else self.STAGES
        if stage == "載入完成":
            self.completed = len(self.stages)
        else:
            if stage == "AI 執行環境已載入":
                stage = "讀取模型資訊"
            elif stage == "建立模型並載入運算裝置":
                stage = "建立辨識模型"
            if stage in self.stages:
                self.completed = max(self.completed, self.stages.index(stage))

    @property
    def total(self):
        return len(self.stages)

    @property
    def percent(self):
        return round(100 * self.completed / self.total)

    def tooltip(self):
        lines = ["百分比依已完成階段計算；各階段耗時不同。"]
        for index, stage in enumerate(self.stages):
            state = "已完成" if index < self.completed else (
                "失敗" if self.failed and index == self.completed else
                "進行中" if index == self.completed else "等待")
            lines.append(f"{index + 1}. {stage}：{state}")
        return "\n".join(lines)


class ModelETA:
    def __init__(self, history_path, key):
        self.path = Path(history_path)
        self.key = key
        self.events = []
        self.deadline = None
        self.expired = False
        self.finished = False
        self.history = {}
        try:
            history = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(history, dict):
                self.history = history
        except (OSError, ValueError):
            pass

    def observe(self, event):
        stage = stage_key(event)
        measured = event.measured_at
        if stage == "載入完成":
            self.finished = True
            # Only successful completion trains the estimate. Delivery delays
            # from queued preloading never count as model work.
            if not self.events:
                return
            mode = event.mode
            samples = self.history.get(self.key, {})
            if not isinstance(samples, dict):
                samples = {}
            runs = samples.get(mode, [])
            if not isinstance(runs, list):
                runs = []
            run = {name: measured - stamp for name, stamp, _ in self.events
                   if 0 <= measured - stamp < 3600}
            samples[mode] = (runs + [run])[-12:]
            self.history[self.key] = samples
            # Keep a bounded local history and write atomically.
            self.history = dict(list(self.history.items())[-32:])
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.history, ensure_ascii=False), encoding="utf-8")
                temporary.replace(self.path)
            except OSError:
                pass  # A read-only history must never prevent model loading.
            return
        self.events.append((stage, measured, event.mode))

    def remaining_text(self, now=None):
        if self.finished:
            return "剩餘 00:00"
        now = time.perf_counter() if now is None else now
        if self.deadline is not None and now >= self.deadline:
            self.expired = True
        if self.expired:
            return "載入超出預估時間"
        if not self.events:
            return "剩餘時間估算中"
        stage, stamp, mode = self.events[-1]
        samples = self.history.get(self.key, {})
        runs = samples.get(mode, []) if isinstance(samples, dict) else []
        values = [run.get(stage) for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []
        values = [v for v in values if isinstance(v, (float, int)) and not isinstance(v, bool)
                  and math.isfinite(v) and 0 < v < 3600]
        if not values:
            if self.deadline is None:
                return "剩餘時間估算中"
        else:
            candidate = stamp + max(values)
            # Actual milestones can bring completion closer, never push the
            # displayed deadline later. An overrun stays explicit until ready.
            self.deadline = candidate if self.deadline is None else min(self.deadline, candidate)
        remaining = self.deadline - now
        if remaining <= 0:
            self.expired = True
            return "載入超出預估時間"
        minutes, seconds = divmod(math.ceil(remaining), 60)
        return f"預估剩餘 {minutes:02d}:{seconds:02d}"
