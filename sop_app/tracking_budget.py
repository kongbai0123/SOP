"""Measured, bounded scheduling of expensive recovery searches (not pose quality)."""
from __future__ import annotations

import math


class TrackingBudget:
    def __init__(self, target_fps=30., max_interval=.25):
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.frame_ms = 1000. / target_fps
        self.max_interval = max_interval
        self.costs = {}
        self.last_runs = {}
        self.pressure = 1.
        self.samples = 0
        self._high = self._low = 0
        self._overloaded = False

    def observe_frame(self, elapsed_ms):
        # Require sustained pressure; one slow search must not switch modes.
        self.samples += 1
        ratio = max(0., elapsed_ms) / self.frame_ms
        self.pressure = .9 * self.pressure + .1 * ratio
        self._high = self._high + 1 if self.pressure > 1.15 else 0
        self._low = self._low + 1 if self.pressure < .8 else 0
        if self._high >= 8:
            self._overloaded = True
        elif self._low >= 20:
            self._overloaded = False

    @property
    def overloaded(self):
        return self._overloaded

    def allow(self, stage, now, elapsed_ms):
        last = self.last_runs.get(stage)
        # Probe unknown costs immediately; force a bounded retry even when busy.
        if last is None or now - last >= self.max_interval:
            return True
        predicted = self.costs.get(stage, 0.)
        interval = min(self.max_interval, max(.033, predicted / 1000. * (2 if self.overloaded else 1)))
        return now - last >= interval and elapsed_ms + predicted <= self.frame_ms

    def record(self, stage, now, elapsed_ms):
        self.last_runs[stage] = now
        self.costs[stage] = .8 * self.costs.get(stage, elapsed_ms) + .2 * elapsed_ms
