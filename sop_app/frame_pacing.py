"""Inference-start rate limit, independent of capture and source-video time.

The caller supplies seconds from one monotonic clock (normally
``time.monotonic()``), checks ``remaining`` before reading a source frame, and
calls ``mark_started`` immediately before processing it. This class neither
sleeps nor buffers frames. Slow inference consumes the interval itself.
"""
from __future__ import annotations

import math
from numbers import Real


class InferencePacer:
    """A maximum start rate: 0 is unlimited; finite limits are 1..240 FPS.

    Every deadline is the actual previous start plus one period. There is no
    catch-up, accumulated credit, or batch after a late wake-up. Changing the
    limit recomputes the deadline from that same last start; ``reset`` allows
    the next frame immediately while preserving the selected limit.
    """

    def __init__(self, fps=0):
        self._fps = 0.
        self._last_started = None
        self._next_due = None
        self.set_fps(fps)

    @property
    def fps(self):
        return self._fps

    def set_fps(self, fps):
        if isinstance(fps, bool) or not isinstance(fps, Real):
            raise ValueError("Inference FPS must be a finite number: 0 or 1..240")
        try:
            value = float(fps)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Inference FPS must be a finite number: 0 or 1..240") from exc
        if not math.isfinite(value) or (value != 0 and not 1 <= value <= 240):
            raise ValueError("Inference FPS must be 0 (unlimited) or between 1 and 240")
        self._fps = value
        self._next_due = (self._last_started + 1. / value
                          if value and self._last_started is not None else None)

    def reset(self):
        self._last_started = None
        self._next_due = None

    def _check_clock(self, now):
        if isinstance(now, bool) or not isinstance(now, Real):
            raise ValueError("Inference pacing requires finite monotonic seconds")
        try:
            value = float(now)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Inference pacing requires finite monotonic seconds") from exc
        if not math.isfinite(value):
            raise ValueError("Inference pacing requires finite monotonic seconds")
        if self._last_started is not None and value < self._last_started:
            raise ValueError("Inference pacing clock moved backwards; use one monotonic clock")
        return value

    def remaining(self, now):
        now = self._check_clock(now)
        return max(0., self._next_due - now) if self._next_due is not None else 0.

    def mark_started(self, now):
        """Record an actual start after the caller has honored ``remaining``."""
        now = self._check_clock(now)
        self._last_started = now
        self._next_due = now + 1. / self._fps if self._fps else None
