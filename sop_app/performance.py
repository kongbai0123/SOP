"""Bounded, observational timings; never used as the SOP's clock or input."""
from __future__ import annotations

from collections import deque
import math
import threading


def observed_mapping(owner, name):
    """A missing/broken optional observer must never fail a business operation."""
    try:
        value = getattr(owner, name, {})
        return value.copy() if isinstance(value, dict) else {}
    except Exception:
        return {}


class PerformanceWindow:
    def __init__(self, capacity=300):
        self._samples = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def observe(self, stages):
        sample = {key: float(value) for key, value in stages.items()
                  if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0}
        with self._lock:
            self._samples.append(sample)

    def clear(self):
        with self._lock:
            self._samples.clear()

    def snapshot(self):
        with self._lock:
            samples = list(self._samples)
        stages = {}
        for key in {key for sample in samples for key in sample}:
            values = sorted(sample[key] for sample in samples if key in sample)
            def percentile(q):
                at = (len(values) - 1) * q
                low = int(at)
                return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (at - low)
            stages[key] = {"count": len(values), "p50": percentile(.50),
                           "p95": percentile(.95), "p99": percentile(.99), "max": values[-1]}
        return {"frames": len(samples), "capacity": self._samples.maxlen, "stages_ms": stages}
