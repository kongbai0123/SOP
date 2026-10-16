"""Read-only summaries of positioning interruptions in historical records."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PositionPause:
    cycle_id: int
    step_index: int
    started_at: str
    recovered_at: str
    duration: float | None

    @property
    def recovered(self) -> bool:
        return bool(self.recovered_at)


def _duration(start, end) -> float | None:
    """Source clocks measure elapsed time; database write times do not."""
    if start is None or end is None:
        return None
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return None
    return end - start if math.isfinite(start) and math.isfinite(end) and end >= start else None


def positioning_pauses(events: list[tuple]) -> list[PositionPause]:
    """Pair ordered events only within a contiguous cycle/step.

    Rows are (cycle, step, kind, source_time, created_at). Repeated lost messages
    belong to the open pause. A boundary or another lost event after the clock
    goes backwards never supplies a recovery that was not actually recorded.
    """
    pauses = []
    active = None

    def unfinished():
        if active is not None:
            cycle, step, _source, created = active
            pauses.append(PositionPause(cycle, step, created or "時間未記錄", "", None))

    for cycle, step, kind, source, created in events:
        if active is not None and active[:2] != (cycle, step):
            unfinished()
            active = None
        if kind == "position_lost":
            if active is None:
                active = (cycle, step, source, created)
            elif source is not None and active[2] is not None and _duration(active[2], source) is None:
                unfinished()
                active = (cycle, step, source, created)
        elif kind == "position_recovered":
            if active is None:
                pauses.append(PositionPause(cycle, step, "", created or "時間未記錄", None))
            else:
                _cycle, _step, start_source, start_time = active
                # A reset source clock cannot establish that this recovery
                # belongs to the earlier pause. Preserve both missing edges.
                if source is not None and start_source is not None and _duration(start_source, source) is None:
                    unfinished()
                    pauses.append(PositionPause(cycle, step, "", created or "時間未記錄", None))
                else:
                    pauses.append(PositionPause(cycle, step, start_time or "時間未記錄",
                                                created or "時間未記錄", _duration(start_source, source)))
                active = None
    unfinished()
    return pauses


def _elapsed(items: list[PositionPause]) -> str:
    known = [item.duration for item in items if item.duration is not None]
    if not known:
        return "未記錄"
    return f"{sum(known):.2f}" + ("（部分區間）" if len(known) < len(items) else "")


def positioning_summary(events: list[tuple], step_names: dict[int, str]) -> tuple[str, list[tuple]]:
    """Counts, cumulative paused time and recovery state; one row per step.

    Rows are (step, pauses, paused seconds, recovery). Individual loss/recovery
    timestamps stay in the database and CSV export instead of the summary.
    """
    pauses = positioning_pauses(events)
    if not pauses:
        return "本輪無定位暫停。", []
    grouped = {}
    for pause in pauses:
        grouped.setdefault((pause.cycle_id, pause.step_index), []).append(pause)
    rows = []
    for (_cycle, step), items in grouped.items():
        lost = [item for item in items if item.started_at]
        pending = sum(not item.recovered for item in lost)
        missing_start = sum(not item.started_at for item in items)
        status = f"{pending} 次未記錄恢復" if pending else "已恢復"
        if missing_start:
            status += f"；{missing_start} 次恢復缺少起點"
        name = f"{step} · {step_names[step]}" if step in step_names else str(step)
        rows.append((name, len(lost), _elapsed(items), status))
    lost_count = sum(bool(item.started_at) for item in pauses)
    pending = sum(bool(item.started_at) and not item.recovered for item in pauses)
    recovered = sum(bool(item.started_at) and item.recovered for item in pauses)
    orphan = sum(not item.started_at for item in pauses)
    summary = (f"判定暫停 {lost_count} 次 · 累計 {_elapsed(pauses)} 秒 · 已恢復 {recovered} 次"
               f" · 未記錄恢復 {pending} 次")
    if orphan:
        summary += f" · {orphan} 次恢復缺少起點"
    return summary, rows
