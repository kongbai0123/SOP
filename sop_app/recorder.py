"""作業紀錄：SQLite + 每道工序完成當下的截圖。"""
from __future__ import annotations

from datetime import datetime
from contextlib import closing
from pathlib import Path
import sqlite3

import cv2
import numpy as np

from .engine import EngineEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sop_name TEXT, sop_version INTEGER, model_version TEXT,
    started_at TEXT, ended_at TEXT, duration_sec REAL, result TEXT
);
CREATE TABLE IF NOT EXISTS step_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER REFERENCES cycles(id), step_index INTEGER, step_name TEXT,
    status TEXT, manual INTEGER, had_alarm INTEGER, duration_sec REAL, finished_at TEXT, snapshot TEXT
);
CREATE TABLE IF NOT EXISTS positioning_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER REFERENCES cycles(id), step_index INTEGER,
    kind TEXT, message TEXT, source_time REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS alarms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER REFERENCES cycles(id), step_index INTEGER, step_name TEXT,
    kind TEXT, message TEXT, created_at TEXT
);
"""


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


class Recorder:
    """只能在建立它的執行緒中使用（SQLite 連線限制）。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.snapshot_dir = self.root / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "records.db")
        self.db.executescript(_SCHEMA)
        self._cycle_ids: dict[int, int] = {}

    def handle(self, events: list[EngineEvent], sop_name: str, sop_version: int, model_version: str,
               frame: np.ndarray | None = None):
        for event in events:
            if event.kind == "cycle_started":
                cursor = self.db.execute(
                    "INSERT INTO cycles (sop_name, sop_version, model_version, started_at, result) VALUES (?,?,?,?,?)",
                    (sop_name, sop_version, model_version, _now(), "RUNNING"))
                self._cycle_ids[event.cycle] = cursor.lastrowid
                continue

            cycle_id = self._cycle_ids.get(event.cycle)
            if cycle_id is None:
                continue
            if event.kind in ("cycle_completed", "cycle_aborted"):
                result = event.data.get("result", "ABORTED")
                self.db.execute("UPDATE cycles SET ended_at=?, duration_sec=?, result=? WHERE id=?",
                                (_now(), round(event.data.get("duration", 0.0), 2), result, cycle_id))
            elif event.kind == "step_completed":
                snapshot = self._save_snapshot(frame, cycle_id, event.step_index)
                self.db.execute(
                    "INSERT INTO step_records (cycle_id, step_index, step_name, status, manual, had_alarm, "
                    "duration_sec, finished_at, snapshot) VALUES (?,?,?,?,?,?,?,?,?)",
                    (cycle_id, event.step_index + 1, event.step_name, event.data.get("status"),
                     int(event.data.get("manual", False)), int(event.data.get("alarm", False)),
                     round(event.data.get("duration", 0.0), 2), _now(), snapshot))
            elif event.kind in ("position_lost", "position_recovered"):
                self.db.execute("INSERT INTO positioning_events "
                                "(cycle_id, step_index, kind, message, source_time, created_at) VALUES (?,?,?,?,?,?)",
                                (cycle_id, event.step_index + 1, event.kind, event.message, event.time, _now()))
            elif event.kind == "alarm":
                self.db.execute(
                    "INSERT INTO alarms (cycle_id, step_index, step_name, kind, message, created_at) VALUES (?,?,?,?,?,?)",
                    (cycle_id, event.step_index + 1, event.step_name, event.data.get("alarm"), event.message, _now()))
        self.db.commit()

    def _save_snapshot(self, frame: np.ndarray | None, cycle_id: int, step_index: int) -> str:
        if frame is None:
            return ""
        path = self.snapshot_dir / f"cycle{cycle_id:05d}_step{step_index + 1:02d}.jpg"
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return ""
        path.write_bytes(encoded.tobytes())      # 避免 cv2.imwrite 在 Windows 非 ASCII 路徑失敗
        return path.name

    def close(self):
        self.db.close()


def query(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    """給介面讀取紀錄用（另開連線，不與背景執行緒共用）。"""
    if not Path(db_path).exists():
        return []
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute(sql, params).fetchall()
