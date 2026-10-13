"""作業紀錄：SQLite + 工序完成與警報當下的截圖。

資料庫以 ``PRAGMA user_version`` 記錄結構版本，開啟時依序套用遷移，舊紀錄可直接沿用。
正式監控透過 :class:`AsyncRecorder` 在獨立執行緒寫入，JPEG 編碼與 SQLite 不會拖慢影像處理。
"""
from __future__ import annotations

import csv
from datetime import datetime
from contextlib import closing
import logging
from pathlib import Path
import queue
import sqlite3
import threading
from typing import Callable

from .native_runtime import cv2
import numpy as np

from .engine import EngineEvent

SCHEMA_VERSION = 2

# 每個版本只做增量變更；已存在的資料庫會從目前版本往後套用。
_MIGRATIONS: dict[int, str] = {
    1: """
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
""",
    2: """
ALTER TABLE cycles ADD COLUMN station TEXT DEFAULT '';
ALTER TABLE cycles ADD COLUMN operator TEXT DEFAULT '';
ALTER TABLE cycles ADD COLUMN work_order TEXT DEFAULT '';
ALTER TABLE cycles ADD COLUMN serial_no TEXT DEFAULT '';
ALTER TABLE alarms ADD COLUMN snapshot TEXT DEFAULT '';
ALTER TABLE alarms ADD COLUMN acknowledged_at TEXT;
ALTER TABLE alarms ADD COLUMN acknowledged_by TEXT DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_cycles_started ON cycles(started_at);
CREATE INDEX IF NOT EXISTS idx_cycles_result ON cycles(result);
CREATE INDEX IF NOT EXISTS idx_steps_cycle ON step_records(cycle_id);
CREATE INDEX IF NOT EXISTS idx_alarms_cycle ON alarms(cycle_id);
CREATE INDEX IF NOT EXISTS idx_positioning_cycle ON positioning_events(cycle_id);
""",
}

RESULT_TEXT = {"OK": "OK", "NG": "NG", "ABORTED": "中止", "INTERRUPTED": "中斷", "RUNNING": "進行中"}
ALARM_TEXT = {"violation": "違規", "timeout": "超時", "too_fast": "工時過短", "sequence": "順序錯誤",
              "skipped": "人工跳過"}


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    """開啟並升級資料庫；WAL 讓介面讀取紀錄時不會擋住背景寫入。"""
    db = sqlite3.connect(db_path)
    try:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        # 舊版資料庫沒有版本號（0）：第 1 版全是 IF NOT EXISTS，可安全重跑以補齊缺少的資料表。
        version = db.execute("PRAGMA user_version").fetchone()[0]
        for target in range(version + 1, SCHEMA_VERSION + 1):
            with db:
                db.executescript(_MIGRATIONS[target])
                db.execute(f"PRAGMA user_version = {target}")
        if version > SCHEMA_VERSION:
            logging.getLogger("sop.records").warning("紀錄資料庫版本 %s 比程式新（%s）", version, SCHEMA_VERSION)
    except Exception:
        db.close()
        raise
    return db


class Recorder:
    """只能在建立它的執行緒中使用（SQLite 連線限制）。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.snapshot_dir = self.root / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.db = connect(self.root / "records.db")
        self._cycle_ids: dict[int, int] = {}
        self.recover_interrupted()

    def recover_interrupted(self) -> int:
        """上次程式異常結束而停在「進行中」的輪次，標記為中斷，避免永遠顯示進行中。"""
        with self.db:
            cursor = self.db.execute("UPDATE cycles SET result='INTERRUPTED', ended_at=COALESCE(ended_at, ?) "
                                     "WHERE result='RUNNING'", (_now(),))
        return cursor.rowcount

    def handle(self, events: list[EngineEvent], sop_name: str, sop_version: int, model_version: str,
               frame: np.ndarray | None = None, context: dict | None = None):
        context = context or {}
        for event in events:
            if event.kind == "cycle_started":
                cursor = self.db.execute(
                    "INSERT INTO cycles (sop_name, sop_version, model_version, started_at, result, "
                    "station, operator, work_order, serial_no) VALUES (?,?,?,?,?,?,?,?,?)",
                    (sop_name, sop_version, model_version, _now(), "RUNNING",
                     context.get("station", ""), context.get("operator", ""),
                     context.get("work_order", ""), context.get("serial_no", "")))
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
                snapshot = self._save_snapshot(frame, f"cycle{cycle_id:05d}_step{event.step_index + 1:02d}")
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
                cursor = self.db.execute(
                    "INSERT INTO alarms (cycle_id, step_index, step_name, kind, message, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (cycle_id, event.step_index + 1, event.step_name, event.data.get("alarm"), event.message, _now()))
                # NG 的證據比 OK 更重要：警報當下也保留畫面。
                snapshot = self._save_snapshot(frame, f"cycle{cycle_id:05d}_step{event.step_index + 1:02d}"
                                                      f"_alarm{cursor.lastrowid}")
                if snapshot:
                    self.db.execute("UPDATE alarms SET snapshot=? WHERE id=?", (snapshot, cursor.lastrowid))
        self.db.commit()

    def acknowledge_alarms(self, by: str = "") -> int:
        with self.db:
            cursor = self.db.execute("UPDATE alarms SET acknowledged_at=?, acknowledged_by=? "
                                     "WHERE acknowledged_at IS NULL", (_now(), by))
        return cursor.rowcount

    def _save_snapshot(self, frame: np.ndarray | None, stem: str) -> str:
        if frame is None:
            return ""
        path = self.snapshot_dir / f"{stem}.jpg"
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return ""
        path.write_bytes(encoded.tobytes())      # 避免 cv2.imwrite 在 Windows 非 ASCII 路徑失敗
        return path.name

    def close(self):
        self.db.close()


class AsyncRecorder:
    """在獨立執行緒擁有 :class:`Recorder`；呼叫端只把工作放入佇列，不等待磁碟。

    ``on_written`` 在每批寫入提交後呼叫，``on_error`` 回報失敗；兩者都在寫入執行緒中執行。
    """

    def __init__(self, root: Path, on_written: Callable[[], None] | None = None,
                 on_error: Callable[[str], None] | None = None):
        self._root = Path(root)
        self._on_written = on_written or (lambda: None)
        self._on_error = on_error or (lambda message: None)
        self._jobs: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sop-recorder", daemon=True)
        self._thread.start()
        self._ready.wait(10)

    def handle(self, events, sop_name, sop_version, model_version, frame=None, context=None):
        has_image = any(e.kind in ("step_completed", "alarm") for e in events)
        image = frame.copy() if has_image and frame is not None else None
        self._jobs.put(("handle", (list(events), sop_name, sop_version, model_version, image, dict(context or {}))))

    def acknowledge_alarms(self, by: str = ""):
        self._jobs.put(("ack", (by,)))

    def close(self, timeout: float = 10.0):
        self._jobs.put(("close", ()))
        self._thread.join(timeout)

    def _run(self):
        try:
            recorder = Recorder(self._root)
        except Exception as exc:          # 無法開啟資料庫：仍持續消化佇列，避免呼叫端堆積
            recorder = None
            self._on_error(f"無法開啟作業紀錄資料庫：{exc}")
        self._ready.set()
        while True:
            command, args = self._jobs.get()
            if command == "close":
                break
            if recorder is None:
                continue
            try:
                if command == "handle":
                    recorder.handle(*args)
                elif command == "ack":
                    recorder.acknowledge_alarms(*args)
                self._on_written()
            except Exception as exc:
                logging.getLogger("sop.records").exception("寫入紀錄失敗")
                self._on_error(f"寫入紀錄失敗：{exc}")
        if recorder is not None:
            recorder.close()


def query(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    """給介面讀取紀錄用（另開連線，不與背景執行緒共用）。"""
    if not Path(db_path).exists():
        return []
    with closing(sqlite3.connect(db_path)) as db:
        return db.execute(sql, params).fetchall()


def has_column(db_path: Path, table: str, column: str) -> bool:
    return bool(query(db_path, f"SELECT 1 FROM pragma_table_info('{table}') WHERE name=?", (column,)))


def cycle_filter_sql(date_from: str = "", date_to: str = "", result: str = "", text: str = "",
                     extended: bool = True) -> tuple[str, list]:
    """組出輪次篩選條件（日期為 YYYY-MM-DD，含當日）。"""
    clauses, params = [], []
    if date_from:
        clauses.append("c.started_at >= ?")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        clauses.append("c.started_at <= ?")
        params.append(f"{date_to} 23:59:59")
    if result:
        clauses.append("c.result = ?")
        params.append(result)
    if text:
        fields = ["c.sop_name"] + (["c.operator", "c.work_order", "c.serial_no", "c.station"] if extended else [])
        clauses.append("(" + " OR ".join(f"{name} LIKE ?" for name in fields) + ")")
        params.extend([f"%{text}%"] * len(fields))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def export_csv(db_path: Path, target: Path, **filters) -> int:
    """匯出逐工序明細（含輪次欄位），UTF-8 BOM 讓 Excel 直接開啟中文。回傳列數。"""
    where, params = cycle_filter_sql(**filters)
    rows = query(db_path, f"""
        SELECT c.id, c.started_at, c.ended_at, c.duration_sec, c.result, c.sop_name, c.sop_version,
               c.model_version, c.station, c.operator, c.work_order, c.serial_no,
               s.step_index, s.step_name, s.status, s.duration_sec, s.manual, s.had_alarm, s.finished_at,
               (SELECT GROUP_CONCAT(a.message, '；') FROM alarms a
                 WHERE a.cycle_id = c.id AND a.step_index = s.step_index)
        FROM cycles c LEFT JOIN step_records s ON s.cycle_id = c.id
        {where} ORDER BY c.id, s.id""", tuple(params))
    headers = ["輪次編號", "開始時間", "結束時間", "本輪耗時(秒)", "結果", "SOP", "SOP版本", "模型版本",
               "工站", "作業員", "工單", "序號", "工序", "工序名稱", "工序狀態", "工序耗時(秒)", "手動",
               "警報", "完成時間", "警報內容"]
    with Path(target).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for row in rows:
            row = list(row)
            row[4] = RESULT_TEXT.get(row[4], row[4])
            writer.writerow(["" if value is None else value for value in row])
    return len(rows)
