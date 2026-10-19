"""作業紀錄：SQLite + 工序完成與警報當下的截圖。

資料庫以 ``PRAGMA user_version`` 記錄結構版本，開啟時依序套用遷移，舊紀錄可直接沿用。
正式監控先持久保存事件與必要截圖，再透過 :class:`AsyncRecorder` 在獨立執行緒轉入正式紀錄。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from contextlib import closing
import logging
from pathlib import Path
import queue
import sqlite3
import threading
import time
from typing import Callable

from .native_runtime import cv2
import numpy as np

from .engine import EngineEvent

SCHEMA_VERSION = 3

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
    3: """
CREATE TABLE sop_snapshots (
    hash TEXT PRIMARY KEY, content TEXT NOT NULL, created_at TEXT NOT NULL
);
ALTER TABLE cycles ADD COLUMN sop_snapshot_hash TEXT REFERENCES sop_snapshots(hash);
ALTER TABLE cycles ADD COLUMN session_id TEXT;
ALTER TABLE cycles ADD COLUMN source_cycle INTEGER;
CREATE UNIQUE INDEX idx_cycle_session ON cycles(session_id, source_cycle);
CREATE TABLE applied_jobs (job_key TEXT PRIMARY KEY, applied_at TEXT NOT NULL);
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
        db.execute("PRAGMA synchronous = FULL")
        # 舊版資料庫沒有版本號（0）：第 1 版全是 IF NOT EXISTS，可安全重跑以補齊缺少的資料表。
        version = db.execute("PRAGMA user_version").fetchone()[0]
        for target in range(version + 1, SCHEMA_VERSION + 1):
            db.executescript("BEGIN IMMEDIATE;\n" + _MIGRATIONS[target]
                             + f"\nPRAGMA user_version = {target};\nCOMMIT;")
        if version > SCHEMA_VERSION:
            raise ValueError(f"紀錄資料庫版本 {version} 比程式新（{SCHEMA_VERSION}），請更新程式")
    except Exception:
        db.close()
        raise
    return db


class Recorder:
    """只能在建立它的執行緒中使用（SQLite 連線限制）。"""

    def __init__(self, root: Path, *, recover=True):
        self.root = Path(root)
        self.snapshot_dir = self.root / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.db = connect(self.root / "records.db")
        self._session = uuid.uuid4().hex
        if recover:
            self.recover_interrupted()

    def recover_interrupted(self, preserve_sessions=()) -> int:
        """上次程式異常結束而停在「進行中」的輪次，標記為中斷，避免永遠顯示進行中。"""
        with self.db:
            placeholders = ",".join("?" for _ in preserve_sessions)
            clause = f" AND (session_id IS NULL OR session_id NOT IN ({placeholders}))" if preserve_sessions else ""
            cursor = self.db.execute("UPDATE cycles SET result='INTERRUPTED', ended_at=COALESCE(ended_at, ?) "
                                     "WHERE result='RUNNING'" + clause, (_now(), *preserve_sessions))
        return cursor.rowcount

    def handle(self, events: list[EngineEvent], sop_name: str, sop_version: int, model_version: str,
               frame: np.ndarray | None = None, context: dict | None = None,
               sop_snapshot: dict | None = None, *, session_id=None, job_key=None, recorded_at=None,
               snapshot_jpeg: bytes | None = None):
        """Write one atomic job; snapshot_jpeg is prevalidated by AsyncRecorder."""
        context = context or {}
        stamp = recorded_at or _now()
        session = session_id or self._session
        if job_key and self.db.execute("SELECT 1 FROM applied_jobs WHERE job_key=?", (job_key,)).fetchone():
            return
        try:
            self._handle_batch(events, sop_name, sop_version, model_version, frame, context,
                               sop_snapshot, session, stamp, snapshot_jpeg)
            if job_key:
                self.db.execute("INSERT INTO applied_jobs VALUES (?,?)", (job_key, stamp))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _handle_batch(self, events, sop_name, sop_version, model_version, frame, context,
                      sop_snapshot, session, stamp, snapshot_jpeg=None):
        snapshot_hash = None
        if sop_snapshot is not None:
            content = json.dumps(sop_snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            snapshot_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            self.db.execute("INSERT OR IGNORE INTO sop_snapshots VALUES (?,?,?)", (snapshot_hash, content, stamp))
        for event in events:
            if event.kind == "cycle_started":
                cursor = self.db.execute(
                    "INSERT INTO cycles (sop_name, sop_version, model_version, started_at, result, "
                    "station, operator, work_order, serial_no, sop_snapshot_hash, session_id, source_cycle) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sop_name, sop_version, model_version, stamp, "RUNNING",
                     context.get("station", ""), context.get("operator", ""),
                     context.get("work_order", ""), context.get("serial_no", ""), snapshot_hash, session, event.cycle))
                continue

            row = self.db.execute("SELECT id FROM cycles WHERE session_id=? AND source_cycle=?",
                                  (session, event.cycle)).fetchone()
            if row is None:
                raise ValueError(f"找不到事件所屬輪次：{session}/{event.cycle}")
            cycle_id = row[0]
            if event.kind in ("cycle_completed", "cycle_aborted"):
                result = event.data.get("result", "ABORTED")
                self.db.execute("UPDATE cycles SET ended_at=?, duration_sec=?, result=? WHERE id=?",
                                (stamp, round(event.data.get("duration", 0.0), 2), result, cycle_id))
            elif event.kind == "step_completed":
                snapshot = self._save_snapshot(frame, f"cycle{cycle_id:05d}_step{event.step_index + 1:02d}",
                                               snapshot_jpeg)
                self.db.execute(
                    "INSERT INTO step_records (cycle_id, step_index, step_name, status, manual, had_alarm, "
                    "duration_sec, finished_at, snapshot) VALUES (?,?,?,?,?,?,?,?,?)",
                    (cycle_id, event.step_index + 1, event.step_name, event.data.get("status"),
                     int(event.data.get("manual", False)), int(event.data.get("alarm", False)),
                     round(event.data.get("duration", 0.0), 2), stamp, snapshot))
            elif event.kind in ("position_lost", "position_recovered"):
                self.db.execute("INSERT INTO positioning_events "
                                "(cycle_id, step_index, kind, message, source_time, created_at) VALUES (?,?,?,?,?,?)",
                                (cycle_id, event.step_index + 1, event.kind, event.message, event.time, stamp))
            elif event.kind == "alarm":
                cursor = self.db.execute(
                    "INSERT INTO alarms (cycle_id, step_index, step_name, kind, message, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (cycle_id, event.step_index + 1, event.step_name, event.data.get("alarm"), event.message, stamp))
                # NG 的證據比 OK 更重要：警報當下也保留畫面。
                snapshot = self._save_snapshot(frame, f"cycle{cycle_id:05d}_step{event.step_index + 1:02d}"
                                                      f"_alarm{cursor.lastrowid}", snapshot_jpeg)
                if snapshot:
                    self.db.execute("UPDATE alarms SET snapshot=? WHERE id=?", (snapshot, cursor.lastrowid))

    def acknowledge_alarms(self, by: str = "", *, session_id=None, job_key=None, recorded_at=None) -> int:
        with self.db:
            if job_key and self.db.execute("SELECT 1 FROM applied_jobs WHERE job_key=?", (job_key,)).fetchone():
                return 0
            clause = " AND cycle_id IN (SELECT id FROM cycles WHERE session_id=?)" if session_id else ""
            params = (recorded_at or _now(), by) + ((session_id,) if session_id else ())
            cursor = self.db.execute("UPDATE alarms SET acknowledged_at=?, acknowledged_by=? "
                                     "WHERE acknowledged_at IS NULL" + clause, params)
            if job_key:
                self.db.execute("INSERT INTO applied_jobs VALUES (?,?)", (job_key, _now()))
        return cursor.rowcount

    def _save_snapshot(self, frame: np.ndarray | None, stem: str, jpeg: bytes | None = None) -> str:
        if frame is None and jpeg is None:
            return ""
        path = self.snapshot_dir / f"{stem}.jpg"
        if jpeg is None:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise OSError("截圖編碼失敗")
            jpeg = encoded.tobytes()
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            stream.write(jpeg)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        return path.name

    def close(self):
        self.db.close()


class AsyncRecorder:
    """Persist jobs before returning; replay in order with transactional deduplication."""

    def __init__(self, root: Path, on_written=None, on_error=None, on_status=None):
        from PySide6.QtCore import QLockFile
        from .station_backup import recover_restore
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._writer_lock = QLockFile(str(self._root / ".writer.lock"))
        self._writer_lock.setStaleLockTime(0)
        if not self._writer_lock.tryLock(0):
            raise RuntimeError("此工站資料已由另一個程式使用，請先關閉另一個 SOP 視窗")
        recover_restore(self._root.parent)
        self._on_written = on_written or (lambda: None)
        self._on_error = on_error or (lambda message: None)
        self._on_status = on_status or (lambda pending, message: None)
        self._last_error = ("", 0.0)
        # Two fixed-size snapshots, independent from the SQLite lock. Readers
        # receive copies; no per-event history or unbounded performance log.
        self._metrics_lock = threading.Lock()
        self._last_enqueue_metrics = {}
        self._last_write_metrics = {}
        self._session = uuid.uuid4().hex
        self._live_sessions = set()
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._closed = False
        self._journal = sqlite3.connect(self._root / "pending.db", check_same_thread=False)
        self._journal.execute("PRAGMA journal_mode=WAL")
        self._journal.execute("PRAGMA synchronous=FULL")
        self._journal.execute("CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                              "job_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, image BLOB)")
        self._journal.commit()
        self._maintenance = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="sop-recorder", daemon=True)
        self._thread.start()
        self._ready.wait(10)

    def _enqueue(self, payload, frame=None):
        started = time.perf_counter()
        image = None
        if frame is not None:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise OSError("無法保存待寫入截圖")
            image = encoded.tobytes()
        encoded_at = time.perf_counter()
        content = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        serialized_at = time.perf_counter()
        with self._lock:
            locked_at = time.perf_counter()
            if self._closed:
                raise RuntimeError("紀錄寫入器已關閉")
            with self._journal:
                self._journal.execute("INSERT INTO jobs(job_key,payload,image) VALUES (?,?,?)",
                                      (uuid.uuid4().hex, content, image))
            committed_at = time.perf_counter()
        with self._metrics_lock:
            self._last_enqueue_metrics = {
                "encode_ms": (encoded_at - started) * 1000 if frame is not None else 0.0,
                "serialize_ms": (serialized_at - encoded_at) * 1000,
                "pending_lock_wait_ms": (locked_at - serialized_at) * 1000,
                "pending_commit_ms": (committed_at - locked_at) * 1000,
                "total_ms": (committed_at - started) * 1000,
            }
        self._wake.set()
        self._on_status(self.pending_count(), "事件已保存，等待寫入正式紀錄")

    def handle(self, events, sop_name, sop_version, model_version, frame=None, context=None, sop_snapshot=None):
        if not events:
            return
        if any(e.kind == "cycle_started" and e.cycle == 1 for e in events):
            self._session = uuid.uuid4().hex
        with self._lock:
            self._live_sessions.add(self._session)
        image = frame if any(e.kind in ("step_completed", "alarm") for e in events) else None
        self._enqueue(dict(command="handle", session=self._session, at=_now(),
                           events=[asdict(e) for e in events], sop_name=sop_name,
                           sop_version=sop_version, model_version=model_version,
                           context=dict(context or {}), sop_snapshot=sop_snapshot), image)

    def acknowledge_alarms(self, by=""):
        self._enqueue(dict(command="ack", session=self._session, at=_now(), by=by))

    def pending_count(self):
        with self._lock:
            return self._journal.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    @property
    def last_enqueue_metrics(self):
        """Last successful durable enqueue, in ms (before status callback)."""
        with self._metrics_lock:
            return self._last_enqueue_metrics.copy()

    @property
    def last_write_metrics(self):
        """Last completed background job, in ms; overlaps the calling thread."""
        with self._metrics_lock:
            return self._last_write_metrics.copy()

    def _drain(self, recorder):
        while True:
            with self._lock:
                row = self._journal.execute("SELECT id,job_key,payload,image FROM jobs ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return
            started = time.perf_counter()
            number, key, content, image = row
            payload = json.loads(content)
            kwargs = dict(session_id=payload["session"], job_key=key, recorded_at=payload["at"])
            decode_ms = 0.0
            if payload["command"] == "ack":
                write_started = time.perf_counter()
                recorder.acknowledge_alarms(payload["by"], **kwargs)
            else:
                decode_started = time.perf_counter()
                frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR) if image else None
                if image and frame is None:
                    raise ValueError("待寫入截圖損毀，已保留待處理資料")
                decode_ms = (time.perf_counter() - decode_started) * 1000 if image else 0.0
                # The pending BLOB already contains a complete JPEG. Keep the
                # decode validation for old journals, but save the same encoded
                # bytes rather than recompressing the decoded pixels again.
                del frame
                write_started = time.perf_counter()
                recorder.handle([EngineEvent(**event) for event in payload["events"]],
                                payload["sop_name"], payload["sop_version"], payload["model_version"],
                                None, payload["context"], payload["sop_snapshot"], snapshot_jpeg=image or None, **kwargs)
            written_at = time.perf_counter()
            with self._lock, self._journal:
                self._journal.execute("DELETE FROM jobs WHERE id=?", (number,))
            deleted_at = time.perf_counter()
            with self._metrics_lock:
                self._last_write_metrics = {
                    "decode_ms": decode_ms,
                    "formal_write_ms": (written_at - write_started) * 1000,
                    "pending_delete_ms": (deleted_at - written_at) * 1000,
                    "total_ms": (deleted_at - started) * 1000,
                }
            self._on_written()
            self._on_status(self.pending_count(), "正式紀錄已寫入")

    def maintenance(self, action, callback, *, allow_unavailable=False):
        self._maintenance.put((action, callback, allow_unavailable))
        self._wake.set()

    def close(self, timeout=10.0):
        with self._lock:
            if self._closed:
                return not self._thread.is_alive()
            self._closed = True
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self):
        recorder = None
        recovered = False
        try:
            while True:
                self._wake.clear()
                try:
                    if recorder is None:
                        from .station_backup import recover_restore
                        recover_restore(self._root.parent)
                        recorder = Recorder(self._root, recover=False)
                    self._drain(recorder)
                    self._last_error = ("", 0.0)
                    if not recovered:
                        with self._lock:
                            recorder.recover_interrupted(tuple(self._live_sessions))
                        recovered = True
                        self._on_status(self.pending_count(), "紀錄系統就緒，待處理紀錄已補寫")
                    self._ready.set()
                    while not self._maintenance.empty():
                        action, callback, _allow_unavailable = self._maintenance.get_nowait()
                        # No events are produced during maintenance (pipeline is stopped).
                        recorder.close()
                        recorder = None
                        try:
                            result = action()
                            recorder = Recorder(self._root, recover=True)
                        except Exception as exc:
                            callback(None, str(exc))
                        else:
                            callback(result, "")
                        finally:
                            if recorder is None:
                                from .station_backup import recover_restore
                                recover_restore(self._root.parent)
                                recorder = Recorder(self._root, recover=False)
                except Exception as exc:
                    last_message, last_time = self._last_error
                    if str(exc) != last_message or time.monotonic() - last_time >= 30:
                        logging.getLogger("sop.records").exception("紀錄尚未寫入，保留待重試資料")
                        self._on_error(f"紀錄尚未寫入，待處理資料已保留，將自動重試：{exc}")
                        self._last_error = (str(exc), time.monotonic())
                    self._on_status(self.pending_count(), f"正式紀錄尚未寫入，將自動重試：{exc}")
                    self._ready.set()
                    if recorder is not None:
                        recorder.close()
                        recorder = None
                    while not self._maintenance.empty():
                        action, callback, allow_unavailable = self._maintenance.get_nowait()
                        if allow_unavailable and self.pending_count() == 0:
                            try:
                                result = action()
                                recorder = Recorder(self._root, recover=True)
                                recovered = True
                            except Exception as maintenance_error:
                                callback(None, str(maintenance_error))
                            else:
                                callback(result, "")
                                self._on_status(0, "工站資料已還原，紀錄系統就緒")
                        else:
                            callback(None, "仍有待寫入紀錄，請排除磁碟／資料庫問題後再備份或還原")
                if self._stop.is_set():
                    break
                self._wake.wait(1.0)
        finally:
            if recorder is not None:
                recorder.close()
            with self._lock:
                self._journal.close()
            self._writer_lock.unlock()


def read_sop_snapshot(db_path: Path, cycle_id: int) -> tuple[dict, str] | None:
    if not has_column(db_path, "cycles", "sop_snapshot_hash"):
        return None
    rows = query(db_path, "SELECT s.content,s.hash FROM cycles c JOIN sop_snapshots s "
                         "ON s.hash=c.sop_snapshot_hash WHERE c.id=?", (cycle_id,))
    if not rows:
        return None
    content, digest = rows[0]
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != digest:
        raise ValueError("SOP 快照完整性檢查失敗")
    return json.loads(content), digest


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
