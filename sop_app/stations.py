"""Station ownership, atomic creation and non-destructive legacy migration."""
from __future__ import annotations

from contextlib import contextmanager, closing
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import uuid

from .station_backup import atomic_json, digest, recover_restore

STATION_ID = re.compile(r"ST[0-9]{3,8}\Z")


@contextmanager
def directory_lock(path):
    from PySide6.QtCore import QLockFile
    lock = QLockFile(str(path))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise RuntimeError("工站資料正在由另一個程式處理，請稍後重試")
    try:
        yield
    finally:
        lock.unlock()


@dataclass(frozen=True)
class Station:
    id: str
    name: str
    root: Path
    legacy_root: Path | None = None

    @property
    def sops(self):
        return self.root / "sops"

    @property
    def records(self):
        return self.root / "records"

    @property
    def settings(self):
        return self.root / "app_settings.json"

    def resolve(self, value):
        path = Path(value)
        if path.is_absolute():
            return path
        local = self.root / path
        # Compatibility for immutable old snapshots and external model paths.
        if not local.exists() and self.legacy_root is not None:
            old = self.legacy_root / path
            if old.exists() and path.parts and path.parts[0] not in ("sops", "records", "backups"):
                return old
        return local

    def relativize(self, value):
        path = Path(value).resolve()
        try:
            return path.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)


class StationStore:
    def __init__(self, data_root):
        self.data_root = Path(data_root).resolve()
        self.root = self.data_root / "data" / "stations"
        self.index = self.root.parent / "stations.json"

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with directory_lock(self.root.parent / ".stations.lock"):
            stations = self.list()
            if not stations:
                self._migrate_default()
                stations = self.list()
            if not self.index.exists():
                atomic_json(self.index, dict(version=1, active_station=stations[0].id))
            else:
                state = json.loads(self.index.read_text(encoding="utf-8"))
                if state.get("version") != 1:
                    raise ValueError("不支援的工站清單版本")
                if state.get("active_station") not in {s.id for s in stations}:
                    raise ValueError("上次使用的工站資料不存在，請先復原工站目錄")
        station = self.active
        if (station.root / ".restore-transaction.json").exists():
            # Read startup settings/model paths only after an interrupted restore is recovered.
            with directory_lock(station.records / ".writer.lock"):
                recover_restore(station.root)
        return station

    def list(self):
        result = []
        if not self.root.exists():
            return result
        for path in sorted(self.root.iterdir()):
            if not STATION_ID.fullmatch(path.name):
                continue
            if path.is_symlink() or path.is_junction() or not path.is_dir():
                raise ValueError("工站目錄不可使用連結")
            metadata = json.loads((path / "station.json").read_text(encoding="utf-8"))
            if metadata.get("id") != path.name or metadata.get("version") != 1:
                raise ValueError(f"工站 {path.name} 的識別資料無效")
            result.append(Station(path.name, metadata["name"], path,
                                  Path(metadata["legacy_root"]) if metadata.get("legacy_root") else None))
        return result

    def get(self, station_id):
        return next((station for station in self.list() if station.id == station_id), None)

    @property
    def active(self):
        state = json.loads(self.index.read_text(encoding="utf-8"))
        station = self.get(state["active_station"])
        if station is None:
            raise ValueError("選定的工站不存在")
        return station

    def activate(self, station_id):
        with directory_lock(self.root.parent / ".stations.lock"):
            station = self.get(station_id)
            if station is None:
                raise ValueError("工站不存在")
            atomic_json(self.index, dict(version=1, active_station=station.id))
        return station

    def create(self, name):
        if any(ord(c) < 32 for c in name):
            raise ValueError("工站名稱不可包含換行或控制字元")
        name = name.strip()
        if not name or len(name) > 80 or any(ord(c) < 32 for c in name):
            raise ValueError("工站名稱需為 1～80 個字，且不可包含換行或控制字元")
        with directory_lock(self.root.parent / ".stations.lock"):
            stations = self.list()
            if any(station.name.casefold() == name.casefold() for station in stations):
                raise ValueError("工站名稱已存在")
            station_id = f"ST{max((int(s.id[2:]) for s in stations), default=0) + 1:03d}"
            stage = self.root / (".new-" + uuid.uuid4().hex)
            stage.mkdir()
            try:
                self._seed(stage, station_id, name)
                os.replace(stage, self.root / station_id)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        return self.get(station_id)

    @staticmethod
    def _seed(root, station_id, name, legacy_root=None):
        for directory in ("sops", "records/snapshots", "backups", "logs"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        metadata = dict(version=1, id=station_id, name=name)
        if legacy_root is not None:
            metadata["legacy_root"] = str(legacy_root)
        atomic_json(root / "station.json", metadata)
        if not (root / "app_settings.json").exists():
            atomic_json(root / "app_settings.json", {"station": name})

    def _migrate_default(self):
        legacy = self.data_root
        records = legacy / "records"
        records.mkdir(exist_ok=True)
        with directory_lock(records / ".writer.lock"):
            recover_restore(legacy)
            stage = self.root / (".migration-" + uuid.uuid4().hex)
            stage.mkdir()
            try:
                copied = {}
                # Copy originals first; no source data is deleted or overwritten.
                for entry in ("sops", "records", "backups", "logs", "app_settings.json"):
                    source = legacy / entry
                    if not source.exists():
                        continue
                    if source.is_symlink() or source.is_junction():
                        raise ValueError("舊資料目錄含連結，請先改為一般目錄")
                    paths = list(source.rglob("*")) if source.is_dir() else [source]
                    for path in paths:
                        if path.is_symlink() or path.is_junction():
                            raise ValueError("舊資料含連結，遷移已停止，原資料保留")
                        managed_db = path.parent == records and (path.name in (".writer.lock", "records.db", "pending.db")
                            or path.name in ("records.db-wal", "records.db-shm", "pending.db-wal", "pending.db-shm"))
                        if not path.is_file() or managed_db:
                            continue
                        relative = path.relative_to(legacy)
                        output = stage / relative
                        output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, output)
                        if digest(output) != digest(path):
                            raise OSError("遷移副本與原檔不一致")
                        copied[relative.as_posix()] = digest(output)
                for filename in ("records.db", "pending.db"):
                    source = records / filename
                    if not source.exists():
                        continue
                    output = stage / "records" / filename
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db, \
                            closing(sqlite3.connect(output)) as destination:
                        db.backup(destination)
                        destination.execute("PRAGMA journal_mode=DELETE")
                        if destination.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                            raise ValueError("舊資料庫完整性檢查失敗，遷移未完成，原資料保留")
                    copied["records/" + filename] = digest(output)
                settings_file = stage / "app_settings.json"
                settings = json.loads(settings_file.read_text(encoding="utf-8")) if settings_file.exists() else {}
                name = str(settings.get("station") or "預設工站")
                self._seed(stage, "ST001", name, legacy)
                last = settings.get("last_sop")
                if last:
                    source = Path(last) if Path(last).is_absolute() else legacy / last
                    try:
                        relative = source.resolve().relative_to((legacy / "sops").resolve())
                        settings["last_sop"] = (Path("sops") / relative).as_posix()
                    except ValueError:
                        if source.is_file():
                            destination = stage / "sops" / source.name
                            if destination.exists():
                                destination = destination.with_name("imported-" + uuid.uuid4().hex[:8] + "-" + destination.name)
                            shutil.copy2(source, destination)
                            settings["last_sop"] = destination.relative_to(stage).as_posix()
                if settings.get("video_path") and not Path(settings["video_path"]).is_absolute():
                    settings["video_path"] = str(legacy / settings["video_path"])
                settings["station"] = name
                atomic_json(settings_file, settings)
                # Rebase editable files; immutable historical snapshots stay unchanged.
                for path in (stage / "sops").rglob("*.json"):
                    try:
                        sop = json.loads(path.read_text(encoding="utf-8"))
                    except (ValueError, UnicodeError):
                        continue  # Preserve unreadable originals for manual repair.
                    if not isinstance(sop, dict):
                        continue
                    model = sop.get("model_path")
                    if isinstance(model, str) and model and not Path(model).is_absolute():
                        sop["model_path"] = str((legacy / model).resolve())
                        atomic_json(path, sop)
                atomic_json(stage / "migration.json", dict(version=1, source=str(legacy),
                    original_files_retained=True, original_hashes=copied))
                os.replace(stage, self.root / "ST001")
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
