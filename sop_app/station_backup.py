"""Verified local station backups and recoverable replacement of managed data."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
import uuid
import zipfile

ENTRIES = ("sops", "app_settings.json", "records/records.db",
           "records/records.db-wal", "records/records.db-shm", "records/snapshots")
MARKER = ".restore-transaction.json"
MAX_BYTES = 100 * 1024 ** 3


class FutureDatabaseError(ValueError):
    pass


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _safe_name(name):
    path = PurePosixPath(name)
    reserved = {"con", "prn", "aux", "nul"} | {f"{prefix}{n}" for prefix in ("com", "lpt") for n in range(1, 10)}
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(not part or part in ("..", ".") or part.endswith((".", " "))
                   or part.split(".")[0].casefold() in reserved for part in name.split("/"))):
        raise ValueError("備份含無效檔案路徑")
    if name not in ("app_settings.json", "records/records.db") and not any(
            name.startswith(entry + "/") for entry in ("sops", "records/snapshots")):
        raise ValueError(f"備份含不允許的內容：{name}")
    return name


def _validate_database(path):
    from .recorder import SCHEMA_VERSION
    with closing(sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("備份資料庫完整性檢查失敗")
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("備份資料庫的關聯資料不完整")
        if db.execute("PRAGMA user_version").fetchone()[0] > SCHEMA_VERSION:
            raise FutureDatabaseError("備份資料庫版本較新，請先更新程式")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"cycles", "step_records", "alarms", "positioning_events"} <= tables:
            raise ValueError("備份不是完整的作業紀錄資料庫")
        if "sop_snapshots" in tables:
            for content, expected in db.execute("SELECT content,hash FROM sop_snapshots"):
                if hashlib.sha256(content.encode("utf-8")).hexdigest() != expected:
                    raise ValueError("備份內的 SOP 快照檢查失敗")
                json.loads(content)


def create_backup(data_root: Path, target: Path):
    """Call after draining the recorder; SQLite backup includes committed WAL data."""
    root, target = Path(data_root).resolve(), Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    for entry in ENTRIES:
        managed = root / entry
        if target == managed or managed in target.parents:
            raise ValueError("備份請存於 SOP、資料庫與截圖資料夾之外")
    with tempfile.TemporaryDirectory(prefix="sop-backup-") as folder:
        stage = Path(folder)
        for entry in ("sops", "app_settings.json", "records/snapshots"):
            source = root / entry
            if not source.exists():
                continue
            if source.is_symlink() or source.is_junction() or (source.is_dir() and any(p.is_symlink() or p.is_junction() for p in source.rglob("*"))):
                raise ValueError("資料目錄含連結，請先改為一般檔案")
            destination = stage / entry
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        source_db = root / "records/records.db"
        if not source_db.is_file():
            raise ValueError("尚無作業紀錄資料庫可備份")
        destination = stage / "records/records.db"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)) as source, \
                closing(sqlite3.connect(destination)) as output:
            source.backup(output)
            output.execute("PRAGMA journal_mode=DELETE")
        _validate_database(destination)
        files = {p.relative_to(stage).as_posix(): {"sha256": digest(p), "size": p.stat().st_size}
                 for p in stage.rglob("*") if p.is_file()}
        manifest = dict(format="sop-station-backup", version=1,
                        created_at=datetime.now(timezone.utc).isoformat(), files=files,
                        excluded="外部 SOP、模型權重、日誌與未寫入的記憶體狀態")
        metadata = root / "station.json"
        if metadata.exists():
            station = json.loads(metadata.read_text(encoding="utf-8"))
            manifest["station_id"] = station["id"]
            manifest["station_name"] = station["name"]
        temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
                for name in files:
                    archive.write(stage / name, name)
            with temp.open("r+b") as stream:
                os.fsync(stream.fileno())
            inspect_backup(temp)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    return str(target)


def _extract_verified(archive_path, stage):
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(name.casefold() for name in names)) or "manifest.json" not in names:
            raise ValueError("備份清單重複或缺少 manifest.json")
        if len(infos) > 1_000_000 or sum(info.file_size for info in infos) > MAX_BYTES:
            raise ValueError("備份超過可接受大小")
        if archive.getinfo("manifest.json").file_size > 64 * 1024 ** 2:
            raise ValueError("備份清單過大")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "sop-station-backup" or manifest.get("version") != 1:
            raise ValueError("不支援的備份格式")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(names) != set(files) | {"manifest.json"}:
            raise ValueError("備份內容與清單不一致")
        if "records/records.db" not in files:
            raise ValueError("備份缺少作業紀錄資料庫")
        for name, metadata in files.items():
            _safe_name(name)
            info = archive.getinfo(name)
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("備份含不允許的目錄或連結")
            if info.file_size != metadata["size"]:
                raise ValueError("備份檔案大小不符")
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            if digest(destination) != metadata["sha256"]:
                raise ValueError(f"備份檔案校驗失敗：{name}")
        _validate_database(stage / "records/records.db")
        for path in (stage / "sops").rglob("*.json") if (stage / "sops").exists() else []:
            from .sop_schema import SOPDefinition
            SOPDefinition.from_dict(json.loads(path.read_text(encoding="utf-8")))
        settings = stage / "app_settings.json"
        if settings.exists() and not isinstance(json.loads(settings.read_text(encoding="utf-8")), dict):
            raise ValueError("備份的應用程式設定格式無效")
        return manifest


def inspect_backup(path):
    with tempfile.TemporaryDirectory(prefix="sop-verify-") as folder:
        return _extract_verified(path, Path(folder))


def _preserve_damaged_data(root, token):
    """Keep byte-for-byte originals when the current SQLite file cannot be read."""
    destination = root / "backups" / ("damaged-before-restore-" + token)
    destination.mkdir(parents=True)
    files = {}
    for entry in ENTRIES:
        source = root / entry
        if not source.exists():
            continue
        if source.is_symlink() or source.is_junction():
            raise ValueError("原始資料含連結，無法安全保存")
        paths = source.rglob("*") if source.is_dir() else [source]
        for path in paths:
            if path.is_symlink() or path.is_junction():
                raise ValueError("原始資料含連結，無法安全保存")
            if path.is_file():
                relative = path.relative_to(root)
                copy = destination / relative
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copy)
                with copy.open("r+b") as stream:
                    os.fsync(stream.fileno())
                if digest(copy) != digest(path):
                    raise OSError("原始資料的安全副本校驗失敗")
                files[relative.as_posix()] = digest(copy)
    atomic_json(destination / "recovery-manifest.json", dict(
        format="sop-damaged-data-copy", files=files,
        note="原資料庫無法驗證；本資料夾保留原始檔供人工救援，不是一般工站備份 ZIP"))
    return destination


def _remove(path):
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def recover_restore(root: Path):
    """Rollback an interrupted restore before opening any application data."""
    root = Path(root).resolve()
    marker = root / MARKER
    if not marker.exists():
        return
    state = json.loads(marker.read_text(encoding="utf-8"))
    token = state["token"]
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("還原復原標記無效")
    transaction = root / (".restore-" + token)
    for entry in reversed(ENTRIES):
        old = transaction / "old" / entry
        live = root / entry
        if old.exists():
            _remove(live)
            live.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old, live)
        elif not state["existing"][entry]:
            _remove(live)
    marker.unlink()
    shutil.rmtree(transaction, ignore_errors=True)


def restore_backup(root: Path, archive_path: Path):
    root = Path(root).resolve()
    recover_restore(root)
    token = uuid.uuid4().hex
    transaction = root / (".restore-" + token)
    stage = transaction / "new"
    stage.mkdir(parents=True)
    marked = False
    try:
        manifest = _extract_verified(archive_path, stage)
        if (root / "station.json").exists():
            station = json.loads((root / "station.json").read_text(encoding="utf-8"))
            source_id = manifest.get("station_id")
            if (source_id and source_id != station["id"]) or (not source_id and station["id"] != "ST001"):
                raise ValueError("備份不屬於目前工站；請先切換至正確工站。無工站資訊的舊備份僅能還原至預設工站 ST001。")
        # A valid safety backup is required before replacing any data.
        safety = root / "backups" / ("before-restore-" + token + ".zip")
        try:
            create_backup(root, safety)
        except (sqlite3.DatabaseError, ValueError) as exc:
            if isinstance(exc, (sqlite3.OperationalError, FutureDatabaseError)):
                raise  # A locked/full disk is not permission to replace live data.
            safety = _preserve_damaged_data(root, token)
        else:
            # Close/checkpoint WAL through SQLite, never delete sidecars manually.
            with closing(sqlite3.connect(root / "records/records.db", timeout=1)) as db:
                if db.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                    raise ValueError("無法取得資料庫獨佔使用權，請關閉其他程式後重試")
            for suffix in ("-wal", "-shm"):
                if (root / ("records/records.db" + suffix)).exists():
                    raise ValueError("作業資料庫仍被使用，請關閉其他程式後重試")
        state = dict(token=token, existing={entry: (root / entry).exists() for entry in ENTRIES})
        atomic_json(root / MARKER, state)
        marked = True
        for entry in ENTRIES:
            live, new, old = root / entry, stage / entry, transaction / "old" / entry
            if live.exists():
                old.parent.mkdir(parents=True, exist_ok=True)
                os.replace(live, old)
            if new.exists():
                live.parent.mkdir(parents=True, exist_ok=True)
                os.replace(new, live)
        (root / MARKER).unlink()
        marked = False
        return str(safety)
    except Exception:
        if marked:
            recover_restore(root)
        raise
    finally:
        # On rollback failure keep the marker and original data for startup recovery.
        if not (root / MARKER).exists():
            shutil.rmtree(transaction, ignore_errors=True)
