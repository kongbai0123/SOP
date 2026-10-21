"""Keep installed resources separate from each Windows user's mutable data."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

PRODUCT_ID = "SOP.WorkInstruction"
FROZEN = bool(getattr(sys, "frozen", False))
SOURCE_ROOT = Path(__file__).resolve().parent.parent
INSTALL_ROOT = Path(sys.executable).resolve().parent if FROZEN else SOURCE_ROOT
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", INSTALL_ROOT)) if FROZEN else SOURCE_ROOT


def user_data_root(local_appdata: str | Path | None = None) -> Path:
    """Return a user-writable location independent of the process cwd."""
    base = local_appdata or os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / "AppData" / "Local") / PRODUCT_ID


# Keep PROJECT_ROOT as the base of saved relative paths for existing consumers.
# In an installed build it deliberately names writable data, never _MEIPASS.
DATA_ROOT = user_data_root() if FROZEN else SOURCE_ROOT
PROJECT_ROOT = DATA_ROOT
SOPS_DIR = DATA_ROOT / "sops"
RECORDS_DIR = DATA_ROOT / "records"
LOGS_DIR = DATA_ROOT / "logs"
APP_SETTINGS = DATA_ROOT / "app_settings.json"
MODELS_DIR = DATA_ROOT / "models"


def initialize_runtime() -> None:
    """Create writable directories and seed a clean example once after install."""
    for directory in (SOPS_DIR, RECORDS_DIR, LOGS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    # Recover a stopped restore before settings/SOP files are read by startup.
    if (DATA_ROOT / ".restore-transaction.json").exists():
        from PySide6.QtCore import QLockFile
        from .station_backup import recover_restore
        recovery_lock = QLockFile(str(RECORDS_DIR / ".writer.lock"))
        recovery_lock.setStaleLockTime(0)
        if not recovery_lock.tryLock(0):
            raise RuntimeError("工站資料仍被另一個程式使用，無法復原未完成的還原")
        try:
            recover_restore(DATA_ROOT)
        finally:
            recovery_lock.unlock()
    if FROZEN:
        source = RESOURCE_ROOT / "sops" / "example_sop.json"
        destination = SOPS_DIR / "example_sop.json"
        if source.is_file():
            try:
                # Exclusive creation protects edits and concurrent first launches.
                with destination.open("xb") as output, source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
            except FileExistsError:
                pass


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def models_directory(root: str | Path | None = None) -> Path:
    """Share model bundles across stations, outside their backed-up data."""
    if root is None:
        return MODELS_DIR
    root = Path(root)
    if root.parent.name == "stations" and root.parent.parent.name == "data":
        return root.parent.parent.parent / "models"
    return root / "models"


def resolve_model(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve shared models while preserving existing station-local paths."""
    path = Path(path)
    if path.is_absolute():
        return path
    local = (Path(root) if root is not None else PROJECT_ROOT) / path
    if path.parts and path.parts[0].casefold() == "models" and not local.exists():
        return models_directory(root).joinpath(*path.parts[1:])
    return local


def relativize(path: str | Path) -> str:
    """Keep paths within the writable data root portable between machines."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
