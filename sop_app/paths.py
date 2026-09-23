"""專案路徑工具：設定檔中的相對路徑一律以專案根目錄為基準。"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOPS_DIR = PROJECT_ROOT / "sops"
RECORDS_DIR = PROJECT_ROOT / "records"
APP_SETTINGS = PROJECT_ROOT / "app_settings.json"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relativize(path: str | Path) -> str:
    """能轉成專案內相對路徑就轉，讓整個專案資料夾搬移後設定檔仍可用。"""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
