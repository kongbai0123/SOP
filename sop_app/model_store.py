"""Import model bundles into a shared, content-addressed writable library."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
import zlib

from .model_bundle import ModelBundleError, ModelInfo, read_model_info

_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def _safe_name(value: str) -> str:
    """Return one portable Windows filename component, without changing sources."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().rstrip(". ")
    if not name or name in (".", ".."):
        name = "model"
    if name.split(".", 1)[0].upper() in _RESERVED:
        name = "_" + name
    # Leave room for a full SHA-256 and the app's data root on Windows.
    return name[:64].rstrip(". ") or "model"


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_files(info: ModelInfo) -> list[tuple[str, Path]]:
    checkpoint = info.checkpoint_path or info.source / info.checkpoint_name
    files = [("model.json", checkpoint.parent / "model.json"),
             (info.checkpoint_name, checkpoint)]
    for candidate in (checkpoint.parent / "export-manifest.json",
                      checkpoint.parent.parent / "export-manifest.json",
                      info.source / "export-manifest.json"):
        if candidate.is_file():
            files.append(("export-manifest.json", candidate))
            break
    return files


def _directory_digest(info: ModelInfo) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(_directory_files(info)):
        # Include boundaries and names so different file layouts cannot alias.
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint(info: ModelInfo) -> None:
    if info.is_zip:
        # Read only this member; no writes beside an external or read-only ZIP.
        with zipfile.ZipFile(info.source) as archive:
            member = info.checkpoint_member or f"model/{info.checkpoint_name}"
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
    else:
        actual = _digest_file(info.checkpoint_path or info.source / info.checkpoint_name)
    if info.checkpoint_sha256 and actual != info.checkpoint_sha256:
        raise ModelBundleError(f"{info.checkpoint_name} 的 SHA-256 與模型包紀錄不符，檔案可能損毀")


def _existing_bundle(target: Path, expected: str, is_zip: bool) -> Path:
    """Reuse a completed concurrent import; never overwrite a damaged library."""
    if is_zip:
        candidates = [path for path in target.iterdir() if path.is_file() and path.suffix.lower() == ".zip"]
        if len(candidates) != 1 or _digest_file(candidates[0]) != expected:
            raise ModelBundleError(f"模型庫內已有內容不符的模型，請檢查：{target}")
        source = candidates[0]
    else:
        source = target
    info = read_model_info(source)
    if not is_zip and _directory_digest(info) != expected:
        raise ModelBundleError(f"模型庫內已有內容不符的模型，請檢查：{target}")
    _verify_checkpoint(info)
    return source


def import_model(source: str | Path, models_dir: Path | None = None) -> Path:
    """Copy a valid ZIP or model directory into the shared models directory.

    Originals are retained. Imports with identical content share one location;
    a staged directory is published only after metadata and weights validate.
    """
    if models_dir is None:
        from .paths import MODELS_DIR
        models_dir = MODELS_DIR
    root = Path(models_dir).resolve()
    stage = None
    try:
        info = read_model_info(Path(source).resolve())
        if info.source.resolve().is_relative_to(root):
            _verify_checkpoint(info)
            return info.source.resolve()
        root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".import-", dir=root))
        if info.is_zip:
            filename = _safe_name(info.source.stem) + ".zip"
            copied = stage / filename
            shutil.copyfile(info.source, copied)
            copied_info = read_model_info(copied)
            content_hash = _digest_file(copied)
        else:
            for name, path in _directory_files(info):
                shutil.copyfile(path, stage / name)
            copied = stage
            copied_info = read_model_info(copied)
            content_hash = _directory_digest(copied_info)
        _verify_checkpoint(copied_info)
        label = _safe_name(copied_info.project_name or copied_info.model_version_id or "model")
        target = root / f"{label}-{content_hash}"
        if target.exists():
            return _existing_bundle(target, content_hash, copied_info.is_zip)
        try:
            stage.rename(target)
        except OSError:
            # Another process may have atomically published the same bundle.
            if target.is_dir():
                return _existing_bundle(target, content_hash, copied_info.is_zip)
            raise
        return target / copied.name if copied_info.is_zip else target
    except ModelBundleError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
        raise ModelBundleError(f"無法匯入模型：{source}\n{exc}") from exc
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
