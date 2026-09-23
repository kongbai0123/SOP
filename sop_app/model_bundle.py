"""讀取 Vision Workbench 匯出的模型包（.zip 或含 model.json 的資料夾）。

只讀取中繼資料不需要 torch，因此編輯器可以在模型尚未載入前就取得類別清單。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile

SUPPORTED_ENGINES = {"maskrcnn_resnet50_fpn"}
_ZIP_MODEL_JSON = "model/model.json"
_ZIP_CHECKPOINT = "model/checkpoint.pt"


class ModelBundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    source: Path
    engine: str
    classes: tuple[str, ...]
    image_size: int
    score_threshold: float
    model_version_id: str
    project_name: str
    checkpoint_sha256: str | None = None

    @property
    def is_zip(self) -> bool:
        return self.source.suffix.lower() == ".zip"


def read_model_info(path: str | Path) -> ModelInfo:
    path = Path(path)
    if not path.exists():
        raise ModelBundleError(f"找不到模型：{path}")
    if path.is_file() and path.name == "model.json":
        path = path.parent

    manifest: dict = {}
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as bundle:
                names = set(bundle.namelist())
                if _ZIP_MODEL_JSON not in names or _ZIP_CHECKPOINT not in names:
                    raise ModelBundleError("模型包缺少 model/model.json 或 model/checkpoint.pt")
                record = json.loads(bundle.read(_ZIP_MODEL_JSON).decode("utf-8"))
                if "export-manifest.json" in names:
                    manifest = json.loads(bundle.read("export-manifest.json").decode("utf-8"))
        except zipfile.BadZipFile as exc:
            raise ModelBundleError(f"模型包損毀：{path}") from exc
    elif path.is_dir():
        record_path = path / "model.json"
        if not record_path.exists() or not (path / "checkpoint.pt").exists():
            raise ModelBundleError("資料夾內需要 model.json 與 checkpoint.pt")
        record = json.loads(record_path.read_text(encoding="utf-8"))
    else:
        raise ModelBundleError("請選擇模型包 .zip 或 model.json")

    engine = record.get("engine", "")
    if engine not in SUPPORTED_ENGINES:
        raise ModelBundleError(f"尚未支援的模型引擎：{engine or '未知'}")

    checkpoint_sha = next((a.get("sha256") for a in manifest.get("artifacts", [])
                           if a.get("path") == _ZIP_CHECKPOINT), None)
    return ModelInfo(
        source=path,
        engine=engine,
        classes=tuple(record["classes"]),
        image_size=int(record.get("image_size", 640)),
        score_threshold=float(record.get("score_threshold", 0.5)),
        model_version_id=str(record.get("model_version_id", "")),
        project_name=str(manifest.get("project_name", "")),
        checkpoint_sha256=checkpoint_sha,
    )


def ensure_checkpoint(info: ModelInfo) -> Path:
    """回傳 checkpoint 路徑；zip 模型包第一次使用時解壓到旁邊的 .extracted 資料夾並驗證 SHA-256。"""
    if not info.is_zip:
        return info.source / "checkpoint.pt"

    target = info.source.parent / ".extracted" / info.source.stem / "checkpoint.pt"
    with zipfile.ZipFile(info.source) as bundle:
        expected_size = bundle.getinfo(_ZIP_CHECKPOINT).file_size
        if target.exists() and target.stat().st_size == expected_size:
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        digest = hashlib.sha256()
        with bundle.open(_ZIP_CHECKPOINT) as src, partial.open("wb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                digest.update(chunk)
                dst.write(chunk)

    if info.checkpoint_sha256 and digest.hexdigest() != info.checkpoint_sha256:
        partial.unlink(missing_ok=True)
        raise ModelBundleError("checkpoint.pt 的 SHA-256 與模型包紀錄不符，檔案可能損毀")
    partial.replace(target)
    return target
