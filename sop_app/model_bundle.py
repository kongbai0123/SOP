"""讀取 Vision Workbench 匯出的模型包（.zip 或含 model.json 的資料夾）。

模型包可能由不同版本的匯出工具產生，外層可以是 ZIP 根目錄，也可以
包在一個專案名稱資料夾內。這個模組只負責找到並驗證模型內容；推論
引擎由 :mod:`sop_app.detector` 依 ``engine`` 建立。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import zipfile


@dataclass(frozen=True)
class EngineSpec:
    name: str
    runtime: str
    has_masks: bool


# 可用於工序判定的偵測引擎。新增模型時只要先登錄引擎規格，再由
# detector.py 提供對應的輸出轉換，就不需要改動 ZIP 讀取流程。
ENGINES = {
    "maskrcnn_resnet50_fpn": EngineSpec("Mask R-CNN · ResNet50 FPN", "torchvision", True),
    "fasterrcnn_mobilenet_v3_large_fpn": EngineSpec("Faster R-CNN · MobileNet V3 FPN", "torchvision", False),
    "fasterrcnn_mobilenet_v3_large_320_fpn": EngineSpec("Faster R-CNN · MobileNet V3 320 FPN", "torchvision", False),
    "fasterrcnn_resnet50_fpn_v2": EngineSpec("Faster R-CNN · ResNet50 FPN V2", "torchvision", False),
    "yolo26n_detect": EngineSpec("YOLO26n Detect", "ultralytics", False),
    "yolo26s_detect": EngineSpec("YOLO26s Detect", "ultralytics", False),
    "yolo26n_seg": EngineSpec("YOLO26n Seg", "ultralytics", True),
    "yolo26s_seg": EngineSpec("YOLO26s Seg", "ultralytics", True),
    "rt_detr_r50": EngineSpec("RT-DETR · ResNet50", "ultralytics", False),
}

# 舊版工作台可以匯出、但沒有個別物件位置，無法用於數量或區域判定的模型。
UNUSABLE_ENGINES = {
    "deeplabv3_mobilenet_v3_large": "語意分割只標出像素類別、沒有個別物件，無法計算數量",
    "deeplabv3_resnet50": "語意分割只標出像素類別、沒有個別物件，無法計算數量",
    "mobilenet_v3_large_classification": "影像分類沒有物件位置，無法配合偵測區域判定",
    "efficientnet_b0_classification": "影像分類沒有物件位置，無法配合偵測區域判定",
    "resnet18_classification": "影像分類沒有物件位置，無法配合偵測區域判定",
    "pixel_prototype_v1": "工作台內建基準模型沒有可部署的 checkpoint",
}

SUPPORTED_ENGINES = frozenset(ENGINES)
RUNTIME_INSTALL = {
    "ultralytics": "YOLO／RT-DETR 模型需要 ultralytics 套件：\n"
                   r"  .venv\Scripts\python.exe -m pip install --no-deps ultralytics==8.4.150",
    "torchvision": "需要 torch 與 torchvision：\n"
                   r"  .venv\Scripts\python.exe -m pip install -r requirements.txt",
}


class ModelBundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelInfo:
    source: Path
    engine: str
    engine_name: str
    runtime: str
    has_masks: bool
    classes: tuple[str, ...]
    image_size: int
    score_threshold: float
    model_version_id: str
    project_name: str
    checkpoint_name: str = "checkpoint.pt"
    checkpoint_sha256: str | None = None
    # ZIP 內實際的成員路徑。它可以包含任意外層資料夾。
    checkpoint_member: str | None = None
    # 非 ZIP 模型的實際 checkpoint 路徑，支援資料夾包在外層資料夾內。
    checkpoint_path: Path | None = None

    @property
    def is_zip(self) -> bool:
        return self.source.suffix.lower() == ".zip"


def read_model_info(path: str | Path) -> ModelInfo:
    """讀取模型中繼資料，接受標準包與任意單一外層資料夾包裝。"""
    source = Path(path)
    if not source.exists():
        raise ModelBundleError(f"找不到模型：{source}")
    if source.is_file() and source.name.lower() == "model.json":
        source = source.parent

    manifest: dict = {}
    checkpoint_member: str | None = None
    checkpoint_path: Path | None = None
    if source.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(source) as bundle:
                record, _model_member, checkpoint_member, manifest = _read_zip_bundle(bundle)
        except zipfile.BadZipFile as exc:
            raise ModelBundleError(f"模型包損毀：{source}") from exc
    elif source.is_dir():
        record, checkpoint_path, manifest = _read_directory_bundle(source)
    else:
        raise ModelBundleError("請選擇模型包 .zip 或 model.json")

    engine = str(record.get("engine", ""))
    spec = ENGINES.get(engine)
    if spec is None:
        if engine in UNUSABLE_ENGINES:
            raise ModelBundleError(f"{engine} 無法用於工序判定：{UNUSABLE_ENGINES[engine]}")
        raise ModelBundleError(f"尚未支援的模型引擎：{engine or '未知'}\n"
                               f"目前支援：{'、'.join(sorted(ENGINES))}")

    try:
        classes = tuple(str(value) for value in record["classes"])
    except (KeyError, TypeError) as exc:
        raise ModelBundleError("model.json 缺少有效的 classes 類別清單") from exc
    if not classes or any(not name for name in classes):
        raise ModelBundleError("model.json 的 classes 不可為空")

    checkpoint_name = _checkpoint_name(record)
    checkpoint_sha = _manifest_checkpoint_sha(manifest, checkpoint_member, checkpoint_path)
    return ModelInfo(
        source=source,
        engine=engine,
        engine_name=str(record.get("engine_name") or spec.name),
        runtime=spec.runtime,
        has_masks=spec.has_masks,
        classes=classes,
        image_size=int(record.get("image_size", 640)),
        score_threshold=float(record.get("score_threshold", 0.5)),
        model_version_id=str(record.get("model_version_id", "")),
        project_name=str(manifest.get("project_name", "")),
        checkpoint_name=checkpoint_name,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_member=checkpoint_member,
        checkpoint_path=checkpoint_path,
    )


def _read_zip_bundle(bundle: zipfile.ZipFile) -> tuple[dict, str, str, dict]:
    """在 ZIP 中尋找唯一一組 model.json 與同層 checkpoint。"""
    members: dict[str, str] = {}
    for info in bundle.infolist():
        normalized = _normalize_member(info.filename)
        if normalized and not normalized.endswith("/"):
            members.setdefault(normalized, info.filename)

    candidates: list[tuple[dict, str, str]] = []
    for model_member in sorted(members):
        if PurePosixPath(model_member).name.lower() != "model.json":
            continue
        try:
            record = _read_json_bytes(bundle.read(members[model_member]), model_member)
            checkpoint_name = _checkpoint_name(record)
        except ModelBundleError:
            continue
        parent = PurePosixPath(model_member).parent
        checkpoint_member = _normalize_member(str(parent / checkpoint_name))
        if checkpoint_member in members:
            candidates.append((record, model_member, checkpoint_member))

    if not candidates:
        if not any(PurePosixPath(name).name.lower() == "model.json" for name in members):
            raise ModelBundleError("模型包找不到 model.json（可位於 model/ 或任意外層資料夾）")
        raise ModelBundleError("模型包內的 model.json 找不到同層 checkpoint.pt")
    if len(candidates) > 1:
        paths = "、".join(item[1] for item in candidates)
        raise ModelBundleError(f"模型包包含多組模型，無法判定要載入哪一組：{paths}")

    record, model_member, checkpoint_member = candidates[0]
    manifest = _read_zip_manifest(bundle, members, model_member)
    return record, model_member, checkpoint_member, manifest


def _read_zip_manifest(bundle: zipfile.ZipFile, members: dict[str, str], model_member: str) -> dict:
    model_path = PurePosixPath(model_member)
    package_root = model_path.parent.parent if model_path.parent.name == "model" else model_path.parent
    choices = [str(package_root / "export-manifest.json"), "export-manifest.json"]
    for candidate in choices:
        candidate = _normalize_member(candidate)
        if candidate in members:
            return _read_json_bytes(bundle.read(members[candidate]), candidate)
    return {}


def _read_directory_bundle(source: Path) -> tuple[dict, Path, dict]:
    candidates: list[tuple[dict, Path, Path]] = []
    for model_path in sorted(source.rglob("model.json")):
        try:
            record = _read_json_bytes(model_path.read_bytes(), str(model_path))
            checkpoint_name = _checkpoint_name(record)
        except (OSError, ModelBundleError):
            continue
        checkpoint_path = model_path.parent / checkpoint_name
        if checkpoint_path.is_file():
            candidates.append((record, model_path, checkpoint_path))

    if not candidates:
        raise ModelBundleError("資料夾內找不到成對的 model.json 與 checkpoint.pt")
    if len(candidates) > 1:
        paths = "、".join(str(item[1]) for item in candidates)
        raise ModelBundleError(f"資料夾包含多組模型，無法判定要載入哪一組：{paths}")

    record, model_path, checkpoint_path = candidates[0]
    manifest = {}
    for candidate in (model_path.parent / "export-manifest.json",
                      model_path.parent.parent / "export-manifest.json",
                      source / "export-manifest.json"):
        try:
            if candidate.is_file():
                manifest = _read_json_bytes(candidate.read_bytes(), str(candidate))
                break
        except OSError:
            continue
    return record, checkpoint_path, manifest


def _read_json_bytes(raw: bytes, name: str) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelBundleError(f"{name} 不是有效的 JSON") from exc
    if not isinstance(value, dict):
        raise ModelBundleError(f"{name} 必須是 JSON 物件")
    return value


def _normalize_member(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def _checkpoint_name(record: dict) -> str:
    """只接受模型目錄內的單純檔名，避免 ZIP 路徑跳脫。"""
    name = str(record.get("checkpoint") or "checkpoint.pt")
    if not name or name in (".", "..") or "/" in name or "\\" in name or PurePosixPath(name).name != name:
        raise ModelBundleError(f"model.json 的 checkpoint 檔名無效：{name}")
    return name


def _manifest_checkpoint_sha(manifest: dict, member: str | None, path: Path | None) -> str | None:
    if not isinstance(manifest, dict):
        return None
    expected = _normalize_member(str(member)) if member else _normalize_member(path.name) if path else ""
    relative = f"model/{path.name}" if path and path.parent.name == "model" else ""
    for artifact in manifest.get("artifacts", []):
        if not isinstance(artifact, dict) or not artifact.get("sha256"):
            continue
        artifact_path = _normalize_member(str(artifact.get("path", "")))
        if (artifact_path == expected or
                (relative and (artifact_path == relative or expected.endswith("/" + artifact_path))) or
                (expected and artifact_path.endswith("/" + PurePosixPath(expected).name))):
            return str(artifact["sha256"])
    return None


def ensure_runtime(info: ModelInfo) -> None:
    """在推論前檢查引擎套件，錯誤訊息直接指出安裝方式。"""
    if importlib.util.find_spec(info.runtime) is None:
        raise ModelBundleError(f"{info.engine_name} 需要的 {info.runtime} 套件尚未安裝。\n"
                               f"{RUNTIME_INSTALL.get(info.runtime, '')}")


def ensure_checkpoint(info: ModelInfo) -> Path:
    """取得 checkpoint；ZIP 會解壓到快取並驗證大小與 SHA-256。"""
    if not info.is_zip:
        checkpoint = info.checkpoint_path or (info.source / info.checkpoint_name)
        if not checkpoint.is_file():
            raise ModelBundleError(f"找不到 checkpoint：{checkpoint}")
        return checkpoint

    member = info.checkpoint_member or f"model/{info.checkpoint_name}"
    target = info.source.parent / ".extracted" / info.source.stem / info.checkpoint_name
    try:
        with zipfile.ZipFile(info.source) as bundle:
            entry = bundle.getinfo(member)
            if target.exists() and target.stat().st_size == entry.file_size:
                if not info.checkpoint_sha256 or _sha256_file(target) == info.checkpoint_sha256:
                    return target

            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            digest = hashlib.sha256()
            with bundle.open(entry) as src, partial.open("wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    digest.update(chunk)
                    dst.write(chunk)
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ModelBundleError(f"無法讀取模型 checkpoint：{member}") from exc

    if info.checkpoint_sha256 and digest.hexdigest() != info.checkpoint_sha256:
        partial.unlink(missing_ok=True)
        raise ModelBundleError(f"{info.checkpoint_name} 的 SHA-256 與模型包紀錄不符，檔案可能損毀")
    partial.replace(target)
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
