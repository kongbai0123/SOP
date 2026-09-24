"""將模型 ZIP 正規化為根目錄模型包。

舊版匯出工具或 Windows 檔案總管可能把整個模型資料夾再包一層；
本工具會移除那一層，輸出固定包含 model/model.json 與
model/checkpoint.pt 的 ZIP，原始檔案不會被覆寫。

用法：
  .venv\\Scripts\\python.exe tools\\normalize_model_bundle.py <模型.zip>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


class BundleNormalizationError(RuntimeError):
    pass


def normalize_model_bundle(source: str | Path, output: str | Path | None = None) -> Path:
    source = Path(source)
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise BundleNormalizationError(f"請選擇模型 ZIP：{source}")
    target = Path(output) if output is not None else source.with_name(f"{source.stem}-flat.zip")
    if target.exists():
        raise FileExistsError(f"輸出檔案已存在：{target}")

    try:
        with zipfile.ZipFile(source) as bundle:
            if bundle.testzip() is not None:
                raise BundleNormalizationError("來源 ZIP CRC 驗證失敗")
            entries = [info for info in bundle.infolist() if not info.is_dir()]
            names = [_normalize(info.filename) for info in entries]
            candidates = [name for name in names if PurePosixPath(name).name.lower() == "model.json"]
            if len(candidates) != 1:
                raise BundleNormalizationError("模型 ZIP 必須包含唯一的 model.json")
            model_member = candidates[0]
            raw = bundle.read(entries[names.index(model_member)])
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BundleNormalizationError("model.json 不是有效的 JSON") from exc
            checkpoint = str(record.get("checkpoint") or "checkpoint.pt")
            if not checkpoint or "/" in checkpoint or "\\" in checkpoint:
                raise BundleNormalizationError(f"model.json 的 checkpoint 檔名無效：{checkpoint}")

            model_path = PurePosixPath(model_member)
            prefix = str(model_path.parent.parent) + "/" if model_path.parent.name == "model" else ""
            mapped: dict[str, bytes] = {}
            for info, name in zip(entries, names):
                canonical = name[len(prefix):] if prefix and name.startswith(prefix) else name
                canonical = canonical.lstrip("/")
                if not canonical:
                    continue
                if canonical in mapped:
                    raise BundleNormalizationError(f"移除外層資料夾後檔名衝突：{canonical}")
                mapped[canonical] = bundle.read(info)

            required = {"model/model.json", f"model/{checkpoint}"}
            missing = sorted(required - mapped.keys())
            if missing:
                raise BundleNormalizationError("模型 ZIP 缺少：" + "、".join(missing))
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".tmp", dir=target.parent)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output_zip:
                    for name in sorted(mapped):
                        output_zip.writestr(name, mapped[name])
                with zipfile.ZipFile(temporary) as output_zip:
                    if output_zip.testzip() is not None:
                        raise BundleNormalizationError("輸出 ZIP CRC 驗證失敗")
                temporary.replace(target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
    except zipfile.BadZipFile as exc:
        raise BundleNormalizationError(f"來源 ZIP 損毀：{source}") from exc
    return target


def _normalize(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(normalize_model_bundle(args.source, args.output))


if __name__ == "__main__":
    main()
