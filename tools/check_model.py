"""檢查模型能否載入並在圖片或影片上偵測。

用法：
  .venv\\Scripts\\python.exe tools\\check_model.py <模型.zip> <圖片/資料夾/影片> [--limit 20]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from sop_app.detector import create_detector  # noqa: E402
from sop_app.model_bundle import read_model_info  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def frames(source: Path, limit: int):
    if source.is_dir():
        for path in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:limit]:
            yield path.name, cv2.imread(str(path))
    elif source.suffix.lower() in IMAGE_SUFFIXES:
        yield source.name, cv2.imread(str(source))
    else:
        capture = cv2.VideoCapture(str(source))
        index = 0
        while index < limit:
            ok, frame = capture.read()
            if not ok:
                break
            yield f"frame {index}", frame
            index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("source")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    info = read_model_info(args.model)
    print(f"模型 {info.model_version_id} · 類別 {', '.join(info.classes)}")
    started = time.perf_counter()
    detector = create_detector(info)
    print(f"載入完成 {time.perf_counter() - started:.1f}s · {detector.device_name}")

    timings = []
    for name, frame in frames(Path(args.source), args.limit):
        started = time.perf_counter()
        detections = detector.detect(frame)
        timings.append(time.perf_counter() - started)
        found = ", ".join(f"{d.label}:{d.score:.2f}" for d in detections) or "（無）"
        print(f"{name} {frame.shape[1]}x{frame.shape[0]} {timings[-1] * 1000:.0f}ms -> {found}")
    if timings:
        print(f"平均 {sum(timings) / len(timings) * 1000:.0f} ms/幀")


if __name__ == "__main__":
    main()
