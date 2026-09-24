"""不開介面，用影片跑完整流程（模型 → 判定引擎），印出事件。用於驗證 SOP 設定。

用法：
  .venv\\Scripts\\python.exe tools\\simulate_sop.py sops\\example_sop.json demo\\demo.mp4 [--fps 10]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from sop_app.detection import FrameResult  # noqa: E402
from sop_app.detector import create_detector  # noqa: E402
from sop_app.engine import SOPEngine  # noqa: E402
from sop_app.model_bundle import read_model_info  # noqa: E402
from sop_app.paths import resolve  # noqa: E402
from sop_app.sop_schema import load_sop  # noqa: E402
from sop_app.tracking import WorkpieceLocator  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sop")
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=10.0, help="取樣頻率，模擬即時推論速度")
    args = parser.parse_args()

    sop = load_sop(args.sop)
    info = read_model_info(resolve(sop.model_path))
    errors, warnings = sop.validate(info.classes)
    for message in errors + warnings:
        print(f"[檢查] {message}")
    if errors:
        raise SystemExit(1)

    detector = create_detector(info)
    engine = SOPEngine(sop)
    locator = WorkpieceLocator(sop.workpiece) if sop.workpiece else None
    capture = cv2.VideoCapture(args.video)
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(video_fps / args.fps))

    for event in engine.start(0.0):
        print(f"{0:7.2f}s  {event.message}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            timestamp = index / video_fps
            result = FrameResult(frame.shape[1], frame.shape[0], timestamp, detector.detect(frame))
            if locator:
                result.workpiece_transform, result.tracking_status = locator.locate(frame)
            for event in engine.update(result):
                if event.kind != "step_started":
                    print(f"{timestamp:7.2f}s  {event.message}")
        index += 1
    snapshot = engine.snapshot()
    print(f"結束：第 {snapshot.cycle} 輪，目前工序 {snapshot.current_index + 1}，階段 {snapshot.phase.value}")


if __name__ == "__main__":
    main()
