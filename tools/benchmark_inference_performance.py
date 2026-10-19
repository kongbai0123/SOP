"""Compare real, batch-one inference on identical video frames without changing station data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root.resolve()))
    from sop_app.native_runtime import cv2
    import numpy as np
    from sop_app.inference_worker import InferenceWorker

    capture = cv2.VideoCapture(str(args.source.resolve()))
    images, input_hashes = [], []
    try:
        for index in range(args.frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index * 3)
            ok, image = capture.read()
            if not ok:
                break
            images.append(image)
            input_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
    finally:
        capture.release()
    if not images:
        raise ValueError("Source contains no readable frames")
    samples, predictions, stage_samples = [], [], {}
    worker = InferenceWorker().start()
    started = time.perf_counter()
    try:
        detector = worker.load_model(args.model.resolve())
        load_ms = (time.perf_counter() - started) * 1000
        for _ in range(5):
            detector.detect(images[0])
        for image in images:
            tick = time.perf_counter()
            detections = detector.detect(image)
            samples.append((time.perf_counter() - tick) * 1000)
            for key, value in getattr(detector, "last_metrics", {}).items():
                if isinstance(value, (int, float)):
                    stage_samples.setdefault(key, []).append(value)
            predictions.append([{"label": d.label, "score": float(d.score), "box": list(d.box),
                                 "mask_sha256": hashlib.sha256(d.mask.tobytes()).hexdigest()
                                 if d.mask is not None else None} for d in detections])
        def summary(values):
            return {"count": len(values), "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99))}
        result = {"project_root": str(args.project_root.resolve()), "device": detector.device_name,
                  "runtime": getattr(detector, "runtime_info", {}), "load_ms": load_ms,
                  "frames": len(images), "shapes_hwc": sorted({tuple(i.shape) for i in images}),
                  "input_hashes": input_hashes, "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  "rpc_ms": summary(samples), "stages_ms": {k: summary(v) for k, v in stage_samples.items()},
                  "predictions": predictions,
                  "notes": "Warm batch-one RPC latency; video decode excluded; no camera or station writes."}
    finally:
        worker.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("predictions", "input_hashes")},
                     ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
