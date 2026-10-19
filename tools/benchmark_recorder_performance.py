"""Compare recorder revisions with identical synthetic images and temporary data.

Run this file from either checkout and select the implementation with
``--project-root``. No model, camera, station settings, or production data is read.
The original WAL/FULL settings and snapshot fsync remain enabled. Run benchmarks
one at a time: other CPU/disk workloads can materially affect the measurements.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import sys
import tempfile
import threading
import time
from unittest.mock import patch


def distribution(values):
    import numpy as np
    if not values:
        return {"samples": 0}
    return {"samples": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(max(values))}


def elapsed_samples(action, count):
    values = []
    for _ in range(count):
        started = time.perf_counter()
        action()
        values.append((time.perf_counter() - started) * 1000)
    return distribution(values)


def synthetic_frame(width, height, seed):
    import numpy as np
    yy, xx = np.indices((height, width))
    gradient = np.stack((xx * 255 // width, yy * 255 // height,
                         (xx + yy) * 255 // (width + height)), axis=2)
    noise = np.random.default_rng(seed).integers(-15, 16, size=gradient.shape, dtype=np.int16)
    return np.clip(gradient + noise, 0, 255).astype(np.uint8)


def benchmark_resolution(width, height, args):
    from sop_app.engine import EngineEvent
    from sop_app.native_runtime import cv2
    from sop_app.recorder import AsyncRecorder, Recorder, query

    frame = synthetic_frame(width, height, args.seed)
    def encode():
        ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Synthetic image could not be encoded")
        return image

    image = encode()
    decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
    for _ in range(args.warmup):
        encode()
        cv2.imdecode(image, cv2.IMREAD_COLOR)
    result = {"width": width, "height": height, "fixture_seed": args.seed,
              "fixture_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
              "jpeg_bytes": int(image.nbytes),
              "first_jpeg_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
              "codec_encode": elapsed_samples(encode, args.codec_samples),
              "codec_decode": elapsed_samples(lambda: cv2.imdecode(image, cv2.IMREAD_COLOR), args.codec_samples),
              "codec_reencode": elapsed_samples(lambda: cv2.imencode(".jpg", decoded,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 85]), args.codec_samples)}

    # Recorder.handle is measured on its own background thread. It overlaps the
    # caller; its duration must not be added to foreground enqueue duration.
    formal_write = []
    complete = threading.Event()
    completion_at = []
    errors = []
    original_handle = Recorder.handle
    def timed_handle(*call_args, **kwargs):
        started = time.perf_counter()
        answer = original_handle(*call_args, **kwargs)
        formal_write.append((time.perf_counter() - started) * 1000)
        return answer

    def written():
        completion_at.append(time.perf_counter())
        complete.set()

    with tempfile.TemporaryDirectory(prefix="sop-recorder-benchmark-") as folder:
        root = Path(folder) / "records"
        with patch.object(Recorder, "handle", new=timed_handle):
            writer = AsyncRecorder(root, on_written=written, on_error=errors.append)
            try:
                def submit(events, *, evidence=None, snapshot=None):
                    complete.clear()
                    started = time.perf_counter()
                    writer.handle(events, "Synthetic benchmark", 1, "synthetic",
                                  frame=evidence, sop_snapshot=snapshot)
                    returned = time.perf_counter()
                    if not complete.wait(10):
                        raise RuntimeError(f"Recorder did not finish: {errors}")
                    return (returned - started) * 1000, (completion_at[-1] - started) * 1000

                submit([EngineEvent("cycle_started", 0, 1)], snapshot={"name": "Synthetic benchmark", "steps": []})
                for index in range(args.warmup):
                    submit([EngineEvent("step_completed", index + 1, 1, index, "Warmup",
                                        data={"status": "done", "duration": 1})], evidence=frame)
                formal_write.clear()
                foreground, end_to_end = [], []
                enqueue_stages, write_stages = {}, {}
                for index in range(args.samples):
                    values = submit([EngineEvent("step_completed", index + args.warmup + 1,
                                    1, index + args.warmup, "Synthetic step",
                                    data={"status": "done", "duration": 1})], evidence=frame)
                    foreground.append(values[0])
                    end_to_end.append(values[1])
                    for attribute, target in (("last_enqueue_metrics", enqueue_stages),
                                              ("last_write_metrics", write_stages)):
                        for key, value in getattr(writer, attribute, {}).items():
                            target.setdefault(key, []).append(value)
                if errors:
                    raise RuntimeError(f"Recorder reported errors: {errors}")
                expected = args.samples + args.warmup
                count = query(root / "records.db", "SELECT COUNT(*) FROM step_records")[0][0]
                pending = writer.pending_count()
                if count != expected or pending:
                    raise AssertionError(f"Lost records: steps={count}, expected={expected}, pending={pending}")
                snapshot_name = query(root / "records.db", "SELECT snapshot FROM step_records ORDER BY id DESC LIMIT 1")[0][0]
                saved = (root / "snapshots" / snapshot_name).read_bytes()
                with closing(sqlite3.connect(root / "pending.db")) as db:
                    pending_mode = db.execute("PRAGMA journal_mode").fetchone()[0]
                result.update(foreground_handle=distribution(foreground),
                              end_to_end_written_callback=distribution(end_to_end),
                              formal_handle_background=distribution(formal_write),
                              enqueue_stages={key: distribution(values) for key, values in enqueue_stages.items()},
                              write_stages={key: distribution(values) for key, values in write_stages.items()},
                              checked_step_rows=count, pending_after_run=pending,
                              saved_jpeg_sha256=hashlib.sha256(saved).hexdigest(),
                              saved_equals_first_encoding=saved == image.tobytes(),
                              pending_journal_mode=pending_mode)
            finally:
                if not writer.close():
                    raise RuntimeError("Recorder failed to stop; temporary data retained by OS until process exit")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--codec-samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--resolutions", nargs="+", choices=("720p", "1080p"), default=("720p", "1080p"))
    args = parser.parse_args()
    if args.samples < 1 or args.codec_samples < 1 or args.warmup < 0:
        parser.error("sample counts must be positive and warmup must be nonnegative")
    project = args.project_root.resolve(strict=True)
    if not (project / "sop_app" / "recorder.py").is_file():
        parser.error("--project-root must contain sop_app/recorder.py")
    sys.path.insert(0, str(project))
    import numpy as np
    from sop_app.native_runtime import cv2
    sizes = {"720p": (1280, 720), "1080p": (1920, 1080)}
    report = {"benchmark": "recorder-performance-v1", "project_root": str(project),
              "run_parameters": {"samples": args.samples, "codec_samples": args.codec_samples,
                                 "warmup": args.warmup, "seed": args.seed,
                                 "resolutions": args.resolutions},
              "python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
              "temporary_data_only": True, "temp_root": tempfile.gettempdir(), "jpeg_quality": 85,
              "durability": "Original recorder settings: WAL, synchronous=FULL and snapshot fsync are unchanged.",
              "definitions": {
                  "foreground_handle": "Caller wall time through encoding, durable pending commit and status callback.",
                  "end_to_end_written_callback": "Wall time from caller entry until background formal commit and pending deletion, using on_written timestamp.",
                  "formal_handle_background": "Recorder.handle on writer thread only; includes formal DB commit and snapshot write, excludes pending decode and deletion.",
                  "stage_metrics": "If supported by selected revision; durations from distinct threads overlap and must not be summed.",
                  "workload": "One job at a time, no model/camera, gradients plus seeded noise; no production data."},
              "resolutions": [benchmark_resolution(*sizes[name], args) for name in args.resolutions]}
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
