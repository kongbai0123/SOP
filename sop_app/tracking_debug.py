"""Opt-in, bounded local capture for reproducing live tracking faults."""
import json
import os
import time
from pathlib import Path

import cv2


class TrackingDebug:
    def __init__(self):
        folder = os.environ.get("SOP_TRACKING_DEBUG")
        self.folder = Path(folder) if folder else None
        self.started = time.monotonic()
        self.last = 0.
        self.index = 0

    def capture(self, packet, matrix, status, tracking_metrics=None, running=False):
        now = time.monotonic()
        if self.folder is None or now - self.started > 600:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            metrics = tracking_metrics or packet.result.tracking_metrics
            trace = dict(elapsed=now-self.started, frame_timestamp=packet.result.timestamp,
                         display_valid=matrix is not None, active=running,
                         decision_valid=packet.result.workpiece_transform is not None if running else None,
                         decision_blocked=any(r.error for r in packet.snapshot.results)
                             if running and packet.snapshot is not None else None,
                         handlebar=[dict(score=round(d.score, 3), box=d.box)
                                    for d in packet.result.detections if d.label == 'handlebar'],
                         tracking=metrics)
            with (self.folder / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(trace, ensure_ascii=False) + '\n')
            if now - self.last < .5:
                return
            self.last = now
            self.index += 1
            filename = f"frame-{self.index:04d}.jpg"
            cv2.imwrite(str(self.folder / filename), packet.frame)
            row = dict(frame=filename, elapsed=now-self.started, status=status,
                       matrix=matrix.tolist() if matrix is not None else None,
                       fps=packet.fps, camera_fps=packet.camera_fps,
                       inference_ms=packet.inference_ms,
                       processing_ms=packet.processing_ms,
                       capture_age_at_dispatch_ms=packet.capture_age_ms,
                       capture_age_at_ui_ms=max(0., (now-packet.result.timestamp)*1000)
                           if packet.capture_age_ms is not None else None,
                       tracking=metrics)
            with (self.folder / 'samples.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        except OSError:
            self.folder = None
