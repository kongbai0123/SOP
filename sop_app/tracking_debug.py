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

    def capture(self, packet, matrix, status, tracking_metrics=None):
        now = time.monotonic()
        if self.folder is None or now - self.started > 600 or now - self.last < .5:
            return
        self.last = now
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
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
                       tracking=tracking_metrics or packet.result.tracking_metrics)
            with (self.folder / 'samples.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        except OSError:
            self.folder = None
