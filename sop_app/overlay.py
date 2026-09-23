"""在畫面上畫出偵測結果（遮罩、框、類別名稱）。中文的區域名稱由 Qt 畫，這裡只畫英文類別。"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .detection import Detection

# BGR，依類別順序循環使用
PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (23, 204, 146), (134, 219, 61), (211, 188, 0), (255, 115, 100)]


def class_color(label: str, classes: Sequence[str]) -> tuple[int, int, int]:
    index = classes.index(label) if label in classes else abs(hash(label))
    return PALETTE[index % len(PALETTE)]


def draw_detections(frame: np.ndarray, detections: list[Detection], classes: Sequence[str],
                    min_score: float = 0.3, show_masks: bool = True) -> np.ndarray:
    canvas = frame.copy()
    scale = max(1.0, frame.shape[1] / 1280)
    thickness = max(2, round(2 * scale))
    for det in detections:
        if det.score < min_score:
            continue
        color = class_color(det.label, classes)
        x1, y1, x2, y2 = det.box
        if show_masks and det.mask is not None:
            region = canvas[y1:y2, x1:x2]
            object_mask = det.mask[y1:y2, x1:x2]
            region[object_mask] = (region[object_mask] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        text = f"{det.label} {det.score:.2f}"
        font_scale = 0.6 * scale
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(canvas, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1)
        cv2.putText(canvas, text, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas
