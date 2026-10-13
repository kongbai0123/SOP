"""在畫面上畫出偵測結果（遮罩、框、類別名稱）。中文的區域名稱由 Qt 畫，這裡只畫英文類別。"""
from __future__ import annotations

from typing import Sequence

from .native_runtime import cv2
import numpy as np

from .detection import Detection

# BGR，依類別順序循環使用
PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (23, 204, 146), (134, 219, 61), (211, 188, 0), (255, 115, 100)]


def class_color(label: str, classes: Sequence[str]) -> tuple[int, int, int]:
    index = classes.index(label) if label in classes else abs(hash(label))
    return PALETTE[index % len(PALETTE)]


def _label_geometry(text, x, y, width, height, font_scale, thickness):
    """Keep complete English detection labels inside the image, including edges."""
    while True:
        lines, line = [], ""
        full_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        if full_size[0] <= width - 8:
            lines = [text]
        else:
            for character in text:
                candidate = line + character
                size = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                if line and size[0] > width - 8:
                    lines.append(line)
                    line = character
                else:
                    line = candidate
            lines.append(line)
        sizes = [cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness) for value in lines]
        line_height = max(h + baseline + 6 for (_w, h), baseline in sizes)
        total_height = line_height * len(lines)
        if total_height <= height - 2 or font_scale <= .1:
            break
        font_scale *= .85
    total_width = max(w for (w, _h), _baseline in sizes) + 6
    left = max(1, min(x, width - total_width - 1))
    top = max(1, min(y - total_height, height - total_height - 1))
    return lines, sizes, font_scale, left, top, total_width, line_height


def draw_detections(frame: np.ndarray, detections: list[Detection], classes: Sequence[str],
                    min_score: float = 0.3, show_masks: bool = True, show_boxes: bool = True) -> np.ndarray:
    canvas = frame.copy()
    scale = max(1.0, frame.shape[1] / 1280)
    thickness = max(2, round(2 * scale))
    for det in detections:
        if det.score < min_score:
            continue
        color = class_color(det.label, classes)
        x1, y1, x2, y2 = det.box
        object_mask = det.box_mask() if show_masks else None
        if object_mask is not None:
            region = canvas[y1:y2, x1:x2]
            object_mask = object_mask[:region.shape[0], :region.shape[1]]
            region[object_mask] = (region[object_mask] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
    # 所有遮罩先畫完，才畫所有框與文字。
    if not show_boxes:
        return canvas
    for det in detections:
        if det.score < min_score:
            continue
        color = class_color(det.label, classes)
        x1, y1, x2, y2 = det.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        text = f"{det.label} {det.score:.2f}"
        font_scale = 0.6 * scale
        lines, sizes, font_scale, left, top, tw, line_height = _label_geometry(
            text, x1, y1, canvas.shape[1], canvas.shape[0], font_scale, thickness)
        cv2.rectangle(canvas, (left, top), (left + tw, top + line_height * len(lines)), color, -1)
        for index, (line, ((_w, th), _baseline)) in enumerate(zip(lines, sizes)):
            cv2.putText(canvas, line, (left + 3, top + index * line_height + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas
