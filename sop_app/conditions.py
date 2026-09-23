"""單一畫面的條件判斷：物件 + 區域 + 數量 + 信心度。時間上的判斷在 engine.py。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .detection import Detection, FrameResult
from .sop_schema import ROI, Condition
from .tracking import transform_roi


@dataclass
class ConditionResult:
    condition: Condition
    count: int
    met: bool
    error: str = ""


@lru_cache(maxsize=64)
def _roi_mask(points: tuple[tuple[float, float], ...], width: int, height: int) -> np.ndarray:
    polygon = np.array([[round(x * width), round(y * height)] for x, y in points], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def in_roi(detection: Detection, roi: ROI, width: int, height: int, min_overlap: float) -> bool:
    """物件落在區域內的比例 ≥ min_overlap。有遮罩用遮罩像素，沒有就用框面積。"""
    if len(roi.points) < 3:
        return False
    region = _roi_mask(tuple(tuple(p) for p in roi.points), width, height)
    x1, y1, x2, y2 = detection.box
    region_crop = region[y1:y2, x1:x2]
    if region_crop.size == 0:
        return False
    if detection.mask is not None:
        object_crop = detection.mask[y1:y2, x1:x2]
        pixels = int(object_crop.sum())
        if pixels:
            return (object_crop & region_crop).sum() / pixels >= min_overlap
    return float(region_crop.mean()) >= min_overlap


def evaluate_condition(cond: Condition, frame: FrameResult, rois: dict[str, ROI]) -> ConditionResult:
    if not frame.detection_valid:
        return ConditionResult(cond, 0, False, "模型尚未就緒，無法判斷")
    roi = None
    if cond.roi:
        roi = rois.get(cond.roi)
        if roi is None:
            # 區域不存在時一律不成立，避免「消失」條件被誤判為成立
            return ConditionResult(cond, 0, False, f"找不到區域 {cond.roi}")
        roi = transform_roi(roi, frame.workpiece_transform)
        if roi is None:
            return ConditionResult(cond, 0, False, "作業位置暫時無法確認或已超出畫面")

    count = sum(
        1 for det in frame.detections
        if det.label == cond.label and det.score >= cond.min_score
        and (roi is None or in_roi(det, roi, frame.width, frame.height, cond.roi_overlap))
    )
    met = count == 0 if cond.type == "disappear" else count >= cond.min_count
    return ConditionResult(cond, count, met)


def evaluate_conditions(conditions: list[Condition], frame: FrameResult,
                        rois: dict[str, ROI]) -> tuple[bool, list[ConditionResult]]:
    """全部成立才回傳 True；沒有條件時回傳 False（需手動確認）。"""
    results = [evaluate_condition(cond, frame, rois) for cond in conditions]
    return bool(results) and all(r.met for r in results), results
