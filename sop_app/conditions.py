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
    detail: str = ""


@lru_cache(maxsize=64)
def _roi_mask(points: tuple[tuple[float, float], ...], width: int, height: int) -> np.ndarray:
    polygon = np.array([[round(x * width), round(y * height)] for x, y in points], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def roi_overlap_ratio(detection: Detection, roi: ROI, width: int, height: int) -> float:
    """回傳物件本身有多少比例落在區域內。"""
    if len(roi.points) < 3:
        return 0.0
    region = _roi_mask(tuple(tuple(p) for p in roi.points), width, height)
    x1, y1, x2, y2 = detection.box
    region_crop = region[y1:y2, x1:x2]
    if region_crop.size == 0:
        return 0.0
    if detection.mask is not None:
        object_mask = np.asarray(detection.mask, dtype=bool)
        if object_mask.shape == region.shape:
            object_pixels = int(object_mask.sum())
            intersection = int((object_mask & region).sum())
            return intersection / object_pixels if object_pixels else 0.0
    return float(region_crop.mean())


def in_roi(detection: Detection, roi: ROI, width: int, height: int, min_overlap: float) -> bool:
    """物件落在區域內的比例 ≥ min_overlap。"""
    return roi_overlap_ratio(detection, roi, width, height) >= min_overlap


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

    labelled = [det for det in frame.detections if det.label == cond.label]
    confident = [det for det in labelled if det.score >= cond.min_score]
    overlaps = [(det, roi_overlap_ratio(det, roi, frame.width, frame.height))
                for det in confident] if roi is not None else [(det, 1.0) for det in confident]
    count = sum(overlap >= cond.roi_overlap for _det, overlap in overlaps)
    met = count == 0 if cond.type == "disappear" else count >= cond.min_count
    if not labelled:
        detail = f"模型未輸出 {cond.label}"
    elif not confident:
        detail = f"最高信心 {max(det.score for det in labelled):.2f}，門檻 {cond.min_score:.2f}"
    elif roi is not None and not count:
        detail = f"物件在區域內最高 {max(overlap for _det, overlap in overlaps):.0%}，門檻 {cond.roi_overlap:.0%}"
    else:
        detail = f"通過 {count} 個"
    return ConditionResult(cond, count, met, detail=detail)


def evaluate_conditions(conditions: list[Condition], frame: FrameResult,
                        rois: dict[str, ROI], mode: str = "all") -> tuple[bool, list[ConditionResult]]:
    """全部成立才回傳 True；沒有條件時回傳 False（需手動確認）。"""
    results = [evaluate_condition(cond, frame, rois) for cond in conditions]
    met = any(r.met for r in results) if mode == "any" else all(r.met for r in results)
    return bool(results) and not any(r.error for r in results) and met, results
