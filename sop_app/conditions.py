"""單一畫面的條件判斷：物件 + 區域 + 數量 + 信心度。時間上的判斷在 engine.py。"""
from __future__ import annotations

from dataclasses import dataclass

from .native_runtime import cv2
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
    # unavailable：模型或位置暫時無法確認；config：設定錯誤（例如區域不存在）
    error_kind: str = ""


def _box_region(roi: ROI, width: int, height: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """只在物件框範圍內畫出區域遮罩，成本與物件大小成正比，而不是整張畫面。"""
    x1, y1, x2, y2 = box
    polygon = np.array([[round(x * width) - x1, round(y * height) - y1] for x, y in roi.points], dtype=np.int32)
    region = np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=np.uint8)
    if region.size:
        cv2.fillPoly(region, [polygon], 1)
    return region.astype(bool)


def roi_overlap_ratio(detection: Detection, roi: ROI, width: int, height: int) -> float:
    """回傳物件本身有多少比例落在區域內。"""
    if len(roi.points) < 3:
        return 0.0
    x1, y1, x2, y2 = detection.box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = _box_region(roi, width, height, (x1, y1, x2, y2))
    object_mask = detection.box_mask()
    if object_mask is not None:
        # 遮罩以原始框為基準；框被裁到畫面內時同步位移。
        ox, oy = x1 - detection.box[0], y1 - detection.box[1]
        object_mask = object_mask[oy:oy + region.shape[0], ox:ox + region.shape[1]]
        if object_mask.shape == region.shape:
            object_pixels = int(object_mask.sum())
            return int((object_mask & region).sum()) / object_pixels if object_pixels else 0.0
    return float(region.mean())


def in_roi(detection: Detection, roi: ROI, width: int, height: int, min_overlap: float) -> bool:
    """物件落在區域內的比例 ≥ min_overlap。"""
    return roi_overlap_ratio(detection, roi, width, height) >= min_overlap


def evaluate_condition(cond: Condition, frame: FrameResult, rois: dict[str, ROI]) -> ConditionResult:
    if not frame.detection_valid:
        return ConditionResult(cond, 0, False, "模型尚未就緒，無法判斷", error_kind="unavailable")
    roi = None
    if cond.roi:
        roi = rois.get(cond.roi)
        if roi is None:
            # 區域不存在時一律不成立，避免「消失」條件被誤判為成立
            return ConditionResult(cond, 0, False, f"找不到區域 {cond.roi}", error_kind="config")
        roi = transform_roi(roi, frame.workpiece_transform)
        if roi is None:
            return ConditionResult(cond, 0, False, f"作業位置暫時無法確認或已超出畫面（{frame.tracking_status}）", error_kind="unavailable")

    labelled = [det for det in frame.detections if det.label == cond.label]
    confident = [det for det in labelled if det.score >= cond.min_score]
    overlaps = [(det, roi_overlap_ratio(det, roi, frame.width, frame.height))
                for det in confident] if roi is not None else [(det, 1.0) for det in confident]
    count = sum(overlap >= cond.roi_overlap for _det, overlap in overlaps)
    met = count == 0 if cond.type == "disappear" else count >= cond.min_count
    if not labelled:
        detail = f"模型未輸出 {cond.label}"
        if roi is not None:
            others = sorted((det for det in frame.detections
                             if det.label != cond.label and det.score >= cond.min_score
                             and in_roi(det, roi, frame.width, frame.height, cond.roi_overlap)),
                            key=lambda det: det.score, reverse=True)
            if others:
                actual = '、'.join(f"{det.label} {det.score:.2f}" for det in others[:3])
                detail += f"；區域內辨識為 {actual}"
    elif not confident:
        detail = f"最高信心 {max(det.score for det in labelled):.2f}，門檻 {cond.min_score:.2f}"
    elif roi is not None and not count:
        detail = f"物件在區域內最高 {max(overlap for _det, overlap in overlaps):.0%}，門檻 {cond.roi_overlap:.0%}"
    else:
        detail = f"通過 {count} 個"
    return ConditionResult(cond, count, met, detail=detail)


def combine_results(results: list[ConditionResult], mode: str = "all") -> tuple[bool, bool]:
    """回傳 (成立, 無法判斷)。

    all：任何條件無法判斷 → 整組無法判斷。
    any：已有一個可判斷的條件成立就算成立；某個位置暫時出畫不會擋住其他位置。
         設定錯誤（例如區域不存在）仍一律視為無法判斷。
    """
    if not results:
        return False, False
    if any(r.error_kind == "config" for r in results):
        return False, True
    valid = [r for r in results if not r.error]
    if mode == "any":
        met = any(r.met for r in valid)
        return met, not met and len(valid) < len(results)
    blocked = len(valid) < len(results)
    return not blocked and all(r.met for r in results), blocked


def evaluate_condition_set(conditions: list[Condition], frame: FrameResult, rois: dict[str, ROI],
                           mode: str = "all") -> tuple[bool, bool, list[ConditionResult]]:
    results = [evaluate_condition(cond, frame, rois) for cond in conditions]
    met, blocked = combine_results(results, mode)
    return met, blocked, results


def evaluate_conditions(conditions: list[Condition], frame: FrameResult,
                        rois: dict[str, ROI], mode: str = "all") -> tuple[bool, list[ConditionResult]]:
    """回傳 (是否成立, 各條件結果)；沒有條件時回傳 False（需手動確認）。"""
    met, _blocked, results = evaluate_condition_set(conditions, frame, rois, mode)
    return met, results
