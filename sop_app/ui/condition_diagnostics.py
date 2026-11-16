"""Explain existing condition errors without changing engine decisions."""
from __future__ import annotations

from ..conditions import ConditionResult
from ..detection import FrameResult, TRACKING_RELOCATING, TRACKING_UNCERTAIN
from ..sop_schema import SOPDefinition
from ..tracking import transform_roi


def _tracking_diagnostic_text(frame: FrameResult) -> str:
    """Describe measured rejection reasons, not guessed physical causes."""
    diagnostics = frame.tracking_metrics.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        return ""
    feature_reasons = {
        "reference_features_insufficient": "參考影像特徵不足",
        "frame_features_insufficient": "當幀特徵不足",
        "matches_insufficient": "特徵配對不足",
        "affine_failed": "特徵配對不足",
        "geometry_rejected": "特徵配對的幾何驗證未通過",
        "scale_rejected": "特徵配對的比例不一致",
        "residual_or_scale_rejected": "特徵配對的幾何驗證未通過",
        "outside_detection_box": "特徵配對位置與偵測範圍不符",
    }
    contour_reasons = {
        "reference_edges_insufficient": "參考影像輪廓不足",
        "frame_edges_insufficient": "當幀輪廓不足",
        "score_below_threshold": "輪廓相似度不足",
        "ambiguous": "有多個相似位置",
        "geometry_rejected": "輪廓的幾何驗證未通過",
    }
    reasons = []
    for stage in ("template", "orb", "local_orb", "sift", "local_sift"):
        observed = diagnostics.get(stage, {})
        if not isinstance(observed, dict):
            continue
        explanation = (contour_reasons if stage == "template" else feature_reasons).get(observed.get("reason"))
        if explanation and explanation not in reasons:
            reasons.append(explanation)
    return "；本幀檢查：" + "、".join(reasons[:2]) if reasons else ""


def condition_error_text(result: ConditionResult, frame: FrameResult | None,
                         sop: SOPDefinition) -> str:
    """Plain text for an unavailable condition; callers escape it for rich text.

    The condition result remains the authority on whether an error exists.
    Structured frame/configuration data only distinguishes its cause for the
    user; no alternate completion or warning evaluation is performed here.
    """
    if not result.error:
        return ""
    condition = result.condition
    roi = sop.roi_map().get(condition.roi)
    if result.error_kind == "config":
        reason = result.error + "；請在工序設定中選擇有效區域。"
    elif frame is not None and not frame.detection_valid:
        reason = "模型尚未就緒；請確認模型已載入，並等待辨識結果。"
    elif roi is not None and roi.anchor == "workpiece" and frame is not None:
        if sop.workpiece is None:
            reason = f"「{roi.name}」未設定參考工件；請在作業區設定參考工件。"
        elif frame.tracking_metrics.get("configuration_error"):
            reason = (f"「{roi.name}」的參考工件設定無效（{frame.tracking_status}）；"
                      "請在作業區重新設定參考工件。")
        elif frame.confirmed_transform is None:
            status = {TRACKING_UNCERTAIN: "位置待確認", TRACKING_RELOCATING: "重新定位中"}.get(
                frame.tracking_state, "等待工件定位")
            reason = (f"「{roi.name}」尚未定位（{status}）{_tracking_diagnostic_text(frame)}；"
                      "請讓工件保持在畫面內並短暫靜止，等待位置確認。")
        elif transform_roi(roi, frame.confirmed_transform) is None:
            reason = f"「{roi.name}」已超出畫面；請將工件移回攝影機範圍內。"
        else:
            reason = result.error
    else:
        reason = result.error
    return f"{condition.describe()}：{reason}"
