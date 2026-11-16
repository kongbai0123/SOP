"""Shared preview categories; independent of detection and SOP judgement."""
from __future__ import annotations

DISPLAY_GROUPS = ("operation", "objects", "installation")
DEFAULT_DISPLAY_GROUPS = frozenset(DISPLAY_GROUPS)
DISPLAY_GROUP_LABELS = {"operation": "操作區", "objects": "物件", "installation": "安裝位置"}
ROI_DISPLAY_GROUPS = ("auto", "operation", "installation")


def normalize_display_groups(groups):
    return frozenset(groups) & DEFAULT_DISPLAY_GROUPS


def roi_display_group(roi):
    """Old SOPs need no migration; unfamiliar names can be classified in the editor.

    A region's coordinate anchor is deliberately unrelated to its display group:
    an installation position can be fixed or follow the workpiece.
    """
    group = getattr(roi, "display_group", "auto")
    if group in ("operation", "installation"):
        return group
    name = roi.name.casefold()
    if any(word in name for word in ("安裝", "安装", "組裝", "组装", "裝配", "装配",
                                     "assembly", "install", "mount")):
        return "installation"
    return "operation"


def filter_display_rois(rois, groups):
    selected = normalize_display_groups(groups)
    return [roi for roi in rois if roi_display_group(roi) in selected]


def display_group_summary(groups):
    selected = normalize_display_groups(groups)
    return "、".join(DISPLAY_GROUP_LABELS[group] for group in DISPLAY_GROUPS if group in selected) or "全部隱藏"
