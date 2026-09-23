"""偵測結果的資料結構（與模型種類無關，判定引擎只依賴這裡）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]          # x1, y1, x2, y2（像素，x2/y2 不含）
    mask: np.ndarray | None = None          # bool，與畫面同尺寸；沒有遮罩時用框判斷


@dataclass
class FrameResult:
    width: int
    height: int
    timestamp: float                        # 秒；攝影機為 monotonic 時間，影片為播放時間
    detections: list[Detection] = field(default_factory=list)
    workpiece_transform: np.ndarray | None = None  # 參考正規化座標 → 目前正規化座標，2x3
    tracking_status: str = "未設定工件"
    detection_valid: bool = True
