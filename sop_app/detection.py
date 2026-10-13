"""偵測結果的資料結構（與模型種類無關，判定引擎只依賴這裡）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]          # x1, y1, x2, y2（像素，x2/y2 不含）
    mask: np.ndarray | None = None          # bool；與框同尺寸（省記憶體）或與畫面同尺寸；沒有遮罩時用框判斷

    def box_mask(self) -> np.ndarray | None:
        """回傳裁切到框內的遮罩；接受框尺寸或整張畫面尺寸的遮罩。"""
        if self.mask is None:
            return None
        x1, y1, x2, y2 = self.box
        mask = np.asarray(self.mask, dtype=bool)
        if mask.shape == (y2 - y1, x2 - x1):
            return mask
        return mask[y1:y2, x1:x2]


@dataclass
class FrameResult:
    width: int
    height: int
    timestamp: float                        # 秒；攝影機為 monotonic 時間，影片為播放時間
    detections: list[Detection] = field(default_factory=list)
    workpiece_transform: np.ndarray | None = None  # 參考正規化座標 → 目前正規化座標，2x3
    tracking_status: str = "未設定工件"
    detection_valid: bool = True
    tracking_metrics: dict = field(default_factory=dict)
    workpiece_key: str = ""                 # 產生定位結果的參考工件識別；介面據此忽略舊設定的定位
