"""偵測結果的資料結構（與模型種類無關，判定引擎只依賴這裡）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 工件定位狀態：只有「已確認」的位置可以累積 SOP 條件。
TRACKING_CONFIRMED = "confirmed"      # 本幀絕對定位通過，或光流延續自核對期限內通過的位置
TRACKING_UNCERTAIN = "uncertain"      # 光流仍追得住，但重新核對沒有通過
TRACKING_RELOCATING = "relocating"    # 沒有可用位置，正在重新定位


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
    tracking_state: str = ""                # TRACKING_*；空字串 = 未經定位器（固定畫面或外部提供的位置）
    detection_valid: bool = True
    tracking_metrics: dict = field(default_factory=dict)
    workpiece_key: str = ""                 # 產生定位結果的參考工件識別；介面據此忽略舊設定的定位

    @property
    def confirmed_transform(self) -> np.ndarray | None:
        """判定只用已確認的工件位置；位置待確認或重新定位中一律當作位置未知。"""
        return self.workpiece_transform if self.tracking_state in ("", TRACKING_CONFIRMED) else None
