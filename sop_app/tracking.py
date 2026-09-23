"""以固定參考影像定位工件；不沿用遺失前的位置、不以預測位置通過條件。"""
from __future__ import annotations

import base64
import binascii
import cv2
import numpy as np

from .sop_schema import ROI, Workpiece


def decode_reference(workpiece: Workpiece) -> np.ndarray:
    try:
        raw = base64.b64decode(workpiece.reference_png, validate=True)
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, binascii.Error, cv2.error) as exc:
        raise ValueError("參考影像損壞，請重新框選工件") from exc
    if frame is None or frame.size == 0:
        raise ValueError("缺少有效參考影像")
    return frame


def capture_workpiece(frame: np.ndarray, points) -> Workpiece:
    height, width = frame.shape[:2]
    if max(height, width) > 1280:
        frame = cv2.resize(frame, (round(width * 1280 / max(height, width)),
                                   round(height * 1280 / max(height, width))))
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("無法建立參考影像")
    result = Workpiece(reference_png=base64.b64encode(encoded).decode("ascii"),
                       points=[tuple(p) for p in points])
    WorkpieceLocator(result)  # 框選時立即檢查，保留舊設定直到新參考有效
    return result


def transform_roi(roi: ROI, matrix: np.ndarray | None) -> ROI | None:
    if roi.anchor == "fixed":
        return roi
    if matrix is None:
        return None
    points = np.asarray(roi.points, np.float64)
    mapped = points @ matrix[:, :2].T + matrix[:, 2]
    if not np.isfinite(mapped).all() or (mapped < 0).any() or (mapped > 1).any():
        return None  # 作業區有一部分出畫，不能據此判定消失
    return ROI(roi.name, [tuple(p) for p in mapped])


def display_rois(rois, matrix):
    return [mapped for roi in rois if (mapped := transform_roi(roi, matrix)) is not None]


class WorkpieceLocator:
    """ORB 配對與 RANSAC 相似變換，支援平移、等比縮放、平面旋轉。

    參考紋理需分散；連續三次定位穩定才輸出矩陣。不能保證區分外觀相同的工件。
    """

    def __init__(self, workpiece: Workpiece):
        reference = decode_reference(workpiece)
        height, width = reference.shape[:2]
        points = np.asarray(workpiece.points, np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError("請框選有效的工件範圍")
        pixels = points * [width, height]
        if cv2.contourArea(pixels.astype(np.float32)) < 400:
            raise ValueError("工件範圍太小，請靠近或放大後重新框選")
        self.reference_size = (width, height)
        self.corners = points
        self.orb = cv2.ORB_create(nfeatures=2000, edgeThreshold=10, fastThreshold=10)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, pixels.astype(np.int32), 255)
        keypoints, self.descriptors = self.orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), mask)
        if self.descriptors is None or len(keypoints) < 16:
            raise ValueError("工件紋理不足，請選擇含清楚圖案、孔位或文字的範圍")
        self.reference_points = np.float32([k.pt for k in keypoints])
        self.span = np.ptp(pixels, axis=0)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reset()

    def reset(self):
        self.previous = None
        self.stable = 0

    def locate(self, frame: np.ndarray):
        try:
            matrix, count = self._estimate(frame)
        except cv2.error:
            matrix, count = None, 0
        if matrix is None:
            self.reset()
            return None, "作業位置暫時無法確認"
        corners = self.corners @ matrix[:, :2].T + matrix[:, 2]
        if self.previous is not None and np.max(np.linalg.norm(corners - self.previous, axis=1)) < 0.12:
            self.stable += 1
        else:
            self.stable = 1
        self.previous = corners
        if self.stable < 3:
            return None, f"定位確認中（{self.stable}/3）"
        return matrix, f"工件定位穩定（{count} 個特徵）"

    def _estimate(self, frame):
        height, width = frame.shape[:2]
        factor = min(1.0, 1280 / max(height, width))
        small = cv2.resize(frame, (round(width * factor), round(height * factor))) if factor < 1 else frame
        h, w = small.shape[:2]
        keypoints, descriptors = self.orb.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
        if descriptors is None or len(descriptors) < 2:
            return None, 0
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair]
                   if a.distance < 0.7 * b.distance and a.distance < 65]
        # 一個當前特徵只允許對應一個參考特徵
        matches = list({m.trainIdx: m for m in sorted(matches, key=lambda m: -m.distance)}.values())
        if len(matches) < 10:
            return None, 0
        src = np.float32([self.reference_points[m.queryIdx] for m in matches])
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3.0)
        if affine is None or inliers is None:
            return None, 0
        keep = inliers.ravel().astype(bool)
        count = int(keep.sum())
        if count < 10 or count / len(matches) < 0.6 or (np.ptp(src[keep], axis=0) < self.span * 0.2).any():
            return None, 0
        scale = np.linalg.norm(affine[:, 0])
        if not 0.3 <= scale <= 3.0:
            return None, 0
        rw, rh = self.reference_size
        normalized = np.diag([1 / w, 1 / h]) @ affine @ np.diag([rw, rh, 1])
        return normalized, count
