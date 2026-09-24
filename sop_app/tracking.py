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
        x, y, template_w, template_h = cv2.boundingRect(pixels.astype(np.int32))
        self._template_origin = (x, y)
        template_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)[y:y + template_h, x:x + template_w]
        self._template_edges = cv2.Canny(cv2.GaussianBlur(template_gray, (5, 5), 0), 40, 120)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reset()

    def reset(self):
        self.previous = None
        self.stable = 0
        self.misses = 0
        self._smoothed_matrix = None
        self._flow_gray = None
        self._flow_src = None
        self._flow_dst = None
        self._flow_age = 0

    def locate(self, frame: np.ndarray):
        try:
            matrix, count = self._estimate(frame)
        except cv2.error:
            matrix, count = None, 0
        if matrix is None:
            self.misses += 1
            # 單一模糊幀不清空已穩定的追蹤上下文，下一幀恢復時不必重新等待三幀。
            # 失效幀仍回傳 None，判定引擎不會拿舊位置通過條件。
            if self.misses >= 3:
                self.reset()
                return None, "位置跟隨已中斷，正在重新定位"
            return None, f"位置跟隨短暫不穩（{self.misses}/3）"
        self.misses = 0
        corners = self.corners @ matrix[:, :2].T + matrix[:, 2]
        if self.previous is not None and np.max(np.linalg.norm(corners - self.previous, axis=1)) < 0.12:
            self.stable += 1
            if self._smoothed_matrix is None:
                self._smoothed_matrix = matrix.copy()
            else:
                self._smoothed_matrix = self._smoothed_matrix * 0.15 + matrix * 0.85
        else:
            self.stable = 1
            self._smoothed_matrix = matrix.copy()
        self.previous = corners
        if self.stable < 3:
            return None, f"定位確認中（{self.stable}/3）"
        quality = f"輪廓 {abs(count)}%" if count < 0 else f"{count} 個特徵"
        return self._smoothed_matrix.copy(), f"位置跟隨穩定（{quality}）"

    def _estimate(self, frame):
        height, width = frame.shape[:2]
        factor = min(1.0, 1280 / max(height, width))
        small = cv2.resize(frame, (round(width * factor), round(height * factor))) if factor < 1 else frame
        h, w = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # 逐幀量測光流；每五幀回到原始參考校正，避免連續追蹤累積漂移。
        if self._flow_gray is not None and self._flow_age < 4:
            flowed = self._estimate_flow(gray)
            if flowed[0] is not None:
                return flowed
        matrix, count = self._estimate_features(gray)
        if matrix is not None:
            return matrix, count
        # 純色、反光工件可能缺少足夠角點，改以邊緣模板做絕對位置復原。
        matrix, count = self._estimate_template(gray)
        if matrix is not None:
            return matrix, count
        # 模糊／局部遮擋時只允許短期、經雙向驗證的追蹤，絕不沿用舊座標。
        if self._flow_gray is not None and self._flow_age < 8:
            return self._estimate_flow(gray)
        return None, 0

    def _estimate_features(self, gray):
        h, w = gray.shape
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return None, 0
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair]
                   if a.distance < 0.7 * b.distance and a.distance < 65]
        # 一個當前特徵只允許對應一個參考特徵
        matches = list({m.trainIdx: m for m in sorted(matches, key=lambda m: -m.distance)}.values())
        if len(matches) < 8:
            return None, 0
        src = np.float32([self.reference_points[m.queryIdx] for m in matches])
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3.0)
        if affine is None or inliers is None:
            return None, 0
        keep = inliers.ravel().astype(bool)
        count = int(keep.sum())
        if count < 8 or count / len(matches) < 0.6 or (np.ptp(src[keep], axis=0) < self.span * 0.15).any():
            return None, 0
        scale = np.linalg.norm(affine[:, 0])
        if not 0.3 <= scale <= 3.0:
            return None, 0
        rw, rh = self.reference_size
        normalized = np.diag([1 / w, 1 / h]) @ affine @ np.diag([rw, rh, 1])
        self._flow_gray = gray.copy()
        self._flow_src = src[keep].copy()
        self._flow_dst = dst[keep].copy()
        self._flow_age = 0
        return normalized, count

    def _estimate_flow(self, gray):
        if gray.shape != self._flow_gray.shape or len(self._flow_dst) < 10:
            return None, 0
        old = self._flow_dst.reshape(-1, 1, 2)
        options = dict(winSize=(21,21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        new, forward, error = cv2.calcOpticalFlowPyrLK(self._flow_gray, gray, old, None, **options)
        if new is None or forward is None:
            return None, 0
        back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, self._flow_gray, new, None, **options)
        if back is None or backward is None:
            return None, 0
        h, w = gray.shape
        dst = new.reshape(-1,2)
        keep = (forward.ravel() > 0) & (backward.ravel() > 0)
        keep &= np.linalg.norm(back.reshape(-1,2) - old.reshape(-1,2), axis=1) < 1.0
        keep &= error.ravel() < 25
        keep &= np.isfinite(dst).all(axis=1) & (dst >= 0).all(axis=1) & (dst < [w,h]).all(axis=1)
        src, dst = self._flow_src[keep], dst[keep]
        if len(src) < 10:
            return None, 0
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if affine is None or inliers is None:
            return None, 0
        good = inliers.ravel().astype(bool)
        if good.sum() < 10 or good.mean() < .7 or (np.ptp(src[good], axis=0) < self.span * .2).any():
            return None, 0
        if not .3 <= np.linalg.norm(affine[:,0]) <= 3:
            return None, 0
        rw, rh = self.reference_size
        matrix = np.diag([1/w, 1/h]) @ affine @ np.diag([rw,rh,1])
        self._flow_gray = gray.copy()
        self._flow_src, self._flow_dst = src[good].copy(), dst[good].copy()
        self._flow_age += 1
        return matrix, int(good.sum())

    def _estimate_template(self, gray):
        template = self._template_edges
        if template.size == 0 or np.count_nonzero(template) < 30:
            return None, 0
        current_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        best = None
        for scale in (0.9, 1.0, 1.1):
            width = round(template.shape[1] * scale)
            height = round(template.shape[0] * scale)
            if width < 12 or height < 12 or width > gray.shape[1] or height > gray.shape[0]:
                continue
            candidate = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
            scores = cv2.matchTemplate(current_edges, candidate, cv2.TM_CCOEFF_NORMED)
            _minimum, score, _min_at, location = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = (float(score), scale, location)
        if best is None or best[0] < 0.58:
            return None, 0
        score, scale, (x, y) = best
        origin_x, origin_y = self._template_origin
        affine = np.array([[scale, 0., x - scale * origin_x],
                           [0., scale, y - scale * origin_y]], dtype=np.float64)
        rw, rh = self.reference_size
        h, w = gray.shape
        normalized = np.diag([1 / w, 1 / h]) @ affine @ np.diag([rw, rh, 1])
        return normalized, -round(score * 100)
