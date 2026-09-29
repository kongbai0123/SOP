"""以固定參考影像定位工件；不沿用遺失前的位置、不以預測位置通過條件。"""
from __future__ import annotations

import base64
import binascii
import time
import cv2
import numpy as np

from .sop_schema import ROI, Workpiece
from .motion_filter import AdaptiveMotionFilter


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

    每幀輸出有效量測；追蹤與參考核對偏差超過工件對角線 2% 時立即校正。不能保證區分外觀相同的工件。
    """

    CORRECTION_RATIO = .02
    MIN_RECHECK_FRAMES = 5
    MAX_RECHECK_FRAMES = 15
    MAX_FLOW_FRAMES = 60

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
        self.reference_points = np.float32([k.pt for k in keypoints])
        # ORB 對低紋理、尺度與光照改變較敏感；保留尺度不變特徵供重新定位。
        self.sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=.015)
        sift_points, self._sift_descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), mask)
        self._sift_points = np.float32([k.pt for k in sift_points])
        self.span = np.ptp(pixels, axis=0)
        x, y, template_w, template_h = cv2.boundingRect(pixels.astype(np.int32))
        self._template_origin = (x, y)
        template_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)[y:y + template_h, x:x + template_w]
        self._template_edges = cv2.Canny(cv2.GaussianBlur(template_gray, (5, 5), 0), 40, 120)
        if (self.descriptors is None or len(keypoints) < 16) and np.count_nonzero(self._template_edges) < 30:
            raise ValueError("工件紋理與輪廓不足，請選擇含清楚圖案、孔位或文字的範圍")
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._template_cache = {}
        self._motion_filter = AdaptiveMotionFilter(self.corners, self.CORRECTION_RATIO)
        self.reset()

    def reset(self):
        self._motion_filter.reset()
        self._measurement_verified = False
        self._absolute_matrix = None
        self.last_deviation = 0.
        self.misses = 0
        self._flow_gray = None
        self._flow_src = None
        self._flow_dst = None
        self._flow_age = 0

    def locate(self, frame: np.ndarray, timestamp: float | None = None):
        timestamp = time.monotonic() if timestamp is None else timestamp
        try:
            matrix, count = self._estimate(frame)
        except cv2.error:
            matrix, count = None, 0
        if matrix is None:
            self._motion_filter.reset()
            self.misses += 1
            # 失效時保留短期追蹤上下文；恢復有效量測的當幀即輸出。
            # 失效幀仍回傳 None，判定引擎不會拿舊位置通過條件。
            if self.misses >= 3:
                self.reset()
                return None, "位置跟隨已中斷，正在重新定位"
            return None, f"位置跟隨短暫不穩（{self.misses}/3）"
        self.misses = 0
        quality_weight = min(1., count / 30) if count > 0 else abs(count) / 100
        matrix = self._motion_filter.update(matrix, timestamp, frame.shape[1], frame.shape[0],
                                             quality_weight, self._measurement_verified)
        quality = f"輪廓 {abs(count)}%" if count < 0 else f"{count} 個特徵"
        return matrix.copy(), f"位置跟隨穩定（{quality}）"

    def _deviation_ratio(self, tracked, verified, width, height):
        before = (self.corners @ tracked[:, :2].T + tracked[:, 2]) * [width, height]
        after = (self.corners @ verified[:, :2].T + verified[:, 2]) * [width, height]
        diagonal = np.linalg.norm(np.ptp(before, axis=0))
        return float(np.max(np.linalg.norm(after - before, axis=1)) / max(diagonal, 1.))

    def _estimate(self, frame):
        self._measurement_verified = False
        height, width = frame.shape[:2]
        factor = min(1.0, 1280 / max(height, width))
        small = cv2.resize(frame, (round(width * factor), round(height * factor))) if factor < 1 else frame
        h, w = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # 光流每幀更新；定期做絕對核對，2% 比較的是同一幀兩個估計的偏差。
        flowed = (None, 0)
        if self._flow_gray is not None and self._flow_age < self.MAX_FLOW_FRAMES:
            flowed = self._estimate_flow(gray)
        if flowed[0] is not None:
            age = self._flow_age
            if age < self.MAX_RECHECK_FRAMES or age % self.MIN_RECHECK_FRAMES:
                return flowed
        for estimate in (self._estimate_features, self._estimate_sift, self._estimate_template):
            matrix, count = estimate(gray)
            if matrix is not None:
                self._measurement_verified = True
                self._absolute_matrix = matrix.copy()
                if flowed[0] is not None:
                    self.last_deviation = self._deviation_ratio(flowed[0], matrix, w, h)
                    if self.last_deviation > self.CORRECTION_RATIO:
                        self._motion_filter.reset()
                return matrix, count
        if flowed[0] is not None:
            return flowed
        return None, 0

    def _estimate_features(self, gray):
        if self.descriptors is None or len(self.reference_points) < 8:
            return None, 0
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

    def _estimate_sift(self, gray):
        if self._sift_descriptors is None or len(self._sift_points) < 12:
            return None, 0
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 12:
            return None, 0
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        reverse = {m.queryIdx: m.trainIdx for m in matcher.match(descriptors, self._sift_descriptors)}
        matches = [a for pair in matcher.knnMatch(self._sift_descriptors, descriptors, k=2)
                   if len(pair) == 2 for a, b in [pair]
                   if a.distance < .7 * b.distance and reverse.get(a.trainIdx) == a.queryIdx]
        if len(matches) < 12:
            return None, 0
        src = np.float32([self._sift_points[m.queryIdx] for m in matches])
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3., maxIters=3000)
        if affine is None or inliers is None:
            return None, 0
        keep = inliers.ravel().astype(bool)
        count = int(keep.sum())
        # 雙向唯一配對、分散的內點與幾何誤差一起驗證，不以少數相似孔位通過。
        if count < 12 or keep.mean() < .35 or (np.ptp(src[keep], axis=0) < self.span * .25).any():
            return None, 0
        residual = np.linalg.norm(src[keep] @ affine[:, :2].T + affine[:, 2] - dst[keep], axis=1)
        if np.median(residual) > 1.5 or not .3 <= np.linalg.norm(affine[:, 0]) <= 3:
            return None, 0
        rw, rh = self.reference_size
        h, w = gray.shape
        self._flow_gray = gray.copy()
        self._flow_src, self._flow_dst = src[keep].copy(), dst[keep].copy()
        self._flow_age = 0
        return np.diag([1/w, 1/h]) @ affine @ np.diag([rw, rh, 1]), count

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
        full_gray = gray
        factor = min(1., 640 / max(gray.shape))
        if factor < 1:
            gray = cv2.resize(gray, (round(gray.shape[1] * factor), round(gray.shape[0] * factor)))
        current_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        if np.count_nonzero(current_edges) < 30:
            return None, 0
        rw, rh = self.reference_size
        sx, sy = gray.shape[1] / rw, gray.shape[0] / rh
        # 優先平移比對；只有失敗才搜尋小角度旋轉，模板依解析度快取。
        if self._template_cache.get('shape') != gray.shape:
            self._template_cache = {'shape': gray.shape}
        best = None
        for angles in ((0,), (-12, -6, 6, 12)):
            for angle in angles:
                for scale in (0.9, 1.0, 1.1):
                    key = (angle, scale)
                    if key not in self._template_cache:
                        local = cv2.getRotationMatrix2D((0, 0), angle, scale)
                        local = np.diag([sx, sy]) @ local
                        th, tw = template.shape
                        corners = np.array([[0,0], [tw,0], [tw,th], [0,th]]) @ local[:, :2].T
                        offset = np.floor(corners.min(axis=0))
                        size = np.ceil(corners.max(axis=0) - offset).astype(int)
                        local[:, 2] -= offset
                        candidate = cv2.warpAffine(template, local, tuple(size))
                        self._template_cache[key] = (candidate, local)
                    candidate, local = self._template_cache[key]
                    height, width = candidate.shape
                    if width < 12 or height < 12 or width > gray.shape[1] or height > gray.shape[0]:
                        continue
                    scores = cv2.matchTemplate(current_edges, candidate, cv2.TM_CCOEFF_NORMED)
                    _minimum, score, _min_at, location = cv2.minMaxLoc(scores)
                    if best is None or score > best[0]:
                        best = (float(score), local, location)
            if best is not None and best[0] >= 0.58:
                break
        if best is None or best[0] < 0.58:
            return None, 0
        score, local, location = best
        affine = local.copy()
        affine[:, 2] += np.asarray(location) - affine[:, :2] @ self._template_origin
        affine = np.diag([full_gray.shape[1] / gray.shape[1], full_gray.shape[0] / gray.shape[0]]) @ affine
        rw, rh = self.reference_size
        h, w = full_gray.shape
        normalized = np.diag([1 / w, 1 / h]) @ affine @ np.diag([rw, rh, 1])
        self._seed_template_flow(full_gray, affine)
        return normalized, -round(score * 100)

    def _seed_template_flow(self, gray, affine):
        """輪廓絕對定位後，建立參考座標對應，讓後續幀使用實測光流。"""
        rw, rh = self.reference_size
        polygon = self.corners * [rw, rh] @ affine[:, :2].T + affine[:, 2]
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(polygon), 255)
        points = cv2.goodFeaturesToTrack(gray, 160, .01, 7, mask=mask)
        if points is None or len(points) < 10:
            return
        dst = points.reshape(-1, 2)
        inverse = cv2.invertAffineTransform(affine)
        src = dst @ inverse[:, :2].T + inverse[:, 2]
        if (np.ptp(src, axis=0) < self.span * .2).any():
            return
        self._flow_gray = gray.copy()
        self._flow_src = np.float32(src)
        self._flow_dst = np.float32(dst)
        self._flow_age = 0
