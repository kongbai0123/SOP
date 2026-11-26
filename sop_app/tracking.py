"""以固定參考影像定位工件；不沿用遺失前的位置、不以預測或未經核對的位置通過條件。"""
from __future__ import annotations

import base64
import binascii
import time
import hashlib
from .native_runtime import cv2
import numpy as np

from .sop_schema import ROI, Workpiece
from .detection import TRACKING_CONFIRMED, TRACKING_RELOCATING, TRACKING_UNCERTAIN
from .motion_filter import AdaptiveMotionFilter
from .tracking_budget import TrackingBudget


def workpiece_key(workpiece: Workpiece | None) -> str:
    """參考影像與框選範圍的識別碼；背景定位結果帶著它，介面才不會套用舊設定的位置。"""
    if workpiece is None:
        return ""
    digest = hashlib.sha1(workpiece.reference_png.encode("ascii", "ignore"))
    digest.update(repr([tuple(map(float, p)) for p in workpiece.points]).encode())
    return digest.hexdigest()[:16]


def decode_reference(workpiece: Workpiece) -> np.ndarray:
    cv2.require()
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
    WorkpieceLocator(result, adaptive_budget=True)  # 與正式管線使用相同比對尺度，失敗保留舊設定
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
    return ROI(roi.name, [tuple(p) for p in mapped], display_group=roi.display_group)


def display_rois(rois, matrix):
    return [mapped for roi in rois if (mapped := transform_roi(roi, matrix)) is not None]


class WorkpieceLocator:
    """ORB 配對與 RANSAC 相似變換，支援平移、等比縮放、平面旋轉。

    每幀輸出有效量測；追蹤與參考核對偏差超過工件對角線 2% 時立即校正。不能保證區分外觀相同的工件。
    state 區分已確認、位置待確認（光流延續但重新核對未通過）與重新定位中；只輸出已確認的位置。
    """

    CORRECTION_RATIO = .02
    TEMPLATE_MIN_SCORE = .58
    # 輪廓最佳位置須比其他位置高出此分數；直管對上桌緣這類平行邊緣的近似解會被擋下。
    TEMPLATE_MIN_MARGIN = .15
    # 中心距離在模板短邊 25% 內（含其他角度與縮放）視為同一位置，不算競爭者。
    TEMPLATE_PEAK_RADIUS = .25
    # 參考自我驗證另用平移非整數像素的副本：比對尺度的取樣相位改變後仍須找回原位。
    SELF_CHECK_SHIFT = (2.5, 1.5)
    MIN_RECHECK_FRAMES = 5
    MAX_RECHECK_FRAMES = 15
    MAX_RECHECK_SECONDS = .5
    MAX_FLOW_FRAMES = 60
    # Work limits only affect the opt-in adaptive path. Matching/geometry
    # acceptance thresholds below are shared with the original fixed strategy.
    MAX_SEARCH_SIDE = 960
    MAX_LOCAL_SEARCH_SIDE = 640
    MAX_FLOW_POINTS = 240
    MAX_RANSAC_ITERS = 1000
    # 距離輪廓：參考輪廓至少此比例在畫面內才評分；出畫部分不算支持也不算反證。
    CONTOUR_MIN_VISIBLE = .75
    # 上次確認角度與光流角度另以補零外框比對（工件尺寸的此比例），找回部分出畫的工件。
    CONTOUR_PAD = .25
    CONTOUR_CACHE_LIMIT = 256
    # ±42° 以外分三組；連續兩次找不到後才與既有兩組交替，不拖慢常見角度的恢復。
    CONTOUR_WIDE_ANGLES = tuple(a for a in range(-174, 181, 6) if abs(a) > 42)
    CONTOUR_WIDE_GROUPS = 3
    # 粗搜 0.9/1.0/1.1 之間的細搜：步距 0.025 連續涵蓋 0.85～1.15，0.95、1.05 也搜得到。
    # 單組粗搜角度相隔 12°，角度細搜取 ±6° 才連續涵蓋。
    CONTOUR_FINE_SCALES = (-.05, -.025, 0., .025, .05)
    CONTOUR_FINE_DEGREES = 6

    def __init__(self, workpiece: Workpiece, *, adaptive_budget=False):
        self.budget = TrackingBudget() if adaptive_budget else None
        self.metrics = {}
        self._other_ms = 0.
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
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(nfeatures=1200 if self.budget else 2000, edgeThreshold=10, fastThreshold=10)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, pixels.astype(np.int32), 255)
        keypoints, self.descriptors = self.orb.detectAndCompute(reference_gray, mask)
        self.reference_points = np.float32([k.pt for k in keypoints])
        # ORB 對低紋理、尺度與光照改變較敏感；保留尺度不變特徵供重新定位。
        self.sift = cv2.SIFT_create(nfeatures=800 if self.budget else 1500, contrastThreshold=.015)
        sift_points, self._sift_descriptors = self.sift.detectAndCompute(
            reference_gray, mask)
        self._sift_points = np.float32([k.pt for k in sift_points])
        self.span = np.ptp(pixels, axis=0)
        x, y, template_w, template_h = cv2.boundingRect(pixels.astype(np.int32))
        template_gray = reference_gray[y:y + template_h, x:x + template_w]
        self._template_edges = cv2.Canny(cv2.GaussianBlur(template_gray, (5, 5), 0), 40, 120)
        if (self.descriptors is None or len(keypoints) < 16) and np.count_nonzero(self._template_edges) < 30:
            raise ValueError("工件紋理與輪廓不足，請選擇含清楚圖案、孔位或文字的範圍")
        # 輪廓模板依比對尺度從參考灰階重建，與當前畫面同樣「先縮小再取邊緣」。
        self._reference_gray = reference_gray
        self._template_polygon = pixels.astype(np.float64)
        self._rotated_reference = {}
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._template_cache = {}
        self._contour_cache = {}
        self._contour_group = 0
        self._contour_hint = None
        self._contour_attempts = 0   # 上次輪廓接受後的連續搜尋次數
        self._contour_wide = 0
        self._flow_matrix = None
        self._last_absolute_stage = None
        self._template_group_index = 0
        self._motion_filter = AdaptiveMotionFilter(self.corners, self.CORRECTION_RATIO)
        self._disabled_stages = set()
        self.reference_check = self._verify_reference(reference)
        self.reset()

    def reset(self):
        self._motion_filter.reset()
        self._measurement_verified = False
        self._verification_failed = False   # 光流延續中，該做的重新核對沒有任何方法通過
        self.state = TRACKING_RELOCATING
        self._absolute_matrix = None
        self.last_deviation = 0.
        self.misses = 0
        self._flow_gray = None
        self._flow_src = None
        self._flow_dst = None
        self._flow_age = 0
        self._flow_matrix = None
        self._last_verified_timestamp = None
        self._previous_source_timestamp = None
        self._time_recheck_due = False
        self._last_local_search = getattr(self, '_last_local_search', 0.)
        # Keep recovery progression across consecutive invalid frames. No pose
        # or image score from those frames is reused by a later search.
        self._template_group_index = getattr(self, '_template_group_index', 0)

    def locate(self, frame: np.ndarray, timestamp: float | None = None, *, other_ms=0., detections=()):
        cv2.require()
        started = time.perf_counter()
        self._started = started
        self._other_ms = max(0., other_ms)
        self.metrics = {"stages_ms": {}, "components_ms": {}, "skipped": [], "outcomes": {}, "diagnostics": {}}
        if self.budget:
            self.budget.begin_frame()
        timestamp = time.monotonic() if timestamp is None else timestamp
        source_timestamp = float(timestamp) if np.isfinite(timestamp) else None
        reference_age = (None if source_timestamp is None or self._last_verified_timestamp is None else
                         source_timestamp - self._last_verified_timestamp)
        discontinuity = source_timestamp is None or (
            self._previous_source_timestamp is not None and source_timestamp <= self._previous_source_timestamp
        ) or (reference_age is not None and reference_age < 0.)
        self._time_recheck_due = discontinuity or reference_age is None or reference_age >= self.MAX_RECHECK_SECONDS
        self.metrics["time_recheck_reason"] = ("timestamp_discontinuity" if discontinuity else
                                              "unverified" if reference_age is None else
                                              "time_limit" if self._time_recheck_due else "")
        if source_timestamp is not None:
            self._previous_source_timestamp = source_timestamp
        # 無效的來源時間不能送入平滑器；仍須以當前影像重新核對，不能讓舊光流保持已確認。
        timestamp = source_timestamp if source_timestamp is not None else time.monotonic()
        try:
            matrix, count = self._estimate(frame, detections)
        except cv2.error:
            matrix, count = None, 0
            self.metrics["error"] = "opencv"
        if matrix is not None and self._measurement_verified:
            self._last_verified_timestamp = source_timestamp
        elapsed = (time.perf_counter() - started) * 1000
        uncertain = matrix is not None and self.metrics.get("source") == "flow" and self._verification_failed
        self.state = (TRACKING_RELOCATING if matrix is None else
                      TRACKING_UNCERTAIN if uncertain else TRACKING_CONFIRMED)
        self.metrics.update(tracking_ms=elapsed, valid=self.state == TRACKING_CONFIRMED, state=self.state)
        if self.budget:
            self.budget.observe_frame(elapsed + self._other_ms)
            self.metrics.update(overloaded=self.budget.overloaded, budget_ms=self.budget.frame_ms,
                                estimated_stage_ms=dict(self.budget.costs))
        if matrix is None:
            self._motion_filter.reset()
            self.misses += 1
            # 失效時保留短期追蹤上下文；恢復有效量測的當幀即輸出。
            # 失效幀仍回傳 None，判定引擎不會拿舊位置通過條件。
            if self.misses >= 3:
                self.reset()
                return None, "位置跟隨已中斷，正在重新定位"
            return None, f"重新定位中（{self.misses}/3）"
        if uncertain:
            # 光流追得住不代表找對工件：不輸出給判定與介面，也不算遺失，光流繼續延續到下次核對。
            self._motion_filter.reset()
            self.misses = 0
            return None, f"位置待確認：重新核對未通過，條件暫停累積（光流 {count} 點）"
        self.misses = 0
        quality_weight = min(1., count / 30) if count > 0 else abs(count) / 100
        filter_started = time.perf_counter()
        matrix = self._motion_filter.update(matrix, timestamp, frame.shape[1], frame.shape[0],
                                             quality_weight, self._measurement_verified)
        self._component("motion_filter", filter_started)
        self.metrics["tracking_ms"] = (time.perf_counter() - started) * 1000
        quality = (f"輪廓 {abs(count)}%" if count < 0 else
                   f"光流 {count} 點" if self.metrics.get("source") == "flow" else f"特徵 {count} 點")
        return matrix.copy(), f"位置已確認（{quality}）"

    def _deviation_ratio(self, tracked, verified, width, height):
        before = (self.corners @ tracked[:, :2].T + tracked[:, 2]) * [width, height]
        after = (self.corners @ verified[:, :2].T + verified[:, 2]) * [width, height]
        diagonal = np.linalg.norm(np.ptp(before, axis=0))
        return float(np.max(np.linalg.norm(after - before, axis=1)) / max(diagonal, 1.))

    _STAGE_NAMES = {"orb": "特徵", "sift": "尺度特徵", "template": "輪廓"}
    _CHECK_REASONS = {
        "reference_features_insufficient": "點數太少", "frame_features_insufficient": "點數太少",
        "matches_insufficient": "配對不足", "affine_failed": "配對不足",
        "geometry_rejected": "配對點過於集中", "scale_rejected": "配對不一致",
        "residual_or_scale_rejected": "配對不一致", "reference_edges_insufficient": "輪廓太少",
        "frame_edges_insufficient": "輪廓太少", "score_below_threshold": "相似度不足",
        "ambiguous": "畫面中有相似位置", "wrong_position": "找到的位置與框選範圍不符",
    }

    def _verify_reference(self, reference):
        """參考影像必須能找回自己：找不回的方法在之後的定位中停用，全部找不回就拒絕這個參考。

        與正式定位走相同的前處理與驗證；同一張圖都找錯或找不到的方法，在現場只會給出錯誤位置。
        """
        height, width = reference.shape[:2]
        factor = min(1.0, 1280 / max(height, width))
        if factor < 1:
            reference = cv2.resize(reference, (round(width * factor), round(height * factor)))
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        dx, dy = self.SELF_CHECK_SHIFT
        shifted = cv2.warpAffine(gray, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        views = ((gray, np.array([[1., 0., 0.], [0., 1., 0.]])),
                 (shifted, np.array([[1., 0., dx / w], [0., 1., dy / h]])))
        results = {}
        # 特徵配對以描述子與 RANSAC 驗證，對取樣相位不敏感，只驗原圖；輪廓比對另驗平移副本。
        # 特徵點通過時略過最慢的尺度特徵：它維持啟用，本身仍有嚴格的幾何驗證。特徵點不通過時
        # 必須驗證尺度特徵，否則重新核對可能輪到一個找不回原位的方法，位置會無故變成待確認。
        for stage, estimate, stage_views in (("orb", self._estimate_features, views[:1]),
                                             ("template", self._estimate_template, views),
                                             ("sift", self._estimate_sift, views[:1])):
            if stage == "sift" and results.get("orb") == "ok":
                break
            results[stage] = "ok"
            for image, expected in stage_views:
                self.metrics = {"diagnostics": {}}
                self._template_group_index = self._contour_attempts = 0
                matrix, _count = estimate(image)
                if matrix is None:
                    results[stage] = self.metrics["diagnostics"].get(stage, {}).get("reason", "no_valid_match")
                elif self._deviation_ratio(expected, matrix, w, h) > self.CORRECTION_RATIO:
                    results[stage] = "wrong_position"
                if results[stage] != "ok":
                    break
        self.metrics = {}
        self._template_group_index = self._contour_attempts = 0
        self._disabled_stages = {stage for stage, result in results.items() if result != "ok"}
        if "ok" not in results.values():
            details = "；".join(f"{self._STAGE_NAMES[stage]}：{self._CHECK_REASONS.get(result, '無法定位')}"
                               for stage, result in results.items())
            raise ValueError(f"參考範圍在原圖中也找不回原位（{details}），"
                             "請改框含接頭、孔位或文字等不重複結構的範圍")
        return results

    def _enabled(self, stage):
        return stage.removeprefix("local_") not in self._disabled_stages

    def _diagnose(self, stage, reason, **values):
        self.metrics.setdefault("diagnostics", {})[stage] = {"reason": reason, **values}

    def _component(self, name, started):
        values = self.metrics.setdefault("components_ms", {})
        values[name] = values.get(name, 0.) + (time.perf_counter() - started) * 1000

    def _seed_flow(self, gray, src, dst):
        if self.budget and len(src) > self.MAX_FLOW_POINTS:
            # Round-robin a spatial grid: keep coverage, rather than selecting
            # only the strongest points from one textured corner of the part.
            span = np.maximum(np.ptp(src, axis=0), 1.)
            cells = np.minimum(((src - src.min(axis=0)) / span * 4).astype(int), 3)
            buckets = [np.flatnonzero((cells == (x, y)).all(axis=1)).tolist()
                       for y in range(4) for x in range(4)]
            # Retain the four extrema before thinning each cell. Otherwise a
            # response-ordered cell can keep only its top/left edge and shrink
            # the geometric support despite visiting every cell.
            chosen = list(dict.fromkeys([int(np.argmin(src[:, 0])), int(np.argmax(src[:, 0])),
                                         int(np.argmin(src[:, 1])), int(np.argmax(src[:, 1]))]))
            selected = set(chosen)
            for level in range(len(src)):
                for bucket in buckets:
                    if level < len(bucket) and bucket[level] not in selected:
                        chosen.append(bucket[level])
                        selected.add(bucket[level])
                        if len(chosen) == self.MAX_FLOW_POINTS:
                            break
                if len(chosen) == self.MAX_FLOW_POINTS:
                    break
            src, dst = src[chosen], dst[chosen]
        self._flow_gray = gray.copy()
        self._flow_src, self._flow_dst = src.copy(), dst.copy()
        self._flow_age = 0
        self.metrics["flow_seed_points"] = len(src)

    @staticmethod
    def _detected_search_box(detections, width, height, factor):
        candidates = [d for d in detections if d.label == "handlebar" and d.score >= .5]
        if not candidates:
            return None
        chosen = max(candidates, key=lambda d: d.score)
        x1, y1, x2, y2 = [float(v) for v in chosen.box]
        pad = max(x2 - x1, y2 - y1) * .15
        box = (max(0, int((x1 - pad) * factor)), max(0, int((y1 - pad) * factor)),
               min(width, int(np.ceil((x2 + pad) * factor))),
               min(height, int(np.ceil((y2 + pad) * factor))))
        return box if box[2] - box[0] >= 30 and box[3] - box[1] >= 30 else None

    def _candidate_in_box(self, matrix, box, width, height):
        corners = (self.corners @ matrix[:, :2].T + matrix[:, 2]) * [width, height]
        x1, y1 = np.min(corners, axis=0)
        x2, y2 = np.max(corners, axis=0)
        area = max(0., (x2 - x1) * (y2 - y1))
        intersection = max(0., min(x2, box[2]) - max(x1, box[0])) * max(
            0., min(y2, box[3]) - max(y1, box[1]))
        return bool(area > 0 and intersection / area >= .7)

    def _estimate(self, frame, detections=()):
        self._measurement_verified = False
        preprocess_started = time.perf_counter()
        height, width = frame.shape[:2]
        factor = min(1.0, 1280 / max(height, width))
        small = cv2.resize(frame, (round(width * factor), round(height * factor))) if factor < 1 else frame
        h, w = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self._component("preprocess", preprocess_started)
        search_box = self._detected_search_box(detections, w, h, factor)
        self.metrics["detector_guided"] = search_box is not None
        # 光流每幀更新；定期做絕對核對，2% 比較的是同一幀兩個估計的偏差。
        flowed = (None, 0)
        if self._flow_gray is not None and self._flow_age < self.MAX_FLOW_FRAMES:
            flow_started = time.perf_counter()
            flowed = self._estimate_flow(gray)
            self.metrics.setdefault("stages_ms", {})["flow"] = (time.perf_counter() - flow_started) * 1000
            self.metrics.setdefault("outcomes", {})["flow"] = "accepted" if flowed[0] is not None else "no_valid_match"
        # 光流當下角度只讓輪廓搜尋多比一個角度，不提供位置或分數。
        self._flow_matrix = flowed[0]
        if flowed[0] is not None:
            age = self._flow_age
            weak = self.budget is not None and flowed[1] < 20 and age >= self.MIN_RECHECK_FRAMES
            # 位置待確認時每幀重新核對：自適應模式每幀只做一種昂貴搜尋，輪到能確認的方法就立即恢復。
            # 幀數與來源時間任一到期就核對，5 FPS 時也不會等到約三秒才確認工件。
            if not weak and not self._verification_failed and not self._time_recheck_due and (
                    age < self.MAX_RECHECK_FRAMES or age % self.MIN_RECHECK_FRAMES):
                self.metrics["source"] = "flow"
                return flowed
        if self.budget:
            return self._estimate_adaptive(gray, search_box, flowed)
        stages = [("orb", lambda: self._estimate_features(gray)),
                  ("sift", lambda: self._estimate_sift(gray)),
                  ("template", lambda: self._estimate_template(gray))]
        if search_box is not None:
            stages.extend((("local_orb", lambda: self._estimate_features(gray, search_box)),
                           ("local_sift", lambda: self._estimate_sift(gray, search_box))))
        stages = [(stage, estimate) for stage, estimate in stages if self._enabled(stage)]
        local_allowed = not self._last_local_search or time.perf_counter() - self._last_local_search >= .5
        for stage, estimate in stages:
            now = time.perf_counter()
            # Preserve the original full-frame result. A detected crop is a
            # bounded extra recovery attempt only when all original methods
            # failed, so detector jitter cannot replace a verified pose.
            if stage.startswith("local_") and not local_allowed:
                self.metrics.setdefault("skipped", []).append(stage)
                self._diagnose(stage, "local_retry_interval")
                continue
            spent = (now - getattr(self, '_started', now)) * 1000 + self._other_ms
            if self.budget and stage != "orb" and not self.budget.allow(stage, now, spent):
                self.metrics.setdefault("skipped", []).append(stage)
                self._diagnose(stage, "compute_budget_deferred")
                continue
            if stage.startswith("local_"):
                self._last_local_search = now
            matrix, count = estimate()
            cost = (time.perf_counter() - now) * 1000
            self.metrics.setdefault("stages_ms", {})[stage] = cost
            self.metrics.setdefault("outcomes", {})[stage] = "accepted" if matrix is not None else "no_valid_match"
            if self.budget:
                self.budget.record(stage, now, cost)
            if matrix is not None:
                self.metrics["source"] = stage
                self._measurement_verified = True
                self._verification_failed = False
                self._absolute_matrix = matrix.copy()
                if flowed[0] is not None:
                    self.last_deviation = self._deviation_ratio(flowed[0], matrix, w, h)
                    if self.last_deviation > self.CORRECTION_RATIO:
                        self._motion_filter.reset()
                return matrix, count
        if flowed[0] is not None:
            # 該重新核對卻沒有任何方法通過：光流位置改為待確認，直到下一次核對通過。
            self.metrics["source"] = "flow"
            self._verification_failed = True
            return flowed
        self.metrics["source"] = "none"
        return None, 0

    def _estimate_adaptive(self, gray, search_box, flowed):
        """Current-frame local recovery, global fallback, then one costly stage.

        A live flow measurement is always checked against the full reference at
        the existing interval; a detector hint cannot replace that verification.
        """
        # A verified whole-part contour must be rechecked before a handful of
        # central ORB features can replace its pose and well-spread flow seed.
        # This also applies after flow drops out: it is a fresh global search,
        # not permission to keep the old contour pose. A failed attempt uses
        # this frame's one expensive slot, then normal recovery gets its turn.
        preferred_attempted = (self._last_absolute_stage == "template" and
                               not self._verification_failed and self._enabled("template"))
        if preferred_attempted:
            matrix, count = self._run_stage("template", lambda: self._estimate_template(gray))
            if matrix is not None:
                return self._accept_measurement("template", matrix, count, flowed, gray.shape)
            self._verification_failed = True
        cheap = []
        if flowed[0] is None and search_box is not None:
            cheap.append(("local_orb", lambda: self._estimate_features(gray, search_box)))
        cheap.append(("orb", lambda: self._estimate_features(gray)))
        for stage, estimate in cheap:
            if not self._enabled(stage):
                continue
            matrix, count = self._run_stage(stage, estimate)
            if matrix is not None:
                return self._accept_measurement(stage, matrix, count, flowed, gray.shape)
        expensive = {"sift": lambda: self._estimate_sift(gray), "template": lambda: self._estimate_template(gray)}
        if search_box is not None:
            expensive = {"local_sift": lambda: self._estimate_sift(gray, search_box), **expensive}
        expensive = {stage: estimate for stage, estimate in expensive.items() if self._enabled(stage)}
        now = time.perf_counter()
        spent = (now - getattr(self, '_started', now)) * 1000 + self._other_ms
        # Once a method has verified the current view, try it first at the
        # next mandatory recheck. A failed retry still goes through the normal
        # recovery rotation on the following frame; no result is reused.
        preferred = self._last_absolute_stage if flowed[0] is not None and not self._verification_failed else None
        selected = (self.budget.select_stage([preferred], now, spent)
                    if not preferred_attempted and preferred in expensive else None)
        if selected is None and not preferred_attempted:
            selected = self.budget.select_stage(list(expensive), now, spent)
        for stage in expensive:
            if stage != selected and stage not in self.metrics.get("stages_ms", {}):
                self.metrics.setdefault("skipped", []).append(stage)
                self._diagnose(stage, "one_recovery_stage_per_frame" if selected else "compute_budget_deferred")
        if selected is not None:
            matrix, count = self._run_stage(selected, expensive[selected])
            if matrix is not None:
                return self._accept_measurement(selected, matrix, count, flowed, gray.shape)
        self.metrics["source"] = "flow" if flowed[0] is not None else "none"
        if flowed[0] is not None:
            self._verification_failed = True   # 重新核對沒有通過，見 _estimate
        return flowed

    def _run_stage(self, stage, estimate):
        began = time.perf_counter()
        result = estimate()
        elapsed = (time.perf_counter() - began) * 1000
        self.metrics.setdefault("stages_ms", {})[stage] = elapsed
        self.metrics.setdefault("outcomes", {})[stage] = "accepted" if result[0] is not None else "no_valid_match"
        self.budget.record(stage, began, elapsed)
        return result

    def _accept_measurement(self, stage, matrix, count, flowed, shape):
        self._last_absolute_stage = stage
        self.metrics["source"] = stage
        self._measurement_verified = True
        self._verification_failed = False
        self._absolute_matrix = matrix.copy()
        if flowed[0] is not None:
            self.last_deviation = self._deviation_ratio(flowed[0], matrix, shape[1], shape[0])
            if self.last_deviation > self.CORRECTION_RATIO:
                self._motion_filter.reset()
        return matrix, count

    @staticmethod
    def _keypoints_in_box(detector, gray, box, max_side=None):
        x1, y1, x2, y2 = box if box is not None else (0, 0, gray.shape[1], gray.shape[0])
        crop = gray[y1:y2, x1:x2]
        factor = min(1., max_side / max(crop.shape)) if max_side else 1.
        original_shape = crop.shape
        if factor < 1:
            crop = cv2.resize(crop, (round(crop.shape[1] * factor), round(crop.shape[0] * factor)))
        points, descriptors = detector.detectAndCompute(crop, None)
        if points is None:
            points = []
        for point in points:
            point.pt = (point.pt[0] * original_shape[1] / crop.shape[1] + x1,
                        point.pt[1] * original_shape[0] / crop.shape[0] + y1)
        return points, descriptors

    def _estimate_features(self, gray, box=None):
        stage = "local_orb" if box is not None else "orb"
        if self.descriptors is None or len(self.reference_points) < 8:
            self._diagnose(stage, "reference_features_insufficient", reference_count=len(self.reference_points))
            return None, 0
        h, w = gray.shape
        started = time.perf_counter()
        limit = (self.MAX_LOCAL_SEARCH_SIDE if box is not None else self.MAX_SEARCH_SIDE) if self.budget else None
        keypoints, descriptors = self._keypoints_in_box(self.orb, gray, box, limit)
        self._component(stage + ".features", started)
        if descriptors is None or len(descriptors) < 2:
            self._diagnose(stage, "frame_features_insufficient", frame_count=len(keypoints))
            return None, 0
        started = time.perf_counter()
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair]
                   if a.distance < 0.7 * b.distance and a.distance < 65]
        # 一個當前特徵只允許對應一個參考特徵
        matches = list({m.trainIdx: m for m in sorted(matches, key=lambda m: -m.distance)}.values())
        self._component(stage + ".matching", started)
        if len(matches) < 8:
            self._diagnose(stage, "matches_insufficient", matches=len(matches))
            return None, 0
        src = np.float32([self.reference_points[m.queryIdx] for m in matches])
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        started = time.perf_counter()
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3.0,
                                                     maxIters=self.MAX_RANSAC_ITERS if self.budget else 2000)
        self._component(stage + ".geometry", started)
        if affine is None or inliers is None:
            self._diagnose(stage, "affine_failed", matches=len(matches))
            return None, 0
        keep = inliers.ravel().astype(bool)
        count = int(keep.sum())
        spread = np.ptp(src[keep], axis=0).tolist() if count else [0., 0.]
        if count < 8 or count / len(matches) < 0.6 or (np.ptp(src[keep], axis=0) < self.span * 0.15).any():
            self._diagnose(stage, "geometry_rejected", matches=len(matches), inliers=count,
                           inlier_ratio=round(count / len(matches), 3), spread=spread)
            return None, 0
        scale = np.linalg.norm(affine[:, 0])
        if not 0.3 <= scale <= 3.0:
            self._diagnose(stage, "scale_rejected", scale=round(float(scale), 3))
            return None, 0
        self._diagnose(stage, "accepted", matches=len(matches), inliers=count, spread=spread)
        rw, rh = self.reference_size
        normalized = np.diag([1 / w, 1 / h]) @ affine @ np.diag([rw, rh, 1])
        if box is not None and not self._candidate_in_box(normalized, box, w, h):
            self._diagnose(stage, "outside_detection_box", matches=len(matches), inliers=count)
            return None, 0
        self._seed_flow(gray, src[keep], dst[keep])
        return normalized, count

    def _estimate_sift(self, gray, box=None):
        stage = "local_sift" if box is not None else "sift"
        if self._sift_descriptors is None or len(self._sift_points) < 12:
            self._diagnose(stage, "reference_features_insufficient", reference_count=len(self._sift_points))
            return None, 0
        started = time.perf_counter()
        limit = (self.MAX_LOCAL_SEARCH_SIDE if box is not None else self.MAX_SEARCH_SIDE) if self.budget else None
        keypoints, descriptors = self._keypoints_in_box(self.sift, gray, box, limit)
        self._component(stage + ".features", started)
        if descriptors is None or len(keypoints) < 12:
            self._diagnose(stage, "frame_features_insufficient", frame_count=len(keypoints))
            return None, 0
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        started = time.perf_counter()
        reverse = {m.queryIdx: m.trainIdx for m in matcher.match(descriptors, self._sift_descriptors)}
        matches = [a for pair in matcher.knnMatch(self._sift_descriptors, descriptors, k=2)
                   if len(pair) == 2 for a, b in [pair]
                   if a.distance < .7 * b.distance and reverse.get(a.trainIdx) == a.queryIdx]
        self._component(stage + ".matching", started)
        if len(matches) < 12:
            self._diagnose(stage, "matches_insufficient", matches=len(matches))
            return None, 0
        src = np.float32([self._sift_points[m.queryIdx] for m in matches])
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        started = time.perf_counter()
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3.,
                                                     maxIters=self.MAX_RANSAC_ITERS if self.budget else 3000)
        self._component(stage + ".geometry", started)
        if affine is None or inliers is None:
            self._diagnose(stage, "affine_failed", matches=len(matches))
            return None, 0
        keep = inliers.ravel().astype(bool)
        count = int(keep.sum())
        spread = np.ptp(src[keep], axis=0).tolist() if count else [0., 0.]
        # 雙向唯一配對、分散的內點與幾何誤差一起驗證，不以少數相似孔位通過。
        if count < 12 or keep.mean() < .35 or (np.ptp(src[keep], axis=0) < self.span * .25).any():
            self._diagnose(stage, "geometry_rejected", matches=len(matches), inliers=count,
                           inlier_ratio=round(float(keep.mean()), 3), spread=spread)
            return None, 0
        residual = np.linalg.norm(src[keep] @ affine[:, :2].T + affine[:, 2] - dst[keep], axis=1)
        if np.median(residual) > 1.5 or not .3 <= np.linalg.norm(affine[:, 0]) <= 3:
            self._diagnose(stage, "residual_or_scale_rejected", median_residual=round(float(np.median(residual)), 3))
            return None, 0
        self._diagnose(stage, "accepted", matches=len(matches), inliers=count, spread=spread,
                       median_residual=round(float(np.median(residual)), 3))
        rw, rh = self.reference_size
        h, w = gray.shape
        normalized = np.diag([1/w, 1/h]) @ affine @ np.diag([rw, rh, 1])
        if box is not None and not self._candidate_in_box(normalized, box, w, h):
            self._diagnose(stage, "outside_detection_box", matches=len(matches), inliers=count)
            return None, 0
        self._seed_flow(gray, src[keep], dst[keep])
        return normalized, count

    def _estimate_flow(self, gray):
        if gray.shape != self._flow_gray.shape or len(self._flow_dst) < 10:
            self._diagnose("flow", "seed_or_resolution_invalid", seed_count=len(self._flow_dst))
            return None, 0
        old = self._flow_dst.reshape(-1, 1, 2)
        options = dict(winSize=(21,21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        started = time.perf_counter()
        new, forward, error = cv2.calcOpticalFlowPyrLK(self._flow_gray, gray, old, None, **options)
        self._component("flow.forward", started)
        if new is None or forward is None:
            self._diagnose("flow", "forward_failed")
            return None, 0
        started = time.perf_counter()
        back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, self._flow_gray, new, None, **options)
        self._component("flow.backward", started)
        if back is None or backward is None:
            self._diagnose("flow", "backward_failed")
            return None, 0
        h, w = gray.shape
        dst = new.reshape(-1,2)
        keep = (forward.ravel() > 0) & (backward.ravel() > 0)
        keep &= np.linalg.norm(back.reshape(-1,2) - old.reshape(-1,2), axis=1) < 1.0
        keep &= error.ravel() < 25
        keep &= np.isfinite(dst).all(axis=1) & (dst >= 0).all(axis=1) & (dst < [w,h]).all(axis=1)
        src, dst = self._flow_src[keep], dst[keep]
        if len(src) < 10:
            self._diagnose("flow", "points_insufficient", valid_points=len(src), seed_count=len(old))
            return None, 0
        started = time.perf_counter()
        affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                                    maxIters=self.MAX_RANSAC_ITERS if self.budget else 2000)
        self._component("flow.geometry", started)
        if affine is None or inliers is None:
            self._diagnose("flow", "affine_failed", valid_points=len(src))
            return None, 0
        good = inliers.ravel().astype(bool)
        if good.sum() < 10 or good.mean() < .7 or (np.ptp(src[good], axis=0) < self.span * .2).any():
            self._diagnose("flow", "geometry_rejected", valid_points=len(src), inliers=int(good.sum()),
                           inlier_ratio=round(float(good.mean()), 3),
                           spread=np.ptp(src[good], axis=0).tolist() if good.any() else [0., 0.])
            return None, 0
        if not .3 <= np.linalg.norm(affine[:,0]) <= 3:
            self._diagnose("flow", "scale_rejected", valid_points=len(src), inliers=int(good.sum()))
            return None, 0
        self._diagnose("flow", "accepted", valid_points=len(src), inliers=int(good.sum()))
        rw, rh = self.reference_size
        matrix = np.diag([1/w, 1/h]) @ affine @ np.diag([rw,rh,1])
        self._flow_gray = gray.copy()
        self._flow_src, self._flow_dst = src[good].copy(), dst[good].copy()
        self._flow_age += 1
        return matrix, int(good.sum())

    @staticmethod
    def _edges(gray):
        """Canny 邊緣再輕微模糊：取樣錯開半個像素時，低對比的細邊緣才不會整條消失或換位置。"""
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        return cv2.GaussianBlur(edges.astype(np.float32), (3, 3), 0)

    def _template_candidate(self, shape, angle, scale):
        """參考灰階在原解析度旋轉，用與當前畫面相同的 cv2.resize 縮到比對尺度，最後才取邊緣。

        先取邊緣再縮小會讓細邊緣變淡、斷裂，連同一張照片都比不回原位。
        回傳 (模板, 參考像素 → 模板座標的 2x3 矩陣)；模板太小或輪廓太少時回傳 None。
        """
        key = (angle, scale)
        if key not in self._template_cache:
            rh, rw = self._reference_gray.shape
            rotate = cv2.getRotationMatrix2D(tuple(self._template_polygon.mean(axis=0)), angle, 1.)
            if angle not in self._rotated_reference:
                self._rotated_reference[angle] = self._reference_gray if angle == 0 else cv2.warpAffine(
                    self._reference_gray, rotate, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            size = (max(1, round(shape[1] * scale)), max(1, round(shape[0] * scale)))
            edges = self._edges(cv2.resize(self._rotated_reference[angle], size))
            shrink = np.array([size[0] / rw, size[1] / rh])
            affine = np.diag(shrink) @ rotate
            affine[:, 2] += .5 * shrink - .5   # cv2.resize 以像素中心對齊：u = s(p + .5) - .5
            corners = self._template_polygon @ affine[:, :2].T + affine[:, 2]
            x1, y1 = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
            x2, y2 = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 1, size)
            entry = None
            if x2 - x1 >= 12 and y2 - y1 >= 12:
                template = edges[y1:y2, x1:x2].copy()
                inside = np.zeros(template.shape, np.uint8)
                cv2.fillConvexPoly(inside, np.int32(np.round(corners - [x1, y1])), 255)
                template[inside == 0] = 0
                affine[:, 2] -= (x1, y1)
                if np.count_nonzero(template) >= 10:
                    entry = (template, affine)
            self._template_cache[key] = entry
        return self._template_cache[key]

    def _template_centre(self, affine):
        return self._template_polygon.mean(axis=0) @ affine[:, :2].T + affine[:, 2]

    def _template_competitor(self, maps, best):
        """最佳位置以外的最高分；其他角度與縮放在同一中心附近的高分屬於同一個解，不算競爭者。"""
        _score, location, _scores, affine, shape = best
        centre = self._template_centre(affine) + location
        radius = max(3, round(self.TEMPLATE_PEAK_RADIUS * min(shape)))
        competitor = -1.
        for _other_score, _other_location, scores, other, _other_shape in maps:
            masked = scores.copy()
            cv2.circle(masked, tuple(int(v) for v in np.round(centre - self._template_centre(other))),
                       radius, -1., -1)
            competitor = max(competitor, float(masked.max()))
        return competitor

    def _estimate_template(self, gray):
        if self._template_edges.size == 0 or np.count_nonzero(self._template_edges) < 30:
            self._diagnose("template", "reference_edges_insufficient")
            return None, 0
        full_gray = gray
        started = time.perf_counter()
        factor = min(1., (480 if self.budget else 640) / max(gray.shape))
        if factor < 1:
            gray = cv2.resize(gray, (round(gray.shape[1] * factor), round(gray.shape[0] * factor)))
        current_edges = self._edges(gray)
        self._component("template.preprocess", started)
        if np.count_nonzero(current_edges) < 30:
            self._diagnose("template", "frame_edges_insufficient")
            return None, 0
        # 優先平移比對；只有失敗才搜尋小角度旋轉，模板依解析度快取。
        if self._template_cache.get('shape') != gray.shape:
            self._template_cache = {'shape': gray.shape}
        groups = ((0,), (-12, -6, 6, 12))
        if self.budget:
            # 每幀只搜一組角度，但都帶 0°：只比旋轉候選時，錯位置可能勝過沒被比到的正確解。
            groups = (((0,), (0, -6, 6), (0, -12, 12))[self._template_group_index % 3],)
            self._template_group_index += 1
        maps = []
        started = time.perf_counter()
        for angles in groups:
            for angle in angles:
                for scale in (0.9, 1.0, 1.1):
                    entry = self._template_candidate(gray.shape, angle, scale)
                    if entry is None:
                        continue
                    template, affine = entry
                    if template.shape[1] > gray.shape[1] or template.shape[0] > gray.shape[0]:
                        continue
                    scores = cv2.matchTemplate(current_edges, template, cv2.TM_CCOEFF_NORMED)
                    _minimum, score, _min_at, location = cv2.minMaxLoc(scores)
                    maps.append((float(score), np.asarray(location, np.float64), scores, affine, template.shape))
            if maps and max(m[0] for m in maps) >= self.TEMPLATE_MIN_SCORE:
                break
        self._component("template.matching", started)
        self.metrics["template_candidates"] = len(maps)
        best = max(maps, key=lambda m: m[0]) if maps else None
        if best is None or best[0] < self.TEMPLATE_MIN_SCORE:
            self._diagnose("template", "score_below_threshold", best_score=round(best[0], 3) if best else None)
            return self._estimate_contour(full_gray)
        score, location, _scores, affine, _shape = best
        competitor = self._template_competitor(maps, best)
        if score - competitor < self.TEMPLATE_MIN_MARGIN:
            # 另一個位置幾乎一樣像：無法確認是哪一個，寧可失效也不接受。
            self._diagnose("template", "ambiguous", best_score=round(score, 3), competitor_score=round(competitor, 3))
            return self._estimate_contour(full_gray)
        self._diagnose("template", "accepted", best_score=round(score, 3), competitor_score=round(competitor, 3))
        affine = affine.copy()
        affine[:, 2] += location
        # 比對尺度 → 原畫面，同樣以像素中心對齊。
        fx, fy = gray.shape[1] / full_gray.shape[1], gray.shape[0] / full_gray.shape[0]
        to_full = np.array([[1 / fx, 0., .5 / fx - .5], [0., 1 / fy, .5 / fy - .5]])
        affine = to_full @ np.vstack([affine, (0., 0., 1.)])
        rw, rh = self.reference_size
        h, w = full_gray.shape
        normalized = np.diag([1 / w, 1 / h]) @ affine @ np.diag([rw, rh, 1])
        self._seed_template_flow(full_gray, affine)
        return normalized, -round(score * 100)

    def _contour_reference(self, shape):
        """Remove long background lines that enter the selection from outside it.

        Rotating the entire selected rectangle also rotates the stationary table
        edge. Keep the object's measured contours instead; never infer a mask
        from its label, colour, or a previous successful frame.
        """
        cache = self._contour_cache
        if cache.get("shape") == shape:
            return cache
        gray = cv2.resize(self._reference_gray, (shape[1], shape[0]))
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 80)
        rh, rw = self._reference_gray.shape
        shrink = np.array([shape[1] / rw, shape[0] / rh])
        polygon = self._template_polygon * shrink
        mask = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(polygon), 255)
        selected = edges.copy()
        selected[mask == 0] = 0
        span = np.ptp(polygon, axis=0)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                                minLineLength=max(30., span[0] * .35), maxLineGap=20)
        if lines is not None:
            for x, y, xx, yy in lines[:, 0]:
                if mask[y, x] and mask[yy, xx]:
                    continue
                direction = np.array([xx - x, yy - y], np.float64)
                direction *= max(shape) * 2 / max(np.linalg.norm(direction), 1.)
                cv2.line(selected, tuple(np.int32(np.round([x, y] - direction))),
                         tuple(np.int32(np.round([xx, yy] + direction))), 0, 5)
        yy, xx = np.where(selected > 0)
        cloud = np.column_stack([xx, yy]).astype(np.float64)
        self._contour_cache = {"shape": shape, "cloud": cloud, "polygon": polygon,
                               "entries": {}, "shrink": shrink, "edges": selected}
        return self._contour_cache

    @classmethod
    def _contour_entry(cls, cache, angle, scale):
        key = (round(float(angle), 3), round(float(scale), 3))
        entries = cache["entries"]
        if key in entries:
            # Full-circle angles and fine steps create many templates; keep
            # only the most recently used ones so memory stays bounded.
            entries[key] = entries.pop(key)
            return entries[key]
        cloud = cache["cloud"]
        affine = cv2.getRotationMatrix2D(tuple(cache["polygon"].mean(axis=0)), float(angle), float(scale))
        moved = cloud @ affine[:, :2].T + affine[:, 2]
        lo = np.floor(moved.min(axis=0)).astype(int) - 1
        hi = np.ceil(moved.max(axis=0)).astype(int) + 2
        # Build in an expanded canvas, so rotation never clips the stem or
        # leaves only a small matching fraction of the reference contour.
        affine[:, 2] -= lo
        template = cv2.warpAffine(cache["edges"], affine, tuple(hi - lo))
        template = np.float32(template > 90)
        if len(entries) >= cls.CONTOUR_CACHE_LIMIT:
            del entries[next(iter(entries))]
        entries[key] = (template, affine)
        return entries[key]

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + 180.) % 360. - 180.

    @staticmethod
    def _visible_mass(template, origin, size, frame_shape):
        """Template mass inside the frame at each matchTemplate position.

        Output (row, col) puts the template's top-left corner at frame pixel
        origin + (col, row); a zero border outside the frame adds nothing.
        """
        th, tw = template.shape
        integral = cv2.integral(template)
        ys, xs = origin[1] + np.arange(size[0]), origin[0] + np.arange(size[1])
        r0, r1 = np.clip(-ys, 0, th), np.clip(frame_shape[0] - ys, 0, th)
        c0, c1 = np.clip(-xs, 0, tw), np.clip(frame_shape[1] - xs, 0, tw)
        return integral[r1][:, c1] - integral[r0][:, c1] - integral[r1][:, c0] + integral[r0][:, c0]

    def _contour_scores(self, image, origin, template, frame_shape):
        """Mean distance-field support over the template contour inside the frame."""
        ccorr = cv2.matchTemplate(image, template, cv2.TM_CCORR)
        total = max(float(template.sum()), 1.)
        if (min(origin) >= 0 and origin[0] + image.shape[1] <= frame_shape[1]
                and origin[1] + image.shape[0] <= frame_shape[0]):
            return ccorr / total
        mass = self._visible_mass(template, origin, ccorr.shape, frame_shape)
        return np.where(mass >= self.CONTOUR_MIN_VISIBLE * total,
                        ccorr / np.maximum(mass, 1.), -1.).astype(np.float32)

    def _contour_angles(self, shape):
        """Angles for this search, and the subset also compared past the frame border.

        Two bounded groups alternate across +/-42 degrees; 0 is always a
        competitor. After two failed searches every other call scans a third
        of the remaining circle, so a part resting at any rotation is found
        without slowing common recovery. Hints only choose angles to match
        across the whole current image; they never supply a pose or score.
        """
        failed = self._contour_attempts
        self._contour_attempts += 1
        if failed >= 2 and failed % 2 == 0:
            step = self.CONTOUR_WIDE_GROUPS
            angles = (0, *self.CONTOUR_WIDE_ANGLES[self._contour_wide % step::step])
            self._contour_wide += 1
        else:
            groups = ((0, -42, -30, -18, -6, 6, 18, 30, 42),
                      (0, -36, -24, -12, 12, 24, 36))
            angles = groups[self._contour_group % 2] if self.budget else tuple(range(-42, 43, 6))
            self._contour_group += 1
        hints = [self._wrap_angle(self._contour_hint if self._contour_hint is not None else 0.)]
        if self._flow_matrix is not None:
            # Normalized flow matrix -> pixel rotation; template angles are counter-clockwise.
            m, (h, w) = self._flow_matrix, shape
            current = self._wrap_angle(round(-np.degrees(np.arctan2(m[1, 0] * h, m[0, 0] * w))))
            # A nearby confirmed angle is the same candidate; pay for one padded search.
            hints = [angle for angle in hints if abs(self._wrap_angle(angle - current)) > 3] + [current]
        return tuple(dict.fromkeys((*map(self._wrap_angle, angles), *hints))), set(hints)

    def _estimate_contour(self, full_gray):
        """Absolute contour recovery for low-texture parts, including rotation.

        A distance field tolerates small projected edge changes of a real 3-D
        object. It still needs a unique location and support across the part;
        neither flow continuity nor last frame's position can confirm it.
        Contour outside the frame is neither support nor counter-evidence, but
        most of the part must be visible and every visible quadrant must agree.
        """
        factor = min(1., 480 / max(full_gray.shape))
        shape = (round(full_gray.shape[0] * factor), round(full_gray.shape[1] * factor))
        cache = self._contour_reference(shape)
        cloud, polygon = cache["cloud"], cache["polygon"]
        angles, padded_angles = self._contour_angles(shape)
        if len(cloud) < 60 or (np.ptp(cloud, axis=0) < np.ptp(polygon, axis=0) * .5).any():
            return None, 0
        gray = cv2.resize(full_gray, (shape[1], shape[0]))
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 80)
        if np.count_nonzero(edges) < 60:
            return None, 0
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        near = np.exp(-distance ** 2 / (2 * 2.5 ** 2)).astype(np.float32)
        # A zero border lets a partly out-of-frame part be compared at all;
        # only hinted angles pay for it, and scores use in-frame contour only.
        pad = int(np.ceil(self.CONTOUR_PAD * np.ptp(polygon, axis=0).max()))
        padded = cv2.copyMakeBorder(near, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        maps = []
        for angle in angles:
            image, origin = (padded, (-pad, -pad)) if angle in padded_angles else (near, (0, 0))
            for scale in (.9, 1., 1.1):
                template, affine = self._contour_entry(cache, angle, scale)
                if template.shape[1] > image.shape[1] or template.shape[0] > image.shape[0]:
                    continue
                scores = self._contour_scores(image, origin, template, shape)
                _, score, _, location = cv2.minMaxLoc(scores)
                maps.append((float(score), np.add(location, origin, dtype=np.float64), scores, affine,
                             template.shape, angle, scale, origin))
        if not maps:
            return None, 0
        best = max(maps, key=lambda m: m[0])
        # Only refine near a plausible measured global peak. Fine matching uses
        # a small crop, rather than repeating a whole-image search 35 times.
        if best[0] >= .55:
            coarse_centre = polygon.mean(axis=0) @ best[3][:, :2].T + best[3][:, 2] + best[1]
            searched = set()

            def refine(angle, scale):
                nonlocal best
                angle, scale = self._wrap_angle(angle), round(float(scale), 4)
                if (angle, scale) in searched:
                    return
                searched.add((angle, scale))
                template, affine = self._contour_entry(cache, angle, scale)
                centre = polygon.mean(axis=0) @ affine[:, :2].T + affine[:, 2]
                predicted = coarse_centre - centre + pad
                lo = np.maximum(np.floor(predicted - 4).astype(int), 0)
                hi = np.minimum(np.ceil(predicted + 5).astype(int) + template.shape[::-1], padded.shape[::-1])
                crop = padded[lo[1]:hi[1], lo[0]:hi[0]]
                if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
                    return
                origin = tuple(lo - pad)
                scores = self._contour_scores(crop, origin, template, shape)
                _, score, _, location = cv2.minMaxLoc(scores)
                if score > best[0]:
                    best = (float(score), np.add(location, origin, dtype=np.float64), scores,
                            affine, template.shape, angle, scale, origin)

            # Coordinate search keeps the crop count near a 3x7 grid: angle
            # across one group's spacing, then scale, then the joint neighbours.
            angle, scale = best[5:7]
            for step in range(-self.CONTOUR_FINE_DEGREES, self.CONTOUR_FINE_DEGREES + 1):
                refine(angle + step, scale)
            for step in self.CONTOUR_FINE_SCALES:
                refine(best[5], scale + step)
            angle, scale = best[5:7]
            for turn in (-1, 0, 1):
                for step in (-.025, 0., .025):
                    refine(angle + turn, scale + step)
        score, location, _, affine, _, angle, scale, _ = best
        centre = polygon.mean(axis=0) @ affine[:, :2].T + affine[:, 2] + location
        competitor = -1.
        for _, _, scores, other, _, other_angle, _, origin in maps:
            excluded = scores.copy()
            other_centre = polygon.mean(axis=0) @ other[:, :2].T + other[:, 2]
            # Nearby angles are the same pose; distinctly different rotations
            # at the same centre are still competing explanations.
            if abs(self._wrap_angle(other_angle - angle)) <= 12:
                cv2.circle(excluded, tuple(np.int32(np.round(centre - other_centre - origin))), 20, -1., -1)
            competitor = max(competitor, float(excluded.max()))
        affine = affine.copy()
        affine[:, 2] += location
        moved = cloud @ affine[:, :2].T + affine[:, 2]
        inside = (moved >= 0).all(axis=1) & (moved <= np.subtract(shape[::-1], 1)).all(axis=1)
        values = dict(best_score=round(score, 3), competitor_score=round(competitor, 3),
                      method="contour_distance", angle=angle, scale=round(float(scale), 3),
                      visible=round(float(inside.mean()), 3))
        if score < .65 or score - competitor < .10:
            self._diagnose("template", "score_below_threshold" if score < .65 else "ambiguous", **values)
            return None, 0
        sampled = cv2.remap(near, np.float32(moved[:, 0]).reshape(1, -1),
                            np.float32(moved[:, 1]).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        cells = (cloud >= (cloud.min(axis=0) + cloud.max(axis=0)) / 2).astype(int)
        # Per-quadrant check on in-frame contour only: a quadrant that has
        # left the frame drops out, but every visible one must still agree.
        quadrants = [(cells == (x, y)).all(axis=1) & inside for y in (0, 1) for x in (0, 1)]
        support = [float(sampled[cell].mean()) for cell in quadrants if np.count_nonzero(cell) >= 20]
        if len(support) < 3 or min(support) < .4:
            self._diagnose("template", "geometry_rejected", support=support, **values)
            return None, 0
        self._diagnose("template", "accepted", support=support, **values)
        self._contour_hint = angle
        self._contour_attempts = 0
        rh, rw = self._reference_gray.shape
        to_full = np.diag([full_gray.shape[1] / shape[1], full_gray.shape[0] / shape[0]])
        affine = to_full @ affine @ np.diag([*cache["shrink"], 1.])
        h, w = full_gray.shape
        normalized = np.diag([1 / w, 1 / h]) @ affine @ np.diag([rw, rh, 1.])
        # Seed only near the contours just verified, excluding stationary
        # background corners elsewhere inside the rectangular selection.
        self._seed_template_flow(full_gray, affine, contour=cache)
        return normalized, -round(score * 100)

    def _seed_template_flow(self, gray, affine, contour=None):
        """輪廓絕對定位後，建立參考座標對應，讓後續幀使用實測光流。"""
        # 新絕對位置無法建立角點時，也不能沿用另一個位置的舊點；下幀改做絕對核對。
        self._flow_gray = self._flow_src = self._flow_dst = None
        self._flow_age = 0
        rw, rh = self.reference_size
        polygon = self.corners * [rw, rh] @ affine[:, :2].T + affine[:, 2]
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(polygon), 255)
        if contour is not None:
            contour_to_frame = affine @ np.diag([*(1 / contour["shrink"]), 1.])
            measured = cv2.warpAffine(contour["edges"], contour_to_frame, gray.shape[::-1])
            measured = cv2.dilate(measured, np.ones((15, 15), np.uint8))
            mask[measured == 0] = 0
        points = cv2.goodFeaturesToTrack(gray, 160, .01, 7, mask=mask)
        if points is None or len(points) < 10:
            return
        dst = points.reshape(-1, 2)
        inverse = cv2.invertAffineTransform(affine)
        src = dst @ inverse[:, :2].T + inverse[:, 2]
        if (np.ptp(src, axis=0) < self.span * .2).any():
            return
        self._seed_flow(gray, np.float32(src), np.float32(dst))
