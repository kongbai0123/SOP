"""背景執行緒：讀取影像 → 模型偵測 → 工件定位 → 判定引擎 → 紀錄，結果以 Qt Signal 傳給介面。

介面只透過指令佇列控制這個執行緒；模型在獨立 AI 程序，判定引擎與 SQLite 留在背景執行緒。
工件定位在背景管線逐幀執行，待機、試跑與正式監控共用同一份結果。
"""
from __future__ import annotations

import copy
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time

from .native_runtime import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .detection import FrameResult
from .engine import EngineSnapshot, Phase, SOPEngine
from .inference_worker import InferenceWorker, InferenceWorkerClosed
from .model_bundle import read_model_info
from .overlay import draw_detections
from .recorder import AsyncRecorder
from .sop_schema import SOPDefinition, Workpiece
from .tracking import WorkpieceLocator, display_rois, workpiece_key


@dataclass
class FramePacket:
    frame: np.ndarray              # 原始畫面
    display: np.ndarray            # 疊上偵測結果的畫面
    result: FrameResult
    snapshot: EngineSnapshot | None
    fps: float
    inference_ms: float
    camera_fps: float | None = None  # 實際擷取速率；與處理／推論速率分開
    classes: tuple[str, ...] = ()     # 供監控頁依目前工序重畫相關物件
    min_score: float = 0.5
    show_masks: bool = True
    # 上次送出畫面後背景處理過的每一幀（含本幀），讓試跑與正式監控看到相同的取樣
    results: tuple[FrameResult, ...] = ()

    def all_results(self) -> tuple[FrameResult, ...]:
        return self.results or (self.result,)
    processing_ms: float = 0.
    capture_age_ms: float | None = None  # 僅攝影機，從 read 完成到送出；不含曝光與 GUI 繪製

    @property
    def tracking_load_ms(self):
        """Preview budget includes pipeline work and, for cameras, queued frame age."""
        age = max(0., (time.monotonic() - self.result.timestamp) * 1000) if self.capture_age_ms is not None else 0.
        return max(self.inference_ms, self.processing_ms, age)


class CameraSource:
    """獨立執行緒持續讀取，只保留最新一幀，避免推論較慢時畫面延遲越積越多。"""

    ended = False

    def __init__(self, index: int, width: int = 1280, height: int = 720):
        self.label = f"攝影機 {index}"
        self.capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"無法開啟攝影機 {index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._frame_times = deque(maxlen=240)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            with self._capture_lock:
                if not self._running:
                    break
                ok, frame = self.capture.read()
                if ok:
                    with self._lock:
                        now = time.monotonic()
                        self._latest = (frame, now)
                        self._frame_times.append(now)
                        while len(self._frame_times) > 2 and self._frame_times[0] < now - 2:
                            self._frame_times.popleft()
            if not ok:
                time.sleep(0.01)

    def read(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            latest, self._latest = self._latest, None
        return latest

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        with self._capture_lock:
            self.capture.release()

    def controls(self, changes):
        from .camera_controls import apply_controls, read_controls
        with self._capture_lock:
            messages = apply_controls(self.capture, changes)
            if "fps" in changes or "resolution" in changes:
                with self._lock:
                    self._latest = None
                    self._frame_times.clear()
            return read_controls(self.capture), messages

    @property
    def capture_fps(self):
        with self._lock:
            if len(self._frame_times) < 2:
                return None
            span = self._frame_times[-1] - self._frame_times[0]
            return (len(self._frame_times) - 1) / span if span > 0 else None


class FileSource:
    """以實際播放速度讀取影片（推論太慢時會跳幀），時間戳記使用影片時間。"""

    def __init__(self, path: str, loop: bool = True):
        self.label = Path(path).name
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"無法開啟影片：{path}")
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 240 else 30.0
        self.loop = loop
        self.ended = False
        self._index = 0            # 已讀取的幀數
        self._offset = 0.0         # 循環播放時累加，讓時間戳記保持遞增
        self._wall_start = time.monotonic()

    def read(self) -> tuple[np.ndarray, float] | None:
        elapsed = time.monotonic() - self._wall_start
        due = self._index / self.fps
        if elapsed < due:
            time.sleep(min(due - elapsed, 0.02))
            return None
        for _ in range(int(elapsed * self.fps) - self._index):
            if not self.capture.grab():
                return self._rewind()
            self._index += 1
        ok, frame = self.capture.read()
        if not ok:
            return self._rewind()
        timestamp = self._offset + self._index / self.fps
        self._index += 1
        return frame, timestamp

    def _rewind(self):
        if not self.loop or self._index == 0:
            self.ended = True
            return None
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._offset += self._index / self.fps
        self._index = 0
        self._wall_start = time.monotonic()
        return None

    def close(self):
        self.capture.release()


class VideoPipeline(QThread):
    packet_ready = Signal(object)          # FramePacket
    events_ready = Signal(object)          # list[EngineEvent]
    model_loaded = Signal(object, str, str)  # ModelInfo | None, device, requested source path
    model_progress = Signal(str, str)       # stage, requested source path
    model_timing = Signal(object)
    source_changed = Signal(str)           # 來源名稱，空字串 = 已中斷
    camera_available = Signal(bool)
    camera_controls_ready = Signal(object, object)
    engine_state = Signal(bool)            # 作業是否進行中
    records_changed = Signal()             # 紀錄已寫入資料庫
    status = Signal(str)
    error = Signal(str)

    ERROR_REPEAT_SEC = 30.0                # 相同錯誤在此秒數內只通知一次，避免洗版
    PENDING_RESULTS = 90                   # 介面忙碌時最多保留幾幀結果給試跑

    def __init__(self, records_dir: Path, parent=None, inference_worker=None):
        super().__init__(parent)
        self._records_dir = Path(records_dir)
        self._commands: queue.Queue = queue.Queue()
        self._alive = True
        self._inference_worker = inference_worker
        self._worker_lock = threading.Lock()
        self._gui_ready = threading.Event()
        self._gui_ready.set()
        self._source = None
        self._detector = None
        self._engine: SOPEngine | None = None
        self._locator: WorkpieceLocator | None = None
        self._locator_key = ""
        self._preview_workpiece: Workpiece | None = None
        self._preview_key = ""
        self._model_generation = 0
        self._context: dict = {}
        self._pending_results: deque[FrameResult] = deque(maxlen=self.PENDING_RESULTS)
        self._error_times: dict[str, float] = {}
        self._now = time.monotonic()
        self._show_masks, self._min_score = True, 0.5
        self._last_display: np.ndarray | None = None
        self._fps, self._last_frame_at = 0.0, None

    # ---- 介面執行緒呼叫 --------------------------------------------------------
    def open_camera(self, index: int):
        self._commands.put(("source", lambda: CameraSource(index)))

    def open_file(self, path: str, loop: bool = True):
        self._commands.put(("source", lambda: FileSource(path, loop)))

    def close_source(self):
        self._commands.put(("source", None))

    def camera_controls(self, changes=None):
        self._commands.put(("camera_controls", copy.deepcopy(changes or {})))

    def load_model(self, path: str | Path):
        self._commands.put(("model", str(path)))

    def clear_model(self):
        self._commands.put(("clear_model", None))

    def set_workpiece(self, workpiece: Workpiece | None):
        """待機時要定位的參考工件（編輯中的 SOP）；正式作業期間改用作業中 SOP 的工件。"""
        workpiece = copy.deepcopy(workpiece)
        self._commands.put(("workpiece", (workpiece, workpiece_key(workpiece))))

    def set_run_context(self, context: dict):
        """作業員、工單、序號等追溯資訊；下一輪開始時寫入紀錄。"""
        self._commands.put(("context", dict(context)))

    def start_engine(self, sop: SOPDefinition):
        self._commands.put(("start", copy.deepcopy(sop)))

    def stop_engine(self):
        self._commands.put(("stop", None))

    def restart_cycle(self):
        self._commands.put(("restart", None))

    def confirm_step(self):
        """只確認無自動條件的工序；有條件的工序必須用 skip_step 明確跳過。"""
        self._commands.put(("confirm", None))

    def skip_step(self):
        self._commands.put(("skip", None))

    def acknowledge_alarms(self):
        self._commands.put(("ack_alarms", None))

    def set_display(self, show_masks: bool, min_score: float):
        self._commands.put(("display", (show_masks, min_score)))

    def acknowledge_packet(self):
        """介面處理完一個畫面後呼叫；介面忙碌時背景執行緒就不會塞更多畫面進佇列。"""
        self._gui_ready.set()

    def shutdown(self):
        with self._worker_lock:
            self._alive = False
            worker = self._inference_worker
        if worker is not None:
            worker.close()
        return self.wait(5000)

    # ---- 背景執行緒 ------------------------------------------------------------
    def run(self):
        recorder = AsyncRecorder(self._records_dir, on_written=self.records_changed.emit,
                                 on_error=self.error.emit)
        try:
            while self._alive:
                self._drain_commands(recorder)
                if self._source is None:
                    time.sleep(0.03)
                    continue
                try:
                    got = self._source.read()
                    if got is None:
                        if self._source.ended:
                            self.status.emit(f"{self._source.label} 播放結束")
                            self._set_source(None, recorder)
                        else:
                            time.sleep(0.003)
                        continue
                    self._process(*got, recorder)
                except Exception as exc:     # 單幀失敗不能讓整條影像管線停止
                    self._report_frame_error(exc)
        finally:
            if self._inference_worker is not None:
                self._inference_worker.close()
            if self._source is not None:
                self._source.close()
            recorder.close()

    def _report_frame_error(self, exc: Exception):
        logging.getLogger("sop.pipeline").exception("影像處理失敗")
        self._gui_ready.set()               # 不讓介面因為少收一個畫面而永遠等不到下一幀
        message = f"影像處理發生錯誤（已略過該幀並持續執行）：{exc}"
        now = time.monotonic()
        if now - self._error_times.get(message, -1e9) >= self.ERROR_REPEAT_SEC:
            self._error_times[message] = now
            self.error.emit(message)

    def _process(self, frame: np.ndarray, timestamp: float, recorder):
        self._now = timestamp
        started = time.perf_counter()
        detections = []
        if self._detector is not None:
            try:
                detections = self._detector.detect(frame)
            except Exception as exc:     # 推論失敗不讓整個執行緒掛掉
                failed_source = str(self._detector.info.source)
                self._detector = None
                if self._alive:
                    logging.getLogger("sop.launcher").exception("模型推論失敗")
                    if getattr(exc, "remote_traceback", None):
                        logging.getLogger("sop.launcher").error("AI 程序例外：\n%s", exc.remote_traceback)
                    self.error.emit(f"模型推論失敗，已停用模型：{exc}")
                self.model_loaded.emit(None, "", failed_source)
        inference_ms = (time.perf_counter() - started) * 1000

        result = FrameResult(frame.shape[1], frame.shape[0], timestamp, detections)
        result.detection_valid = self._detector is not None
        if self._locator is not None:
            try:
                tracking_load = max(inference_ms, (time.monotonic() - timestamp) * 1000) if isinstance(self._source, CameraSource) else inference_ms
                result.workpiece_transform, result.tracking_status = self._locator.locate(
                    frame, timestamp, other_ms=tracking_load, detections=detections)
                result.tracking_metrics = self._locator.metrics.copy()
            except Exception as exc:
                result.workpiece_transform, result.tracking_status = None, f"位置跟隨失敗：{exc}"
            result.workpiece_key = self._locator_key
        events = []
        if self._engine is not None:
            try:
                events = self._engine.update(result)
            except Exception as exc:
                self.error.emit(f"判定引擎錯誤，已停止作業：{exc}")
                self._engine = None
                self._sync_locator()
                self.engine_state.emit(False)

        # 只讀一次：介面可能在本函式執行途中確認上一個畫面，前後判斷必須一致。
        ready = self._gui_ready.is_set()
        classes = tuple(self._detector.classes) if self._detector is not None else ()
        if events or ready or self._last_display is None:
            display = draw_detections(frame, detections, classes, self._min_score, self._show_masks)
            if self._engine is not None:
                anchored = {r.name for r in self._engine.sop.rois if r.anchor == "workpiece"}
                for roi in display_rois(self._engine.sop.rois, result.workpiece_transform):
                    if roi.name in anchored:
                        polygon = np.int32(np.asarray(roi.points) * [result.width, result.height])
                        cv2.polylines(display, [polygon], True, (0, 220, 255), 2)
            self._last_display = display
        if events:
            self._emit_events(events, recorder)

        now = time.perf_counter()
        if self._last_frame_at is not None:
            instant = 1.0 / max(now - self._last_frame_at, 1e-6)
            self._fps = instant if self._fps == 0 else self._fps * 0.9 + instant * 0.1
        self._last_frame_at = now

        self._pending_results.append(result)
        if ready:
            self._gui_ready.clear()
            snapshot = self._engine.snapshot() if self._engine is not None else None
            camera_fps = self._source.capture_fps if isinstance(self._source, CameraSource) else None
            results = tuple(self._pending_results)
            self._pending_results.clear()
            self.packet_ready.emit(FramePacket(frame, self._last_display, result, snapshot,
                                               self._fps, inference_ms, camera_fps,
                                               classes, self._min_score, self._show_masks, results,
                                               processing_ms=(time.perf_counter() - started) * 1000,
                                               capture_age_ms=max(0., (time.monotonic() - timestamp) * 1000)
                                               if isinstance(self._source, CameraSource) else None))

    def _emit_events(self, events, recorder):
        sop = self._engine.sop
        model_version = self._detector.info.model_version_id if self._detector is not None else ""
        try:
            recorder.handle(events, sop.name, sop.version, model_version, self._last_display, self._context)
        except Exception as exc:
            self.error.emit(f"寫入紀錄失敗：{exc}")
        self.events_ready.emit(events)

    def _drain_commands(self, recorder: Recorder):
        while self._alive:
            try:
                command, payload = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_command(command, payload, recorder)
            except Exception as exc:
                if self._alive:
                    self.error.emit(str(exc))

    def _handle_command(self, command: str, payload, recorder):
        if command == "source":
            self._set_source(payload, recorder)
        elif command == "model":
            self._start_model_load(payload)
        elif command == "model_ready":
            generation, detector, info, path = payload
            if generation == self._model_generation:
                self._install_model(detector, info, path)
        elif command == "clear_model":
            self._model_generation += 1
            self._detector = None
        elif command == "display":
            self._show_masks, self._min_score = payload
        elif command == "workpiece":
            self._preview_workpiece, self._preview_key = payload
            self._sync_locator()
        elif command == "context":
            self._context.update(payload)
        elif command == "ack_alarms":
            recorder.acknowledge_alarms(self._context.get("operator", ""))
        elif command == "camera_controls":
            if not isinstance(self._source, CameraSource):
                self.camera_controls_ready.emit({}, ["攝影機已中斷，請重新連線。"])
            else:
                changes = dict(payload)
                blocked = []
                if self._engine is not None and self._engine.phase != Phase.IDLE:
                    for key in ("resolution", "fps"):
                        if key in changes:
                            changes.pop(key)
                            blocked.append("解析度" if key == "resolution" else "攝影機 FPS")
                try:
                    values, messages = self._source.controls(changes)
                    if "fps" in changes or "resolution" in changes:
                        self._fps, self._last_frame_at = 0.0, None
                except Exception as exc:
                    values, messages = {}, [f"未完成：讀取／設定攝影機失敗（{exc}）"]
                if blocked:
                    messages.insert(0, f"{'、'.join(blocked)}：未完成（請先停止作業，再變更影像格式）")
                self.camera_controls_ready.emit(values, messages)
        elif command == "start":
            self._stop_engine(recorder)
            self._engine = SOPEngine(payload)
            if self._source is not None:
                self._now = time.monotonic() if isinstance(self._source, CameraSource) else self._now
            events = self._engine.start(self._now)
            self._sync_locator()              # 引擎進入作業狀態後才切換為作業中 SOP 的工件
            self._emit_events(events, recorder)
            self.engine_state.emit(True)
        elif command == "stop":
            self._stop_engine(recorder)
        elif self._engine is not None and self._engine.phase != Phase.IDLE:
            if command == "restart":
                events = self._engine.restart_cycle(self._now)
            elif command == "skip":
                events = self._engine.skip_step(self._now)
            else:
                events = self._engine.confirm_manual_step(self._now)
            if events:
                self._emit_events(events, recorder)

    def _sync_locator(self):
        """正式作業用作業中 SOP 的工件，否則用編輯中的工件；同一工件沿用定位狀態。"""
        if self._engine is not None and self._engine.phase != Phase.IDLE:
            workpiece = self._engine.sop.workpiece
            key = workpiece_key(workpiece)
        else:
            workpiece, key = self._preview_workpiece, self._preview_key
        if key == self._locator_key and (self._locator is not None or workpiece is None):
            return
        self._locator, self._locator_key = None, key
        if workpiece is not None:
            try:
                self._locator = WorkpieceLocator(workpiece)
            except (ValueError, TypeError) as exc:
                self.status.emit(f"工件定位設定無效：{exc}")

    def _stop_engine(self, recorder):
        if self._engine is not None and self._engine.phase != Phase.IDLE:
            self._emit_events(self._engine.stop(self._now), recorder)
            self.engine_state.emit(False)
        self._sync_locator()

    def _set_source(self, factory, recorder):
        if self._engine is not None and self._engine.phase != Phase.IDLE:
            # 不同來源的時間基準不同，換來源時停止作業避免超時判斷錯亂
            self._stop_engine(recorder)
            self.status.emit("影像來源變更，已停止作業")
        if self._source is not None:
            self._source.close()
            self._source = None
        if self._locator is not None:
            self._locator.reset()
        self._pending_results.clear()
        self._last_display = None
        self.camera_available.emit(False)
        self.source_changed.emit("")
        self._fps, self._last_frame_at = 0.0, None
        if factory is None:
            return
        self.status.emit("正在開啟影像來源…")
        self._source = factory()
        self._now = time.monotonic()
        self.source_changed.emit(self._source.label)
        self.camera_available.emit(isinstance(self._source, CameraSource))

    # ---- 模型 ------------------------------------------------------------------
    def _start_model_load(self, path: str):
        """在獨立執行緒載入；載入期間影像與定位照常更新，完成後才換上新模型。"""
        self._model_generation += 1
        generation = self._model_generation
        self._detector = None
        self._model_thread = threading.Thread(target=self._model_worker, args=(path, generation),
                                              name="sop-model-loader", daemon=True)
        self._model_thread.start()

    def _model_worker(self, path: str, generation: int):
        try:
            built = self._build_detector(path)
            if built is None or not self._alive:
                return
            detector, info = built
        except Exception as exc:
            if self._alive and generation == self._model_generation:
                self.model_loaded.emit(None, "", path)
                self.error.emit(str(self._load_error(exc)))
            return
        self._commands.put(("model_ready", (generation, detector, info, path)))

    def _load_model(self, path: str):
        """同步載入（工具程式與測試使用）。"""
        if not self._alive:
            return
        self._detector = None
        try:
            built = self._build_detector(path)
            if built is None or not self._alive:
                return
            detector, info = built
        except Exception as exc:
            self.model_loaded.emit(None, "", path)
            raise self._load_error(exc) from exc
        self._install_model(detector, info, path)

    def _build_detector(self, path: str):
        if not self._alive:
            return
        started = time.perf_counter()
        def report(message):
            if hasattr(message, "measured_at"):
                self.model_timing.emit(message)
            self.model_progress.emit(message, path)
            logging.getLogger("sop.launcher").info("模型載入 %.2fs · %s", time.perf_counter() - started, message)

        self.status.emit("模型載入中…（第一次使用需解壓縮，請稍候；影像會持續更新）")
        try:
            report("讀取模型資訊")
            info = read_model_info(path)
            with self._worker_lock:
                if not self._alive:
                    raise InferenceWorkerClosed("Pipeline closed")
                if self._inference_worker is None:
                    self._inference_worker = InferenceWorker().start()
            detector = self._inference_worker.load_model(path, progress=report)
            if not self._alive:
                return
        except InferenceWorkerClosed:
            if self._alive:
                raise
            return
        except Exception as exc:
            logging.getLogger("sop.launcher").exception("模型載入失敗：%s", path)
            if getattr(exc, "remote_traceback", None):
                logging.getLogger("sop.launcher").error("AI 程序例外：\n%s", exc.remote_traceback)
            report("載入失敗")
            raise
        report("載入完成")
        return detector, info

    def _install_model(self, detector, info, path: str):
        self._detector = detector
        self.model_loaded.emit(info, detector.device_name, path)
        self.status.emit(f"模型 {info.model_version_id} 已載入（{detector.device_name}）")

    @staticmethod
    def _load_error(exc: Exception) -> RuntimeError:
        if getattr(exc, "winerror", None) == 4551:
            return RuntimeError(
                "Windows 應用程式控制已封鎖 PyTorch 模型 DLL（錯誤 4551）。\n"
                "目前無法使用 AI 辨識；仍可編輯 SOP、連線影像與設定工件定位。\n"
                "請由此電腦的管理者檢查 Smart App Control／應用程式控制原則及套件信任狀態。\n\n"
                f"詳細資訊：{exc}")
        return RuntimeError(f"模型載入失敗：{exc}")
