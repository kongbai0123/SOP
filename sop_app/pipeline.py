"""背景執行緒：讀取影像 → 模型偵測 → 判定引擎 → 紀錄，結果以 Qt Signal 傳給介面。

介面只透過指令佇列控制這個執行緒，模型、引擎、SQLite 都只在背景執行緒內使用，不需要鎖。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .detection import FrameResult
from .engine import EngineSnapshot, Phase, SOPEngine
from .model_bundle import read_model_info
from .overlay import draw_detections
from .recorder import Recorder
from .sop_schema import SOPDefinition
from .tracking import WorkpieceLocator, display_rois


@dataclass
class FramePacket:
    frame: np.ndarray              # 原始畫面
    display: np.ndarray            # 疊上偵測結果的畫面
    result: FrameResult
    snapshot: EngineSnapshot | None
    fps: float
    inference_ms: float


class CameraSource:
    """獨立執行緒持續讀取，只保留最新一幀，避免推論較慢時畫面延遲越積越多。"""

    ended = False

    def __init__(self, index: int, width: int = 1280, height: int = 720):
        self.label = f"攝影機 {index}"
        self.capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"無法開啟攝影機 {index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = (frame, time.monotonic())

    def read(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            latest, self._latest = self._latest, None
        return latest

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.capture.release()


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
    model_loaded = Signal(object, str)     # ModelInfo | None, 裝置名稱
    source_changed = Signal(str)           # 來源名稱，空字串 = 已中斷
    engine_state = Signal(bool)            # 作業是否進行中
    status = Signal(str)
    error = Signal(str)

    def __init__(self, records_dir: Path, parent=None):
        super().__init__(parent)
        self._records_dir = Path(records_dir)
        self._commands: queue.Queue = queue.Queue()
        self._alive = True
        self._gui_ready = threading.Event()
        self._gui_ready.set()

    # ---- 介面執行緒呼叫 --------------------------------------------------------
    def open_camera(self, index: int):
        self._commands.put(("source", lambda: CameraSource(index)))

    def open_file(self, path: str, loop: bool = True):
        self._commands.put(("source", lambda: FileSource(path, loop)))

    def close_source(self):
        self._commands.put(("source", None))

    def load_model(self, path: str | Path):
        self._commands.put(("model", str(path)))

    def start_engine(self, sop: SOPDefinition):
        self._commands.put(("start", copy.deepcopy(sop)))

    def stop_engine(self):
        self._commands.put(("stop", None))

    def restart_cycle(self):
        self._commands.put(("restart", None))

    def confirm_step(self):
        self._commands.put(("confirm", None))

    def set_display(self, show_masks: bool, min_score: float):
        self._commands.put(("display", (show_masks, min_score)))

    def acknowledge_packet(self):
        """介面處理完一個畫面後呼叫；介面忙碌時背景執行緒就不會塞更多畫面進佇列。"""
        self._gui_ready.set()

    def shutdown(self):
        self._alive = False
        self.wait(5000)

    # ---- 背景執行緒 ------------------------------------------------------------
    def run(self):
        self._source = None
        self._detector = None
        self._engine: SOPEngine | None = None
        self._locator = None
        self._now = time.monotonic()
        self._show_masks, self._min_score = True, 0.5
        self._last_display: np.ndarray | None = None
        self._fps, self._last_frame_at = 0.0, None
        recorder = Recorder(self._records_dir)
        try:
            while self._alive:
                self._drain_commands(recorder)
                if self._source is None:
                    time.sleep(0.03)
                    continue
                got = self._source.read()
                if got is None:
                    if self._source.ended:
                        self.status.emit(f"{self._source.label} 播放結束")
                        self._set_source(None, recorder)
                    else:
                        time.sleep(0.003)
                    continue
                self._process(*got, recorder)
        finally:
            if self._source is not None:
                self._source.close()
            recorder.close()

    def _process(self, frame: np.ndarray, timestamp: float, recorder: Recorder):
        self._now = timestamp
        started = time.perf_counter()
        detections = []
        if self._detector is not None:
            try:
                detections = self._detector.detect(frame)
            except Exception as exc:     # 推論失敗不讓整個執行緒掛掉
                self._detector = None
                self.error.emit(f"模型推論失敗，已停用模型：{exc}")
                self.model_loaded.emit(None, "")
        inference_ms = (time.perf_counter() - started) * 1000

        result = FrameResult(frame.shape[1], frame.shape[0], timestamp, detections)
        result.detection_valid = self._detector is not None
        if self._locator is not None:
            result.workpiece_transform, result.tracking_status = self._locator.locate(frame)
        events = []
        if self._engine is not None:
            try:
                events = self._engine.update(result)
            except Exception as exc:
                self.error.emit(f"判定引擎錯誤，已停止作業：{exc}")
                self._engine = None
                self.engine_state.emit(False)

        if events or self._gui_ready.is_set():
            classes = self._detector.classes if self._detector is not None else ()
            self._last_display = draw_detections(frame, detections, classes, self._min_score, self._show_masks)
            if self._engine is not None:
                for roi in display_rois(self._engine.sop.rois, result.workpiece_transform):
                    if any(r.name == roi.name and r.anchor == "workpiece" for r in self._engine.sop.rois):
                        polygon = np.int32(np.asarray(roi.points) * [result.width, result.height])
                        cv2.polylines(self._last_display, [polygon], True, (0, 220, 255), 2)
        if events:
            self._emit_events(events, recorder)

        now = time.perf_counter()
        if self._last_frame_at is not None:
            instant = 1.0 / max(now - self._last_frame_at, 1e-6)
            self._fps = instant if self._fps == 0 else self._fps * 0.9 + instant * 0.1
        self._last_frame_at = now

        if self._gui_ready.is_set():
            self._gui_ready.clear()
            snapshot = self._engine.snapshot() if self._engine is not None else None
            self.packet_ready.emit(FramePacket(frame, self._last_display, result, snapshot, self._fps, inference_ms))

    def _emit_events(self, events, recorder: Recorder):
        sop = self._engine.sop
        model_version = self._detector.info.model_version_id if self._detector is not None else ""
        try:
            recorder.handle(events, sop.name, sop.version, model_version, self._last_display)
        except Exception as exc:
            self.error.emit(f"寫入紀錄失敗：{exc}")
        self.events_ready.emit(events)

    def _drain_commands(self, recorder: Recorder):
        while True:
            try:
                command, payload = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_command(command, payload, recorder)
            except Exception as exc:
                self.error.emit(str(exc))

    def _handle_command(self, command: str, payload, recorder: Recorder):
        if command == "source":
            self._set_source(payload, recorder)
        elif command == "model":
            self._load_model(payload)
        elif command == "display":
            self._show_masks, self._min_score = payload
        elif command == "start":
            locator = WorkpieceLocator(payload.workpiece) if payload.workpiece else None
            self._stop_engine(recorder)
            self._locator = locator
            self._engine = SOPEngine(payload)
            if self._source is not None:
                self._now = time.monotonic() if isinstance(self._source, CameraSource) else self._now
            self._emit_events(self._engine.start(self._now), recorder)
            self.engine_state.emit(True)
        elif command == "stop":
            self._stop_engine(recorder)
        elif self._engine is not None and self._engine.phase != Phase.IDLE:
            if command == "restart":
                events = self._engine.restart_cycle(self._now)
            else:
                events = self._engine.confirm_step(self._now)
            if events:
                self._emit_events(events, recorder)

    def _stop_engine(self, recorder: Recorder):
        self._locator = None
        if self._engine is not None and self._engine.phase != Phase.IDLE:
            self._emit_events(self._engine.stop(self._now), recorder)
            self.engine_state.emit(False)

    def _set_source(self, factory, recorder: Recorder):
        if self._engine is not None and self._engine.phase != Phase.IDLE:
            # 不同來源的時間基準不同，換來源時停止作業避免超時判斷錯亂
            self._stop_engine(recorder)
            self.status.emit("影像來源變更，已停止作業")
        if self._source is not None:
            self._source.close()
            self._source = None
        self._fps, self._last_frame_at = 0.0, None
        if factory is None:
            self.source_changed.emit("")
            return
        self.status.emit("正在開啟影像來源…")
        self._source = factory()
        self._now = time.monotonic()
        self.source_changed.emit(self._source.label)

    def _load_model(self, path: str):
        from .detector import MaskRCNNDetector      # 延遲匯入：torch 載入較慢

        self.status.emit("模型載入中…（第一次使用需解壓縮，請稍候）")
        self._detector = None
        try:
            info = read_model_info(path)
            self._detector = MaskRCNNDetector(info)
        except Exception as exc:
            self.model_loaded.emit(None, "")
            if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 4551:
                raise RuntimeError(
                    "Windows 應用程式控制已封鎖 PyTorch 模型 DLL（錯誤 4551）。\n"
                    "目前無法使用 AI 辨識；仍可編輯 SOP、連線影像與設定工件定位。\n"
                    "請由此電腦的管理者檢查 Smart App Control／應用程式控制原則及套件信任狀態。\n\n"
                    f"詳細資訊：{exc}"
                ) from exc
            raise RuntimeError(f"模型載入失敗：{exc}") from exc
        self.model_loaded.emit(info, self._detector.device_name)
        self.status.emit(f"模型 {info.model_version_id} 已載入（{self._detector.device_name}）")
