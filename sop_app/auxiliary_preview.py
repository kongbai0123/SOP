"""Independent camera previews with one latest frame per selected device."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import logging
from threading import Event, Lock, Thread
import time

import numpy as np

from .native_runtime import cv2


@dataclass(frozen=True)
class PreviewFrame:
    image: np.ndarray | None = None
    status: str = ""


class AuxiliaryPreview:
    """Own secondary camera streams without involving the SOP pipeline."""

    FRAME_INTERVAL = 1 / 12
    NO_FRAME_TIMEOUT = 2.0
    MAX_SIZE = (640, 360)

    def __init__(self):
        self._lock = Lock()
        self._frames: dict[int, PreviewFrame] = {}
        self._workers: dict[int, Event] = {}
        self._closed = False

    def set_cameras(self, indices: Iterable[int]):
        """Request the listed DirectShow cameras and release the others."""
        wanted = set(indices)
        with self._lock:
            if self._closed:
                return
            for index in self._frames.keys() - wanted:
                del self._frames[index]
                stop = self._workers.get(index)
                if stop is not None:
                    stop.set()
            for index in wanted - self._frames.keys():
                self._frames[index] = PreviewFrame(status="連線中…")
                if index not in self._workers:
                    self._start_worker(index)

    def snapshot(self) -> dict[int, PreviewFrame]:
        with self._lock:
            return self._frames.copy()

    def is_released(self, index: int) -> bool:
        """Check whether a deselected camera is ready for primary capture."""
        with self._lock:
            return index not in self._workers

    def close(self):
        with self._lock:
            self._closed = True
            self._frames.clear()
            for stop in self._workers.values():
                stop.set()

    def _start_worker(self, index: int):
        # Called with the lock held, after the previous capture has released.
        stop = self._workers[index] = Event()
        worker = Thread(target=self._capture, args=(index, stop), name=f"camera-preview-{index}", daemon=True)
        try:
            worker.start()
        except RuntimeError:
            del self._workers[index]
            self._frames[index] = PreviewFrame(status="無法啟動攝影機預覽")
            logging.getLogger(__name__).exception("無法啟動攝影機 %d 預覽", index)

    def _publish(self, index: int, stop: Event, result: PreviewFrame):
        with self._lock:
            if not stop.is_set():
                self._frames[index] = result

    def _capture(self, index: int, stop: Event):
        camera = None
        pending = "無法開啟攝影機"
        logger = logging.getLogger(__name__)
        try:
            if stop.is_set():
                return
            # Source indexes come from DirectShow. CAP_ANY can address a
            # different camera using the same index in another backend.
            camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not camera.isOpened():
                self._publish(index, stop, PreviewFrame(status=pending))
                return
            pending = "無法讀取攝影機影像"
            last_frame_at = time.monotonic()
            while not stop.is_set():
                started_at = time.monotonic()
                ok, frame = camera.read()
                if ok and frame is not None and frame.size and frame.ndim == 3 and frame.shape[2] == 3:
                    height, width = frame.shape[:2]
                    max_width, max_height = self.MAX_SIZE
                    scale = min(1., max_width / width, max_height / height)
                    if scale < 1.:
                        size = (max(1, round(width * scale)), max(1, round(height * scale)))
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    last_frame_at = time.monotonic()
                    self._publish(index, stop, PreviewFrame(image=frame))
                elif time.monotonic() - last_frame_at >= self.NO_FRAME_TIMEOUT:
                    self._publish(index, stop, PreviewFrame(status=pending))
                    break
                stop.wait(max(0., self.FRAME_INTERVAL - (time.monotonic() - started_at)))
        except Exception:
            logger.exception("%s %d", pending, index)
            self._publish(index, stop, PreviewFrame(status=pending))
        finally:
            try:
                if camera is not None:
                    camera.release()
            except Exception:
                logger.exception("無法釋放攝影機 %d", index)
            finally:
                with self._lock:
                    del self._workers[index]
                    # A quick off/on toggle can be requested as the old
                    # worker is still finishing a read. Do not double-open.
                    if stop.is_set() and index in self._frames:
                        self._start_worker(index)
