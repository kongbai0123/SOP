"""Keep the model and its CUDA context in one process, separate from Qt.

This module deliberately imports only the standard library at module scope so
the GUI can start while the spawned process imports PyTorch. Calls are serial;
frames share one reusable memory buffer and boolean masks travel packed as bits.
"""
from __future__ import annotations

import gc
import logging
import multiprocessing
from multiprocessing import shared_memory
import os
from pathlib import Path
import threading
import time
import traceback


_POLL_SECONDS = 0.05


class InferenceWorkerClosed(RuntimeError):
    """The owner cancelled a pending operation or closed the worker."""


class InferenceWorkerError(OSError):
    """A remote failure, including the original Windows error and traceback."""

    def __init__(self, message, *, remote_type="WorkerError", winerror=None,
                 remote_traceback=""):
        super().__init__(message)
        self.remote_type = remote_type
        self.winerror = winerror
        self.remote_traceback = remote_traceback


def _normalized_path(path):
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _error_record(exc):
    # A runtime may wrap an OSError; retain its useful code through that chain.
    winerror, current, visited = None, exc, set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        winerror = getattr(current, "winerror", None)
        if winerror is not None:
            break
        current = current.__cause__ or current.__context__
    return {"type": type(exc).__name__, "message": str(exc), "winerror": winerror,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))}


def _remote_error(record):
    return InferenceWorkerError(record["message"], remote_type=record["type"],
                                winerror=record.get("winerror"),
                                remote_traceback=record.get("traceback", ""))


class RemoteDetector:
    """The detector interface used by VideoPipeline, backed by the worker."""

    def __init__(self, worker, metadata):
        self._worker = worker
        self._generation = metadata["generation"]
        self.info = metadata["info"]
        self.classes = tuple(metadata["classes"])
        self.device_name = metadata["device_name"]

    def detect(self, frame):
        return self._worker._detect(frame, self._generation)


class InferenceWorker:
    """One persistent spawned child; start explicitly before constructing Qt.

    ``load_model`` and ``detect`` may run on a background thread. ``close`` can
    run concurrently on the GUI thread and cancels their wait immediately.
    ``worker_target`` is a spawn-compatible target for protocol tests; production
    uses the PyTorch-first target below.
    """

    def __init__(self, preload_path=None, *, worker_target=None):
        self._preload_path = _normalized_path(preload_path) if preload_path else None
        self._worker_target = worker_target or _worker_main
        self._context = multiprocessing.get_context("spawn")
        self._closed = threading.Event()
        self._rpc_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._connection = None
        self._process = None
        self._cancel = None
        self._next_id = 0
        self._preload_id = None
        self._responses = {}
        self._fatal_error = None
        self._runtime_ready = False
        self._frame_memory = None

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    @property
    def is_alive(self):
        return self._process is not None and self._process.is_alive()

    def start(self):
        """Start imports and optionally queue the initial model without waiting."""
        with self._lifecycle_lock:
            if self._closed.is_set():
                raise InferenceWorkerClosed("模型程序已關閉")
            if self._process is not None:
                return self
            parent, child = self._context.Pipe()
            self._cancel = self._context.Event()
            process = self._context.Process(target=self._worker_target,
                                            args=(child, self._cancel),
                                            name="SOP inference", daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            self._connection, self._process = parent, process
            if self._preload_path is not None:
                self._preload_id = self._send("load", path=self._preload_path)
        return self

    def _check_open(self):
        if self._closed.is_set():
            raise InferenceWorkerClosed("模型程序已關閉")
        if self._process is None:
            raise RuntimeError("請先啟動模型程序（InferenceWorker.start）")
        if self._fatal_error is not None:
            raise self._fatal_error

    def _send(self, operation, **payload):
        self._check_open()
        self._next_id += 1
        request_id = self._next_id
        try:
            self._connection.send({"id": request_id, "operation": operation, **payload})
        except (OSError, EOFError) as exc:
            if self._closed.is_set():
                raise InferenceWorkerClosed("模型程序已關閉") from exc
            raise InferenceWorkerError(f"無法連線至模型程序：{exc}") from exc
        return request_id

    def _receive(self, request_id, progress=None):
        while True:
            self._check_open()
            response = self._responses.pop(request_id, None)
            if response is not None:
                if response["kind"] == "error":
                    raise _remote_error(response["error"])
                return response["result"]
            try:
                if not self._connection.poll(_POLL_SECONDS):
                    if not self._process.is_alive():
                        raise InferenceWorkerError(
                            f"模型程序意外結束（exit code {self._process.exitcode}）",
                            remote_type="WorkerExited")
                    continue
                response = self._connection.recv()
            except (OSError, EOFError, ValueError) as exc:
                if self._closed.is_set():
                    raise InferenceWorkerClosed("模型程序已關閉") from exc
                if isinstance(exc, InferenceWorkerError):
                    raise
                # Pipe EOF can precede the process handle becoming signalled.
                self._process.join(timeout=_POLL_SECONDS)
                raise InferenceWorkerError(
                    f"模型程序連線中斷（exit code {self._process.exitcode}）：{exc}",
                    remote_type="WorkerExited") from exc
            self._check_open()
            if response["kind"] == "fatal":
                self._fatal_error = _remote_error(response["error"])
                raise self._fatal_error
            if response["kind"] == "progress":
                logging.getLogger("sop.launcher").info(
                    "AI 程序 %.2fs · %s", response.get("elapsed", 0.), response["message"])
                if progress is not None and response["id"] in (None, request_id):
                    progress(response["message"])
            else:
                self._responses[response["id"]] = response

    def load_model(self, path, progress=None):
        """Wait for a ready, warmed model; reuse a matching queued preload."""
        normalized = _normalized_path(path)
        with self._rpc_lock:
            self._check_open()
            if not self._runtime_ready:
                # Consume startup failures before sending into a dead pipe so
                # DLL error 4551 remains available even without a preload.
                self._receive(0, progress)
                self._runtime_ready = True
            preload_id, self._preload_id = self._preload_id, None
            request_id = (preload_id if preload_id is not None and normalized == self._preload_path
                          else self._send("load", path=normalized))
            try:
                metadata = self._receive(request_id, progress)
                self._check_open()
            finally:
                if preload_id is not None and preload_id != request_id:
                    self._responses.pop(preload_id, None)
            return RemoteDetector(self, metadata)

    def _detect(self, frame, generation):
        import numpy as np
        from .detection import Detection

        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("辨識影像必須是非空的 uint8 BGR 影像")
        frame = np.ascontiguousarray(frame)
        with self._rpc_lock:
            # Close only holds this lock for buffer setup, never for inference.
            with self._lifecycle_lock:
                self._check_open()
                if self._frame_memory is None or self._frame_memory.size < frame.nbytes:
                    self._release_frame_memory()
                    self._frame_memory = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                shared_frame = np.ndarray(frame.shape, dtype=frame.dtype,
                                          buffer=self._frame_memory.buf)
                np.copyto(shared_frame, frame)
                del shared_frame
                request_id = self._send("detect", generation=generation,
                                        memory=self._frame_memory.name,
                                        shape=frame.shape, dtype=frame.dtype.str)
            packed = self._receive(request_id)
            detections = []
            for label, score, box, mask_shape, mask_bytes in packed:
                mask = None
                if mask_shape is not None:
                    count = int(np.prod(mask_shape))
                    mask = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8),
                                         count=count, bitorder="little").reshape(mask_shape).astype(bool)
                detections.append(Detection(label, score, tuple(box), mask))
            return detections

    def _release_frame_memory(self):
        memory, self._frame_memory = self._frame_memory, None
        if memory is not None:
            memory.close()
            try:
                memory.unlink()
            except FileNotFoundError:
                pass

    def close(self):
        """Cancel waits and stop even a child stuck importing DLLs or warming up."""
        self._closed.set()
        with self._lifecycle_lock:
            if self._cancel is not None:
                self._cancel.set()
            process = self._process
            if process is not None:
                process.join(timeout=0.1)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.7)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=0.2)
            if self._connection is not None:
                self._connection.close()
            self._release_frame_memory()


def _worker_main(connection, cancelled):
    def bootstrap(report):
        report("載入 AI 執行環境")
        started = time.perf_counter()
        import torch  # noqa: F401  This process never imports Qt.
        report(f"AI 執行環境已載入（{time.perf_counter() - started:.2f} 秒）")

    def load_detector(path, report):
        from .model_bundle import read_model_info
        from .detector import create_detector

        report("讀取模型資訊")
        return create_detector(read_model_info(path), progress=report)

    _serve(connection, cancelled, load_detector, bootstrap=bootstrap)


def _serve(connection, cancelled, load_detector, *, bootstrap=None):
    """Serial child protocol, also exercised with a lightweight test detector."""
    child_started = time.perf_counter()

    def report(request_id, message):
        connection.send({"kind": "progress", "id": request_id, "message": str(message),
                         "elapsed": time.perf_counter() - child_started})

    detector, generation = None, 0
    try:
        if bootstrap is not None:
            try:
                bootstrap(lambda message: report(None, message))
            except Exception as exc:
                connection.send({"kind": "fatal", "id": None, "error": _error_record(exc)})
                return
        if cancelled.is_set():
            return
        connection.send({"kind": "result", "id": 0, "result": None})
        while not cancelled.is_set():
            if not connection.poll(_POLL_SECONDS):
                continue
            request = connection.recv()
            request_id = request["id"]
            try:
                if request["operation"] == "load":
                    # Release the old GPU model before allocating another one.
                    had_model = detector is not None
                    detector = None
                    if had_model:
                        gc.collect()
                    generation += 1
                    detector = load_detector(Path(request["path"]),
                                             lambda message: report(request_id, message))
                    result = {"info": detector.info, "classes": tuple(detector.classes),
                              "device_name": detector.device_name, "generation": generation}
                elif request["operation"] == "detect":
                    if detector is None or request["generation"] != generation:
                        raise RuntimeError("模型已變更或不可用，請重新載入模型")
                    result = _detect_shared_frame(detector, request)
                else:
                    raise ValueError(f"未知的模型程序指令：{request['operation']}")
                if not cancelled.is_set():
                    connection.send({"kind": "result", "id": request_id, "result": result})
            except Exception as exc:
                if not cancelled.is_set():
                    connection.send({"kind": "error", "id": request_id, "error": _error_record(exc)})
    except (EOFError, BrokenPipeError, OSError):
        # The parent may close its pipe while this process reports progress.
        pass
    finally:
        connection.close()


def _detect_shared_frame(detector, request):
    import numpy as np

    memory = shared_memory.SharedMemory(name=request["memory"])
    frame = None
    try:
        frame = np.ndarray(request["shape"], dtype=np.dtype(request["dtype"]), buffer=memory.buf)
        packed = []
        for detection in detector.detect(frame):
            shape, mask_bytes = None, None
            if detection.mask is not None:
                mask = np.asarray(detection.mask, dtype=bool)
                shape = mask.shape
                mask_bytes = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
            packed.append((detection.label, float(detection.score), tuple(detection.box),
                           shape, mask_bytes))
        return packed
    finally:
        del frame
        memory.close()
