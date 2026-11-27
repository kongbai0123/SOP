"""ONNX Runtime CPU backend for a fixed-shape YOLO26 raw detection model."""
from __future__ import annotations

from ast import literal_eval
import hashlib
import os
import time

import numpy as np

from .detection import Detection
from .detector import _collect
from .model_bundle import ModelBundleError, ModelInfo, ensure_checkpoint
from .native_runtime import cv2


def _nms_indices(boxes, scores, labels, threshold=.7, limit=300):
    """Apply class-aware NMS in network-image coordinates."""
    size = np.maximum(0, boxes[:, 2:] - boxes[:, :2])
    areas = size[:, 0] * size[:, 1]
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size and len(keep) < limit:
        first, others = order[0], order[1:]
        keep.append(first)
        top_left = np.maximum(boxes[first, :2], boxes[others, :2])
        bottom_right = np.minimum(boxes[first, 2:], boxes[others, 2:])
        overlap = np.maximum(0, bottom_right - top_left)
        intersection = overlap[:, 0] * overlap[:, 1]
        union = areas[first] + areas[others] - intersection
        suppress = (labels[others] == labels[first]) & (intersection > threshold * union)
        order = others[~suppress]
    return np.asarray(keep, dtype=np.intp)


class OnnxYolo26Detector:
    def __init__(self, info: ModelInfo, device="auto", score_floor=.3, progress=None):
        if device not in ("auto", "cpu"):
            raise ModelBundleError("此 ONNX 模型使用 CPU 執行")
        if not np.isfinite(score_floor) or not 0 <= score_floor <= 1:
            raise ValueError("辨識信心門檻須介於 0 與 1")

        report = progress or (lambda message: None)
        report("讀取與驗證 ONNX 模型")
        model_bytes = ensure_checkpoint(info).read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != info.checkpoint_sha256:
            raise ModelBundleError("ONNX 模型的 SHA-256 與模型包清單不符")

        report("載入 ONNX Runtime CPU")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        # Pass the verified bytes so the runtime has no model file directory
        # from which it could resolve an unverified external tensor file.
        self._session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"],
        )
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ModelBundleError("ONNX 模型必須只有一個影像輸入與一個偵測輸出")
        source, result = inputs[0], outputs[0]
        shape = tuple(source.shape or ())
        if (source.type != "tensor(float)" or len(shape) != 4 or shape[:2] != (1, 3)
                or any(type(size) is not int or size <= 0 or size % 32 for size in shape[2:])
                or max(shape[2:]) != info.image_size):
            raise ModelBundleError("ONNX 模型須提供固定大小的 FP32 NCHW 影像輸入，且與模型包尺寸一致")
        height, width = shape[2:]
        proposals = sum((height // stride) * (width // stride) for stride in (8, 16, 32))
        output_shape = (1, 4 + len(info.classes), proposals)
        if result.type != "tensor(float)" or tuple(result.shape or ()) != output_shape:
            raise ModelBundleError("ONNX 輸出須為 YOLO26 原始偵測格式 [1, 4 + 類別數, 候選數]")

        metadata = self._session.get_modelmeta().custom_metadata_map
        try:
            names = literal_eval(metadata["names"])
            if (not isinstance(names, dict) or any(type(index) is not int for index in names)
                    or set(names) != set(range(len(info.classes)))):
                raise ValueError("Invalid class indices")
            classes = tuple(names[index] for index in range(len(info.classes)))
        except (KeyError, TypeError, ValueError, SyntaxError, RecursionError) as exc:
            raise ModelBundleError("ONNX 模型缺少有效的類別索引") from exc
        if (classes != info.classes or metadata.get("task") != "detect"
                or metadata.get("end2end", "").casefold() != "false"):
            raise ModelBundleError("ONNX 類別順序或偵測格式與模型包不符")

        self.info = info
        self.classes = classes
        self.score_floor = score_floor
        self._input, self._output = source.name, result.name
        self._height, self._width = height, width
        self._output_shape = output_shape
        self.device_name = "CPU（ONNX Runtime）"
        self.runtime_info = {
            "backend": "onnxruntime", "runtime_version": ort.__version__,
            "device": "cpu", "cuda": False, "dtype": "float32", "fp16": False,
            "input_shape": list(shape), "shape_stage": "network_input",
        }
        self.last_metrics = {}
        report("預熱 ONNX 模型，準備首次辨識")
        self.detect(np.zeros((height, width, 3), dtype=np.uint8))

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        if (not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3
                or not bgr.size or bgr.dtype != np.uint8):
            raise ValueError("ONNX 推論需要非空的 uint8 BGR 三色影像")
        height, width = bgr.shape[:2]
        scale = min(self._height / height, self._width / width)
        resized_width, resized_height = round(width * scale), round(height * scale)
        if not resized_width or not resized_height:
            raise ValueError("影像寬高比超過模型的縮放範圍")
        left = round((self._width - resized_width) / 2 - .1)
        top = round((self._height - resized_height) / 2 - .1)
        resized = bgr
        if (width, height) != (resized_width, resized_height):
            resized = cv2.resize(bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._height, self._width, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = resized
        rgb = np.ascontiguousarray(canvas[np.newaxis, :, :, ::-1].transpose(0, 3, 1, 2), dtype=np.float32)
        rgb /= np.float32(255)
        prepared = time.perf_counter()

        values = self._session.run([self._output], {self._input: rgb})
        inferred = time.perf_counter()
        if (len(values) != 1 or not isinstance(values[0], np.ndarray)
                or values[0].shape != self._output_shape or values[0].dtype != np.float32):
            raise ModelBundleError("ONNX 推論結果的尺寸或資料類型與宣告不符")
        raw = values[0][0]
        class_scores = raw[4:]
        if (not np.isfinite(raw).all() or (class_scores < 0).any() or (class_scores > 1).any()):
            raise ModelBundleError("ONNX 輸出含有無效的偵測結果")
        scores = class_scores.max(axis=0)
        passing = scores > self.score_floor
        labels = class_scores[:, passing].argmax(axis=0)
        scores = scores[passing]
        center_and_size = raw[:4, passing].T
        if (center_and_size[:, 2:] < 0).any():
            raise ModelBundleError("ONNX 偵測框的寬度或高度為負值")
        centers, half_sizes = center_and_size[:, :2], center_and_size[:, 2:] / 2
        boxes = np.concatenate((centers - half_sizes, centers + half_sizes), axis=1)
        if not np.isfinite(boxes).all():
            raise ModelBundleError("ONNX 偵測框的座標超出範圍")
        # Keep the original FP32 box decoding for matching PyTorch rounding;
        # use higher precision when computing intersection areas for NMS.
        selected = _nms_indices(boxes.astype(np.float64), scores, labels)
        boxes = boxes[selected]
        boxes[:, 0::2] = (boxes[:, 0::2] - left) / scale
        boxes[:, 1::2] = (boxes[:, 1::2] - top) / scale
        if not np.isfinite(boxes).all():
            raise ModelBundleError("ONNX 偵測框的座標超出範圍")
        detections = _collect(self.classes, scores[selected], labels[selected], np.rint(boxes), None, width, height)
        self.last_metrics = {
            "model_preprocess_ms": (prepared - started) * 1000,
            "model_inference_ms": (inferred - prepared) * 1000,
            "result_convert_ms": (time.perf_counter() - inferred) * 1000,
        }
        return detections
