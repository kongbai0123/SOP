"""模型推論：把 TorchVision、YOLO 與 RT-DETR 輸出統一成 Detection。

每個引擎都遵守同一個介面，SOP 與畫面層不需要知道模型是由哪個
訓練框架產生。模型結構仍必須和 checkpoint 的 ``engine`` 相符。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .detection import Detection
from .model_bundle import ModelBundleError, ModelInfo, ensure_checkpoint, ensure_runtime


def create_detector(info: ModelInfo, device: str = "auto", score_floor: float = 0.3, progress=None):
    """依模型包的引擎建立偵測器；所有偵測器都回傳 Detection。"""
    ensure_runtime(info)
    if info.runtime == "ultralytics":
        return UltralyticsDetector(info, device, score_floor, progress)
    if info.engine == "maskrcnn_resnet50_fpn":
        return MaskRCNNDetector(info, device, score_floor, progress)
    return FasterRCNNDetector(info, device, score_floor, progress)


def _select_device(torch, device: str):
    use_cuda = device in ("auto", "cuda") and torch.cuda.is_available()
    if device == "cuda" and not use_cuda:
        raise ModelBundleError("指定使用 CUDA，但目前環境無法使用 GPU")
    return torch.device("cuda" if use_cuda else "cpu")


def _collect(classes, scores, labels, boxes, masks, width: int, height: int) -> list[Detection]:
    """將 0 起算的類別索引和框轉成統一 Detection。"""
    detections = []
    for index, (score, label, box) in enumerate(zip(scores, labels, boxes)):
        label = int(label)
        if not 0 <= label < len(classes):
            continue
        x1, y1, x2, y2 = (int(np.clip(box[0], 0, width)), int(np.clip(box[1], 0, height)),
                          int(np.clip(box[2], 0, width)), int(np.clip(box[3], 0, height)))
        if x2 <= x1 or y2 <= y1:
            continue
        mask = masks[index] if masks is not None else None
        detections.append(Detection(classes[label], float(score), (x1, y1, x2, y2), mask))
    return detections


class _TorchVisionDetector:
    """TorchVision 偵測模型的共用載入、裝置與輸出流程。"""

    def __init__(self, info: ModelInfo, device: str = "auto", score_floor: float = 0.3, progress=None):
        report = progress or (lambda message: None)
        report("載入推論套件")
        import torch

        self.info = info
        self.classes = info.classes
        self.score_floor = score_floor
        self._torch = torch
        report("選擇運算裝置")
        self.device = _select_device(torch, device)

        report("讀取與驗證模型權重")
        checkpoint = torch.load(ensure_checkpoint(info), map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        image_size = int(checkpoint.get("image_size", info.image_size))
        report("建立模型並載入運算裝置")
        self.model = self._build(state, image_size).to(self.device).eval()
        report("預熱模型，準備首次辨識")
        self.detect(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    def _build(self, state, image_size):
        raise NotImplementedError

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return f"GPU（{self._torch.cuda.get_device_name(self.device)}）"
        return "CPU"

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        torch = self._torch
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float().div_(255.0)
        with torch.inference_mode():
            output = self.model([tensor])[0]
            labels = output["labels"]
            keep = (output["scores"] >= self.score_floor) & (labels >= 1) & (labels <= len(self.classes))
            scores = output["scores"][keep].cpu().tolist()
            labels = [label - 1 for label in labels[keep].cpu().tolist()]
            boxes = output["boxes"][keep].round().int().cpu().numpy()
            masks = (output["masks"][keep, 0] >= 0.5).cpu().numpy() if "masks" in output else None
        return _collect(self.classes, scores, labels, boxes, masks, width, height)


class MaskRCNNDetector(_TorchVisionDetector):
    def _build(self, state, image_size: int):
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        builders = [getattr(torchvision.models.detection, name) for name in
                    ("maskrcnn_resnet50_fpn_v2", "maskrcnn_resnet50_fpn")
                    if hasattr(torchvision.models.detection, name)]
        last_error: Exception | None = None
        for build in builders:
            model = build(weights=None, weights_backbone=None, min_size=image_size, max_size=image_size)
            box_features = model.roi_heads.box_predictor.cls_score.in_features
            mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.box_predictor = FastRCNNPredictor(box_features, len(self.classes) + 1)
            model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, len(self.classes) + 1)
            try:
                model.load_state_dict(state)
                return model
            except RuntimeError as exc:
                last_error = exc
        raise ModelBundleError(f"checkpoint 與模型結構不符：{last_error}")


class FasterRCNNDetector(_TorchVisionDetector):
    def _build(self, state, image_size: int):
        import torchvision

        builder = getattr(torchvision.models.detection, self.info.engine, None)
        if builder is None:
            raise ModelBundleError(f"目前的 TorchVision 版本沒有 {self.info.engine}")
        model = builder(weights=None, weights_backbone=None, num_classes=len(self.classes) + 1,
                        min_size=image_size, max_size=image_size)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise ModelBundleError(f"checkpoint 與模型結構不符：{exc}") from exc
        return model


class UltralyticsDetector:
    """YOLO Detect／Seg 與 RT-DETR；checkpoint.pt 是 Ultralytics 完整模型。"""

    def __init__(self, info: ModelInfo, device: str = "auto", score_floor: float = 0.3, progress=None):
        report = progress or (lambda message: None)
        report("載入推論套件")
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("MPLBACKEND", "Agg")
        import torch
        from ultralytics import RTDETR, YOLO

        self.info = info
        self.classes = info.classes
        self.score_floor = score_floor
        self._torch = torch
        report("選擇運算裝置")
        self.device = _select_device(torch, device)
        self.image_size = int(info.image_size)

        constructor = RTDETR if info.engine.startswith("rt_detr_") else YOLO
        report("讀取與驗證模型權重")
        checkpoint = ensure_checkpoint(info)
        report("建立辨識模型")
        self.model = constructor(str(checkpoint))
        names = getattr(self.model, "names", None) or {}
        ordered = tuple(str(names[key]) for key in sorted(names))
        if ordered and ordered != tuple(self.classes):
            raise ModelBundleError(f"checkpoint 的類別與 model.json 不符：\n"
                                   f"  checkpoint：{'、'.join(ordered)}\n"
                                   f"  model.json：{'、'.join(self.classes)}")
        report("預熱模型，準備首次辨識")
        self.detect(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return f"GPU（{self._torch.cuda.get_device_name(self.device)}）"
        return "CPU"

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        height, width = bgr.shape[:2]
        result = self.model.predict(source=bgr, imgsz=self.image_size, conf=self.score_floor,
                                    retina_masks=self.info.has_masks,
                                    device="cuda" if self.device.type == "cuda" else "cpu",
                                    verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        scores = boxes.conf.cpu().tolist()
        labels = boxes.cls.cpu().tolist()
        xyxy = boxes.xyxy.round().int().cpu().numpy()
        keep = [index for index, label in enumerate(labels) if 0 <= int(label) < len(self.classes)]

        masks = None
        if self.info.has_masks and getattr(result, "masks", None) is not None:
            raw = result.masks.data.cpu().numpy() >= 0.5
            if raw.shape[1:] != (height, width):
                raw = np.stack([cv2.resize(m.astype(np.uint8), (width, height),
                                           interpolation=cv2.INTER_NEAREST).astype(bool) for m in raw])
            masks = raw

        return _collect(self.classes, [scores[i] for i in keep], [labels[i] for i in keep],
                        xyxy[keep], masks[keep] if masks is not None else None, width, height)
