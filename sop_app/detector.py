"""TorchVision Mask R-CNN 推論。

模型建構方式必須與 graph_catch/workbench/maskrcnn_engine.py 訓練時一致：
maskrcnn_resnet50_fpn_v2（若可用）、min_size = max_size = image_size、類別數 + 1（背景）。
"""
from __future__ import annotations

import cv2
import numpy as np

from .detection import Detection
from .model_bundle import ModelBundleError, ModelInfo, ensure_checkpoint


class MaskRCNNDetector:
    def __init__(self, info: ModelInfo, device: str = "auto", score_floor: float = 0.3):
        import torch
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        self.info = info
        self.classes = info.classes
        # 偵測器保留較低分數的結果，實際門檻由每個條件的「信心度≥」決定
        self.score_floor = score_floor
        self._torch = torch
        use_cuda = device in ("auto", "cuda") and torch.cuda.is_available()
        if device == "cuda" and not use_cuda:
            raise ModelBundleError("指定使用 CUDA，但目前環境無法使用 GPU")
        self.device = torch.device("cuda" if use_cuda else "cpu")

        checkpoint = torch.load(ensure_checkpoint(info), map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        image_size = int(checkpoint.get("image_size", info.image_size))

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
                break
            except RuntimeError as exc:
                last_error = exc
        else:
            raise ModelBundleError(f"checkpoint 與模型結構不符：{last_error}")

        self.model = model.to(self.device).eval()
        self.detect(np.zeros((image_size, image_size, 3), dtype=np.uint8))  # 預熱，避免第一幀卡頓

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
            labels = labels[keep].cpu().tolist()
            boxes = output["boxes"][keep].round().int().cpu().numpy()
            masks = (output["masks"][keep, 0] >= 0.5).cpu().numpy()

        detections = []
        for score, label, box, mask in zip(scores, labels, boxes, masks):
            x1, y1, x2, y2 = (int(np.clip(box[0], 0, width)), int(np.clip(box[1], 0, height)),
                              int(np.clip(box[2], 0, width)), int(np.clip(box[3], 0, height)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(self.classes[label - 1], float(score), (x1, y1, x2, y2), mask))
        return detections
