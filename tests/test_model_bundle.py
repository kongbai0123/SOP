"""模型包讀取測試（不需要載入 torch 或實際模型）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sop_app.model_bundle import ModelBundleError, ensure_checkpoint, read_model_info  # noqa: E402

CHECKPOINT = b"not a real checkpoint"


def record(engine="yolo26n_detect", **extra):
    return {"engine": engine, "classes": ["grasp", "hand"], "image_size": 800,
            "score_threshold": 0.5, "model_version_id": "M004", **extra}


def write_folder(root: Path, data: dict, checkpoint_name="checkpoint.pt"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.json").write_text(json.dumps(data), encoding="utf-8")
    (root / checkpoint_name).write_bytes(CHECKPOINT)
    return root


def write_zip(path: Path, data: dict, prefix="", sha: str | None = None):
    prefix = prefix.rstrip("/")
    base = f"{prefix}/" if prefix else ""
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(base + "model/model.json", json.dumps(data))
        bundle.writestr(base + "model/checkpoint.pt", CHECKPOINT)
        bundle.writestr(base + "export-manifest.json", json.dumps(
            {"project_name": "grasp",
             "artifacts": [{"path": "model/checkpoint.pt",
                            "sha256": sha if sha is not None else hashlib.sha256(CHECKPOINT).hexdigest()}]}))
    return path


class ModelBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_detection_engine_reports_no_masks(self):
        info = read_model_info(write_folder(self.root / "yolo", record()))
        self.assertEqual((info.engine, info.runtime, info.has_masks), ("yolo26n_detect", "ultralytics", False))
        self.assertEqual(info.classes, ("grasp", "hand"))
        self.assertEqual(info.image_size, 800)

    def test_engine_name_and_masks_per_engine(self):
        for engine, runtime, masks in [("maskrcnn_resnet50_fpn", "torchvision", True),
                                       ("fasterrcnn_resnet50_fpn_v2", "torchvision", False),
                                       ("yolo26s_seg", "ultralytics", True),
                                       ("rt_detr_r50", "ultralytics", False)]:
            info = read_model_info(write_folder(self.root / engine, record(engine)))
            self.assertEqual((info.runtime, info.has_masks), (runtime, masks), engine)
            self.assertTrue(info.engine_name)

    def test_model_json_path_is_accepted(self):
        folder = write_folder(self.root / "yolo", record())
        self.assertEqual(read_model_info(folder / "model.json").engine, "yolo26n_detect")

    def test_zip_bundle_with_outer_folder_is_accepted(self):
        path = write_zip(self.root / "nested.zip", record("maskrcnn_resnet50_fpn"),
                         prefix="project-M004-model")
        info = read_model_info(path)
        self.assertEqual(info.project_name, "grasp")
        self.assertTrue(info.checkpoint_member.endswith("/model/checkpoint.pt"))
        extracted = ensure_checkpoint(info)
        self.assertEqual(extracted.read_bytes(), CHECKPOINT)

    def test_unknown_engine_lists_supported_ones(self):
        folder = write_folder(self.root / "other", record("some_future_net"))
        with self.assertRaises(ModelBundleError) as caught:
            read_model_info(folder)
        self.assertIn("some_future_net", str(caught.exception))
        self.assertIn("maskrcnn_resnet50_fpn", str(caught.exception))

    def test_known_but_unusable_engine_explains_why(self):
        folder = write_folder(self.root / "seg", record("deeplabv3_resnet50"))
        with self.assertRaisesRegex(ModelBundleError, "語意分割"):
            read_model_info(folder)

    def test_missing_checkpoint_is_reported(self):
        folder = self.root / "broken"
        folder.mkdir()
        (folder / "model.json").write_text(json.dumps(record()), encoding="utf-8")
        with self.assertRaisesRegex(ModelBundleError, "checkpoint.pt"):
            read_model_info(folder)

    def test_checkpoint_name_cannot_escape_the_bundle(self):
        folder = write_folder(self.root / "escape", record(checkpoint="../../evil.pt"))
        with self.assertRaisesRegex(ModelBundleError, "checkpoint"):
            read_model_info(folder)

    def test_zip_bundle_extracts_and_verifies_checkpoint(self):
        path = write_zip(self.root / "bundle.zip", record("maskrcnn_resnet50_fpn"))
        info = read_model_info(path)
        self.assertEqual(info.project_name, "grasp")
        extracted = ensure_checkpoint(info)
        self.assertEqual(extracted.read_bytes(), CHECKPOINT)
        self.assertEqual(ensure_checkpoint(info), extracted)

    def test_zip_bundle_rejects_corrupted_checkpoint(self):
        path = write_zip(self.root / "bad.zip", record(), sha="0" * 64)
        with self.assertRaisesRegex(ModelBundleError, "SHA-256"):
            ensure_checkpoint(read_model_info(path))


if __name__ == "__main__":
    unittest.main()
