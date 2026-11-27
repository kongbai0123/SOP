"""ONNX model validation and detector conversion without requiring native ONNX Runtime."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from sop_app.detector import create_detector
from sop_app.model_bundle import ModelBundleError, ensure_checkpoint, read_model_info


WEIGHTS = b"test model"
DIGEST = hashlib.sha256(WEIGHTS).hexdigest()
MODEL = {"engine": "yolo26n_detect_onnx", "classes": ["part", "hand"], "image_size": 64,
         "checkpoint": "weights.onnx", "model_version_id": "test"}
ARTIFACT = {"path": "model/weights.onnx", "sha256": DIGEST}


class OnnxBackendTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        cache = patch("sop_app.model_bundle.MODELS_DIR", self.root / "cache")
        cache.start()
        self.addCleanup(cache.stop)

    def make_bundle(self, artifacts=None, *, zipped=False, **overrides):
        parts = {
            "model/model.json": json.dumps({**MODEL, **overrides}).encode("utf-8"),
            "model/weights.onnx": WEIGHTS,
            "export-manifest.json": json.dumps({"artifacts": [ARTIFACT] if artifacts is None else artifacts}).encode("utf-8"),
        }
        if zipped:
            archive = self.root / "example.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for name, data in parts.items():
                    bundle.writestr("outer/" + name, data)
            return archive
        for name, data in parts.items():
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
        return self.root

    def fake_runtime(self, *, metadata=None, source_shape=None, result_shape=None):
        runtime = ModuleType("onnxruntime")
        runtime.__spec__ = importlib.util.spec_from_loader(runtime.__name__, loader=None)
        runtime.__version__ = "test"
        runtime.SessionOptions = SimpleNamespace
        session = Mock()
        session.get_inputs.return_value = [
            SimpleNamespace(name="images", type="tensor(float)", shape=[1, 3, 32, 64] if source_shape is None else source_shape)
        ]
        session.get_outputs.return_value = [
            SimpleNamespace(name="output", type="tensor(float)", shape=[1, 6, 42] if result_shape is None else result_shape)
        ]
        session.get_modelmeta.return_value = SimpleNamespace(
            custom_metadata_map=metadata if metadata is not None
            else {"names": "{0: 'part', 1: 'hand'}", "task": "detect", "end2end": "False"})
        session.run.return_value = [np.zeros((1, 6, 42), dtype=np.float32)]
        runtime.InferenceSession = Mock(return_value=session)
        sys_modules = patch.dict(sys.modules, {"onnxruntime": runtime})
        sys_modules.start()
        self.addCleanup(sys_modules.stop)
        return runtime, session

    def test_nested_zip_honors_exact_artifact_path_and_accepts_uppercase_sha(self):
        archive = self.make_bundle([{**ARTIFACT, "sha256": DIGEST.upper()}], zipped=True)
        info = read_model_info(archive)
        self.assertEqual((info.runtime, info.has_masks, info.checkpoint_sha256),
                         ("onnxruntime", False, DIGEST))
        self.assertEqual(ensure_checkpoint(info).read_bytes(), WEIGHTS)

    def test_missing_malformed_or_ambiguous_sha_is_rejected(self):
        for entries in ([], [{**ARTIFACT, "path": "elsewhere/weights.onnx"}],
                        [{**ARTIFACT, "sha256": "a" * 63}], [ARTIFACT, ARTIFACT]):
            with self.subTest(entries=entries), self.assertRaisesRegex(ModelBundleError, "SHA-256"):
                read_model_info(self.make_bundle(entries, zipped=True))

    def test_model_bundle_requires_unique_strings_for_classes(self):
        for classes in (["part", "part"], ["part", True], "part", ["part", " "]):
            with self.subTest(classes=classes), self.assertRaisesRegex(ModelBundleError, "類別"):
                read_model_info(self.make_bundle(classes=classes))

    def test_directory_checkpoint_bytes_are_verified(self):
        info = read_model_info(self.make_bundle())
        info.checkpoint_path.write_bytes(b"modified bytes")
        with self.assertRaisesRegex(ModelBundleError, "SHA-256"):
            ensure_checkpoint(info)

    def test_runtime_receives_verified_bytes_and_only_requests_cpu(self):
        runtime, _session = self.fake_runtime()
        info = read_model_info(self.make_bundle())
        detector = create_detector(info)
        self.assertEqual(runtime.InferenceSession.call_args.args, (WEIGHTS,))
        self.assertEqual(runtime.InferenceSession.call_args.kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(detector.runtime_info["input_shape"], [1, 3, 32, 64])
        with self.assertRaisesRegex(ModelBundleError, "CPU"):
            create_detector(info, device="cuda")

    def test_model_graph_must_match_input_output_and_ordered_class_metadata(self):
        info = read_model_info(self.make_bundle())
        for overrides, message in [
            ({"source_shape": [1, 3, "height", "width"]}, "固定大小"),
            ({"result_shape": [1, 300, 6]}, "原始偵測格式"),
            ({"metadata": {"names": "{0: 'hand', 1: 'part'}", "task": "detect", "end2end": "False"}}, "類別順序"),
            ({"metadata": {"names": "{False: 'part', 1: 'hand'}", "task": "detect", "end2end": "False"}}, "類別索引"),
        ]:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ModelBundleError, message):
                self.fake_runtime(**overrides)
                create_detector(info)

    def test_letterbox_and_same_class_nms_keep_distinct_class(self):
        _runtime, session = self.fake_runtime()
        detector = create_detector(read_model_info(self.make_bundle()))
        raw = np.zeros((1, 6, 42), dtype=np.float32)
        raw[0, :4, :3] = np.array([32, 16, 20, 8], dtype=np.float32)[:, None]
        raw[0, 4:, :3] = [[.95, .8, .01], [.01, .01, .9]]
        session.run.return_value = [raw]
        bgr = np.broadcast_to(np.array([0, 32, 255], np.uint8), (16, 16, 3))
        results = detector.detect(bgr)
        self.assertEqual([(item.label, item.box) for item in results],
                         [("part", (3, 6, 13, 10)), ("hand", (3, 6, 13, 10))])
        image_tensor = session.run.call_args.args[1]["images"]
        np.testing.assert_allclose(image_tensor[0, :, 0, 16], [1., 32. / 255, 0.])
        np.testing.assert_allclose(image_tensor[0, :, 0, 15], [114. / 255] * 3)

    def test_runtime_rejects_malformed_result_instead_of_confirming_absence(self):
        _runtime, session = self.fake_runtime()
        detector = create_detector(read_model_info(self.make_bundle()))
        bgr = np.zeros((32, 64, 3), dtype=np.uint8)
        for array in (np.zeros((1, 6, 41), dtype=np.float32), np.zeros((1, 6, 42), dtype=np.float64),
                      np.full((1, 6, 42), np.nan, dtype=np.float32)):
            with self.subTest(shape=array.shape, dtype=array.dtype), self.assertRaises(ModelBundleError):
                session.run.return_value = [array]
                detector.detect(bgr)


if __name__ == "__main__":
    unittest.main()
