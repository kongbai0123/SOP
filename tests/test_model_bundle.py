"""模型包讀取測試（不需要載入 torch 或實際模型）。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
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


def write_zip(path: Path, data: dict, prefix="", sha: str | None = None, checkpoint: bytes = CHECKPOINT):
    prefix = prefix.rstrip("/")
    base = f"{prefix}/" if prefix else ""
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(base + "model/model.json", json.dumps(data))
        bundle.writestr(base + "model/checkpoint.pt", checkpoint)
        bundle.writestr(base + "export-manifest.json", json.dumps(
            {"project_name": "grasp",
             "artifacts": [{"path": "model/checkpoint.pt",
                            "sha256": sha if sha is not None else hashlib.sha256(checkpoint).hexdigest()}]}))
    return path


class ModelBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.models_dir = self.root / "models"
        self.models_patch = patch("sop_app.model_bundle.MODELS_DIR", self.models_dir)
        self.models_patch.start()
        self.addCleanup(self.models_patch.stop)

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
        self.assertEqual(extracted.parent.parent, self.models_dir / ".cache")
        self.assertFalse((path.parent / ".extracted").exists())
        self.assertEqual(ensure_checkpoint(info), extracted)

    def test_same_named_zip_bundles_have_separate_caches(self):
        first_dir = self.root / "first"
        second_dir = self.root / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        first = write_zip(first_dir / "bundle.zip", record())
        second = write_zip(second_dir / "bundle.zip", record(), checkpoint=b"another checkpoint")
        first_checkpoint = ensure_checkpoint(read_model_info(first))
        second_checkpoint = ensure_checkpoint(read_model_info(second))
        self.assertNotEqual(first_checkpoint, second_checkpoint)
        self.assertEqual(first_checkpoint.read_bytes(), CHECKPOINT)
        self.assertEqual(second_checkpoint.read_bytes(), b"another checkpoint")

    def test_replacing_zip_at_same_path_refreshes_cache(self):
        path = write_zip(self.root / "bundle.zip", record())
        before = path.stat()
        first_checkpoint = ensure_checkpoint(read_model_info(path))
        replacement = b"x" * len(CHECKPOINT)
        write_zip(path, record(), checkpoint=replacement)
        # The replacement deliberately has the same size; mtime must distinguish
        # it even on filesystems where a fast rewrite gets the same timestamp.
        self.assertEqual(path.stat().st_size, before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        second_checkpoint = ensure_checkpoint(read_model_info(path))
        self.assertNotEqual(first_checkpoint, second_checkpoint)
        self.assertEqual(second_checkpoint.read_bytes(), replacement)
        self.assertEqual(first_checkpoint.read_bytes(), CHECKPOINT)

    def test_existing_cache_is_verified_and_repaired(self):
        info = read_model_info(write_zip(self.root / "bundle.zip", record()))
        extracted = ensure_checkpoint(info)
        extracted.write_bytes(b"x" * len(CHECKPOINT))
        self.assertEqual(ensure_checkpoint(info), extracted)
        self.assertEqual(extracted.read_bytes(), CHECKPOINT)

    def test_zip_bundle_rejects_corrupted_checkpoint(self):
        path = write_zip(self.root / "bad.zip", record(), sha="0" * 64)
        with self.assertRaisesRegex(ModelBundleError, "SHA-256"):
            ensure_checkpoint(read_model_info(path))
        self.assertEqual([path for path in self.models_dir.rglob("*") if path.is_file()], [])

    def test_zip_read_failure_removes_partial_checkpoint(self):
        info = read_model_info(write_zip(self.root / "bundle.zip", record()))
        with patch("sop_app.model_bundle.zipfile.ZipFile.open", side_effect=OSError("read failed")):
            with self.assertRaisesRegex(ModelBundleError, "無法讀取模型 checkpoint"):
                ensure_checkpoint(info)
        self.assertEqual([path for path in self.models_dir.rglob("*") if path.is_file()], [])

    def test_corrupt_compressed_metadata_and_weights_raise_bundle_errors(self):
        for corrupt_member in ("model/model.json", "model/checkpoint.pt"):
            with self.subTest(member=corrupt_member):
                path = self.root / "compressed.zip"
                with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("model/model.json", json.dumps(record()))
                    archive.writestr("model/checkpoint.pt", CHECKPOINT)
                    entry = archive.getinfo(corrupt_member)
                raw = bytearray(path.read_bytes())
                raw[entry.header_offset + 30 + len(entry.filename.encode("utf-8"))] = 0x07
                path.write_bytes(raw)
                with self.assertRaises(ModelBundleError):
                    ensure_checkpoint(read_model_info(path))
                self.assertEqual([file for file in self.models_dir.rglob("*") if file.is_file()], [])

    def test_locked_concurrently_published_checkpoint_is_verified_and_reused(self):
        info = read_model_info(write_zip(self.root / "bundle.zip", record()))

        def concurrent_publish(_partial, target):
            target.write_bytes(CHECKPOINT)
            raise PermissionError("checkpoint opened by another loader")

        with patch.object(Path, "replace", autospec=True, side_effect=concurrent_publish):
            result = ensure_checkpoint(info)
        self.assertEqual(result.read_bytes(), CHECKPOINT)
        self.assertEqual([path for path in self.models_dir.rglob("*") if path.is_file()], [result])

    def test_concurrent_extractions_use_unique_partial_files(self):
        info = read_model_info(write_zip(self.root / "bundle.zip", record()))
        create_temporary_file = tempfile.NamedTemporaryFile
        barrier = threading.Barrier(2)
        partials = []

        def synchronized_temporary_file(*args, **kwargs):
            stream = create_temporary_file(*args, **kwargs)
            partials.append(Path(stream.name))
            try:
                barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                stream.close()
                raise
            return stream

        with patch("sop_app.model_bundle.tempfile.NamedTemporaryFile", side_effect=synchronized_temporary_file):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(ensure_checkpoint, [info, info]))
        self.assertEqual(len(set(partials)), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0].read_bytes(), CHECKPOINT)
        self.assertTrue(all(not path.exists() for path in partials))


if __name__ == "__main__":
    unittest.main()
