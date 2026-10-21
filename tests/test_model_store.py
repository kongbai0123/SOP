"""Model-library imports use temporary synthetic bundles, without AI runtimes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from sop_app import paths
from sop_app.model_bundle import ModelBundleError, read_model_info
from sop_app.model_store import import_model

CHECKPOINT = b"synthetic checkpoint"


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / "models"

    def folder(self, name="source", *, weights=CHECKPOINT, project="grips", sha=None):
        source = self.root / name
        model = source / "export" / "model"
        model.mkdir(parents=True)
        (model / "model.json").write_text(json.dumps({
            "engine": "yolo26n_detect", "classes": ["grip"],
            "model_version_id": "M001", "checkpoint": "weights.pt",
        }), encoding="utf-8")
        (model / "weights.pt").write_bytes(weights)
        (model.parent / "export-manifest.json").write_text(json.dumps({
            "project_name": project,
            "artifacts": [{"path": "model/weights.pt", "sha256":
                           sha if sha is not None else hashlib.sha256(weights).hexdigest()}],
        }), encoding="utf-8")
        (source / "training-data.txt").write_text("must stay outside library", encoding="utf-8")
        return source

    def bundle(self, name="model.zip", **kwargs):
        folder = self.folder("input-" + Path(name).stem, **kwargs)
        source = self.root / name
        with zipfile.ZipFile(source, "w") as archive:
            for path in (folder / "export").rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(folder).as_posix())
        return source

    def test_zip_is_copied_intact_without_writing_beside_original(self):
        source = self.bundle()
        original = source.read_bytes()
        imported = import_model(source, self.models)
        self.assertEqual(imported.read_bytes(), original)
        self.assertEqual(imported.name, source.name)
        self.assertEqual(imported.parent.name, "grips-" + hashlib.sha256(original).hexdigest())
        self.assertFalse((source.parent / ".extracted").exists())
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(read_model_info(imported).model_version_id, "M001")

    def test_identical_zip_with_different_filename_reuses_first_import(self):
        source = self.bundle()
        imported = import_model(source, self.models)
        renamed = self.root / "renamed.zip"
        shutil.copyfile(source, renamed)
        self.assertEqual(import_model(renamed, self.models), imported)
        self.assertEqual(len(list(self.models.iterdir())), 1)

    def test_different_content_never_overwrites_same_named_bundle(self):
        source = self.bundle()
        first = import_model(source, self.models)
        first_bytes = first.read_bytes()
        replacement = self.bundle("new.zip", weights=b"new checkpoint")
        shutil.copyfile(replacement, source)
        second = import_model(source, self.models)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), first_bytes)
        self.assertEqual(second.read_bytes(), source.read_bytes())

    def test_directory_copies_only_model_files_and_retains_manifest_checksum(self):
        source = self.folder()
        imported = import_model(source, self.models)
        self.assertEqual({path.name for path in imported.iterdir()},
                         {"model.json", "weights.pt", "export-manifest.json"})
        info = read_model_info(imported)
        self.assertEqual(info.checkpoint_path, imported / "weights.pt")
        self.assertEqual(info.checkpoint_sha256, hashlib.sha256(CHECKPOINT).hexdigest())
        self.assertEqual(info.project_name, "grips")
        self.assertTrue((source / "training-data.txt").exists())
        self.assertEqual(import_model(source, self.models), imported)

    def test_model_json_selection_and_equivalent_folder_share_import(self):
        source = self.folder()
        model_json = source / "export" / "model" / "model.json"
        imported = import_model(model_json, self.models)
        self.assertEqual(import_model(source, self.models), imported)
        self.assertTrue(model_json.exists())

    def test_directory_without_manifest_uses_version_name(self):
        source = self.folder()
        (source / "export" / "export-manifest.json").unlink()
        imported = import_model(source, self.models)
        self.assertTrue(imported.name.startswith("M001-"))
        self.assertEqual({path.name for path in imported.iterdir()}, {"model.json", "weights.pt"})
        self.assertIsNone(read_model_info(imported).checkpoint_sha256)

    def test_existing_library_zip_and_model_json_are_reused(self):
        imported_zip = import_model(self.bundle(), self.models)
        self.assertEqual(import_model(imported_zip, self.models), imported_zip)
        imported_folder = import_model(self.folder("directory"), self.models)
        self.assertEqual(import_model(imported_folder / "model.json", self.models), imported_folder)
        self.assertEqual(len(list(self.models.iterdir())), 2)

    def test_windows_reserved_and_invalid_project_names_are_safe(self):
        for index, (project, expected) in enumerate([
                ("CON", "_CON"), ("LPT1.data", "_LPT1.data"),
                (' test:<name>?* . ', "test__name___"), ("...", "model")]):
            with self.subTest(project=project):
                imported = import_model(self.folder(f"source-{index}", project=project), self.models)
                self.assertTrue(imported.name.startswith(expected + "-"), imported.name)
                self.assertEqual(imported.parent, self.models)

    def test_checksum_failure_leaves_no_partial_import_and_preserves_source(self):
        for index, source in enumerate((self.folder("bad-folder", sha="0" * 64),
                                       self.bundle("bad.zip", sha="0" * 64))):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ModelBundleError, "SHA-256"):
                    import_model(source, self.models)
                self.assertTrue(source.exists())
                self.assertEqual(list(self.models.iterdir()), [])

    def test_invalid_bundle_does_not_create_library(self):
        invalid = self.root / "invalid.zip"
        invalid.write_bytes(b"invalid")
        with self.assertRaises(ModelBundleError):
            import_model(invalid, self.models)
        self.assertFalse(self.models.exists())

    def test_corrupt_compressed_weights_report_import_error_and_clean_stage(self):
        source = self.root / "damaged-weights.zip"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("model/model.json", json.dumps({
                "engine": "yolo26n_detect", "classes": ["part"]}))
            archive.writestr("model/checkpoint.pt", CHECKPOINT)
            entry = archive.getinfo("model/checkpoint.pt")
        raw = bytearray(source.read_bytes())
        # Force the first DEFLATE block to use the reserved BTYPE=3 value.
        raw[entry.header_offset + 30 + len(entry.filename.encode("utf-8"))] = 0x07
        source.write_bytes(raw)
        self.assertEqual(read_model_info(source).classes, ("part",))
        with self.assertRaisesRegex(ModelBundleError, "無法匯入模型"):
            import_model(source, self.models)
        self.assertEqual(list(self.models.iterdir()), [])

    def test_copy_failure_cleans_stage_and_preserves_original(self):
        source = self.bundle()
        original = source.read_bytes()
        with patch("sop_app.model_store.shutil.copyfile", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(ModelBundleError, "disk full"):
                import_model(source, self.models)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(list(self.models.iterdir()), [])

    def test_existing_damaged_target_is_reported_without_overwriting(self):
        source = self.bundle()
        imported = import_model(source, self.models)
        imported.write_bytes(b"damaged")
        with self.assertRaisesRegex(ModelBundleError, "內容不符"):
            import_model(source, self.models)
        self.assertEqual(imported.read_bytes(), b"damaged")
        self.assertFalse(any(path.name.startswith(".import-") for path in self.models.iterdir()))

    def test_concurrent_completed_import_is_reused(self):
        source = self.bundle()
        def concurrent_publish(stage, target):
            shutil.copytree(stage, target)
            raise FileExistsError("another process completed import")
        with patch.object(Path, "rename", concurrent_publish):
            imported = import_model(source, self.models)
        self.assertEqual(imported.read_bytes(), source.read_bytes())
        self.assertEqual(len(list(self.models.iterdir())), 1)

    def test_default_uses_shared_paths_models_dir(self):
        source = self.bundle()
        with patch.object(paths, "MODELS_DIR", self.models, create=True):
            imported = import_model(source)
        self.assertEqual(imported.parent.parent, self.models)


if __name__ == "__main__":
    unittest.main()
