"""Shared model imports and saved references, without inference or production data."""
from __future__ import annotations

import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from PySide6.QtWidgets import QApplication

from sop_app.paths import models_directory, resolve_model
from sop_app.sop_schema import SOPDefinition, Step, save_sop, load_sop
from sop_app.startup import initial_model_path
from sop_app.stations import Station
from sop_app.ui.editor_page import EditorPage
from sop_app.ui.main_window import MainWindow


def write_bundle(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("model/model.json", json.dumps({
            "engine": "yolo26n_detect", "classes": ["part"], "model_version_id": "M001"}))
        bundle.writestr("model/checkpoint.pt", b"test weights")
    return path


class SharedModelPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.first = Station("ST001", "A", self.root / "data/stations/ST001")
        self.second = Station("ST002", "B", self.root / "data/stations/ST002")

    def test_stations_share_portable_model_references(self):
        model = write_bundle(self.root / "models/assembly/model.zip")
        reference = self.first.relativize(model)
        self.assertEqual(reference, "models/assembly/model.zip")
        self.assertEqual(self.first.models, self.root / "models")
        self.assertEqual(self.first.models, self.second.models)
        self.assertEqual(self.first.resolve(reference), model)
        self.assertEqual(self.second.resolve(reference), model)

    def test_relocated_data_uses_new_shared_model_location(self):
        moved = self.root / "moved"
        model = write_bundle(moved / "models/assembly/model.zip")
        station = Station("ST001", "A", moved / "data/stations/ST001")
        self.assertEqual(station.resolve("models/assembly/model.zip"), model)

    def test_legacy_absolute_and_station_local_models_keep_resolving(self):
        external = write_bundle(self.root / "old-model/model.zip")
        local = write_bundle(self.first.root / "models/old.zip")
        write_bundle(self.root / "models/old.zip")
        self.assertEqual(self.first.resolve(external), external)
        self.assertEqual(self.first.resolve("models/old.zip"), local)
        self.assertEqual(self.first.relativize(local), "models/old.zip")
        legacy = Station("ST003", "C", self.root / "data/stations/ST003", self.root)
        self.assertEqual(legacy.resolve("old-model/model.zip"), external)

    def test_preload_uses_same_shared_model_as_station(self):
        model = write_bundle(self.root / "models/assembly/model.zip")
        save_sop(SOPDefinition(model_path="models/assembly/model.zip"), self.first.sops / "a.json")
        with patch("pathlib.Path.cwd", return_value=self.root / "unrelated"):
            self.assertEqual(initial_model_path(root=self.first.root), model)
        self.assertEqual(initial_model_path(root=self.first.root), self.first.resolve("models/assembly/model.zip"))

    def test_plain_data_root_has_its_own_models_directory(self):
        self.assertEqual(models_directory(self.root), self.root / "models")
        self.assertEqual(resolve_model("model/legacy.zip", self.root), self.root / "model/legacy.zip")

    def test_external_sop_import_keeps_other_stations_shared_model_reference(self):
        model = write_bundle(self.root / "models/assembly/model.zip")
        source = self.first.sops / "a.json"
        save_sop(SOPDefinition(model_path="models/assembly/model.zip"), source)
        self.second.sops.mkdir(parents=True)
        window = SimpleNamespace(current_station=self.second, _sops_dir=self.second.sops, _apply_sop=Mock())
        MainWindow._open_path(window, source)
        imported, destination = window._apply_sop.call_args.args
        self.assertEqual(imported.model_path, "models/assembly/model.zip")
        self.assertEqual(self.second.resolve(imported.model_path), model)
        self.assertEqual(destination, self.second.sops / "a.json")
        self.assertEqual(load_sop(source).model_path, "models/assembly/model.zip")


class ModelImportEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.station = Station("ST001", "A", self.root / "data/stations/ST001")
        self.editor = EditorPage()
        self.addCleanup(self.editor.deleteLater)
        self.addCleanup(self.editor.close)
        self.editor._models_dir = self.station.models
        self.editor._resolve_path = self.station.resolve
        self.editor._relativize_path = self.station.relativize
        self.editor.set_sop(SOPDefinition(steps=[Step("A")]))
        self.selected, self.modified = [], []
        self.editor.model_selected.connect(self.selected.append)
        self.editor.modified.connect(lambda: self.modified.append(True))

    def test_selection_imports_copy_and_saves_shared_reference(self):
        source = write_bundle(self.root / "downloads/組裝模型.zip")
        with patch("sop_app.ui.editor_page.QFileDialog.getOpenFileName", return_value=(str(source), "")) as dialog:
            self.editor._choose_model()
        imported = self.station.resolve(self.editor.sop.model_path)
        self.assertTrue(imported.is_relative_to(self.station.models))
        self.assertEqual(imported.read_bytes(), source.read_bytes())
        self.assertTrue(self.editor.sop.model_path.startswith("models/"))
        self.assertEqual(self.selected, [str(imported)])
        self.assertEqual(self.modified, [True])
        self.assertEqual(dialog.call_args.args[2], str(self.station.models))
        source.unlink()
        self.assertTrue(imported.is_file())

    def test_invalid_import_preserves_selected_model_and_draft(self):
        source = self.root / "bad.zip"
        source.write_bytes(b"invalid ZIP")
        self.editor.sop.model_path = "models/previous.zip"
        self.editor.set_model_state("available")
        with patch("sop_app.ui.editor_page.QFileDialog.getOpenFileName", return_value=(str(source), "")), \
                patch("sop_app.ui.editor_page.QMessageBox.warning") as warning:
            self.editor._choose_model()
        warning.assert_called_once()
        self.assertEqual(self.editor.sop.model_path, "models/previous.zip")
        self.assertEqual(self.editor._model_state, "available")
        self.assertEqual(self.selected, [])
        self.assertEqual(self.modified, [])

    def test_cancel_does_not_import_or_modify_sop(self):
        with patch("sop_app.ui.editor_page.QFileDialog.getOpenFileName", return_value=("", "")), \
                patch("sop_app.ui.editor_page.import_model") as importer:
            self.editor._choose_model()
        importer.assert_not_called()
        self.assertEqual(self.modified, [])


if __name__ == "__main__":
    unittest.main()
