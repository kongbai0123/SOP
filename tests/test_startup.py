import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from sop_app.startup import (StartupProgress, initial_model_path, initial_sop_path,
                             read_progress_messages)


class InitialModelPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sops = self.root / "sops"
        self.sops.mkdir()
        self.settings = self.root / "app_settings.json"
        self.example = self.sops / "example_sop.json"
        self.model = self.root / "model.zip"
        self.model.touch()
        self.addCleanup(patch.stopall)
        patch.multiple("sop_app.startup", APP_SETTINGS=self.settings, SOPS_DIR=self.sops).start()
        patch("sop_app.startup.resolve", side_effect=self.resolve).start()

    def resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_last_sop_and_relative_model_resolve_from_project_root(self):
        saved = self.sops / "saved.json"
        self.write_json(self.settings, {"last_sop": "sops/saved.json"})
        self.write_json(saved, {"model_path": "model.zip"})
        self.assertEqual(initial_sop_path({"last_sop": "sops/saved.json"}), saved)
        self.assertEqual(initial_model_path(), self.model)

    def test_absolute_sop_and_model_paths_are_preserved(self):
        saved = self.root / "saved.json"
        self.write_json(self.settings, {"last_sop": str(saved)})
        self.write_json(saved, {"model_path": str(self.model)})
        self.assertEqual(initial_model_path(), self.model)

    def test_missing_last_sop_uses_example(self):
        self.write_json(self.settings, {"last_sop": "missing.json"})
        self.write_json(self.example, {"model_path": "model.zip"})
        self.assertEqual(initial_sop_path({"last_sop": "missing.json"}), self.example)
        self.assertEqual(initial_model_path(), self.model)

    def test_existing_malformed_last_sop_does_not_preload_example(self):
        saved = self.sops / "saved.json"
        self.write_json(self.settings, {"last_sop": str(saved)})
        saved.write_text("invalid JSON", encoding="utf-8")
        self.write_json(self.example, {"model_path": "model.zip"})
        self.assertEqual(initial_sop_path({"last_sop": str(saved)}), saved)
        self.assertIsNone(initial_model_path())

    def test_missing_or_malformed_settings_use_example(self):
        self.write_json(self.example, {"model_path": "model.zip"})
        self.assertEqual(initial_model_path(), self.model)
        self.settings.write_text("invalid JSON", encoding="utf-8")
        self.assertEqual(initial_model_path(), self.model)

    def test_settings_with_invalid_types_use_example(self):
        self.write_json(self.example, {"model_path": "model.zip"})
        for value in ([], None, 1, {"last_sop": ["saved.json"]}, {"last_sop": 1}):
            with self.subTest(value=value):
                self.write_json(self.settings, value)
                self.assertEqual(initial_model_path(), self.model)

    def test_invalid_sop_or_model_types_do_not_preload(self):
        for value in ([], None, 1, {}, {"model_path": None}, {"model_path": 1},
                      {"model_path": ["model.zip"]}, {"model_path": ""},
                      {"model_path": "\u0000"}):
            with self.subTest(value=value):
                self.write_json(self.example, value)
                self.assertIsNone(initial_model_path())

    def test_missing_model_or_sop_does_not_preload(self):
        self.assertIsNone(initial_sop_path({}))
        self.assertIsNone(initial_model_path())
        self.write_json(self.example, {"model_path": "missing.zip"})
        self.assertIsNone(initial_model_path())

    def test_directory_models_can_preload(self):
        model_dir = self.root / "model"
        model_dir.mkdir()
        self.write_json(self.example, {"model_path": "model"})
        self.assertEqual(initial_model_path(), model_dir)

    def test_import_keeps_torch_and_qt_out_of_ui_parent(self):
        script = ("import sys; import sop_app.startup; "
                  "assert not any(name == 'torch' or name.startswith('torch.') "
                  "or name == 'PySide6' or name.startswith('PySide6.') "
                  "for name in sys.modules)")
        subprocess.run([sys.executable, "-c", script], check=True, timeout=10,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class StartupProgressTests(unittest.TestCase):
    def test_utf8_pipe_is_independent_of_windows_text_encoding(self):
        message = {"progress": 40, "message": "2 / 4 · 正在載入介面元件…", "elapsed": 1.7}
        wire = (json.dumps(message, ensure_ascii=False) + "\r\n").encode("utf-8")
        stdin = io.TextIOWrapper(io.BytesIO(wire), encoding="cp950", errors="surrogateescape")
        self.assertEqual(list(read_progress_messages(stdin.buffer)), [message])

    def test_bad_packet_does_not_discard_following_status(self):
        wire = b'\xff\nnot json\n{"close": true}\n'
        self.assertEqual(list(read_progress_messages(io.BytesIO(wire))), [{"close": True}])

    def test_real_child_process_reads_utf8_under_cp950_environment(self):
        message = {"progress": 65, "message": "3 / 4 · 正在建立工作視窗…", "elapsed": 2.0}
        script = ("import sys,json; from sop_app.startup import read_progress_messages; "
                  "print(json.dumps(list(read_progress_messages(sys.stdin.buffer)), ensure_ascii=True))")
        environment = dict(os.environ, PYTHONIOENCODING="cp950", PYTHONUTF8="0")
        result = subprocess.run([sys.executable, "-c", script],
                                input=(json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
                                timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout.decode("ascii")), [message])

    def test_progress_is_delivered_as_unicode_json(self):
        progress = StartupProgress()
        progress.process = Mock(stdin=io.StringIO())
        progress.report(40, "正在載入介面")
        message = json.loads(progress.process.stdin.getvalue())
        self.assertEqual(message["message"], "正在載入介面")
        self.assertEqual(message["progress"], 40)
        self.assertGreaterEqual(message["elapsed"], 0)

    def test_closed_splash_does_not_break_startup(self):
        progress = StartupProgress()
        progress.process = Mock()
        progress.process.stdin.write.side_effect = BrokenPipeError
        progress.report(40, "繼續啟動")
        progress.close()
        progress.process.stdin.close.assert_called_once()

    def test_failed_splash_launch_allows_startup_to_continue(self):
        with patch("sop_app.startup.subprocess.Popen", side_effect=OSError), self.assertLogs("sop.launcher"):
            progress = StartupProgress()
            progress.start()
        progress.report(100, "介面已開啟")
        progress.close()


if __name__ == "__main__":
    unittest.main()
