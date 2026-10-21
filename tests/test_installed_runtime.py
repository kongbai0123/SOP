"""Installation-safe paths and entry dispatch, without loading a real model."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from sop_app import paths
from sop_app.launcher import start
from sop_app.startup import StartupProgress, read_progress_file


class RuntimePathsTests(unittest.TestCase):
    def load_paths(self, frozen, root):
        spec = importlib.util.spec_from_file_location("isolated_sop_paths", paths.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch.object(sys, "frozen", frozen, create=True), \
                patch.object(sys, "_MEIPASS", str(root / "resources"), create=True), \
                patch.object(sys, "executable", str(root / "install" / "SOP.exe")), \
                patch.dict(os.environ, {"LOCALAPPDATA": str(root / "user")}):
            spec.loader.exec_module(module)
        return module

    def test_source_layout_remains_project_relative(self):
        module = self.load_paths(False, Path("C:/unused"))
        self.assertEqual(module.PROJECT_ROOT, Path(paths.__file__).resolve().parent.parent)
        self.assertEqual(module.RECORDS_DIR, module.PROJECT_ROOT / "records")
        self.assertEqual(module.MODELS_DIR, module.PROJECT_ROOT / "models")
        self.assertEqual(module.RESOURCE_ROOT, module.PROJECT_ROOT)

    def test_installed_paths_and_first_run_never_modify_install_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resource = root / "resources" / "sops" / "example_sop.json"
            resource.parent.mkdir(parents=True)
            resource.write_text('{"name":"clean example","model_path":""}', encoding="utf-8")
            module = self.load_paths(True, root)
            self.assertEqual(module.PROJECT_ROOT, root / "user" / "SOP.WorkInstruction")
            self.assertEqual(module.INSTALL_ROOT, root / "install")
            module.initialize_runtime()
            self.assertFalse(module.INSTALL_ROOT.exists())
            self.assertEqual(module.MODELS_DIR, module.DATA_ROOT / "models")
            self.assertTrue(module.MODELS_DIR.is_dir())
            sample = module.SOPS_DIR / "example_sop.json"
            self.assertEqual(sample.read_bytes(), resource.read_bytes())
            sample.write_text("user change", encoding="utf-8")
            module.initialize_runtime()
            self.assertEqual(sample.read_text(encoding="utf-8"), "user change")
            self.assertEqual(module.resolve("models/new.zip"), module.DATA_ROOT / "models/new.zip")
            self.assertEqual(module.relativize(sample), "sops/example_sop.json")


class FrozenEntryTests(unittest.TestCase):
    def test_normal_start_imports_main_and_restores_redirected_streams(self):
        original_cwd, original_stdout, original_stderr = Path.cwd(), sys.stdout, sys.stderr
        self.addCleanup(os.chdir, original_cwd)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            fake = Mock()
            fake.main.return_value = 7
            with patch("sop_app.launcher.multiprocessing.freeze_support"), \
                    patch.multiple(paths, PROJECT_ROOT=root, LOGS_DIR=logs), \
                    patch.object(paths, "initialize_runtime"), patch.dict(sys.modules, {"main": fake}):
                self.assertEqual(start([]), 7)
            fake.main.assert_called_once_with()
            self.assertIs(sys.stdout, original_stdout)
            self.assertIs(sys.stderr, original_stderr)
            self.assertIn("啟動 SOP", (logs / "startup.log").read_text(encoding="utf-8"))
            os.chdir(original_cwd)

    def test_freeze_support_runs_before_smoke_dispatch(self):
        events = []
        fake = Mock()
        fake.run.side_effect = lambda path: events.append(("smoke", path)) or 0
        with patch("sop_app.launcher.multiprocessing.freeze_support", side_effect=lambda: events.append("freeze")), \
                patch.dict(sys.modules, {"sop_app.installer_smoke": fake}):
            self.assertEqual(start(["--installer-smoke-test", "result.json"]), 0)
        self.assertEqual(events, ["freeze", ("smoke", Path("result.json"))])

    def test_missing_smoke_argument_does_not_open_normal_gui(self):
        with patch("sop_app.launcher.multiprocessing.freeze_support"):
            self.assertEqual(start(["--installer-smoke-test"]), 2)

    def test_unknown_or_undispatched_worker_arguments_cannot_open_production_gui(self):
        with patch("sop_app.launcher.multiprocessing.freeze_support"), \
                patch.object(paths, "initialize_runtime") as initialize:
            for arguments in (["--multiprocessing-fork", "parent_pid=1", "pipe_handle=2"],
                              ["--installer-runtime-test result.json"], ["-c", "pass"]):
                with self.subTest(arguments=arguments):
                    self.assertEqual(start(arguments), 2)
            initialize.assert_not_called()

    def test_opt_in_diagnostics_capture_effective_arguments_and_process_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnostic = Path(temporary) / "dispatch.jsonl"
            with patch.dict(os.environ, {"SOP_LAUNCH_DIAGNOSTICS": str(diagnostic)}), \
                    patch("sop_app.launcher.multiprocessing.freeze_support"):
                self.assertEqual(start(["--unexpected"]), 2)
            records = [json.loads(line) for line in diagnostic.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["stage"] for record in records],
                             ["before-freeze-support", "dispatch", "unsupported-arguments"])
            self.assertEqual(records[1]["arguments"], ["--unexpected"])
            self.assertEqual(records[1]["pid"], os.getpid())

    def test_runtime_flag_explicitly_enables_isolated_ai_verification(self):
        fake = Mock()
        fake.run.return_value = 0
        with patch("sop_app.launcher.multiprocessing.freeze_support"), \
                patch.dict(sys.modules, {"sop_app.installer_smoke": fake}):
            self.assertEqual(start(["--installer-runtime-test", "result.json"]), 0)
        fake.run.assert_called_once_with(Path("result.json"), check_ai_runtime=True)

    def test_failed_frozen_splash_launch_cleans_up_status_file(self):
        with patch.object(sys, "frozen", True, create=True), \
                patch("sop_app.startup.subprocess.Popen", side_effect=OSError), \
                self.assertLogs("sop.launcher"):
            progress = StartupProgress()
            progress.start()
        self.assertIsNone(progress.status_file)
        progress.close()

    def test_frozen_splash_uses_dispatch_flag_and_unicode_file(self):
        with patch.object(sys, "frozen", True, create=True), patch("sop_app.startup.subprocess.Popen") as popen:
            popen.return_value.stdin = None
            progress = StartupProgress()
            progress.start()
            self.assertEqual(popen.call_args.args[0][:2], [sys.executable, "--sop-startup-window"])
            progress.report(40, "正在載入介面")
            progress.close()
            progress.close()
            packets = list(read_progress_file(progress.status_file))
            self.assertEqual(packets[0]["message"], "正在載入介面")
            self.assertEqual(packets[1:], [{"close": True}])
            self.assertFalse(progress.status_file.exists())

    def test_orphan_splash_reader_expires_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "status.jsonl"
            status.write_bytes(b"")
            self.assertEqual(list(read_progress_file(status, timeout=0.01)), [])
            self.assertFalse(status.exists())

    def test_real_smoke_opens_ui_spawns_worker_and_isolates_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            command = [sys.executable, "-c", "from sop_app.launcher import start; raise SystemExit(start())",
                       "--installer-smoke-test", str(report)]
            result = subprocess.run(command, cwd=paths.SOURCE_ROOT,
                                    env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
                                    capture_output=True, timeout=45)
            self.assertTrue(report.exists(), result.stderr.decode("utf-8", errors="replace"))
            evidence = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result.returncode, 0, evidence)
            self.assertTrue(evidence["ok"], evidence)
            self.assertTrue(all(evidence["checks"].values()))
            self.assertFalse(evidence["camera_opened"])
            self.assertFalse(evidence["model_loaded"])
            self.assertFalse(Path(evidence["paths"]["smoke_data"]).exists())


if __name__ == "__main__":
    unittest.main()
