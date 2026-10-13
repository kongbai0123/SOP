"""Startup must survive an OS refusal to load OpenCV (no native CV dependency)."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from sop_app.native_runtime import LazyOpenCV, NativeDependencyError


class StartupTests(unittest.TestCase):
    def test_failed_import_is_explicit_and_cached(self):
        runtime = LazyOpenCV()
        with patch("sop_app.native_runtime.importlib.import_module", side_effect=ImportError("policy blocked")) as load:
            for _ in range(2):
                with self.assertRaisesRegex(NativeDependencyError, "policy blocked"):
                    runtime.require()
            self.assertEqual(load.call_count, 1)

    def test_editor_and_records_start_when_opencv_blocked(self):
        script = '''
import sys, tempfile
from pathlib import Path
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QMessageBox
from sop_app.native_runtime import cv2
assert "cv2" not in sys.modules
from sop_app.ui.main_window import MainWindow
from sop_app.sop_schema import SOPDefinition, Step
assert "cv2" not in sys.modules
cv2._failure = "OpenCV policy blocked (test)"
app = QApplication([])
with tempfile.TemporaryDirectory() as directory:
    def initial(window):
        sop = SOPDefinition(steps=[Step("manual")])
        window.editor.set_sop(sop)
        window.run_page.set_sop(sop)
    with patch("sop_app.ui.main_window.RECORDS_DIR", Path(directory)), \\
         patch.object(MainWindow, "_load_settings", return_value={}), \\
         patch.object(MainWindow, "_save_settings"), \\
         patch.object(MainWindow, "_open_initial_sop", initial):
        window = MainWindow()
        window.show()
        app.processEvents()
        assert window.isVisible()
        assert window._vision_error
        assert window.pipeline.isRunning()
        window.editor._add_step()
        assert len(window.editor.sop.steps) == 2
        with patch.object(QMessageBox, "warning") as warning:
            window._connect_source()
            window._start_run()
            assert warning.call_count == 2
        window.records_page.refresh()
        window.dirty = False
        window.close()
        assert not window.pipeline.isRunning()
print("BLOCKED_NATIVE_STARTUP_OK")
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("BLOCKED_NATIVE_STARTUP_OK", result.stdout)
