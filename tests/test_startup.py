import io
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from sop_app.startup import StartupProgress, read_progress_messages


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
