"""Pipeline integration with the isolated inference worker, without loading a GPU."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sop_app.inference_worker import InferenceWorkerClosed, InferenceWorkerError, ModelProgress
from sop_app.pipeline import VideoPipeline


class PipelineInferenceTests(unittest.TestCase):
    def setUp(self):
        self.path = str(Path("model.zip").resolve())
        self.info = SimpleNamespace(model_version_id="M-test", classes=("tool",))
        self.proxy = SimpleNamespace(info=self.info, classes=self.info.classes,
                                     device_name="GPU（test）", detect=Mock(return_value=[]))
        self.worker = Mock()
        self.worker.load_model.return_value = self.proxy
        self.pipeline = VideoPipeline(".", inference_worker=self.worker)
        self.loaded = []
        self.errors = []
        self.progress = []
        self.status = []
        self.pipeline.model_loaded.connect(lambda info, device: self.loaded.append((info, device)))
        self.pipeline.error.connect(self.errors.append)
        self.pipeline.model_progress.connect(self.progress.append)
        self.pipeline.status.connect(self.status.append)
        self.read_info = self.enterContext(patch("sop_app.pipeline.read_model_info", return_value=self.info))

    def test_model_load_reuses_prestarted_worker_and_forwards_progress(self):
        def load(path, progress):
            progress("預熱模型，準備首次辨識")
            return self.proxy

        self.worker.load_model.side_effect = load
        with patch("sop_app.pipeline.InferenceWorker") as factory:
            self.pipeline._load_model(self.path)
        factory.assert_not_called()
        self.worker.start.assert_not_called()
        self.worker.load_model.assert_called_once()
        self.assertEqual(self.worker.load_model.call_args.args, (self.path,))
        self.assertEqual(self.loaded, [(self.info, "GPU（test）")])
        self.assertIn("預熱模型，準備首次辨識", self.progress)
        self.assertEqual(self.progress[-1], "載入完成")
        self.assertIs(self.pipeline._detector, self.proxy)
        self.assertIn("M-test", self.status[-1])

    def test_missing_worker_is_created_only_when_a_model_is_requested(self):
        with patch("sop_app.pipeline.InferenceWorker") as factory:
            pipeline = VideoPipeline(".")
            factory.assert_not_called()
            factory.return_value.start.return_value = self.worker
            pipeline._load_model(self.path)
            factory.assert_called_once_with()
            factory.return_value.start.assert_called_once_with()
            pipeline._load_model(self.path)
            factory.assert_called_once_with()
        self.assertIs(pipeline._inference_worker, self.worker)
        self.assertIs(pipeline._detector, self.proxy)

    def test_child_timing_reaches_ui_without_losing_measurement(self):
        timing = ModelProgress({"message": "預熱模型", "measured_at": 123.5,
                                "mode": "reload"})
        received = []
        self.pipeline.model_timing.connect(received.append)

        def load(path, progress):
            progress(timing)
            return self.proxy

        self.worker.load_model.side_effect = load
        self.pipeline._load_model(self.path)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].measured_at, 123.5)
        self.assertEqual(received[0].mode, "reload")
        self.assertIn("預熱模型", self.progress)

    def test_remote_application_control_failure_has_ui_message_and_traceback_log(self):
        remote_error = InferenceWorkerError(
            "remote worker: blocked torch_cuda.dll", winerror=4551,
            remote_type="OSError", remote_traceback="Remote traceback: LoadLibrary torch_cuda.dll")
        self.worker.load_model.side_effect = remote_error
        with self.assertLogs("sop.launcher", level="ERROR") as captured:
            self.pipeline.load_model(self.path)
            self.pipeline._drain_commands(Mock())
        self.assertEqual(self.loaded, [(None, "")])
        self.assertIsNone(self.pipeline._detector)
        self.assertEqual(len(self.errors), 1)
        self.assertIn("錯誤 4551", self.errors[0])
        self.assertIn("blocked torch_cuda.dll", self.errors[0])
        self.assertEqual(self.progress[-1], "載入失敗")
        log = "\n".join(captured.output)
        self.assertIn("Traceback", log)
        self.assertIn("blocked torch_cuda.dll", log)
        self.assertIn(self.path, log)
        self.assertIn("Remote traceback: LoadLibrary torch_cuda.dll", log)

    def test_shutdown_cancels_worker_before_waiting_for_pipeline(self):
        order = []
        self.worker.close.side_effect = lambda: order.append(("close", self.pipeline._alive))
        def wait(*args):
            order.append(("wait", args))
            return True

        with patch.object(self.pipeline, "wait", side_effect=wait):
            self.assertTrue(self.pipeline.shutdown())
        self.assertEqual(order[0], ("close", False))
        self.assertEqual(order[1][0], "wait")
        self.worker.close.assert_called_once_with()

    def test_shutdown_reports_thread_that_has_not_stopped(self):
        with patch.object(self.pipeline, "wait", return_value=False):
            self.assertFalse(self.pipeline.shutdown())
        self.assertFalse(self.pipeline._alive)
        self.worker.close.assert_called_once_with()

    def test_cancelled_load_on_close_does_not_publish_error_or_keep_old_detector(self):
        self.pipeline._detector = self.proxy

        def cancel(path, progress):
            self.pipeline._alive = False
            raise InferenceWorkerClosed("worker closed")

        self.worker.load_model.side_effect = cancel
        with patch("sop_app.pipeline.logging.getLogger") as logger:
            self.pipeline._handle_command("model", self.path, Mock())
        self.assertIsNone(self.pipeline._detector)
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.errors, [])
        self.assertNotIn("載入失敗", self.progress)
        logger.return_value.exception.assert_not_called()

    def test_load_finishing_after_close_does_not_publish_stale_detector(self):
        def finish_during_close(path, progress):
            self.pipeline._alive = False
            return self.proxy

        self.worker.load_model.side_effect = finish_during_close
        self.pipeline._load_model(self.path)
        self.assertIsNone(self.pipeline._detector)
        self.assertEqual(self.loaded, [])
        self.assertNotIn("載入完成", self.progress)
        self.assertEqual(self.errors, [])

    def test_model_request_after_close_does_not_start_an_inference_process(self):
        pipeline = VideoPipeline(".")
        pipeline._alive = False
        with patch("sop_app.pipeline.InferenceWorker") as factory:
            pipeline._load_model(self.path)
        factory.assert_not_called()
        self.read_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
