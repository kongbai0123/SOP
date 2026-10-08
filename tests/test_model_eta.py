import json
from pathlib import Path
import tempfile
import unittest

from sop_app.inference_worker import ModelProgress
from sop_app.model_eta import ModelETA, model_key


def event(stage, stamp, mode="startup"):
    return ModelProgress({"message": stage, "measured_at": stamp, "mode": mode})


class ModelETATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "history.json"

    def train(self, duration=10, mode="startup"):
        eta = ModelETA(self.path, "model-a")
        eta.observe(event("預熱模型", 100, mode))
        eta.observe(event("載入完成", 100 + duration, mode))

    def test_delayed_preload_delivery_uses_child_measurements(self):
        self.train(10)
        eta = ModelETA(self.path, "model-a")
        eta.observe(event("預熱模型", 200))
        self.assertEqual(eta.remaining_text(now=204), "預估剩餘約 6 秒")

    def test_range_and_expired_estimate_never_claim_zero_seconds(self):
        self.train(10)
        self.train(20)
        eta = ModelETA(self.path, "model-a")
        eta.observe(event("預熱模型", 200))
        self.assertEqual(eta.remaining_text(now=202), "預估剩餘 8 秒～18 秒")
        self.assertIn("接近歷史範圍上限", eta.remaining_text(now=212))
        self.assertIn("重新評估", eta.remaining_text(now=220))

    def test_initial_and_reload_measurements_are_separate(self):
        self.train(10)
        self.train(2, "reload")
        eta = ModelETA(self.path, "model-a")
        eta.observe(event("預熱模型", 200, "reload"))
        self.assertEqual(eta.remaining_text(now=200), "預估剩餘約 2 秒")

    def test_model_and_unknown_stage_have_no_invented_estimate(self):
        self.train()
        eta = ModelETA(self.path, "model-b")
        eta.observe(event("預熱模型", 200))
        self.assertIn("尚無", eta.remaining_text(now=200))
        eta = ModelETA(self.path, "model-a")
        eta.observe(event("新階段", 200))
        self.assertIn("尚無", eta.remaining_text(now=200))

    def test_incomplete_or_failed_load_does_not_train(self):
        eta = ModelETA(self.path, "model-a")
        eta.observe(event("預熱模型", 100))
        eta.observe(event("載入失敗", 110))
        self.assertFalse(self.path.exists())

    def test_runtime_duration_suffix_is_normalized(self):
        eta = ModelETA(self.path, "model-a")
        eta.observe(event("AI 執行環境已載入（1.46 秒）", 100))
        eta.observe(event("載入完成", 105))
        eta = ModelETA(self.path, "model-a")
        eta.observe(event("AI 執行環境已載入（47.16 秒）", 200))
        self.assertEqual(eta.remaining_text(now=200), "預估剩餘約 5 秒")

    def test_bad_history_is_ignored_and_samples_are_bounded(self):
        self.path.write_text("broken", encoding="utf-8")
        for _ in range(15):
            self.train()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["model-a"]["startup"]), 12)

    def test_replaced_model_invalidates_history_key(self):
        model = Path(self.temp.name) / "model.zip"
        model.write_bytes(b"model")
        original = model_key(model)
        model.write_bytes(b"different model")
        self.assertNotEqual(original, model_key(model))
