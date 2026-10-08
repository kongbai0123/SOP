import json
from pathlib import Path
import tempfile
import unittest

from sop_app.inference_worker import ModelProgress
from sop_app.model_eta import ModelETA, ModelLoadProgress, model_key


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

    def test_progress_counts_only_completed_stages(self):
        work = ModelLoadProgress()
        work.observe(event("載入 AI 執行環境", 100))
        self.assertEqual((work.completed, work.total, work.percent), (0, 7, 0))
        work.observe(event("AI 執行環境已載入（1.46 秒）", 101))
        self.assertEqual(work.completed, 1)
        work.observe(event("讀取模型資訊", 102))
        self.assertEqual(work.completed, 1)  # No double-counting the same milestone.
        work.observe(event("預熱模型，準備首次辨識", 103))
        self.assertEqual((work.completed, work.percent), (6, 86))
        self.assertIn("預熱模型，準備首次辨識：進行中", work.tooltip())
        work.observe(event("載入完成", 104))
        self.assertEqual(work.percent, 100)
        self.assertNotIn("進行中", work.tooltip())

    def test_reload_omits_runtime_and_torchvision_uses_same_milestone(self):
        work = ModelLoadProgress()
        work.observe(event("讀取模型資訊", 100, "reload"))
        self.assertEqual((work.completed, work.total), (0, 6))
        work.observe(event("建立模型並載入運算裝置", 101, "reload"))
        self.assertEqual(work.completed, 4)
        work.observe(event("預熱模型，準備首次辨識", 102, "reload"))
        self.assertEqual((work.completed, work.percent), (5, 83))

    def test_unknown_events_do_not_invent_work_and_failure_stays_incomplete(self):
        work = ModelLoadProgress()
        work.observe(event("未知訊息", 100))
        self.assertEqual(work.completed, 0)
        work.observe(event("讀取與驗證模型權重", 101))
        work.failed = True
        self.assertLess(work.percent, 100)
        self.assertIn("讀取與驗證模型權重：失敗", work.tooltip())
