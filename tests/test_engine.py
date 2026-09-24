"""判定引擎單元測試（不需要模型與攝影機）。

執行：.venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sop_app.conditions import evaluate_condition  # noqa: E402
from sop_app.detection import Detection, FrameResult  # noqa: E402
from sop_app.engine import Phase, SOPEngine, StepStatus  # noqa: E402
from sop_app.sop_schema import (ROI, Condition, CycleSettings, SOPDefinition, Step,  # noqa: E402
                                load_sop, save_sop)

W, H, FPS = 200, 100, 10


def det(label, box=(10, 10, 30, 30), score=0.9, mask=None):
    return Detection(label, score, box, mask)


def frame(t, *detections):
    return FrameResult(W, H, t, list(detections))


def appear(label, **kw):
    return Condition("appear", label, **kw)


def run(engine, start, seconds, *detections):
    """以固定 FPS 餵同樣的偵測結果，回傳所有事件與結束時間。"""
    events, t = [], start
    for i in range(int(seconds * FPS)):
        t = start + i / FPS
        events += engine.update(frame(t, *detections))
    return events, t


def kinds(events):
    return [e.kind for e in events]


def make_sop(*steps, **cycle):
    return SOPDefinition(name="測試", steps=list(steps), cycle=CycleSettings(**cycle))


class EngineTests(unittest.TestCase):
    def test_step_completes_after_hold(self):
        engine = SOPEngine(make_sop(Step("A", conditions=[appear("a")], hold_sec=1.0)))
        engine.start(0.0)
        events, _ = run(engine, 0.0, 0.8, det("a"))
        self.assertNotIn("step_completed", kinds(events))
        events, _ = run(engine, 0.8, 0.5, det("a"))
        self.assertIn("step_completed", kinds(events))
        self.assertIn("cycle_completed", kinds(events))

    def test_flicker_does_not_complete(self):
        engine = SOPEngine(make_sop(Step("A", conditions=[appear("a")], hold_sec=1.0, ratio=0.8)))
        engine.start(0.0)
        events = []
        for i in range(50):                      # 每 3 幀只有 1 幀偵測到
            t = i / FPS
            events += engine.update(frame(t, det("a")) if i % 3 == 0 else frame(t))
        self.assertNotIn("step_completed", kinds(events))

    def test_strict_order_next_step_needs_own_hold(self):
        engine = SOPEngine(make_sop(Step("A", conditions=[appear("a")]), Step("B", conditions=[appear("b")])))
        engine.start(0.0)
        # B 的物件一開始就在，但 A 還沒完成，所以 B 不能先完成
        events, t = run(engine, 0.0, 3.0, det("b"))
        self.assertEqual(kinds(events), [])
        self.assertEqual(engine.index, 0)
        events, t = run(engine, t + 0.1, 1.3, det("a"), det("b"))
        self.assertEqual(engine.runtimes[0].status, StepStatus.DONE)
        self.assertEqual(engine.runtimes[1].status, StepStatus.ACTIVE)   # B 需在自己開始後重新累積
        events, _ = run(engine, t + 0.1, 1.3, det("b"))
        self.assertEqual(engine.phase, Phase.CYCLE_DONE)

    def test_disappear_and_count(self):
        sop = make_sop(Step("放 2 個", conditions=[appear("a", min_count=2)], hold_sec=0.5),
                       Step("拿走", conditions=[Condition("disappear", "a")], hold_sec=0.5))
        engine = SOPEngine(sop)
        engine.start(0.0)
        events, t = run(engine, 0.0, 1.0, det("a"))
        self.assertEqual(engine.index, 0)                 # 只有 1 個不夠
        events, t = run(engine, t + 0.1, 1.0, det("a"), det("a", box=(50, 10, 70, 30)))
        self.assertEqual(engine.index, 1)
        events, t = run(engine, t + 0.1, 1.0)
        self.assertEqual(engine.phase, Phase.CYCLE_DONE)

    def test_low_score_ignored(self):
        engine = SOPEngine(make_sop(Step("A", conditions=[appear("a", min_score=0.6)], hold_sec=0.5)))
        engine.start(0.0)
        events, _ = run(engine, 0.0, 2.0, det("a", score=0.4))
        self.assertNotIn("step_completed", kinds(events))

    def test_roi_uses_mask_overlap(self):
        roi = ROI("左半", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)])
        cond = appear("a", roi="左半")
        rois = {"左半": roi}
        left = det("a", box=(10, 10, 40, 40))
        right = det("a", box=(150, 10, 190, 40))
        self.assertTrue(evaluate_condition(cond, frame(0, left), rois).met)
        self.assertFalse(evaluate_condition(cond, frame(0, right), rois).met)

        # 框橫跨兩邊，但遮罩只有在右半 → 不算在區域內
        mask = np.zeros((H, W), dtype=bool)
        mask[20:40, 120:180] = True
        wide = det("a", box=(20, 20, 180, 40), mask=mask)
        self.assertFalse(evaluate_condition(cond, frame(0, wide), rois).met)

    def test_overlap_is_measured_from_object_and_reports_failed_gate(self):
        roi = ROI("安裝位置", [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
        cond = appear("a", roi="安裝位置")
        # 區域大小可能受框選影響，因此維持以「物件有多少位於區域內」為基準。
        covering = det("a", box=(20, 20, 180, 80))
        result = evaluate_condition(cond, frame(0, covering), {roi.name: roi})
        self.assertFalse(result.met)
        self.assertIn('物件在區域內最高', result.detail)

        partial = det("a", box=(0, 0, 90, 45))
        self.assertFalse(evaluate_condition(cond, frame(0, partial), {roi.name: roi}).met)

        low_score = evaluate_condition(Condition(label='a', min_score=.8),
                                       frame(0, det('a', score=.61)), {})
        self.assertEqual(low_score.detail, '最高信心 0.61，門檻 0.80')

    def test_missing_roi_never_met(self):
        result = evaluate_condition(Condition("disappear", "a", roi="不存在"), frame(0), {})
        self.assertFalse(result.met)

    def test_forbidden_raises_violation_once(self):
        sop = make_sop(Step("A", conditions=[appear("a")], forbidden=[appear("b")], hold_sec=0.5))
        engine = SOPEngine(sop)
        engine.start(0.0)
        events, _ = run(engine, 0.0, 3.0, det("b"))
        alarms = [e for e in events if e.kind == "alarm"]
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0].data["alarm"], "violation")
        self.assertFalse(engine.cycle_ok)

    def test_timeout_alarm_and_ng(self):
        engine = SOPEngine(make_sop(Step("A", conditions=[appear("a")], timeout_sec=2.0, hold_sec=0.5)))
        engine.start(0.0)
        events, t = run(engine, 0.0, 3.0)
        self.assertEqual([e.data.get("alarm") for e in events if e.kind == "alarm"], ["timeout"])
        events, _ = run(engine, t + 0.1, 1.0, det("a"))
        completed = [e for e in events if e.kind == "cycle_completed"]
        self.assertEqual(completed[0].data["result"], "NG")

    def test_auto_restart(self):
        engine = SOPEngine(make_sop(Step("A", conditions=[appear("a")], hold_sec=0.5), auto_restart_sec=2.0))
        engine.start(0.0)
        events, t = run(engine, 0.0, 1.0, det("a"))
        self.assertEqual(engine.phase, Phase.CYCLE_DONE)
        events, _ = run(engine, t + 0.1, 2.5)
        self.assertIn("cycle_started", kinds(events))
        self.assertEqual(engine.cycle, 2)

    def test_reset_condition_waits_for_clear(self):
        engine = SOPEngine(make_sop(Step("A", conditions=[appear("a")], hold_sec=0.5),
                                    reset_conditions=[Condition("disappear", "a")], reset_hold_sec=1.0))
        engine.start(0.0)
        events, t = run(engine, 0.0, 1.0, det("a"))
        events, t = run(engine, t + 0.1, 5.0, det("a"))     # 工件還在，不能開始下一輪
        self.assertEqual(engine.cycle, 1)
        events, _ = run(engine, t + 0.1, 1.5)
        self.assertEqual(engine.cycle, 2)

    def test_manual_step_and_skip(self):
        sop = make_sop(Step("人工確認"), Step("B", conditions=[appear("b")]))
        engine = SOPEngine(sop)
        engine.start(0.0)
        events, t = run(engine, 0.0, 5.0, det("b"))
        self.assertEqual(engine.index, 0)                  # 無條件工序不會自動完成
        engine.confirm_step(t)
        self.assertEqual(engine.runtimes[0].status, StepStatus.DONE)
        self.assertTrue(engine.cycle_ok)
        engine.confirm_step(t)                              # 跳過有條件的工序 → NG
        self.assertEqual(engine.runtimes[1].status, StepStatus.SKIPPED)
        self.assertFalse(engine.cycle_ok)

    def test_restart_cycle_emits_abort(self):
        engine = SOPEngine(make_sop(Step("A", conditions=[appear("a")])))
        engine.start(0.0)
        events = engine.restart_cycle(1.0)
        self.assertEqual(kinds(events)[:2], ["cycle_aborted", "cycle_started"])
        self.assertEqual(engine.cycle, 2)


class SchemaTests(unittest.TestCase):
    def test_roundtrip_and_validate(self):
        sop = make_sop(Step("A", conditions=[appear("a", roi="區")], forbidden=[appear("zzz")]))
        sop.rois = [ROI("區", [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)])]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sop.json"
            save_sop(sop, path)
            loaded = load_sop(path)
        self.assertEqual(loaded, sop)
        errors, warnings = loaded.validate(["a", "b"])
        self.assertTrue(any("zzz" in e for e in errors))

    def test_example_sop_is_valid(self):
        root = Path(__file__).resolve().parent.parent
        sop = load_sop(root / "sops" / "example_sop.json")
        errors, _ = sop.validate(["grip", "L_grip", "R_grip", "mid"])
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
