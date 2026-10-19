"""Replay deterministic SOP fixtures through a real pipeline and durable recorder.

Select baseline/new code using --project-root. Detection/positioning are scripted;
no model, camera, GUI window or production station data is opened. Compare the
``semantics`` member of the two JSON reports (or --compare a previous report).
JPEG bytes intentionally differ after removing the second lossy encoding, so
evidence is checked for existence, decodability and dimensions, not byte identity.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace


def normalized(value):
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def fixtures():
    import numpy as np
    from sop_app.native_runtime import cv2
    from sop_app.sop_schema import Condition, CycleSettings, ROI, SOPDefinition, Step, Workpiece

    def step(name, label, **kwargs):
        return Step(name, conditions=[Condition(label=label)], hold_sec=.2, ratio=1., **kwargs)

    def sop(name, steps, **cycle):
        return SOPDefinition(name=name, version=7, steps=steps, cycle=CycleSettings(enabled=False, **cycle))

    def segment(times, labels=(), pose="valid"):
        return [(float(at), tuple(labels), pose) for at in times]

    cases = [
        (sop("normal", [step("A", "a"), step("B", "b")]),
         segment([0., .1, .2, .3], ["a"]) + segment([.4, .5, .6, .7], ["b"]), ["OK"]),
        (sop("violation", [step("A", "a", forbidden=[Condition(label="bad")], forbidden_hold_sec=.1)]),
         segment([0., .1, .2], ["bad"]) + segment([.3, .4, .5, .6], ["a"]), ["NG"]),
        (sop("sequence", [step("A", "a"), step("B", "b")], detect_sequence=True),
         segment([0., .1, .2, .3], ["b"]) + segment([.4, .5, .6, .7], ["a"]) +
         segment([.8, .9, 1., 1.1], ["b"]), ["NG"]),
        (sop("disappear_and_count", [Step("Put two", conditions=[Condition(label="a", min_count=2)],
                                              hold_sec=.2, ratio=1.),
                                           Step("Remove", conditions=[Condition("disappear", "a")],
                                                hold_sec=.2, ratio=1.)]),
         segment([0., .1], ["a"]) + segment([.2, .3, .4, .5], ["a", "a"]) +
         segment([.6, .7, .8, .9]), ["OK"]),
    ]
    reset = SOPDefinition(name="reset", version=7, steps=[step("A", "a")],
        cycle=CycleSettings(enabled=True, reset_conditions=[Condition("disappear", "a")],
                            reset_hold_sec=.2, reset_ratio=1.))
    cases.append((reset, segment([0., .1, .2, .3], ["a"]) + segment([.4, .5, .6, .7, .8]) +
                  segment([.9, 1., 1.1, 1.2, 1.3], ["a"]), ["OK", "OK"]))

    # A valid PNG is embedded in the SOP snapshot, but no real locator is built.
    reference = np.zeros((100, 200, 3), dtype=np.uint8)
    png = cv2.imencode(".png", reference)[1].tobytes()
    def positioned(name):
        return SOPDefinition(name=name, version=7,
            rois=[ROI("workpiece", [(0., 0.), (.8, 0.), (.8, .8), (0., .8)], "workpiece")],
            workpiece=Workpiece(points=[(.1, .1), (.9, .1), (.9, .9), (.1, .9)],
                                reference_png=base64.b64encode(png).decode("ascii")),
            steps=[Step("A", conditions=[Condition(label="a", roi="workpiece")], hold_sec=1.)],
            cycle=CycleSettings(enabled=False))
    initial = segment([0., .3, .6], ["a"])
    cases.extend([
        (positioned("position_short_loss"), initial + segment([.7, .9], ["a"], "lost") +
         segment([1., 1.31, 1.4], ["a"]), ["OK"]),
        (positioned("position_long_loss"), initial + segment([.7, 1.2], ["a"], "lost") +
         segment([1.3, 2.2, 2.31, 2.4], ["a"]), ["OK"]),
        (positioned("position_shifted_recovery"), initial + segment([.7], ["a"], "lost") +
         segment([.9, 1.8, 1.91, 2.], ["a"], "shifted"), ["OK"]),
    ])
    return cases


def database_semantics(root):
    from sop_app.native_runtime import cv2
    def rows(sql):
        with closing(sqlite3.connect(root / "records.db")) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql)]

    cycles = rows("SELECT id, sop_name, sop_version, model_version, duration_sec, result, station, operator, "
                  "work_order, serial_no, sop_snapshot_hash FROM cycles ORDER BY id")
    steps = rows("SELECT cycle_id, step_index, step_name, status, manual, had_alarm, duration_sec, snapshot "
                 "FROM step_records ORDER BY id")
    alarms = rows("SELECT cycle_id, step_index, step_name, kind, message, acknowledged_by, snapshot "
                  "FROM alarms ORDER BY id")
    positions = rows("SELECT cycle_id, step_index, kind, message, source_time FROM positioning_events ORDER BY id")
    snapshots = rows("SELECT hash, content FROM sop_snapshots ORDER BY hash")
    for item in snapshots:
        if hashlib.sha256(item["content"].encode("utf-8")).hexdigest() != item["hash"]:
            raise AssertionError("Invalid persisted SOP snapshot hash")
        item["content"] = json.loads(item["content"])
    for item in steps + alarms:
        name = item.pop("snapshot")
        item["evidence"] = None
        if name:
            frame = cv2.imread(str(root / "snapshots" / name))
            if frame is None or frame.shape != (100, 200, 3):
                raise AssertionError("Missing, corrupt, or incorrectly sized evidence")
            item["evidence"] = {"decodable": True, "shape": list(frame.shape)}
    with closing(sqlite3.connect(root / "pending.db")) as db:
        pending = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    if pending:
        raise AssertionError(f"Unwritten pending jobs: {pending}")
    return dict(cycles=cycles, steps=steps, alarms=alarms, positioning_events=positions,
                sop_snapshots=snapshots, pending=pending)


def replay(sop, samples, expected_results, root):
    import numpy as np
    from sop_app.detection import Detection
    from sop_app.engine import SOPEngine
    from sop_app.pipeline import VideoPipeline
    from sop_app.recorder import AsyncRecorder
    from sop_app.tracking import workpiece_key

    current = {"labels": (), "pose": "valid"}
    classes = ("a", "b", "bad")
    def detect(_frame):
        return [Detection(label, .91, (10 + index * 40, 10, 30 + index * 40, 30))
                for index, label in enumerate(current["labels"])]

    class ScriptedLocator:
        metrics = {}
        def locate(self, frame, timestamp, **kwargs):
            if current["pose"] == "lost":
                return None, "scripted loss"
            shift = .1 if current["pose"] == "shifted" else 0.
            return np.array([[1., 0., shift], [0., 1., 0.]]), "scripted valid"

    pipe = VideoPipeline(root)
    pipe._context = {"station": "ST-REPLAY", "operator": "fixture", "work_order": "WO", "serial_no": "SN"}
    pipe._engine = SOPEngine(sop)
    pipe._detector = SimpleNamespace(detect=detect, classes=classes,
                                     info=SimpleNamespace(model_version_id="fixture-model"))
    if sop.workpiece:
        pipe._locator = ScriptedLocator()
        pipe._locator_key = workpiece_key(sop.workpiece)
    traces, frames, errors = [], [], []
    pipe.events_ready.connect(lambda events: traces.extend(asdict(event) for event in events))
    pipe.error.connect(errors.append)
    writer = AsyncRecorder(root, on_error=errors.append)
    try:
        pipe._emit_events(pipe._engine.start(0.), writer)
        image = np.random.default_rng(31).integers(0, 256, (100, 200, 3), dtype=np.uint8)
        for index, (timestamp, labels, pose) in enumerate(samples):
            current.update(labels=labels, pose=pose)
            # Exercise both ready and busy GUI paths; every engine input remains.
            if index % 3 == 0:
                pipe.acknowledge_packet()
            pipe._process(image, timestamp, writer)
            state = pipe._engine.snapshot()
            frames.append(dict(timestamp=timestamp, cycle=state.cycle, phase=state.phase.value,
                               current_index=state.current_index, progress=state.progress,
                               cycle_ok=state.cycle_ok, steps=[asdict(step) for step in state.steps]))
        if not writer.close():
            raise AssertionError("Recorder failed to stop")
    finally:
        writer.close()
    if errors:
        raise AssertionError(f"Pipeline/recording errors: {errors}")
    data = database_semantics(root)
    if [row["result"] for row in data["cycles"]] != expected_results:
        raise AssertionError(f"Fixture {sop.name} did not produce expected cycle results: {data['cycles']}")
    return normalized(dict(events=traces, frame_states=frames, records=data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, help="Fail if normalized semantics differ from this JSON report")
    args = parser.parse_args()
    project = args.project_root.resolve(strict=True)
    if not (project / "sop_app" / "pipeline.py").is_file():
        parser.error("--project-root must contain sop_app/pipeline.py")
    sys.path.insert(0, str(project))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance() or QCoreApplication([])
    with tempfile.TemporaryDirectory(prefix="sop-semantic-replay-") as folder:
        semantics = {sop.name: replay(sop, samples, results, Path(folder) / sop.name / "records")
                     for sop, samples, results in fixtures()}
    report = {"check": "sop-semantic-replay-v1", "project_root": str(project),
              "temporary_data_only": True, "detector": "scripted CPU fixture", "locator": "scripted known poses",
              "evidence_comparison": "Existence, decodability, dimensions; excludes intentional JPEG encoding changes",
              "semantics": semantics}
    if args.compare:
        previous = json.loads(args.compare.read_text(encoding="utf-8"))
        if previous["semantics"] != semantics:
            raise SystemExit("SOP replay semantics differ from comparison report")
        report["comparison_matches"] = True
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
