"""Compare locator availability and cost on saved frames; no accuracy labels."""
import argparse
import collections
import importlib.util
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import cv2

from sop_app.detector import create_detector
from sop_app.model_bundle import read_model_info
from sop_app.sop_schema import load_sop
from sop_app.tracking import WorkpieceLocator


def baseline_locator(sop):
    source = subprocess.check_output(['git', 'show', '6a03db3:sop_app/tracking.py'])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / '_baseline_tracker.py'
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('sop_app._baseline_tracker', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.WorkpieceLocator(sop.workpiece, adaptive_budget=True)


def summarize(items):
    durations, missing_start = [], None
    for item in items:
        if not item['valid'] and missing_start is None:
            missing_start = item['t']
        elif item['valid'] and missing_start is not None:
            durations.append(item['t'] - missing_start)
            missing_start = None
    timings = sorted(x['tracking_ms'] for x in items)
    return dict(frames=len(items), valid=sum(x['valid'] for x in items),
                median_tracking_ms=round(statistics.median(timings), 1),
                p95_tracking_ms=round(timings[int(.95 * (len(timings)-1))], 1),
                recovered_runs=len(durations),
                median_recovery_sec=round(statistics.median(durations), 2) if durations else None,
                sources=dict(collections.Counter(x['source'] for x in items if x['source'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sop', default='sops/grip_mcp_demo_20260924.json')
    parser.add_argument('--capture', default='logs/tracking-live-20260929')
    parser.add_argument('--start', type=int, default=35)
    parser.add_argument('--count', type=int, default=160)
    parser.add_argument('--output', default='logs/reacquisition-replay.json')
    args = parser.parse_args()
    root = Path(args.capture)
    lines = (root / 'samples.jsonl').read_text(encoding='utf-8').splitlines()
    rows = [json.loads(line) for line in lines if line.strip()][args.start:args.start + args.count]
    sop = load_sop(args.sop)
    detector = create_detector(read_model_info(sop.model_path))
    locators = {'baseline': baseline_locator(sop),
                'guided': WorkpieceLocator(sop.workpiece, adaptive_budget=True)}
    results = {name: [] for name in locators}
    for row in rows:
        frame = cv2.imread(str(root / row['frame']))
        if frame is None:
            continue
        detections = detector.detect(frame)
        for name, locator in locators.items():
            started = time.perf_counter()
            if name == 'guided':
                matrix, _ = locator.locate(frame, row['elapsed'], detections=detections)
            else:
                matrix, _ = locator.locate(frame, row['elapsed'])
            results[name].append(dict(t=row['elapsed'], valid=matrix is not None,
                                      tracking_ms=round((time.perf_counter()-started)*1000, 2),
                                      source=locator.metrics.get('source') if name == 'guided' else None,
                                      handlebar_detected=any(d.label == 'handlebar' and d.score >= .5
                                                             for d in detections)))
    baseline, guided = results['baseline'], results['guided']
    report = dict(method='same saved frames, same M004 detections, two tracker versions',
                  baseline=summarize(baseline), guided=summarize(guided),
                  guided_only_valid=sum(not a['valid'] and b['valid'] for a, b in zip(baseline, guided)),
                  baseline_only_valid=sum(a['valid'] and not b['valid'] for a, b in zip(baseline, guided)),
                  detections_present=sum(x['handlebar_detected'] for x in guided))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(report=report, frames=results), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
