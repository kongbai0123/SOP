"""Summarize per-frame tracking traces and optional manually labelled corners.

Labels JSON format: {"frame-0001.jpg": [[x,y], [x,y], [x,y], [x,y]], ...}
The four points must match SOP workpiece.points order in pixel coordinates.
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from sop_app.sop_schema import load_sop


def run_lengths(rows, key):
    periods, began = [], None
    for row in rows:
        if not row.get(key) and began is None:
            began = row['elapsed']
        elif row.get(key) and began is not None:
            periods.append(round(row['elapsed'] - began, 3))
            began = None
    return periods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', help='Directory containing samples.jsonl / metrics.jsonl')
    parser.add_argument('--sop', default='sops/grip_mcp_demo_20260924.json')
    parser.add_argument('--labels', help='Optional JSON of human-labelled workpiece corners')
    args = parser.parse_args()
    root = Path(args.capture)
    path = root / 'metrics.jsonl'
    if not path.exists():
        path = root / 'samples.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [r for r in rows if 'display_valid' in r or 'matrix' in r]
    for row in rows:
        row['valid'] = row.get('display_valid', row.get('matrix') is not None)
    valid = sum(r['valid'] for r in rows)
    durations = run_lengths(rows, 'valid')
    reasons = collections.Counter()
    for row in rows:
        for stage, detail in (row.get('tracking') or {}).get('diagnostics', {}).items():
            if detail.get('reason') != 'accepted':
                reasons[(stage, detail.get('reason'))] += 1
    report = dict(samples=len(rows), valid=valid,
                  valid_ratio=round(valid / len(rows), 3) if rows else None,
                  recovered_losses=len(durations),
                  median_recovery_sec=round(statistics.median(durations), 3) if durations else None,
                  max_recovery_sec=max(durations, default=None),
                  failure_reasons={f'{stage}:{reason}': count for (stage, reason), count in reasons.most_common()})
    if args.labels:
        labels = json.loads(Path(args.labels).read_text(encoding='utf-8'))
        sop = load_sop(args.sop)
        reference = np.asarray(sop.workpiece.points)
        corner_errors = []
        for row in rows:
            if row.get('frame') not in labels or row.get('matrix') is None:
                continue
            frame = cv2.imread(str(root / row['frame']))
            if frame is None:
                continue
            truth = np.asarray(labels[row['frame']], dtype=float)
            if truth.shape != (4, 2):
                raise ValueError(f"{row['frame']} needs four labelled corners")
            matrix = np.asarray(row['matrix'], dtype=float)
            estimate = (reference @ matrix[:, :2].T + matrix[:, 2]) * [frame.shape[1], frame.shape[0]]
            corner_errors.extend(np.linalg.norm(estimate - truth, axis=1).tolist())
        report['labelled_corners'] = len(corner_errors)
        report['median_corner_error_px'] = round(statistics.median(corner_errors), 2) if corner_errors else None
        report['p95_corner_error_px'] = round(float(np.percentile(corner_errors, 95)), 2) if corner_errors else None
    else:
        report['pose_accuracy'] = 'unavailable: no human-labelled corners'
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
