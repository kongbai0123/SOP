"""Read-only video/SOP regression replay using frozen detector predictions.

No model, GPU, camera, recorder, or station writes. Replays the inference
benchmark's exact frames (index * 3), verifies every decoded image hash, and
uses source-video timestamps with the real locator and SOP engine. Comparison
detects behavior changes; without labelled pose truth it cannot prove accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]


def check_signature(results):
    return [(r['met'], r['count'], r['error_kind'], bool(r['error'])) for r in results]


def decision_signature(steps):
    """Include count changes for review, even if boolean decisions agree."""
    return [{**{role: check_signature(step[role]) for role in ('completion', 'forbidden')},
             **{flag: step[flag] for flag in ('completion_met', 'completion_blocked',
                                            'forbidden_met', 'forbidden_blocked')}} for step in steps]


def compare_reports(before, after):
    for key in ('sop_sha256', 'predictions_sha256', 'input_hashes'):
        if before[key] != after[key]:
            raise ValueError(f'Cannot compare different replay inputs: {key}')
    if len(before['frames']) != len(after['frames']):
        raise ValueError('Cannot compare different frame counts')
    validity_changes, condition_changes, state_changes, deltas = [], [], [], []
    for old, new in zip(before['frames'], after['frames']):
        index = new['sample_index']
        if old['source_frame_index'] != new['source_frame_index'] or old['timestamp'] != new['timestamp']:
            raise ValueError('Source frame/time alignment differs')
        if old['pose_valid'] != new['pose_valid']:
            validity_changes.append({'sample_index': index, 'before': old['pose_valid'], 'after': new['pose_valid']})
        if old['corners_px'] is not None and new['corners_px'] is not None:
            delta = np.linalg.norm(np.asarray(old['corners_px']) - np.asarray(new['corners_px']), axis=1)
            deltas.append({'sample_index': index, 'max_corner_delta_px': float(delta.max())})
        old_signature, new_signature = decision_signature(old['steps']), decision_signature(new['steps'])
        if old_signature != new_signature or check_signature(old['reset']) != check_signature(new['reset']):
            condition_changes.append({'sample_index': index, 'before': old_signature, 'after': new_signature,
                                      'reset_before': old['reset'], 'reset_after': new['reset']})
        if old['engine'] != new['engine']:
            state_changes.append({'sample_index': index, 'before': old['engine'], 'after': new['engine']})
    values = [r['max_corner_delta_px'] for r in deltas]
    return {'pose_validity_changes': validity_changes, 'condition_changes': condition_changes,
            'engine_state_changes': state_changes, 'events_identical': before['events'] == after['events'],
            'common_valid_pose_frames': len(values),
            'corner_delta_px': {k: float(np.percentile(values, p)) for k, p in [('p50', 50), ('p95', 95), ('max', 100)]}
                if values else None,
            'per_frame_corner_deltas': deltas,
            'interpretation': 'Behavior comparison only. Neither version is pose ground truth; review differences.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=WORKSPACE)
    parser.add_argument('--strategy', choices=('fixed', 'adaptive'), required=True)
    parser.add_argument('--video', type=Path, default=WORKSPACE / 'demo/demo_sop.mp4')
    parser.add_argument('--sop', type=Path, default=WORKSPACE / 'data/stations/ST001/sops/grip_mcp_demo_20260924.json')
    parser.add_argument('--predictions', type=Path, default=WORKSPACE / '.report-build/performance/inference_after_repeat.json')
    parser.add_argument('--frame-stride', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--fail-on-decision-change', action='store_true')
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error('--frame-stride must be positive')
    output = args.output.resolve()
    if any(output.is_relative_to(root.resolve() / 'data') for root in (WORKSPACE, args.project_root)):
        raise ValueError('Replay output must be outside station data')
    inputs = [args.video, args.sop, args.predictions] + ([args.compare] if args.compare else [])
    if output in {p.resolve() for p in inputs}:
        raise ValueError('Replay output cannot overwrite an input')
    sys.path.insert(0, str(args.project_root.resolve()))
    from sop_app.conditions import evaluate_condition
    from sop_app.detection import Detection, FrameResult
    from sop_app.engine import SOPEngine, evaluate_step
    from sop_app.sop_schema import SOPDefinition
    from sop_app.tracking import WorkpieceLocator, workpiece_key

    frozen = json.loads(args.predictions.read_text(encoding='utf-8'))
    predictions, hashes = frozen['predictions'], frozen['input_hashes']
    if not predictions or len(predictions) != len(hashes):
        raise ValueError('Frozen predictions and input hashes must have equal nonzero length')
    if any(d.get('mask_sha256') is not None for ds in predictions for d in ds):
        raise ValueError('A mask hash cannot reconstruct masks; use complete frozen masks or a detection-only fixture')
    sop_bytes = args.sop.read_bytes()
    sop = SOPDefinition.from_dict(json.loads(sop_bytes))
    locator = WorkpieceLocator(sop.workpiece, adaptive_budget=args.strategy == 'adaptive') if sop.workpiece else None
    engine = SOPEngine(sop)
    events = [asdict(e) for e in engine.start(0.)]
    roi_map = sop.roi_map()
    key = workpiece_key(sop.workpiece)
    capture = cv2.VideoCapture(str(args.video.resolve()))
    if not capture.isOpened():
        raise ValueError('Cannot open replay video')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('Video has no usable source FPS')
    rows = []

    def checks(results):
        return [{'met': bool(r.met), 'count': int(r.count), 'error_kind': r.error_kind,
                 'error': r.error, 'detail': r.detail} for r in results]

    try:
        for index, (detections, expected_hash) in enumerate(zip(predictions, hashes)):
            source_index = index * args.frame_stride
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f'Cannot decode frozen sample {index}')
            actual_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f'Decoded image hash differs at sample {index}; refusing unmatched predictions')
            timestamp = source_index / fps
            detected = [Detection(d['label'], float(d['score']), tuple(map(int, d['box']))) for d in detections]
            matrix, status = locator.locate(image, timestamp=timestamp, detections=detected) if locator else (None, 'no workpiece')
            result = FrameResult(image.shape[1], image.shape[0], timestamp, detected,
                                 workpiece_transform=matrix, tracking_status=status, workpiece_key=key)
            judged = [evaluate_step(step, result, roi_map) for step in sop.steps]
            new_events = [asdict(e) for e in engine.update(result)]
            events.extend(new_events)
            snapshot = engine.snapshot()
            corners = ((np.asarray(sop.workpiece.points) @ matrix[:, :2].T + matrix[:, 2]) *
                       [image.shape[1], image.shape[0]]).tolist() if matrix is not None and sop.workpiece else None
            rows.append({'sample_index': index, 'source_frame_index': source_index, 'timestamp': timestamp,
                         'pose_valid': matrix is not None, 'matrix': matrix.tolist() if matrix is not None else None,
                         'corners_px': corners, 'tracking_status': status,
                         'tracking_source': locator.metrics.get('source') if locator else None,
                         'steps': [{'step_index': i, 'completion_met': j.completion_met,
                                    'completion_blocked': j.completion_blocked,
                                    'forbidden_met': j.forbidden_met, 'forbidden_blocked': j.forbidden_blocked,
                                    'completion': checks(j.completion_results), 'forbidden': checks(j.forbidden_results)}
                                   for i, j in enumerate(judged)],
                         'reset': checks([evaluate_condition(c, result, roi_map) for c in sop.cycle.reset_conditions]),
                         'engine': {'phase': snapshot.phase.value, 'cycle': snapshot.cycle,
                                    'current_index': snapshot.current_index, 'cycle_ok': snapshot.cycle_ok,
                                    'progress': snapshot.progress,
                                    'step_statuses': [s.status.value for s in snapshot.steps]},
                         'events': new_events})
    finally:
        capture.release()
    kinds = Counter(e['kind'] for e in events)
    report = {'tool_version': 1, 'project_root': str(args.project_root.resolve()), 'strategy': args.strategy,
              'sop_sha256': hashlib.sha256(sop_bytes).hexdigest(),
              'predictions_sha256': hashlib.sha256(json.dumps(predictions, sort_keys=True).encode()).hexdigest(),
              'input_hashes': hashes, 'video_fps': fps, 'video_total_frames': total_frames,
              'frame_stride': args.frame_stride, 'sample_count': len(rows),
              'last_source_timestamp': rows[-1]['timestamp'],
              'pose_valid_frames': sum(r['pose_valid'] for r in rows),
              'event_counts': dict(kinds), 'events': events, 'frames': rows,
              'coverage': {'cycle_completed': bool(kinds['cycle_completed']),
                           'visited_step_indices': sorted({e['step_index'] for e in events if e['kind'] == 'step_started'}),
                           'sop_step_count': len(sop.steps),
                           'note': 'Sampled video only; no manual confirmation/skip or synthetic stop event. '
                                   'No complete cycle in this replay.' if not kinds['cycle_completed'] else
                                   'Sampled video produced a cycle-completed event; this is not a ground-truth correctness label.'},
              'limitations': 'No human-labelled pose ground truth. Same decoded images and frozen detector outputs; '
                             'CPU-only behavior regression, not a live latency benchmark or full production acceptance.'}
    if args.compare:
        report['comparison'] = compare_reports(json.loads(args.compare.read_text(encoding='utf-8')), report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    comparison = report.get('comparison')
    print(json.dumps({k: v for k, v in report.items() if k not in ('frames', 'events', 'input_hashes', 'comparison')}, ensure_ascii=True))
    if comparison:
        print(json.dumps({'pose_validity_changes': len(comparison['pose_validity_changes']),
                          'condition_changes': len(comparison['condition_changes']),
                          'engine_state_changes': len(comparison['engine_state_changes']),
                          'events_identical': comparison['events_identical'],
                          'corner_delta_px': comparison['corner_delta_px']}, ensure_ascii=True))
        if args.fail_on_decision_change and (comparison['condition_changes'] or comparison['engine_state_changes'] or
                                            not comparison['events_identical']):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
