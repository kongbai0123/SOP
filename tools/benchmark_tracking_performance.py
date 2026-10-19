"""Reproducible CPU-only tracking replay; no camera, model, or station data.

Run this *same script* against two checkouts with --project-root. Fixtures are
generated before importing either checkout and their SHA-256 is included in the
report. Source timestamps advance at 30 Hz, but calls run back-to-back: recovery
seconds describe the fixture timeline, not live camera latency or an FPS promise.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np


SEED = 472
POINTS = np.array([(.25, .25), (.7, .25), (.7, .72), (.25, .72)])


def fixtures(width=1280, height=720, *, reference_image=None, reference_points=None):
    rng = np.random.default_rng(SEED)
    points = np.asarray(reference_points if reference_points is not None else POINTS, np.float64)
    if reference_image is None:
        reference = np.zeros((height, width, 3), np.uint8)
        for _ in range(750):
            x, y = rng.integers([int(width * .265), int(height * .27)],
                                [int(width * .68), int(height * .70)])
            cv2.circle(reference, (int(x), int(y)), int(rng.integers(2, 7)),
                       tuple(map(int, rng.integers(70, 255, 3))), -1)
    else:
        reference = reference_image.copy()
        height, width = reference.shape[:2]
    noise = rng.integers(0, 256, reference.shape, np.uint8)
    blank = np.zeros_like(reference)
    identity = np.float64([[1, 0, 0], [0, 1, 0]])

    def view(angle=0., scale=1., dx=0., dy=0., distractor=False):
        affine = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        affine[:, 2] += [dx, dy]
        image = cv2.warpAffine(reference, affine, (width, height))
        if distractor:
            image[:int(height * .2), :int(width * .2)] = noise[:int(height * .2), :int(width * .2)]
        return image, np.diag([1 / width, 1 / height]) @ affine @ np.diag([width, height, 1])

    cases = {}
    cases['stable_motion'] = [(*view(dx=i * .2, dy=i * .1, angle=i * .03), 'motion', False)
                              for i in range(40)]
    cases['occlusion_reappear'] = [(reference, identity, 'initial', False)] * 8
    cases['occlusion_reappear'] += [(blank, None, 'occluded', False)] * 8
    cases['occlusion_reappear'] += [(*view(dx=35, dy=18), 'reappear', False)] * 12
    cases['noise_reappear'] = [(reference, identity, 'initial', False)] * 8
    cases['noise_reappear'] += [(noise, None, 'unrelated_texture', False)] * 12
    cases['noise_reappear'] += [(*view(angle=8, dx=50, dy=20), 'reappear', False)] * 12
    cases['large_move'] = [(reference, identity, 'initial', False)] * 8
    cases['large_move'] += [(*view(angle=28, scale=.85, dx=width * .12, dy=height * .06),
                            'large_move', False)] * 16
    cases['wrong_detector_distractor'] = [(reference, identity, 'initial', False)] * 8
    cases['wrong_detector_distractor'] += [(*view(dx=60, dy=25, distractor=True),
                                           'distractor_wrong_hint', True)] * 16
    cases['guided_reappear'] = [(reference, identity, 'initial', False)] * 8
    cases['guided_reappear'] += [(blank, None, 'occluded', False)] * 6
    cases['guided_reappear'] += [(*view(angle=10, dx=55, dy=20), 'guided_reappear', 'correct')] * 12
    digest = hashlib.sha256(reference.tobytes())
    digest.update(points.tobytes())
    for name, rows in cases.items():
        digest.update(name.encode())
        for image, matrix, phase, hint in rows:
            digest.update(image.tobytes())
            digest.update(json.dumps([matrix.tolist() if matrix is not None else None, phase, hint]).encode())
    return reference, cases, digest.hexdigest()


def distribution(values):
    if not values:
        return None
    return {name: round(float(np.percentile(values, percentile)), 4)
            for name, percentile in [('p50', 50), ('p95', 95), ('p99', 99), ('max', 100)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--strategy', choices=['fixed', 'adaptive'], default='adaptive')
    parser.add_argument('--reference-sop', type=Path,
                        help='Read only this SOP reference photo and points; apply artificial known whole-image transforms')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--opencv-threads', type=int, default=1)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--max-corner-error-px', type=float, default=8.)
    parser.add_argument('--include-samples', action='store_true')
    args = parser.parse_args()
    if args.repeat < 1 or min(args.width, args.height) < 240:
        parser.error('repeat must be positive and fixture dimensions must be at least 240')
    cv2.setNumThreads(args.opencv_threads)
    cv2.setRNGSeed(SEED)
    points = POINTS.copy()
    reference_image = None
    reference_sop_hash = None
    if args.reference_sop:
        raw_sop = args.reference_sop.read_bytes()
        reference_sop_hash = hashlib.sha256(raw_sop).hexdigest()
        raw_workpiece = json.loads(raw_sop).get('workpiece')
        if not raw_workpiece:
            parser.error('--reference-sop has no workpiece reference')
        points = np.asarray(raw_workpiece['points'], np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            parser.error('--reference-sop workpiece points must be four normalized finite coordinates')
        png = base64.b64decode(raw_workpiece['reference_png'], validate=True)
        reference_image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if reference_image is None:
            parser.error('--reference-sop reference image cannot be decoded')
    reference, cases, fixture_hash = fixtures(args.width, args.height,
                                            reference_image=reference_image, reference_points=points)
    args.height, args.width = reference.shape[:2]
    if args.output and args.reference_sop and args.output.resolve() == args.reference_sop.resolve():
        parser.error('output cannot overwrite reference SOP')
    if args.output and any(args.output.resolve().is_relative_to(root / 'data')
                           for root in (Path(__file__).resolve().parents[1], args.project_root.resolve())):
        parser.error('benchmark output must be outside station data')
    sys.path.insert(0, str(args.project_root.resolve()))
    from sop_app.tracking import WorkpieceLocator
    from sop_app.sop_schema import Workpiece, ROI, Condition
    from sop_app.detection import Detection, FrameResult
    from sop_app.conditions import evaluate_condition

    _, encoded = cv2.imencode('.png', reference)
    workpiece = Workpiece(reference_png=base64.b64encode(encoded).decode('ascii'), points=points.tolist())
    roi = ROI('fixture_workpiece', points.tolist(), 'workpiece')
    absent_condition = Condition('disappear', 'tool', roi.name)
    report = {'fixture_version': 2, 'seed': SEED, 'fixture_sha256': fixture_hash,
              'reference_kind': 'SOP photo with artificial whole-image transforms' if args.reference_sop else 'synthetic dots',
              'reference_sop_sha256': reference_sop_hash, 'reference_points': points.tolist(),
              'project_root': str(args.project_root.resolve()), 'strategy': args.strategy,
              'opencv_version': cv2.__version__, 'opencv_threads': cv2.getNumThreads(),
              'shape': [args.width, args.height], 'repeat': args.repeat,
              'corner_tolerance_px': args.max_corner_error_px,
              'timing_semantics': 'back-to-back CPU calls, source timestamps 30Hz, no camera/model/UI; not live FPS',
              'quality_scope': 'Known artificial affine ground truth, blackout and unrelated-noise absence. '
                               'Photo mode moves the entire reference scene including background; '
                               'it is not a real workcell-video or complete SOP acceptance test.',
              'cases': {}}
    for name, rows in cases.items():
        samples, setup_ms, recovery = [], [], []
        for repeat in range(args.repeat):
            cv2.setRNGSeed(SEED)
            began = time.perf_counter()
            locator = WorkpieceLocator(workpiece, adaptive_budget=args.strategy == 'adaptive')
            setup_ms.append((time.perf_counter() - began) * 1000)
            reappeared_at = None
            was_absent = False
            for index, (image, truth, phase, hint) in enumerate(rows):
                timestamp = index / 30.
                hints = []
                if hint == 'correct':
                    projected = (points @ truth[:, :2].T + truth[:, 2]) * [args.width, args.height]
                    low, high = projected.min(axis=0), projected.max(axis=0)
                    hints = [Detection('handlebar', .95, tuple(map(int, [*low, *high])))]
                elif hint:
                    hints = [Detection('handlebar', .95, (0, 0, args.width // 6, args.height // 6))]
                call_started = time.perf_counter()
                matrix, _ = locator.locate(image, timestamp=timestamp, detections=hints)
                call_ms = (time.perf_counter() - call_started) * 1000
                error = None
                if matrix is not None and truth is not None:
                    expected = (points @ truth[:, :2].T + truth[:, 2]) * [args.width, args.height]
                    actual = (points @ matrix[:, :2].T + matrix[:, 2]) * [args.width, args.height]
                    error = float(np.max(np.linalg.norm(actual - expected, axis=1)))
                if truth is None:
                    was_absent = True
                elif was_absent and reappeared_at is None:
                    reappeared_at = index
                if reappeared_at is not None and error is not None and error <= args.max_corner_error_px:
                    recovery.append({'repeat': repeat, 'frames': index - reappeared_at,
                                     'source_seconds': round((index - reappeared_at) / 30., 4)})
                    was_absent, reappeared_at = False, None
                unsafe = False
                if truth is None:
                    result = FrameResult(args.width, args.height, timestamp, [], matrix)
                    unsafe = evaluate_condition(absent_condition, result, {roi.name: roi}).met
                metrics = locator.metrics
                samples.append({'repeat': repeat, 'index': index, 'phase': phase,
                                'expected_visible': truth is not None, 'valid': matrix is not None,
                                'max_corner_error_px': error, 'unsafe_absence_decision': unsafe,
                                # Measure the same call boundary across old/new
                                # versions, including filtering and bookkeeping.
                                'tracking_ms': call_ms, 'reported_tracking_ms': metrics['tracking_ms'],
                                'source': metrics.get('source'),
                                'stages_ms': metrics.get('stages_ms', {}),
                                'components_ms': metrics.get('components_ms', {}),
                                'skipped': metrics.get('skipped', [])})
            if was_absent:
                recovery.append({'repeat': repeat, 'frames': None, 'source_seconds': None})
        stages, components, phases = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in samples:
            phases[row['phase']].append(row['tracking_ms'])
            for stage, elapsed in row['stages_ms'].items():
                stages[stage].append(elapsed)
            for component, elapsed in row['components_ms'].items():
                components[component].append(elapsed)
        summary = {'samples': len(samples), 'setup_ms': distribution(setup_ms),
                   'tracking_ms': distribution([r['tracking_ms'] for r in samples]),
                   'phase_ms': {p: distribution(times) for p, times in phases.items()},
                   'stages_ms': {p: distribution(times) for p, times in stages.items()},
                   'components_ms': {p: distribution(times) for p, times in components.items()},
                   'stage_calls': {p: len(times) for p, times in stages.items()},
                   'sources': dict(Counter(r['source'] for r in samples)),
                   'skips': dict(Counter(s for r in samples for s in r['skipped'])),
                   'visible_frames': sum(r['expected_visible'] for r in samples),
                   'missed_visible': sum(r['expected_visible'] and not r['valid'] for r in samples),
                   'false_valid_absent': sum(not r['expected_visible'] and r['valid'] for r in samples),
                   'wrong_pose': sum(r['max_corner_error_px'] is not None and r['max_corner_error_px'] > args.max_corner_error_px for r in samples),
                   'unsafe_absence_decisions': sum(r['unsafe_absence_decision'] for r in samples),
                   'corner_error_px': distribution([r['max_corner_error_px'] for r in samples if r['max_corner_error_px'] is not None]),
                   'recovery': recovery}
        if args.include_samples:
            summary['frames'] = samples
        report['cases'][name] = summary
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)


if __name__ == '__main__':
    main()
