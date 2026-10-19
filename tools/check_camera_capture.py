"""Check saved camera mode negotiation and capture arrivals across reconnects.

Running this CLI opens the selected camera; importing it and its statistics
helpers does not. No model, SOP engine, recorder, settings writes, or image
files are involved. The caller should wrap the process in a timeout because
native camera open/read/close calls may block inside a driver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from numbers import Real
from pathlib import Path
import sys
import time


WORKSPACE = Path(__file__).resolve().parents[1]


def percentiles(values):
    if not values:
        return None
    ordered = sorted(values)
    result = {}
    for name, fraction in [('p50', .5), ('p95', .95), ('max', 1.)]:
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        result[name] = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return result


class ArrivalStatistics:
    """Statistics of unique source timestamps, independent of polling speed."""

    def __init__(self):
        self._timestamps = set()
        self._dimensions = Counter()
        self._last_received = None
        self.duplicates = 0
        self.out_of_order = 0
        self.invalid_timestamps = 0

    def add(self, timestamp, width, height):
        if (isinstance(timestamp, bool) or not isinstance(timestamp, Real) or
                not math.isfinite(timestamp)):
            self.invalid_timestamps += 1
            return False
        stamp = float(timestamp)
        if stamp in self._timestamps:
            self.duplicates += 1
            return False
        if self._last_received is not None and stamp < self._last_received:
            self.out_of_order += 1
        self._last_received = stamp
        self._timestamps.add(stamp)
        self._dimensions[(int(width), int(height))] += 1
        return True

    def summary(self):
        ordered = sorted(self._timestamps)
        intervals = [(right - left) * 1000 for left, right in zip(ordered, ordered[1:])]
        span = ordered[-1] - ordered[0] if len(ordered) >= 2 else None
        return {'unique_frames': len(ordered),
                'arrival_fps': (len(ordered) - 1) / span if span is not None and span > 0 else None,
                'first_timestamp': ordered[0] if ordered else None,
                'last_timestamp': ordered[-1] if ordered else None,
                'first_to_last_seconds': span, 'interval_ms': percentiles(intervals),
                'frame_dimensions': [{'width': width, 'height': height, 'frames': count}
                                     for (width, height), count in sorted(self._dimensions.items())],
                'duplicate_timestamps': self.duplicates, 'out_of_order_timestamps': self.out_of_order,
                'invalid_timestamps': self.invalid_timestamps, 'unique_timestamps': ordered}


def camera_preferences_from_settings(settings, index):
    """Use the same canonical-first selection as MainWindow.

    A canonical profile, even an empty one, is complete. Never merge old values
    into it and revive cleared controls. CameraSource normalizes legacy aliases
    and automatic modes with saved_camera_changes.
    """
    if not isinstance(settings, dict):
        raise ValueError('Settings JSON must contain an object')
    key = f'camera:{index}'
    profiles = settings.get('camera_profiles', {})
    if isinstance(profiles, dict) and isinstance(profiles.get(key), dict):
        return copy.deepcopy(profiles[key])
    modes = settings.get('camera_modes', {})
    controls = settings.get('camera_controls', {})
    profile = copy.deepcopy(modes.get(key, {})) if isinstance(modes, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    if isinstance(controls, dict) and isinstance(controls.get(key), dict):
        profile['controls'] = copy.deepcopy(controls[key])
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--project-root', type=Path, default=WORKSPACE)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--preferences', type=Path, help='JSON camera preference object')
    source.add_argument('--settings', type=Path, help='Read canonical camera_profiles, else legacy camera_modes/camera_controls')
    parser.add_argument('--warmup', type=float, default=3.)
    parser.add_argument('--duration', type=float, default=8.)
    parser.add_argument('--reconnects', type=int, default=2, help='Number of fresh open/measure/close attempts')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.index < 0 or args.reconnects < 1 or not math.isfinite(args.warmup) or args.warmup < 0 or
            not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('index/warmup must be nonnegative; reconnects and duration must be positive')
    output = args.output.resolve()
    if any(output.is_relative_to(root.resolve() / 'data') for root in (WORKSPACE, args.project_root)):
        parser.error('output must be outside station data')
    if any(path is not None and output == path.resolve() for path in (args.preferences, args.settings)):
        parser.error('output cannot overwrite an input file')
    preferences = {}
    if args.preferences:
        preferences = json.loads(args.preferences.read_text(encoding='utf-8'))
        if not isinstance(preferences, dict):
            parser.error('preferences JSON must contain an object')
    elif args.settings:
        preferences = camera_preferences_from_settings(json.loads(args.settings.read_text(encoding='utf-8')), args.index)
    sys.path.insert(0, str(args.project_root.resolve()))
    # Import only after CLI validation. Constructing CameraSource below is the
    # sole hardware entry point; statistics tests never reach it.
    from sop_app.pipeline import CameraSource

    report = {'tool_version': 1, 'camera_index': args.index,
              'preferences_source': str((args.preferences or args.settings).resolve())
                  if args.preferences or args.settings else None,
              'preferences': preferences, 'warmup_seconds': args.warmup,
              'requested_measurement_seconds': args.duration, 'attempt_count': args.reconnects,
              'poll_sleep_seconds': .001, 'attempts': [],
              'notes': 'No images are stored. arrival_fps uses unique read-completion timestamps (N-1)/(last-first). '
                       'Latest-slot polling can miss frames under scheduling delays; compare source_capture_fps. '
                       'Neither metric proves unique sensor exposures or includes sensor/driver latency. '
                       'This tests capture only, without model/GUI/SOP processing.'}

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    for attempt in range(args.reconnects):
        row = {'attempt': attempt + 1, 'phase': 'opening', 'close_completed': False}
        report['attempts'].append(row)
        save()  # Preserve the last completed phase if a native driver call hangs.
        camera = None
        try:
            opened = time.monotonic()
            camera = CameraSource(args.index, preferences=copy.deepcopy(preferences))
            row['open_seconds'] = time.monotonic() - opened
            row['requested_changes'] = copy.deepcopy(getattr(camera, 'requested_changes', None))
            row['reported_controls'] = copy.deepcopy(getattr(camera, 'reported_controls', None))
            row['startup_messages'] = copy.deepcopy(getattr(camera, 'startup_messages', None))
            row['phase'] = 'warmup'
            save()
            warmup_deadline = time.monotonic() + args.warmup
            while time.monotonic() < warmup_deadline:
                camera.read()  # Discard warmup frames; do not preserve image data.
                time.sleep(.001)
            row['source_capture_fps_before_measurement'] = camera.capture_fps
            stats = ArrivalStatistics()
            polls, empty_polls, outside_window = 0, 0, 0
            began = time.monotonic()
            deadline = began + args.duration
            row['phase'] = 'measuring'
            while time.monotonic() < deadline:
                packet = camera.read()
                polls += 1
                if packet is None:
                    empty_polls += 1
                else:
                    frame, timestamp = packet
                    if isinstance(timestamp, Real) and math.isfinite(timestamp) and not began <= timestamp <= deadline:
                        outside_window += 1
                    else:
                        stats.add(timestamp, frame.shape[1], frame.shape[0])
                time.sleep(.001)
            row.update(measurement_wall_seconds=time.monotonic() - began,
                       polls=polls, empty_polls=empty_polls, frames_outside_measurement_window=outside_window,
                       source_capture_fps=camera.capture_fps, arrivals=stats.summary())
        except Exception as exc:
            row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            row['phase'] = 'closing' if camera is not None else 'open_failed'
            save()
            if camera is not None:
                try:
                    camera.close()
                    row['close_completed'] = True
                except Exception as exc:
                    row['close_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            row['phase'] = 'failed' if 'error' in row or 'close_error' in row else 'complete'
            save()
    print(json.dumps({'output': str(output), 'attempts': [
        {'attempt': row['attempt'], 'phase': row['phase'],
         'arrival_fps': row.get('arrivals', {}).get('arrival_fps'),
         'source_capture_fps': row.get('source_capture_fps'), 'error': row.get('error')}
        for row in report['attempts']]}, ensure_ascii=True))
    if any(row['phase'] != 'complete' for row in report['attempts']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
