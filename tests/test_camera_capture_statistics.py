"""Pure capture-statistics/preferences tests: no OpenCV or camera imports."""
import copy
import math
import unittest

from tools.check_camera_capture import ArrivalStatistics, camera_preferences_from_settings


class CaptureArrivalStatisticsTests(unittest.TestCase):
    def test_fps_uses_unique_timestamps_first_to_last_not_poll_count(self):
        stats = ArrivalStatistics()
        for i in range(121):
            stats.add(100. + i / 60., 1920, 1080)
        result = stats.summary()
        self.assertEqual(result['unique_frames'], 121)
        self.assertAlmostEqual(result['arrival_fps'], 60.)
        self.assertAlmostEqual(result['interval_ms']['p50'], 1000 / 60)
        self.assertAlmostEqual(result['interval_ms']['p95'], 1000 / 60)
        self.assertEqual(result['frame_dimensions'], [{'width': 1920, 'height': 1080, 'frames': 121}])

    def test_duplicates_do_not_inflate_frame_count_or_rate(self):
        stats = ArrivalStatistics()
        for timestamp in (0., 0., .02, .02, .04):
            stats.add(timestamp, 640, 480)
        result = stats.summary()
        self.assertEqual(result['unique_frames'], 3)
        self.assertEqual(result['duplicate_timestamps'], 2)
        self.assertAlmostEqual(result['arrival_fps'], 50.)

    def test_empty_and_single_frame_do_not_claim_zero_fps_or_intervals(self):
        stats = ArrivalStatistics()
        self.assertIsNone(stats.summary()['arrival_fps'])
        self.assertIsNone(stats.summary()['interval_ms'])
        stats.add(10., 640, 480)
        self.assertIsNone(stats.summary()['arrival_fps'])
        self.assertIsNone(stats.summary()['interval_ms'])

    def test_interval_percentiles_and_dimension_changes_are_recorded(self):
        stats = ArrivalStatistics()
        for timestamp, size in [(0., (640, 480)), (.01, (640, 480)), (.03, (1280, 720)), (.06, (1280, 720))]:
            stats.add(timestamp, *size)
        result = stats.summary()
        self.assertAlmostEqual(result['interval_ms']['p50'], 20.)
        self.assertAlmostEqual(result['interval_ms']['p95'], 29.)
        self.assertEqual([r['frames'] for r in result['frame_dimensions']], [2, 2])

    def test_invalid_and_out_of_order_samples_are_visible_in_report(self):
        stats = ArrivalStatistics()
        for timestamp in (math.nan, math.inf, -math.inf, True, '2', None):
            self.assertFalse(stats.add(timestamp, 640, 480))
        for timestamp in (1., 1.04, 1.02):
            stats.add(timestamp, 640, 480)
        result = stats.summary()
        self.assertEqual(result['invalid_timestamps'], 6)
        self.assertEqual(result['out_of_order_timestamps'], 1)
        self.assertAlmostEqual(result['arrival_fps'], 50.)


class CapturePreferencesTests(unittest.TestCase):
    def test_selected_camera_profile_merges_legacy_controls_without_mutation(self):
        settings = {'camera_modes': {'camera:0': {'fps': 60, 'width': 1920, 'height': 1080,
                    'pixel_format': 'MJPG'}},
                    'camera_controls': {'camera:0': {'exposure': {'auto': True, 'value': -6},
                                        'brightness': {'auto': False, 'value': 128}},
                                        'camera:1': {'gain': 99}}}
        original = copy.deepcopy(settings)
        preferences = camera_preferences_from_settings(settings, 0)
        self.assertEqual(preferences['fps'], 60)
        self.assertEqual(preferences['controls']['brightness']['value'], 128)
        self.assertEqual(preferences['controls']['exposure'], {'auto': True, 'value': -6})
        self.assertNotIn('gain', preferences['controls'])
        self.assertEqual(settings, original)

    def test_absent_camera_does_not_inherit_another_camera(self):
        settings = {'camera_modes': {'camera:0': {'fps': 60}},
                    'camera_controls': {'camera:0': {'gain': 4}}}
        self.assertEqual(camera_preferences_from_settings(settings, 1), {})

    def test_non_object_settings_fail_and_malformed_legacy_sections_are_ignored(self):
        with self.assertRaises(ValueError):
            camera_preferences_from_settings([], 0)
        for settings in ({'camera_modes': []}, {'camera_controls': []},
                         {'camera_modes': {'camera:0': []}}):
            self.assertEqual(camera_preferences_from_settings(settings, 0), {})

    def test_canonical_profile_is_independent_and_never_merges_legacy_controls(self):
        canonical = {'fps': 30, 'controls': {'brightness': {'mode': 'manual', 'value': 42}}}
        settings = {'camera_profiles': {'camera:0': canonical},
                    'camera_modes': {'camera:0': {'fps': 60, 'pixel_format': 'MJPG'}},
                    'camera_controls': {'camera:0': {'exposure': {'auto': True, 'value': -6}}}}
        preferences = camera_preferences_from_settings(settings, 0)
        self.assertEqual(preferences, canonical)
        preferences['controls']['brightness']['value'] = 12
        self.assertEqual(canonical['controls']['brightness']['value'], 42)

    def test_empty_canonical_profile_clears_legacy_choices(self):
        settings = {'camera_profiles': {'camera:0': {}},
                    'camera_modes': {'camera:0': {'fps': 60}},
                    'camera_controls': {'camera:0': {'gain': 8}}}
        self.assertEqual(camera_preferences_from_settings(settings, 0), {})


if __name__ == '__main__':
    unittest.main()
