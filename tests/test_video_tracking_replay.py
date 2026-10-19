"""Replay comparison must detect semantic changes and reject unequal inputs."""
import copy
import unittest

from tools.check_video_tracking_replay import compare_reports


def report():
    check = {'met': True, 'count': 1, 'error_kind': '', 'error': '', 'detail': 'one object'}
    step = {'completion_met': True, 'completion_blocked': False,
            'forbidden_met': False, 'forbidden_blocked': False,
            'completion': [check], 'forbidden': []}
    return {'sop_sha256': 'sop', 'predictions_sha256': 'predictions', 'input_hashes': ['image'],
            'events': [], 'frames': [{'sample_index': 0, 'source_frame_index': 0, 'timestamp': 0.,
                                     'pose_valid': True, 'corners_px': [[0, 0], [1, 0], [1, 1], [0, 1]],
                                     'steps': [step], 'reset': [], 'engine': {'phase': 'running'}}]}


class VideoReplayComparisonTests(unittest.TestCase):
    def test_same_inputs_and_decisions_have_no_changes(self):
        before = report()
        result = compare_reports(before, copy.deepcopy(before))
        self.assertFalse(result['condition_changes'])
        self.assertFalse(result['pose_validity_changes'])
        self.assertTrue(result['events_identical'])
        self.assertEqual(result['corner_delta_px']['max'], 0.)

    def test_different_hashes_or_timestamps_cannot_be_compared(self):
        for change in ('sop_sha256', 'predictions_sha256', 'input_hashes', 'timestamp'):
            before, after = report(), report()
            if change == 'timestamp':
                after['frames'][0]['timestamp'] = 1.
            else:
                after[change] = ['different'] if change == 'input_hashes' else 'different'
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_reports(before, after)

    def test_counts_and_aggregate_decisions_are_not_hidden(self):
        for field in ('count', 'completion_met'):
            before, after = report(), report()
            if field == 'count':
                after['frames'][0]['steps'][0]['completion'][0]['count'] = 2
            else:
                after['frames'][0]['steps'][0]['completion_met'] = False
            self.assertEqual(len(compare_reports(before, after)['condition_changes']), 1)

    def test_visual_pose_difference_is_separate_from_decision_difference(self):
        before, after = report(), report()
        after['frames'][0]['corners_px'] = [[3, 4], [4, 4], [4, 5], [3, 5]]
        result = compare_reports(before, after)
        self.assertFalse(result['condition_changes'])
        self.assertEqual(result['corner_delta_px']['max'], 5.)

    def test_no_common_valid_pose_does_not_claim_zero_error(self):
        before, after = report(), report()
        for value in (before, after):
            value['frames'][0]['pose_valid'] = False
            value['frames'][0]['corners_px'] = None
        result = compare_reports(before, after)
        self.assertIsNone(result['corner_delta_px'])
        self.assertEqual(result['common_valid_pose_frames'], 0)


if __name__ == '__main__':
    unittest.main()
