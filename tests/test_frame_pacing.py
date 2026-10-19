"""A pacing cap controls starts, without catch-up or sleeping after slow work."""
import math
import unittest

from sop_app.frame_pacing import InferencePacer


class InferencePacerTests(unittest.TestCase):
    def test_unlimited_never_waits(self):
        pacer = InferencePacer()
        self.assertEqual(pacer.fps, 0)
        self.assertEqual(pacer.remaining(100.), 0.)
        pacer.mark_started(100.)
        self.assertEqual(pacer.remaining(100.), 0.)
        self.assertEqual(pacer.remaining(100.001), 0.)

    def test_first_frame_is_immediate_then_waits_until_next_start(self):
        pacer = InferencePacer(30)
        self.assertEqual(pacer.remaining(12.), 0.)
        pacer.mark_started(12.)
        self.assertAlmostEqual(pacer.remaining(12.), 1 / 30)
        self.assertAlmostEqual(pacer.remaining(12. + 1 / 60), 1 / 60)
        self.assertEqual(pacer.remaining(12. + 1 / 30), 0.)

    def test_slow_inference_needs_no_extra_period(self):
        pacer = InferencePacer(30)
        pacer.mark_started(1.)
        self.assertEqual(pacer.remaining(1.08), 0.)
        pacer.mark_started(1.08)
        self.assertAlmostEqual(pacer.remaining(1.08), 1 / 30)

    def test_late_wakeup_does_not_accumulate_burst_credit(self):
        pacer = InferencePacer(60)
        pacer.mark_started(1.)
        self.assertEqual(pacer.remaining(100.), 0.)
        pacer.mark_started(100.)
        self.assertAlmostEqual(pacer.remaining(100.), 1 / 60)
        self.assertGreater(pacer.remaining(100.001), 0.)

    def test_jitter_uses_actual_start_not_an_old_deadline(self):
        pacer = InferencePacer(30)
        start = 5.
        for _ in range(20):
            self.assertEqual(pacer.remaining(start), 0.)
            pacer.mark_started(start)
            self.assertAlmostEqual(pacer.remaining(start), 1 / 30)
            start += 1 / 30 + .005

    def test_limit_change_reuses_last_actual_start(self):
        pacer = InferencePacer(30)
        pacer.mark_started(1.)
        pacer.set_fps(10)
        self.assertAlmostEqual(pacer.remaining(1.05), .05)
        pacer.set_fps(60)
        self.assertEqual(pacer.remaining(1.05), 0.)
        pacer.set_fps(60)
        self.assertAlmostEqual(pacer.remaining(1.), 1 / 60)

    def test_turning_cap_on_uses_the_last_unlimited_start(self):
        pacer = InferencePacer(30)
        pacer.mark_started(1.)
        pacer.set_fps(0)
        self.assertEqual(pacer.remaining(1.001), 0.)
        pacer.mark_started(1.001)
        pacer.set_fps(60)
        self.assertAlmostEqual(pacer.remaining(1.001), 1 / 60)

    def test_reset_preserves_limit_and_releases_old_deadline(self):
        pacer = InferencePacer(1)
        pacer.mark_started(100.)
        pacer.reset()
        self.assertEqual(pacer.fps, 1)
        self.assertEqual(pacer.remaining(0.), 0.)
        pacer.mark_started(2.)
        self.assertAlmostEqual(pacer.remaining(2.), 1.)

    def test_fractional_limits_and_maximum_boundary(self):
        for fps in (1, 29.97, 240):
            with self.subTest(fps=fps):
                pacer = InferencePacer(fps)
                pacer.mark_started(5.)
                self.assertAlmostEqual(pacer.remaining(5.), 1 / fps)

    def test_invalid_limit_preserves_existing_schedule(self):
        pacer = InferencePacer(30)
        pacer.mark_started(5.)
        for value in (True, False, None, '', '30', -.1, .5, 241, 10**400, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pacer.set_fps(value)
            self.assertEqual(pacer.fps, 30)
            self.assertAlmostEqual(pacer.remaining(5.), 1 / 30)

    def test_invalid_clock_or_backwards_source_time_is_rejected(self):
        pacer = InferencePacer(30)
        pacer.mark_started(10.)
        for value in (9., math.nan, math.inf, -math.inf, 10**400, None, '10', True):
            for action in (pacer.remaining, pacer.mark_started):
                with self.subTest(value=value, action=action.__name__), self.assertRaises(ValueError):
                    action(value)
        self.assertAlmostEqual(pacer.remaining(10.), 1 / 30)


if __name__ == '__main__':
    unittest.main()
