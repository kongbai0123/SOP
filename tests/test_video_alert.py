"""Rendered trial alerts remain separate from image data and locator status."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from sop_app.detection import Detection
from sop_app.sop_schema import ROI
from sop_app.ui.video_widget import VideoWidget


class VideoAlertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def video(self, width=400, height=240):
        video = VideoWidget()
        video.resize(width, height)
        video.set_frame(np.zeros((height, width, 3), np.uint8))
        self.addCleanup(video.close)
        return video

    def assert_red(self, image, x, y):
        color = image.pixelColor(x, y)
        self.assertGreater(color.red(), 150)
        self.assertLess(color.green(), 110)
        self.assertLess(color.blue(), 110)

    def test_alert_banner_covers_detection_labels_and_clears(self):
        video = self.video()
        video.set_frame(np.zeros((240, 400, 3), np.uint8),
                        [Detection("hand", .9, (0, 10, 400, 220))], ["hand"])
        before = video.grab().toImage()
        video.set_alert("Violation: hand in C2", ["C2"])
        alerted = video.grab().toImage()
        self.assert_red(alerted, 2, 15)
        self.assertNotEqual(before, alerted)
        video.set_alert("")
        self.assertEqual(video.grab().toImage(), before)

    def test_hidden_regions_only_reveal_current_alert_region(self):
        video = self.video()
        video.set_rois([
            ROI("C2", [(.1, .4), (.4, .4), (.4, .9), (.1, .9)]),
            ROI("C3", [(.6, .4), (.9, .4), (.9, .9), (.6, .9)]),
        ])
        video.set_regions_visible(False)
        hidden = video.grab().toImage()
        video.set_alert("Violation", ["C2"])
        alerted = video.grab().toImage()
        self.assert_red(alerted, 40, 170)
        fill = alerted.pixelColor(100, 170)
        self.assertGreater(fill.red(), fill.green())
        self.assertEqual(alerted.pixelColor(240, 170), hidden.pixelColor(240, 170))
        self.assertFalse(video.regions_visible)
        video.set_alert("Violation", ["C3"])
        switched = video.grab().toImage()
        self.assert_red(switched, 240, 170)
        self.assertEqual(switched.pixelColor(40, 170), hidden.pixelColor(40, 170))
        video.set_alert("")
        self.assertEqual(video.grab().toImage(), hidden)

    def test_empty_alert_does_not_reveal_regions_or_leave_red_effect(self):
        video = self.video()
        video.set_rois([ROI("C2", [(.1, .4), (.4, .4), (.4, .9), (.1, .9)])])
        video.set_regions_visible(False)
        before = video.grab().toImage()
        video.set_alert("", ["C2"])
        self.assertEqual(video.grab().toImage(), before)
        video.set_alert("  ", ["C2"])
        self.assertEqual(video.grab().toImage(), before)

    def test_locator_banner_is_restored_with_latest_status(self):
        video = self.video()
        video.set_banner("Tracking ready", "#1565c0", single_line=True)
        video.set_alert("Violation", ["C2"])
        video.set_banner("Tracking lost", "#ff9800", single_line=True)
        self.assert_red(video.grab().toImage(), 2, 15)
        video.set_alert("")
        restored = video.grab().toImage()
        reference = self.video()
        reference.set_banner("Tracking lost", "#ff9800", single_line=True)
        self.assertEqual(restored, reference.grab().toImage())
        self.assertEqual(video.toolTip(), "Tracking lost")

    def test_selected_edit_handles_remain_and_original_style_returns(self):
        video = self.video()
        video.set_rois([ROI("C2", [(.1, .4), (.4, .4), (.4, .9), (.1, .9)])],
                       selected="C2", stale=["C2"])
        video.set_editable(["C2"])
        before = video.grab().toImage()
        video.set_alert("Violation", ["C2"])
        alerted = video.grab().toImage()
        self.assert_red(alerted, 40, 170)
        handle = alerted.pixelColor(160, 216)
        self.assertEqual((handle.red(), handle.green(), handle.blue()), (255, 255, 255))
        video.set_alert("")
        self.assertEqual(video.grab().toImage(), before)

    def test_alert_does_not_mutate_frame_roi_points_or_image_geometry(self):
        video = self.video()
        source = np.full((180, 320, 3), (12, 23, 34), np.uint8)
        original_source = source.copy()
        roi = ROI("C2", [(.1, .4), (.4, .4), (.4, .9), (.1, .9)])
        original_points = deepcopy(roi.points)
        video.set_frame(source)
        video.set_rois([roi])
        original_image = video._image.copy()
        original_rect = video._image_rect()
        original_size = video.size()
        video.set_alert("Violation", ["C2"])
        video.grab()
        np.testing.assert_array_equal(source, original_source)
        self.assertEqual(video._image, original_image)
        self.assertEqual(roi.points, original_points)
        self.assertEqual(video._image_rect(), original_rect)
        self.assertEqual(video.size(), original_size)

    def test_long_alert_wraps_at_small_video_width(self):
        video = self.video(320, 180)
        video.set_alert("Violation")
        short = video.grab().toImage()
        video.set_alert("Violation: hand in C2 during left-hand installation; "
                        "remove the hand before restarting the trial.")
        wrapped = video.grab().toImage()
        short_rows = sum(short.pixelColor(2, y).red() > 150 for y in range(180))
        wrapped_rows = sum(wrapped.pixelColor(2, y).red() > 150 for y in range(180))
        self.assertGreater(wrapped_rows, short_rows)
        self.assertLessEqual(wrapped_rows, 180)

    def test_alert_also_shows_without_a_frame(self):
        video = self.video()
        video.clear_frame("No frame")
        before = video.grab().toImage()
        video.set_alert("Trial stopped: violation")
        self.assert_red(video.grab().toImage(), 2, 15)
        video.set_alert("")
        self.assertEqual(video.grab().toImage(), before)


if __name__ == "__main__":
    unittest.main()
