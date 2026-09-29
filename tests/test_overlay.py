import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from sop_app.detection import Detection
from sop_app.overlay import draw_detections, class_color
from sop_app.sop_schema import ROI
from sop_app.ui.video_widget import VideoWidget


class OverlayTests(unittest.TestCase):
    def test_later_mask_does_not_cover_earlier_box(self):
        frame = np.zeros((200, 320, 3), np.uint8)
        mask = np.ones((200, 320), bool)
        detections = [Detection('a', .9, (40, 60, 240, 160)),
                      Detection('b', .9, (0, 0, 320, 200), mask)]
        painted = draw_detections(frame, detections, ['a', 'b'])
        np.testing.assert_array_equal(painted[100, 40], class_color('a', ['a', 'b']))

    def test_detection_box_is_above_region_and_stale_tag(self):
        app = QApplication.instance() or QApplication([])
        video = VideoWidget()
        video.resize(320, 200)
        video.set_frame(np.zeros((200, 320, 3), np.uint8),
                        [Detection('a', .9, (40, 60, 240, 160))], ['a'])
        video.set_rois([ROI('installation', [(.1,.25),(.8,.25),(.8,.85),(.1,.85)])],
                       stale=['installation'])
        rendered = video.grab().toImage()
        color = rendered.pixelColor(40, 65)
        b, g, r = class_color('a', ['a'])
        self.assertEqual((color.red(), color.green(), color.blue()), (r, g, b))
        video.close()


class MonitorDetailsTests(unittest.TestCase):
    def test_details_hidden_by_default_and_toggleable(self):
        from sop_app.ui.run_page import RunPage
        app = QApplication.instance() or QApplication([])
        page = RunPage()
        self.assertTrue(page.details_panel.isHidden())
        self.assertEqual(page.condition_label.parentWidget(), page.details_panel)
        self.assertEqual(page.tracking_label.parentWidget(), page.details_panel)
        page.details_button.click()
        self.assertFalse(page.details_panel.isHidden())
        page.details_button.click()
        self.assertTrue(page.details_panel.isHidden())
        page.close()
