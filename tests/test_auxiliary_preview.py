"""Camera selection and handoff without using physical devices."""
from __future__ import annotations

from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sop_app.auxiliary_preview import AuxiliaryPreview


def wait_for(predicate, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.005)
    return predicate()


class AuxiliaryPreviewTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock(CAP_DSHOW=700, INTER_AREA=3)
        patcher = patch("sop_app.auxiliary_preview.cv2", self.backend)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.preview = AuxiliaryPreview()
        self.addCleanup(self.preview.close)

    def test_selected_camera_does_not_depend_on_another_camera_opening(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = True, frame
        missing = Mock()
        missing.isOpened.return_value = False
        self.backend.VideoCapture.side_effect = lambda index, _: {1: camera, 4: missing}[index]
        self.backend.resize.return_value = np.zeros((360, 640, 3), dtype=np.uint8)
        self.backend.VideoCapture.assert_not_called()

        self.preview.set_cameras([1, 4])
        self.assertTrue(wait_for(lambda: self.preview.snapshot()[1].image is not None))
        self.assertTrue(wait_for(lambda: self.preview.is_released(4)))
        self.assertEqual(self.preview.snapshot()[4].status, "無法開啟攝影機")
        self.assertEqual(self.preview.snapshot()[1].image.shape, (360, 640, 3))
        self.backend.resize.assert_called_with(frame, (640, 360), interpolation=3)
        self.assertCountEqual([call.args for call in self.backend.VideoCapture.call_args_list],
                              [(1, self.backend.CAP_DSHOW), (4, self.backend.CAP_DSHOW)])
        self.preview.close()
        self.assertEqual(self.preview.snapshot(), {})
        self.assertTrue(wait_for(lambda: self.preview.is_released(1)))
        camera.release.assert_called_once()
        missing.release.assert_called_once()

    def test_rapid_reselection_waits_until_old_capture_is_released(self):
        started, allow_read_to_finish = Event(), Event()
        self.addCleanup(allow_read_to_finish.set)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        def read():
            started.set()
            allow_read_to_finish.wait(timeout=3.0)
            return True, frame

        before, after = Mock(), Mock()
        before.isOpened.return_value = after.isOpened.return_value = True
        before.read.side_effect = read
        after.read.return_value = (True, frame)
        self.backend.VideoCapture.side_effect = (before, after)
        self.preview.set_cameras([2])
        self.assertTrue(started.wait(timeout=2.0))

        self.preview.set_cameras([])
        self.assertNotIn(2, self.preview.snapshot())
        self.assertFalse(self.preview.is_released(2))
        self.preview.set_cameras([2])
        self.assertEqual(self.preview.snapshot()[2].status, "連線中…")
        self.backend.VideoCapture.assert_called_once()

        allow_read_to_finish.set()
        self.assertTrue(wait_for(lambda: self.preview.snapshot()[2].image is not None))
        before.release.assert_called_once()
        self.assertEqual(self.backend.VideoCapture.call_count, 2)
        self.preview.set_cameras([])
        self.assertTrue(wait_for(lambda: self.preview.is_released(2)))
        after.release.assert_called_once()

    def test_stalled_camera_discards_a_previous_frame_and_releases_device(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        started = False
        camera = Mock()
        camera.isOpened.return_value = True

        def read():
            nonlocal started
            if not started:
                started = True
                return True, frame
            return False, None

        camera.read.side_effect = read
        self.backend.VideoCapture.return_value = camera
        self.preview.FRAME_INTERVAL = .01
        self.preview.NO_FRAME_TIMEOUT = .05
        self.preview.set_cameras([0])
        self.assertTrue(wait_for(lambda: self.preview.is_released(0)))
        self.assertIsNone(self.preview.snapshot()[0].image)
        self.assertEqual(self.preview.snapshot()[0].status, "無法讀取攝影機影像")
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
