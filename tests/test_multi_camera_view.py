"""Auxiliary video tiles display only their assigned camera."""
import os
import unittest

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sop_app.ui.multi_camera_view import AuxiliaryCameraGrid, CameraPreviewPane
from sop_app.ui.video_widget import VideoWidget


class CameraPreviewPaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def pane(self):
        primary = VideoWidget()
        pane = CameraPreviewPane(primary)
        self.addCleanup(pane.close)
        pane.resize(960, 640)
        pane.show()
        pane.set_devices([(1, "副攝影機")])
        self.app.processEvents()
        return pane, primary

    def test_split_direction_changes_geometry_but_keeps_both_frame_aspects(self):
        pane, primary = self.pane()
        auxiliary = pane.auxiliary._tiles[1].video
        primary.set_frame(np.zeros((90, 160, 3), dtype=np.uint8))
        pane.auxiliary.set_frame(1, np.zeros((90, 120, 3), dtype=np.uint8))

        self.assertAlmostEqual(primary.width(), pane.auxiliary.width(), delta=1)
        self.assertEqual(primary.y(), pane.auxiliary.y())
        pane.set_split_layout("vertical", .6)
        self.app.processEvents()
        self.assertLess(primary.y(), pane.auxiliary.y())
        self.assertEqual(primary.width(), pane.auxiliary.width())
        self.assertAlmostEqual(primary.height() / sum(pane._splitter.sizes()), .6, delta=.01)
        self.assertAlmostEqual(primary._image_rect().width() / primary._image_rect().height(), 16 / 9)
        self.assertAlmostEqual(auxiliary._image_rect().width() / auxiliary._image_rect().height(), 4 / 3)

    def test_handle_ratio_survives_resizing_and_auxiliary_close(self):
        pane, primary = self.pane()
        ratios = []
        pane.split_ratio_changed.connect(ratios.append)
        handle = pane._splitter.handle(1)
        self.assertTrue(handle.isVisible())
        initial = primary.width()
        QTest.keyClick(handle, Qt.Key.Key_Right)
        self.assertGreater(primary.width(), initial)
        self.assertEqual(len(ratios), 1)

        pane.set_devices([])
        self.app.processEvents()
        self.assertTrue(handle.isHidden())
        pane.resize(1250, 650)
        pane.set_devices([(1, "副攝影機")])
        self.app.processEvents()
        self.assertTrue(handle.isVisible())
        self.assertEqual(len(ratios), 1)
        self.assertAlmostEqual(primary.width() / sum(pane._splitter.sizes()), ratios[0], delta=.01)

    def test_minimum_preview_height_respects_vertical_children(self):
        pane, primary = self.pane()
        pane.set_split_layout("vertical", .5)
        self.assertGreater(pane.minimum_preview_height, primary.minimumHeight())
        self.assertGreaterEqual(pane.minimum_preview_height, pane.minimumSizeHint().height())
        pane.set_devices([])
        self.assertEqual(pane.minimum_preview_height, primary.minimumHeight())

    def test_a_narrow_window_does_not_replace_the_requested_ratio(self):
        pane, primary = self.pane()
        ratios = []
        pane.split_ratio_changed.connect(ratios.append)
        pane.set_split_layout("horizontal", .25)
        pane.resize(500, 360)
        self.app.processEvents()
        self.assertGreaterEqual(primary.width(), primary.minimumWidth())

        pane.resize(1500, 600)
        self.app.processEvents()
        self.assertAlmostEqual(primary.width() / sum(pane._splitter.sizes()), .25, delta=.01)
        self.assertEqual(ratios, [])


class AuxiliaryCameraGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def grid(self):
        grid = AuxiliaryCameraGrid()
        self.addCleanup(grid.close)
        grid.resize(640, 210)
        grid.set_devices([(2, "左側攝影機"), (5, "右側攝影機")])
        grid.show()
        self.app.processEvents()
        return grid

    def test_tiles_reflow_and_respect_device_order(self):
        grid = self.grid()
        left, right = grid._tiles.values()
        self.assertLess(left.x(), right.x())
        self.assertEqual(left.y(), right.y())
        grid.resize(250, 500)
        self.app.processEvents()
        self.assertEqual(left.x(), right.x())
        self.assertLess(left.y(), right.y())

    def test_frame_and_error_only_update_the_matching_tile(self):
        grid = self.grid()
        left, right = grid._tiles.values()
        frame = np.full((24, 32, 3), (0, 0, 240), dtype=np.uint8)
        grid.set_frame(2, frame)
        self.assertEqual(left.video._image.pixelColor(0, 0).red(), 240)
        self.assertIsNone(right.video._image)
        grid.set_status(5, "攝影機無法開啟")
        self.assertEqual(right.video._placeholder, "攝影機無法開啟")
        self.assertIsNotNone(left.video._image)
        grid.set_devices([(5, "右側攝影機")])
        grid.set_frame(2, frame)  # A queued frame from the removed stream is ignored.
        self.assertEqual(set(grid._tiles), {5})
        grid.set_frame(5, frame)
        self.assertEqual(right.video._image.pixelColor(0, 0).red(), 240)
        grid.set_devices([])
        self.assertTrue(grid.isHidden())


if __name__ == "__main__":
    unittest.main()
