"""Camera discovery thread behavior without opening hardware."""

import os
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from sop_app.camera_discovery import CameraDevice, CameraDiscovery, CameraDiscoveryResult


class CameraDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for_results(self, results, count):
        deadline = time.monotonic() + 2
        while len(results) < count and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertEqual(len(results), count)

    def test_runs_off_the_gui_thread_and_coalesces_an_in_flight_request(self):
        discovery = CameraDiscovery()
        results = []
        gui_thread = threading.get_ident()
        discovery.finished.connect(lambda result: results.append((threading.get_ident(), result)))
        worker_calls = []
        release = threading.Event()
        devices = (CameraDevice(0, "Camera A"), CameraDevice(1, "Camera B"))

        def enumerate_devices():
            worker_calls.append(threading.get_ident())
            release.wait(2)
            return devices

        with patch("sop_app.camera_discovery.enumerate_cameras", side_effect=enumerate_devices):
            try:
                self.assertTrue(discovery.request())
                self.assertFalse(discovery.request())
            finally:
                release.set()
            self.wait_for_results(results, 1)
        self.assertEqual(len(worker_calls), 1)
        self.assertNotEqual(worker_calls[0], gui_thread)
        self.assertEqual(results, [(gui_thread, CameraDiscoveryResult(devices))])

    def test_failed_search_reports_an_error_and_allows_another_search(self):
        discovery = CameraDiscovery()
        results = []
        discovery.finished.connect(results.append)

        with patch("sop_app.camera_discovery.enumerate_cameras", side_effect=[OSError("未能列舉"), ()]):
            self.assertTrue(discovery.request())
            self.wait_for_results(results, 1)
            self.assertEqual(results[0], CameraDiscoveryResult(error="未能列舉"))
            self.assertTrue(discovery.request())
            self.wait_for_results(results, 2)
        self.assertEqual(results[1], CameraDiscoveryResult())


if __name__ == "__main__":
    unittest.main()
