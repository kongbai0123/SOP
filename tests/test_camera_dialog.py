import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

from PySide6.QtWidgets import QApplication

from sop_app.ui.camera_dialog import CameraDialog


class CameraDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.dialog = CameraDialog()
        self.values = {"resolution": (1280, 720), "fps": 30, "exposure": -6, "brightness": 128}
        self.dialog.update_controls(self.values, [])
        self.requests = []
        self.dialog.requested.connect(self.requests.append)

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()

    def test_drag_without_selecting_mode_sends_only_changed_control(self):
        self.assertFalse(self.dialog.apply.isEnabled())
        self.dialog.sliders["exposure"].setValue(-700)
        self.assertEqual(self.dialog.rows["exposure"][2].value(), -7)
        self.assertTrue(self.dialog.apply.isEnabled())
        self.dialog.apply.click()
        self.assertEqual(self.requests, [{"exposure": ("manual", -7)}])

    def test_format_presets_send_resolution_and_fps_together(self):
        self.dialog.resolution_preset.setCurrentIndex(self.dialog.resolution_preset.findData("1920x1080"))
        self.dialog.fps_presets[-1].click()
        self.dialog.apply.click()
        self.assertEqual(self.requests, [{"resolution": ("manual", (1920, 1080)), "fps": ("manual", 60)}])
        self.assertFalse(self.dialog.width_input.isEnabled())

    def test_failed_change_stays_pending_success_uses_actual_driver_value(self):
        self.dialog._choose_fps(60)
        self.dialog.sliders["exposure"].setValue(-700)
        self.dialog.apply.click()
        self.dialog.update_controls(self.values, ["攝影機 FPS：要求 60，驅動回報 30", "曝光：未完成（不支援）"])
        self.assertEqual(self.dialog._dirty, {"exposure"})
        self.assertEqual(self.dialog.rows["fps"][2].value(), 30)
        self.assertEqual(self.dialog.rows["exposure"][2].value(), -7)
        self.dialog.apply.click()
        self.assertEqual(self.requests[-1], {"exposure": ("manual", -7)})

    def test_refresh_preserves_unsaved_values_and_discard_never_writes(self):
        self.dialog._choose_fps(60)
        self.dialog.refresh.click()
        self.dialog.update_controls(self.values, [])
        self.assertEqual(self.dialog.rows["fps"][2].value(), 60)
        self.dialog.discard.click()
        self.assertEqual(self.dialog.rows["fps"][2].value(), 30)
        self.assertEqual(self.requests, [{}])
        self.assertFalse(self.dialog.apply.isEnabled())

    def test_running_disables_format_but_keeps_image_controls(self):
        self.dialog.set_running(True)
        self.assertFalse(self.dialog.sliders["fps"].isEnabled())
        self.assertFalse(self.dialog.width_input.isEnabled())
        self.assertTrue(self.dialog.sliders["brightness"].isEnabled())
        self.dialog.set_running(False)
        self.assertTrue(self.dialog.sliders["fps"].isEnabled())

    def test_auto_mode_disables_manual_slider(self):
        self.dialog.rows["exposure"][1].setCurrentIndex(2)
        self.assertFalse(self.dialog.sliders["exposure"].isEnabled())
        self.dialog.apply.click()
        self.assertEqual(self.requests, [{"exposure": ("auto", -6)}])


if __name__ == "__main__":
    unittest.main()
