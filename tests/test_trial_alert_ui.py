"""Trial warnings follow live engine results without changing saved SOPs."""
from copy import deepcopy
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest

import numpy as np
from PySide6.QtCore import QEvent, QTimer
from PySide6.QtWidgets import QApplication

from sop_app.detection import Detection, FrameResult
from sop_app.pipeline import FramePacket
from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step
from sop_app.ui.editor_page import EditorPage


def packet(at, region=None, *, valid=True):
    image = np.zeros((200, 400, 3), np.uint8)
    boxes = {"C2": (60, 100, 100, 150), "C3": (280, 100, 320, 150)}
    detections = [Detection("hand", .95, boxes[region])] if region else []
    result = FrameResult(400, 200, at, detections, detection_valid=valid)
    return FramePacket(image, image.copy(), result, None, 10, 5)


class TrialAlertUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def settle(self):
        for _ in range(5):
            self.app.processEvents()

    def dispose(self, page):
        for timer in page.findChildren(QTimer):
            timer.stop()
        page.close()
        page.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settle()

    def page(self):
        page = EditorPage()
        self.addCleanup(self.dispose, page)
        page._validation_watchdog.stop()
        page.resize(1480, 820)
        page.set_sop(SOPDefinition(name="Trial alerts", rois=[
            ROI("C2", [(.1, .35), (.4, .35), (.4, .9), (.1, .9)]),
            ROI("C3", [(.6, .35), (.9, .35), (.9, .9), (.6, .9)]),
        ], steps=[Step("first"), Step("left handle", conditions=[Condition(label="grip_done")],
            forbidden=[Condition(label="hand", roi="C2"), Condition(label="hand", roi="C3")],
            forbidden_mode="any", forbidden_hold_sec=.3, hold_sec=.3)]))
        page.set_classes(["hand", "grip_done"])
        page.set_model_state("available")
        page.validation_step.setCurrentIndex(1)
        page.show()
        page.on_packet(packet(0))
        self.settle()
        return page

    def alarm(self, page):
        page.validation_start.click()
        self.assertTrue(page._validation.running)
        for at in (.1, .2, .3, .4):
            page.on_packet(packet(at, "C2"))
        self.settle()
        self.assertTrue(page._validation.alert()["active"])

    def test_c2_then_c3_updates_image_and_trial_warning(self):
        page = self.page()
        before_sop = deepcopy(page.sop.to_dict())
        self.alarm(page)
        self.assertTrue(page.validation_alert.isVisible())
        self.assertIn("C2", page.validation_alert.text())
        self.assertIn("工序 2", page.video._alert)
        self.assertEqual(page.video._alert_regions, {"C2"})
        page.on_packet(packet(.5, "C3"))
        self.settle()
        self.assertIn("C3", page.validation_alert.text())
        self.assertIn("C3", page.video._alert)
        self.assertEqual(page.video._alert_regions, {"C3"})
        self.assertEqual(page.validation_status.property("state"), "ng")
        alarms = [e for e in page._validation.events if e.kind == "alarm"]
        self.assertEqual(len(alarms), 1)
        self.assertFalse(page._validation.engine.cycle_ok)
        self.assertNotIn("C3", page.validation_details.toPlainText())
        page.on_packet(packet(2.4, "C3"))
        self.assertIn("C3", page.validation_details.toPlainText())
        self.assertEqual(len([e for e in page._validation.events if e.kind == "trial_notice"]), 1)
        self.assertEqual(page.sop.to_dict(), before_sop)
        self.assertFalse(page._packet.frame.any())

    def test_stop_labels_history_and_restart_clears_it(self):
        page = self.page()
        self.alarm(page)
        page.on_packet(packet(.5, "C3"))
        page.validation_stop.click()
        self.settle()
        self.assertFalse(page._validation.running)
        self.assertIn("已停止", page.video._alert)
        self.assertEqual(page.video._alert_regions, set())
        before = page.validation_details.toPlainText()
        page.on_packet(packet(.6, "C2"))
        self.assertEqual(page.validation_details.toPlainText(), before)
        page.validation_start.click()
        self.settle()
        self.assertTrue(page._validation.running)
        self.assertEqual(page.video._alert, "")
        self.assertFalse(page.validation_alert.isVisible())

    def test_invalid_model_clears_live_region_warning(self):
        page = self.page()
        self.alarm(page)
        page.on_packet(packet(.5, valid=False))
        self.assertFalse(page._validation.alert()["active"])
        self.assertEqual(page.video._alert_regions, set())
        self.assertIn("無法", page.video._alert)

    def test_edit_freeze_source_and_production_clear_obsolete_warnings(self):
        for operation in ("edit", "freeze", "source", "production"):
            with self.subTest(operation=operation):
                page = self.page()
                self.alarm(page)
                if operation == "edit":
                    page.step_editors[1].instruction_edit.setPlainText("changed")
                elif operation == "freeze":
                    page.freeze_check.setChecked(True)
                elif operation == "source":
                    page.clear_video("Disconnected")
                else:
                    page.set_production_running(True)
                self.settle()
                self.assertFalse(page._validation.running)
                self.assertEqual(page.video._alert, "")
                self.assertEqual(page.video._alert_regions, set())
                self.assertFalse(page.validation_alert.isVisible())

    def test_one_frame_preview_without_trial_does_not_show_trial_alert(self):
        page = self.page()
        page.on_packet(packet(.1, "C2"))
        self.assertEqual(page.video._alert, "")
        self.assertFalse(page.validation_alert.isVisible())

    def test_message_interval_edits_survive_rebuild_and_sop_roundtrip(self):
        page = self.page()
        page._open_step_settings(page.sop.steps[1])
        field = page.step_editors[1].trial_notice_interval_spin
        self.assertEqual(field.value(), 2.)
        field.setValue(1.75)
        self.assertEqual(page.sop.steps[1].trial_notice_interval_sec, 1.75)
        page._rebuild_steps()
        self.settle()
        self.assertEqual(page.step_settings.current_editor.trial_notice_interval_spin.value(), 1.75)
        restored = SOPDefinition.from_dict(page.sop.to_dict())
        self.assertEqual(restored.steps[1].trial_notice_interval_sec, 1.75)
        self.assertEqual(restored.validate(["hand", "grip_done"])[0], [])

    def test_long_loaded_message_interval_survives_an_unrelated_edit(self):
        page = self.page()
        page.sop.steps[1].trial_notice_interval_sec = 7200
        page._rebuild_steps()
        self.settle()
        page._open_step_settings(page.sop.steps[1])
        card = page.step_editors[1]
        self.assertEqual(card.trial_notice_interval_spin.value(), 7200)
        card.instruction_edit.setPlainText("Use the left handle")
        self.assertEqual(card.step.trial_notice_interval_sec, 7200)


class TrialNoticeSchemaTests(unittest.TestCase):
    def test_legacy_sops_default_to_two_second_message_interval(self):
        restored = SOPDefinition.from_dict({"steps": [{"name": "legacy"}]})
        self.assertEqual(restored.steps[0].trial_notice_interval_sec, 2.)

    def test_message_interval_accepts_zero_but_rejects_invalid_seconds(self):
        for value in (0, .1, 2, 120):
            definition = SOPDefinition(steps=[Step("A", trial_notice_interval_sec=value)])
            self.assertEqual(definition.validate()[0], [])
        for value in (-1, float("nan"), float("inf")):
            definition = SOPDefinition(steps=[Step("A", trial_notice_interval_sec=value)])
            self.assertTrue(any("訊息間隔" in error for error in definition.validate()[0]))


if __name__ == "__main__":
    unittest.main()
