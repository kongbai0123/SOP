"""A missing reference must not trap orphaned following regions in the editor."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
import unittest
from unittest.mock import patch

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from sop_app.sop_schema import Condition, ROI, SOPDefinition, Step
from sop_app.ui.editor_page import EditorPage


class OrphanedFollowingRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def page(self):
        points = [(.1, .1), (.4, .1), (.4, .4), (.1, .4)]
        definition = SOPDefinition(rois=[ROI("follow", points, "workpiece"), ROI("bin", points)],
                                   steps=[Step("install", conditions=[Condition(label="part", roi="follow"),
                                                                      Condition(label="hand", roi="bin")])])
        page = EditorPage()
        def dispose():
            for timer in page.findChildren(QTimer):
                timer.stop()
            page.close()
        self.addCleanup(dispose)
        page.set_sop(definition)
        return page

    def test_missing_reference_can_clear_orphaned_regions_after_confirmation(self):
        page = self.page()
        self.assertTrue(page.clear_target_button.isEnabled())
        page.target_button.click()
        self.assertIn("清除", page.region_creation_label.text())
        self.assertFalse(page._capture_target)
        with patch("sop_app.ui.editor_page.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            page.clear_target_button.click()
        self.assertIsNone(page.sop.workpiece)
        self.assertEqual([roi.name for roi in page.sop.rois], ["bin"])
        self.assertEqual([condition.roi for condition in page.sop.steps[0].conditions], ["bin"])
        self.assertFalse(page.clear_target_button.isEnabled())

    def test_declining_clear_keeps_regions_and_conditions(self):
        page = self.page()
        before = deepcopy(page.sop.to_dict())
        with patch("sop_app.ui.editor_page.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
            page.clear_target_button.click()
        self.assertEqual(page.sop.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
