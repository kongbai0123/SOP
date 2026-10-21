"""Unlimited object counts still require a detected object, using the old schema."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest

from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from sop_app.conditions import evaluate_condition
from sop_app.detection import Detection, FrameResult
from sop_app.sop_schema import Condition, SOPDefinition, Step, load_sop, save_sop
from sop_app.ui.condition_rows import ConditionRows
from sop_app.ui.condition_table import ConditionTable


def frame(count):
    return FrameResult(320, 180, 0, [Detection("hand", .9, (10, 10, 20, 20)) for _ in range(count)])


class UnlimitedConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def rows(self, role="completion", condition=None):
        rows = ConditionRows(role=role)
        rows.set_options(["hand"], [])
        rows.set_conditions([condition or Condition(label="hand")])
        rows.show()
        self.addCleanup(rows.close)
        return rows

    def table(self, condition=None):
        table = ConditionTable()
        table.set_options(["hand"], [])
        table.set_conditions([condition or Condition(label="hand")])
        table.show()
        self.addCleanup(table.close)
        return table

    def test_all_condition_roles_edit_unlimited_and_specific_minimum(self):
        for role in ("completion", "violation", "reset"):
            with self.subTest(role=role):
                rows = self.rows(role)
                count = rows.field_widget(0, "min_count")
                self.assertEqual(count.minimum(), 1)
                self.assertEqual(count.text(), "不限")
                self.assertIn("至少 1 個", count.toolTip())
                self.assertIn("至少 N 個", count.toolTip())
                count.setValue(3)
                self.assertEqual(count.text(), "3")
                self.assertEqual(rows.conditions()[0].min_count, 3)
                count.stepDown()
                count.stepDown()
                self.assertEqual(count.value(), 1)
                self.assertEqual(rows.conditions()[0].min_count, 1)
                count.stepUp()
                self.assertEqual(count.text(), "2")
                count.stepDown()
                self.assertEqual(count.text(), "不限")

    def test_unlimited_requires_one_or_more_objects_in_all_roles(self):
        for role in ("completion", "violation", "reset"):
            rows = self.rows(role)
            condition = rows.conditions()[0]
            for detected in (0, 1, 2, 10):
                with self.subTest(role=role, detected=detected):
                    result = evaluate_condition(condition, frame(detected), {})
                    self.assertEqual(result.met, detected >= 1)
                    self.assertEqual(result.count, detected)

    def test_specific_minimum_is_still_at_least_not_exactly_n(self):
        rows = self.rows()
        rows.field_widget(0, "min_count").setValue(3)
        condition = rows.conditions()[0]
        for detected in (0, 1, 2, 3, 4):
            with self.subTest(detected=detected):
                self.assertEqual(evaluate_condition(condition, frame(detected), {}).met, detected >= 3)

    def test_save_and_reload_keep_existing_integer_schema_for_every_role(self):
        completion = self.rows("completion", Condition(label="hand", min_count=2))
        violation = self.rows("violation", Condition(label="hand", min_count=3))
        reset = self.rows("reset", Condition(label="hand", min_count=4))
        for rows in (completion, violation, reset):
            rows.field_widget(0, "min_count").setValue(1)
        sop = SOPDefinition(steps=[Step("test", conditions=completion.conditions(), forbidden=violation.conditions())])
        sop.cycle.reset_conditions = reset.conditions()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unlimited.json"
            save_sop(sop, path)
            restored = load_sop(path)
        collections = [restored.steps[0].conditions, restored.steps[0].forbidden, restored.cycle.reset_conditions]
        for role, conditions in zip(("completion", "violation", "reset"), collections):
            with self.subTest(role=role):
                self.assertEqual(conditions[0].min_count, 1)
                self.assertIs(type(conditions[0].min_count), int)
                reloaded = self.rows(role, conditions[0])
                self.assertEqual(reloaded.field_widget(0, "min_count").text(), "不限")
        self.assertEqual(sop.to_dict(), restored.to_dict())

    def test_disappear_stays_zero_and_restores_appearance_count(self):
        for role in ("completion", "violation", "reset"):
            with self.subTest(role=role):
                rows = self.rows(role, Condition(label="hand", min_count=3))
                state = rows.field_widget(0, "type")
                state.setCurrentIndex(state.findData("disappear"))
                self.assertTrue(rows.field_widget(0, "min_count").isHidden())
                self.assertEqual(rows.field_widget(0, "min_count_note").text(), "需為 0")
                condition = rows.conditions()[0]
                self.assertEqual(condition.min_count, 3)
                self.assertTrue(evaluate_condition(condition, frame(0), {}).met)
                for detected in (1, 2, 3, 4):
                    self.assertFalse(evaluate_condition(condition, frame(detected), {}).met)
                state.setCurrentIndex(state.findData("appear"))
                self.assertEqual(rows.field_widget(0, "min_count").text(), "3")
                rows.field_widget(0, "min_count").setValue(1)
                state.setCurrentIndex(state.findData("disappear"))
                state.setCurrentIndex(state.findData("appear"))
                self.assertEqual(rows.field_widget(0, "min_count").text(), "不限")

    def test_legacy_table_supports_unlimited_reload_and_real_thresholds(self):
        table = self.table(Condition(label="hand", min_count=3))
        count = table.table.cellWidget(0, 3)
        count.setValue(1)
        self.assertEqual(count.text(), "不限")
        self.assertIn("至少 1 個", count.toolTip())
        self.assertFalse(evaluate_condition(table.conditions()[0], frame(0), {}).met)
        self.assertTrue(evaluate_condition(table.conditions()[0], frame(1), {}).met)
        self.assertTrue(evaluate_condition(table.conditions()[0], frame(2), {}).met)
        table.set_conditions(table.conditions())
        count = table.table.cellWidget(0, 3)
        self.assertEqual(count.text(), "不限")
        count.setValue(2)
        self.assertFalse(evaluate_condition(table.conditions()[0], frame(1), {}).met)
        self.assertTrue(evaluate_condition(table.conditions()[0], frame(2), {}).met)
        self.assertTrue(evaluate_condition(table.conditions()[0], frame(3), {}).met)

    def test_legacy_table_disappear_display_does_not_overwrite_saved_minimum(self):
        table = self.table(Condition(label="hand", min_count=3))
        changes = QSignalSpy(table.changed)
        state = table.table.cellWidget(0, 0)
        count = table.table.cellWidget(0, 3)
        state.setCurrentIndex(state.findData("disappear"))
        self.assertEqual(changes.count(), 1)
        self.assertEqual(count.text(), "需為 0")
        self.assertFalse(count.isEnabled())
        self.assertEqual(table.conditions()[0].min_count, 3)
        self.assertTrue(evaluate_condition(table.conditions()[0], frame(0), {}).met)
        self.assertFalse(evaluate_condition(table.conditions()[0], frame(1), {}).met)
        table.set_conditions(table.conditions())
        self.assertEqual(table.table.cellWidget(0, 3).text(), "需為 0")
        state = table.table.cellWidget(0, 0)
        state.setCurrentIndex(state.findData("appear"))
        self.assertEqual(table.table.cellWidget(0, 3).text(), "3")
        self.assertEqual(table.conditions()[0].min_count, 3)


if __name__ == "__main__":
    unittest.main()
