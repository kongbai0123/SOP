"""Invalid tracking settings stay visible and cannot start a dependent SOP."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sop_app.detection import TRACKING_RELOCATING
from sop_app.pipeline import VideoPipeline
from sop_app.recorder import Recorder
from sop_app.sop_schema import Condition, CycleSettings, ROI, SOPDefinition, Step, Workpiece
from sop_app.tracking import workpiece_key


POINTS = [(.2, .2), (.6, .2), (.6, .6), (.2, .6)]


def definition(*, following=True, referenced=True):
    return SOPDefinition(workpiece=Workpiece(reference_png="invalid-reference", points=POINTS),
                         rois=[ROI("target", POINTS, "workpiece" if following else "fixed")],
                         steps=[Step("install", conditions=[Condition(label="part", roi="target")]
                                    if referenced else [])])


class TrackingRequirementTests(unittest.TestCase):
    def test_only_referenced_following_regions_require_tracking(self):
        self.assertTrue(definition().requires_tracking())
        self.assertFalse(definition(following=False).requires_tracking())
        self.assertFalse(definition(referenced=False).requires_tracking())
        sop = definition(referenced=False)
        sop.steps[0].forbidden = [Condition(label="part", roi="target")]
        self.assertTrue(sop.requires_tracking())

    def test_only_enabled_cycle_reset_requires_tracking(self):
        sop = definition(referenced=False)
        sop.cycle = CycleSettings(reset_conditions=[Condition(label="part", roi="target")])
        self.assertTrue(sop.requires_tracking())
        sop.cycle.enabled = False
        self.assertFalse(sop.requires_tracking())

    def test_invalid_reference_blocks_dependent_conditions_but_only_warns_for_manual(self):
        for referenced in (True, False):
            sop = definition(referenced=referenced)
            with patch("sop_app.tracking.WorkpieceLocator", side_effect=ValueError("reference ambiguous")) as locator:
                errors, warnings = sop.validate()
            self.assertTrue(any("reference ambiguous" in line for line in (errors if referenced else warnings)))
            self.assertEqual(bool(errors), referenced)
            locator.assert_called_once_with(sop.workpiece, adaptive_budget=True)


class PipelineTrackingConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.recorder = Recorder(Path(self.folder.name))
        self.addCleanup(self.recorder.close)
        self.pipe = VideoPipeline(Path(self.folder.name))
        self.pipe._detector = SimpleNamespace(classes=["part"], info=SimpleNamespace(model_version_id="fixture"),
                                              detect=lambda frame: [])

    def test_failed_configuration_is_cached_and_reported_on_every_frame(self):
        sop = definition()
        key = workpiece_key(sop.workpiece)
        packets = []
        self.pipe.packet_ready.connect(packets.append)
        with patch("sop_app.pipeline.WorkpieceLocator", side_effect=ValueError("choose a joint")) as locator:
            self.pipe._handle_command("workpiece", (sop.workpiece, key), self.recorder)
            self.pipe._handle_command("workpiece", (sop.workpiece, key), self.recorder)
            locator.assert_called_once_with(sop.workpiece, adaptive_budget=True)
        image = np.zeros((120, 160, 3), np.uint8)
        for timestamp in (0., .1):
            self.pipe.acknowledge_packet()
            self.pipe._process(image, timestamp, self.recorder)
        self.assertEqual(len(packets), 2)
        for packet in packets:
            self.assertEqual(packet.result.workpiece_key, key)
            self.assertIn("choose a joint", packet.result.tracking_status)
            self.assertEqual(packet.result.tracking_state, TRACKING_RELOCATING)
            self.assertIsNone(packet.result.confirmed_transform)
            self.assertTrue(packet.result.tracking_metrics["configuration_error"])

    def test_direct_start_does_not_record_a_cycle_for_invalid_required_reference(self):
        errors = []
        states = []
        self.pipe.error.connect(errors.append)
        self.pipe.engine_state.connect(states.append)
        with patch.object(self.pipe, "_emit_events") as events, \
             patch("sop_app.pipeline.WorkpieceLocator", side_effect=ValueError("ambiguous reference")):
            self.pipe._handle_command("start", definition(), self.recorder)
        self.assertIsNone(self.pipe._engine)
        self.assertEqual(states, [False])
        self.assertEqual(len(errors), 1)
        self.assertIn("ambiguous reference", errors[0])
        events.assert_not_called()

    def test_unreferenced_following_region_does_not_block_manual_sop(self):
        with patch.object(self.pipe, "_emit_events") as events, \
             patch("sop_app.pipeline.WorkpieceLocator", side_effect=ValueError("ambiguous reference")):
            self.pipe._handle_command("start", definition(referenced=False), self.recorder)
        self.assertIsNotNone(self.pipe._engine)
        events.assert_called_once()

    def test_new_reference_or_clearing_reference_clears_old_error(self):
        sop = definition()
        with patch("sop_app.pipeline.WorkpieceLocator", side_effect=ValueError("old error")):
            self.pipe._handle_command("workpiece", (sop.workpiece, workpiece_key(sop.workpiece)), self.recorder)
        revised = Workpiece(reference_png="different-reference", points=POINTS)
        accepted = object()
        with patch("sop_app.pipeline.WorkpieceLocator", return_value=accepted):
            self.pipe._handle_command("workpiece", (revised, workpiece_key(revised)), self.recorder)
        self.assertIs(self.pipe._locator, accepted)
        self.assertEqual(self.pipe._locator_error, "")
        self.pipe._handle_command("workpiece", (None, ""), self.recorder)
        self.assertIsNone(self.pipe._locator)
        self.assertEqual(self.pipe._locator_key, "")
        self.assertEqual(self.pipe._locator_error, "")


if __name__ == "__main__":
    unittest.main()
