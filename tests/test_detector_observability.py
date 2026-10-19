"""Backend observations use the executing model and never alter its inputs."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from sop_app.detector import UltralyticsDetector


class Device:
    type = "cuda"

    def __str__(self):
        return "cuda:0"


class Backend:
    device = Device()
    fp16 = True

    def __init__(self):
        self.model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="torch.float16")]))
        self.hooks = []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))


class DetectorObservabilityTests(unittest.TestCase):
    def test_shape_hook_attaches_after_setup_and_reads_executing_copy(self):
        detector = UltralyticsDetector.__new__(UltralyticsDetector)
        detector.device = Device()
        detector.image_size = 800
        detector.score_floor = .3
        detector.info = SimpleNamespace(has_masks=False)
        detector._observed_backend = None
        detector._input_shape = None
        backend = Backend()
        original = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="torch.float32")]))

        def predict(**kwargs):
            # Mirrors predictor.setup_model's deepcopy of the source model.
            # A hook bound to detector on this source would drag its state in.
            copy.deepcopy(original)
            for hook in backend.hooks:
                hook(backend, (SimpleNamespace(shape=(1, 3, 480, 800)),))
            return [SimpleNamespace(boxes=None, speed={"preprocess": 1., "inference": 2., "postprocess": 3.})]

        detector.model = SimpleNamespace(model=original, predict=predict,
                                         predictor=SimpleNamespace(model=backend))
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.assertEqual(detector.detect(frame), [])
        self.assertEqual(len(backend.hooks), 1)
        detector.detect(frame)
        self.assertEqual(len(backend.hooks), 1)
        self.assertEqual(detector.runtime_info["input_shape"], [1, 3, 480, 800])
        self.assertEqual(detector.runtime_info["dtype"], "torch.float16")
        self.assertTrue(detector.runtime_info["cuda"])
        self.assertEqual(detector.last_metrics["model_inference_ms"], 2.)


if __name__ == "__main__":
    unittest.main()
