"""Small CPU tests for the public RGT fine-tuning release."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from cigfaciesloss import compute_SegmentLoss
from examples.make_synthetic_case import write_case
from utils import hr_loss


class CoreTests(unittest.TestCase):
    def test_interpreted_horizon_loss_is_zero_for_constant_prediction(self):
        prediction = torch.ones((2, 1, 8, 8), dtype=torch.float32)
        masks = torch.zeros((2, 2, 8, 8), dtype=torch.float32)
        masks[:, 0, 2, :] = 1.0
        masks[:, 1, 6, :] = 1.0
        self.assertTrue(torch.isclose(hr_loss()(masks, prediction), torch.tensor(0.0)))

    def test_segment_loss_penalizes_variation_and_backpropagates(self):
        prediction = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 4, 4)
        prediction.requires_grad_(True)
        segments = torch.ones_like(prediction)
        loss = compute_SegmentLoss(prediction, segments, beta=0.01)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_paper_evaluation_prefers_adapted_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            case_dir = Path(directory) / "case"
            write_case(case_dir)

            def evaluate(name):
                command = [
                    sys.executable,
                    "eval/eval_paper_protocol.py",
                    "--volume",
                    str(case_dir / f"prediction_{name}.dat"),
                    "--shape",
                    "24,16,64",
                    "--frame",
                    str(case_dir / "horizon_frame.dat"),
                    "--centers",
                    "0.25,0.72",
                    "--input-idx",
                    "0",
                    "--valid-maps",
                    str(case_dir / "validation_horizon.npy"),
                ]
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                line = next(
                    value for value in result.stdout.splitlines() if value.startswith("VALID")
                )
                return float(line.split(":", 1)[1].split()[0])

            self.assertLess(evaluate("adapted"), evaluate("direct"))


if __name__ == "__main__":
    unittest.main()
