"""TDD test suite for FourierBaseSolver device propagation — deterministic, no hardware deps."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from torchrdit.device import DeviceResolution
from torchrdit.solver import FourierBaseSolver, create_solver
from torchrdit.constants import Precision


class TestDeviceSolverInitFallback(unittest.TestCase):

    def _make_cpu_fallback_resolution(self, requested: str = "cuda") -> DeviceResolution:
        return DeviceResolution(
            requested_device=requested,
            resolved_device=torch.device("cpu"),
            fell_back=True,
            reason="CUDA not available. Falling back to cpu.",
        )

    def test_solver_tensors_on_resolved_device(self):
        """All solver-level tensors land on the resolved device from Cell3D, not raw request."""
        with patch("torchrdit.cell.resolve_device") as mock_resolve:
            mock_resolve.return_value = self._make_cpu_fallback_resolution("cuda")

            solver = FourierBaseSolver(
                lam0=np.array([1.0]),
                lengthunit="um",
                grids=[16, 16],
                harmonics=[3, 3],
                precision=Precision.SINGLE,
                device="cuda",
            )

            self.assertEqual(solver.device, torch.device("cpu"))

            solver_tensors = {
                "mesh_fp": solver.mesh_fp,
                "mesh_fq": solver.mesh_fq,
                "reci_t1": solver.reci_t1,
                "reci_t2": solver.reci_t2,
                "tlam0": solver.tlam0,
                "k_0": solver.k_0,
                "kinc": solver.kinc,
            }
            for name, tensor in solver_tensors.items():
                self.assertEqual(
                    tensor.device.type,
                    "cpu",
                    f"Tensor '{name}' should be on cpu but is on {tensor.device}",
                )

    def test_create_solver_with_unavailable_cuda_falls_back(self):
        """create_solver(device='cuda') on a CUDA-less machine falls back to cpu end-to-end."""
        with patch("torch.cuda.is_available", return_value=False):
            import warnings

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                solver = create_solver(
                    lam0=np.array([1.0]),
                    grids=[16, 16],
                    harmonics=[3, 3],
                    precision=Precision.SINGLE,
                    device="cuda",
                )

            self.assertEqual(solver.device, torch.device("cpu"))

            for name in ("mesh_fp", "mesh_fq", "reci_t1", "reci_t2", "tlam0", "k_0", "kinc"):
                tensor = getattr(solver, name)
                self.assertEqual(
                    tensor.device.type,
                    "cpu",
                    f"Tensor '{name}' should be on cpu after fallback",
                )

    def test_solver_cpu_device_no_fallback(self):
        """FourierBaseSolver with device='cpu' creates tensors on cpu without fallback."""
        solver = FourierBaseSolver(
            lam0=np.array([1.0]),
            lengthunit="um",
            grids=[16, 16],
            harmonics=[3, 3],
            precision=Precision.SINGLE,
            device="cpu",
        )

        self.assertEqual(solver.device, torch.device("cpu"))
        self.assertFalse(solver._device_resolution.fell_back)

        for name in ("mesh_fp", "mesh_fq", "reci_t1", "reci_t2", "tlam0", "k_0", "kinc"):
            tensor = getattr(solver, name)
            self.assertEqual(tensor.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
