"""TDD test suite for Cell3D device-fallback safety net — deterministic, no hardware deps."""

import unittest
from unittest.mock import patch

import torch

from torchrdit.cell import Cell3D
from torchrdit.device import DeviceResolution


class TestDeviceCellFallback(unittest.TestCase):

    def test_cell_initialization_falls_back_before_grid_allocation(self):
        """Given unavailable device, Cell3D resolves to cpu before allocating grid tensors."""
        with patch("torchrdit.cell.resolve_device") as mock_resolve:
            mock_resolve.return_value = DeviceResolution(
                requested_device="cuda",
                resolved_device=torch.device("cpu"),
                fell_back=True,
                reason="CUDA not available. Falling back to cpu.",
            )

            cell = Cell3D(device="cuda")

            mock_resolve.assert_called_once_with("cuda")
            self.assertEqual(cell.device, torch.device("cpu"))
            self.assertEqual(cell.vec_p.device, torch.device("cpu"))
            self.assertEqual(cell.vec_q.device, torch.device("cpu"))
            self.assertEqual(cell.XO.device, torch.device("cpu"))
            self.assertEqual(cell.YO.device, torch.device("cpu"))

    def test_cell_records_requested_and_resolved_device_metadata(self):
        """Given fallback, Cell3D persists the full DeviceResolution for downstream visibility."""
        with patch("torchrdit.cell.resolve_device") as mock_resolve:
            resolution = DeviceResolution(
                requested_device="mps",
                resolved_device=torch.device("cpu"),
                fell_back=True,
                reason="mps backend rejected: complex-dtype operations are unsupported on mps.",
            )
            mock_resolve.return_value = resolution

            cell = Cell3D(device="mps")

            self.assertTrue(hasattr(cell, "_device_resolution"))
            self.assertIsInstance(cell._device_resolution, DeviceResolution)
            self.assertEqual(cell._device_resolution.requested_device, "mps")
            self.assertEqual(cell._device_resolution.resolved_device, torch.device("cpu"))
            self.assertTrue(cell._device_resolution.fell_back)
            self.assertIn("mps", cell._device_resolution.reason)

    def test_cell_cpu_device_no_fallback(self):
        """Given 'cpu', Cell3D resolves without fallback and stores clean metadata."""
        with patch("torchrdit.cell.resolve_device") as mock_resolve:
            mock_resolve.return_value = DeviceResolution(
                requested_device="cpu",
                resolved_device=torch.device("cpu"),
                fell_back=False,
                reason=None,
            )

            cell = Cell3D(device="cpu")

            self.assertEqual(cell.device, torch.device("cpu"))
            self.assertFalse(cell._device_resolution.fell_back)
            self.assertIsNone(cell._device_resolution.reason)


if __name__ == "__main__":
    unittest.main()
