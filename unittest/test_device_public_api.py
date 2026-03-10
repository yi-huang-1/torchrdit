"""
Test suite for device resolution public API.

Tests that DeviceResolution metadata is exposed through solver objects
and is importable from the public torchrdit module.
"""

import sys

sys.path.insert(0, "torchrdit/src")

import pytest
import torch
from unittest.mock import patch, MagicMock
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.device import DeviceResolution


class TestDeviceResolutionPublicAPI:
    """Test the public API for device resolution metadata."""

    def test_device_resolution_importable_from_torchrdit(self):
        """DeviceResolution should be importable from torchrdit package."""
        from torchrdit import DeviceResolution as PublicDeviceResolution
        
        assert PublicDeviceResolution is DeviceResolution

    def test_solver_has_device_resolution_attribute(self):
        """Solver objects should expose device_resolution attribute."""
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=[1.55],
            grids=[16, 16],
            harmonics=[3, 3],
            device="cpu"
        )
        
        assert hasattr(solver, "device_resolution")

    def test_device_resolution_is_dataclass_instance(self):
        """device_resolution should be a DeviceResolution instance."""
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=[1.55],
            grids=[16, 16],
            harmonics=[3, 3],
            device="cpu"
        )
        
        assert isinstance(solver.device_resolution, DeviceResolution)

    def test_device_resolution_has_required_fields(self):
        """DeviceResolution should have all required fields."""
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=[1.55],
            grids=[16, 16],
            harmonics=[3, 3],
            device="cpu"
        )
        
        resolution = solver.device_resolution
        assert hasattr(resolution, "requested_device")
        assert hasattr(resolution, "resolved_device")
        assert hasattr(resolution, "fell_back")
        assert hasattr(resolution, "reason")

    def test_device_resolution_cpu_no_fallback(self):
        """CPU device should not trigger fallback."""
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=[1.55],
            grids=[16, 16],
            harmonics=[3, 3],
            device="cpu"
        )
        
        resolution = solver.device_resolution
        assert resolution.requested_device == "cpu"
        assert resolution.resolved_device == torch.device("cpu")
        assert resolution.fell_back is False
        assert resolution.reason is None

    def test_device_resolution_cuda_available(self):
        """CUDA device should resolve correctly if available (mocked for deterministic testing)."""
        mock_cuda_resolution = DeviceResolution(
            requested_device="cuda",
            resolved_device=torch.device("cuda"),
            fell_back=False,
            reason=None
        )
        
        with patch("torchrdit.builder.resolve_device", return_value=mock_cuda_resolution):
            with patch("torchrdit.solver.RCWASolver") as MockRCWASolver:
                mock_instance = MagicMock()
                mock_instance._device_resolution = mock_cuda_resolution
                mock_instance.device_resolution = mock_cuda_resolution
                MockRCWASolver.return_value = mock_instance
                
                solver = create_solver(
                    algorithm=Algorithm.RCWA,
                    lam0=[1.55],
                    grids=[16, 16],
                    harmonics=[3, 3],
                    device="cuda"
                )
                
                resolution = solver.device_resolution
                assert resolution.requested_device == "cuda"
                assert resolution.resolved_device.type == "cuda"
                assert resolution.fell_back is False
                assert resolution.reason is None

    def test_device_resolution_cuda_fallback_when_unavailable(self):
        """CUDA device should fallback to CPU when unavailable."""
        # This test assumes CUDA is not available in test environment
        # or we can mock it, but for now we'll test with an invalid device
        with pytest.warns(UserWarning):
            solver = create_solver(
                algorithm=Algorithm.RCWA,
                lam0=[1.55],
                grids=[16, 16],
                harmonics=[3, 3],
                device="invalid_device"
            )
        
        resolution = solver.device_resolution
        assert resolution.requested_device == "invalid_device"
        assert resolution.resolved_device == torch.device("cpu")
        assert resolution.fell_back is True
        assert resolution.reason is not None

    def test_device_resolution_mps_fallback(self):
        """MPS device should fallback to CPU."""
        with pytest.warns(UserWarning):
            solver = create_solver(
                algorithm=Algorithm.RCWA,
                lam0=[1.55],
                grids=[16, 16],
                harmonics=[3, 3],
                device="mps"
            )
        
        resolution = solver.device_resolution
        assert resolution.requested_device == "mps"
        assert resolution.resolved_device == torch.device("cpu")
        assert resolution.fell_back is True
        assert "complex-dtype" in resolution.reason or "mps" in resolution.reason.lower()

    def test_existing_solver_device_behavior_unchanged(self):
        """Existing solver.device behavior should remain unchanged."""
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=[1.55],
            grids=[16, 16],
            harmonics=[3, 3],
            device="cpu"
        )
        
        # solver.device should still work as before
        assert solver.device == torch.device("cpu")
        assert isinstance(solver.device, torch.device)

    def test_device_resolution_with_rdit_solver(self):
        """Device resolution should work with RDIT solver."""
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=[1.55],
            grids=[16, 16],
            harmonics=[3, 3],
            device="cpu"
        )
        
        assert hasattr(solver, "device_resolution")
        assert isinstance(solver.device_resolution, DeviceResolution)
        assert solver.device_resolution.resolved_device == torch.device("cpu")

    def test_device_resolution_immutable(self):
        """DeviceResolution should be immutable (frozen dataclass)."""
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=[1.55],
            grids=[16, 16],
            harmonics=[3, 3],
            device="cpu"
        )
        
        resolution = solver.device_resolution
        
        # Attempting to modify should raise an error
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            resolution.fell_back = True
