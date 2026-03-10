"""TDD test suite for config/helper-path device resolution regression coverage.

This module verifies that all documented solver-creation paths pass through
the device resolver and do not bypass it. Tests cover:
- create_solver_from_config() with device keys
- create_solver_from_builder() with device config
- get_solver_builder() path
- from_config() path
- MPS via config falls back to cpu with complex-dtype warning
"""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch

from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.solver import (
    create_solver_from_config,
    create_solver_from_builder,
    get_solver_builder,
    RCWASolver,
    RDITSolver,
)


class TestDeviceConfigIntegration(unittest.TestCase):
    """Test device resolution through all solver-creation helper paths."""

    def setUp(self):
        """Set up test fixtures."""
        self.lam0 = np.array([1.0])
        self.air = create_material(name="air", permittivity=1.0, permeability=1.0)
        self.silicon = create_material(
            name="silicon", permittivity=11.7, permeability=1.0
        )

    def test_create_solver_from_config_with_cuda_falls_back(self):
        """Test create_solver_from_config() resolves device at build time.

        When create_solver_from_config() is called with device="cuda"
        but CUDA is unavailable, the solver should fall back to CPU.
        """
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "cuda",
        }

        with patch("torchrdit.device.torch.cuda.is_available", return_value=False):
            with self.assertWarns(UserWarning):
                solver = create_solver_from_config(config)

            # Verify solver was created and device is CPU (fallback)
            self.assertIsInstance(solver, RCWASolver)
            self.assertEqual(str(solver.device), "cpu")

    def test_create_solver_from_config_with_mps_falls_back_with_complex_warning(self):
        """Test create_solver_from_config() rejects MPS with complex-dtype reason.

        When create_solver_from_config() is called with device="mps",
        it should fall back to CPU with a warning mentioning complex-dtype.
        """
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "mps",
        }

        with self.assertWarns(UserWarning) as cm:
            solver = create_solver_from_config(config)

        # Verify solver was created and device is CPU (fallback)
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(str(solver.device), "cpu")

        # Verify warning was emitted with complex-dtype reason
        warning_text = str(cm.warning).lower()
        self.assertIn("mps", warning_text)
        self.assertIn("complex", warning_text)

    def test_create_solver_from_config_cpu_no_fallback(self):
        """Test create_solver_from_config() with cpu device has no fallback."""
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "cpu",
        }

        solver = create_solver_from_config(config)

        # Verify solver was created on CPU
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(str(solver.device), "cpu")

    def test_create_solver_from_builder_with_cuda_falls_back(self):
        """Test create_solver_from_builder() resolves device at build time.

        When create_solver_from_builder() is called with a config function
        that sets device="cuda" but CUDA is unavailable, the solver should
        fall back to CPU.
        """

        def configure_solver(builder):
            return (
                builder.with_algorithm(Algorithm.RCWA)
                .with_wavelengths(1.55)
                .with_k_dimensions([5, 5])
                .with_materials([self.air, self.silicon])
                .with_device("cuda")
            )

        with patch("torchrdit.device.torch.cuda.is_available", return_value=False):
            with self.assertWarns(UserWarning):
                solver = create_solver_from_builder(configure_solver)

            # Verify solver was created and device is CPU (fallback)
            self.assertIsInstance(solver, RCWASolver)
            self.assertEqual(str(solver.device), "cpu")

    def test_create_solver_from_builder_with_mps_falls_back_with_complex_warning(
        self,
    ):
        """Test create_solver_from_builder() rejects MPS with complex-dtype reason.

        When create_solver_from_builder() is called with a config function
        that sets device="mps", it should fall back to CPU with a warning
        mentioning complex-dtype.
        """

        def configure_solver(builder):
            return (
                builder.with_algorithm(Algorithm.RCWA)
                .with_wavelengths(1.55)
                .with_k_dimensions([5, 5])
                .with_materials([self.air, self.silicon])
                .with_device("mps")
            )

        with self.assertWarns(UserWarning) as cm:
            solver = create_solver_from_builder(configure_solver)

        # Verify solver was created and device is CPU (fallback)
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(str(solver.device), "cpu")

        # Verify warning was emitted with complex-dtype reason
        warning_text = str(cm.warning).lower()
        self.assertIn("mps", warning_text)
        self.assertIn("complex", warning_text)

    def test_get_solver_builder_with_device_resolution(self):
        """Test get_solver_builder() path resolves device at build time.

        When get_solver_builder() is used with .with_device("cuda")
        but CUDA is unavailable, build() should resolve and fall back to CPU.
        """
        with patch("torchrdit.device.torch.cuda.is_available", return_value=False):
            with self.assertWarns(UserWarning):
                builder = get_solver_builder()
                solver = (
                    builder.with_algorithm(Algorithm.RCWA)
                    .with_wavelengths(1.55)
                    .with_k_dimensions([5, 5])
                    .with_materials([self.air, self.silicon])
                    .with_device("cuda")
                    .build()
                )

            # Verify solver was created and device is CPU (fallback)
            self.assertIsInstance(solver, RCWASolver)
            self.assertEqual(str(solver.device), "cpu")

    def test_get_solver_builder_with_mps_falls_back_with_complex_warning(self):
        """Test get_solver_builder() path rejects MPS with complex-dtype reason.

        When get_solver_builder() is used with .with_device("mps"),
        build() should fall back to CPU with a warning mentioning complex-dtype.
        """
        with self.assertWarns(UserWarning) as cm:
            builder = get_solver_builder()
            solver = (
                builder.with_algorithm(Algorithm.RCWA)
                .with_wavelengths(1.55)
                .with_k_dimensions([5, 5])
                .with_materials([self.air, self.silicon])
                .with_device("mps")
                .build()
            )

        # Verify solver was created and device is CPU (fallback)
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(str(solver.device), "cpu")

        # Verify warning was emitted with complex-dtype reason
        warning_text = str(cm.warning).lower()
        self.assertIn("mps", warning_text)
        self.assertIn("complex", warning_text)

    def test_from_config_path_with_cuda_falls_back(self):
        """Test builder.from_config() path resolves device at build time.

        When builder.from_config(config) is called with device="cuda"
        but CUDA is unavailable, build() should resolve and fall back to CPU.
        """
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "cuda",
        }

        with patch("torchrdit.device.torch.cuda.is_available", return_value=False):
            with self.assertWarns(UserWarning):
                builder = get_solver_builder()
                solver = builder.from_config(config).build()

            # Verify solver was created and device is CPU (fallback)
            self.assertIsInstance(solver, RCWASolver)
            self.assertEqual(str(solver.device), "cpu")

    def test_from_config_path_with_mps_falls_back_with_complex_warning(self):
        """Test builder.from_config() path rejects MPS with complex-dtype reason.

        When builder.from_config(config) is called with device="mps",
        build() should fall back to CPU with a warning mentioning complex-dtype.
        """
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "mps",
        }

        with self.assertWarns(UserWarning) as cm:
            builder = get_solver_builder()
            solver = builder.from_config(config).build()

        # Verify solver was created and device is CPU (fallback)
        self.assertIsInstance(solver, RCWASolver)
        self.assertEqual(str(solver.device), "cpu")

        # Verify warning was emitted with complex-dtype reason
        warning_text = str(cm.warning).lower()
        self.assertIn("mps", warning_text)
        self.assertIn("complex", warning_text)

    def test_all_paths_use_same_resolver(self):
        """Test that all creation paths use the same device resolver.

        This verifies that no path bypasses the resolver by checking
        that all paths produce the same fallback behavior for the same
        invalid device request.
        """
        # Test with invalid CUDA index
        with patch("torchrdit.device.torch.cuda.is_available", return_value=True), \
             patch("torchrdit.device.torch.cuda.device_count", return_value=1):

            # Path 1: create_solver_from_config
            config = {
                "algorithm": "RCWA",
                "wavelengths": [1.55],
                "grids": [16, 16],
                "harmonics": [3, 3],
                "device": "cuda:99",
            }
            with self.assertWarns(UserWarning):
                solver1 = create_solver_from_config(config)

            # Path 2: create_solver_from_builder
            def configure_solver(builder):
                return (
                    builder.with_algorithm(Algorithm.RCWA)
                    .with_wavelengths(1.55)
                    .with_k_dimensions([5, 5])
                    .with_materials([self.air, self.silicon])
                    .with_device("cuda:99")
                )

            with self.assertWarns(UserWarning):
                solver2 = create_solver_from_builder(configure_solver)

            # Path 3: get_solver_builder
            with self.assertWarns(UserWarning):
                builder = get_solver_builder()
                solver3 = (
                    builder.with_algorithm(Algorithm.RCWA)
                    .with_wavelengths(1.55)
                    .with_k_dimensions([5, 5])
                    .with_materials([self.air, self.silicon])
                    .with_device("cuda:99")
                    .build()
                )

            # Path 4: from_config
            with self.assertWarns(UserWarning):
                builder = get_solver_builder()
                solver4 = builder.from_config(config).build()

            # All paths should produce CPU device (fallback)
            self.assertEqual(str(solver1.device), "cpu")
            self.assertEqual(str(solver2.device), "cpu")
            self.assertEqual(str(solver3.device), "cpu")
            self.assertEqual(str(solver4.device), "cpu")

    def test_rdit_solver_all_paths_resolve_device(self):
        """Test device resolution works for RDIT solver through all paths."""
        config = {
            "algorithm": "RDIT",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "cuda",
        }

        with patch("torchrdit.device.torch.cuda.is_available", return_value=False):
            # Path 1: create_solver_from_config
            with self.assertWarns(UserWarning):
                solver1 = create_solver_from_config(config)
            self.assertIsInstance(solver1, RDITSolver)
            self.assertEqual(str(solver1.device), "cpu")

            # Path 2: create_solver_from_builder
            def configure_solver(builder):
                return (
                    builder.with_algorithm(Algorithm.RDIT)
                    .with_wavelengths(1.55)
                    .with_k_dimensions([5, 5])
                    .with_materials([self.air, self.silicon])
                    .with_device("cuda")
                )

            with self.assertWarns(UserWarning):
                solver2 = create_solver_from_builder(configure_solver)
            self.assertIsInstance(solver2, RDITSolver)
            self.assertEqual(str(solver2.device), "cpu")

            # Path 3: get_solver_builder
            with self.assertWarns(UserWarning):
                builder = get_solver_builder()
                solver3 = (
                    builder.with_algorithm(Algorithm.RDIT)
                    .with_wavelengths(1.55)
                    .with_k_dimensions([5, 5])
                    .with_materials([self.air, self.silicon])
                    .with_device("cuda")
                    .build()
                )
            self.assertIsInstance(solver3, RDITSolver)
            self.assertEqual(str(solver3.device), "cpu")

            # Path 4: from_config
            with self.assertWarns(UserWarning):
                builder = get_solver_builder()
                solver4 = builder.from_config(config).build()
            self.assertIsInstance(solver4, RDITSolver)
            self.assertEqual(str(solver4.device), "cpu")

    def test_config_device_override_with_with_device(self):
        """Test that with_device() override takes precedence over config device.

        When both from_config(device=X) and with_device(Y) are used,
        with_device() should override the config value before resolution.
        """
        config = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "cuda",
        }

        with patch("torchrdit.device.torch.cuda.is_available", return_value=False):
            builder = get_solver_builder()
            solver = (
                builder.from_config(config)
                .with_device("cpu")  # Override config device
                .build()
            )

            # Verify solver was created with CPU device (override wins)
            self.assertIsInstance(solver, RCWASolver)
            self.assertEqual(str(solver.device), "cpu")

    def test_mps_rejected_in_all_paths(self):
        """Test that MPS is rejected unconditionally in all creation paths."""
        # Test with create_solver_from_config
        config_mps = {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "mps",
        }

        with self.assertWarns(UserWarning) as cm:
            solver = create_solver_from_config(config_mps)

        self.assertEqual(str(solver.device), "cpu")
        warning_text = str(cm.warning).lower()
        self.assertIn("complex", warning_text)


if __name__ == "__main__":
    unittest.main()
