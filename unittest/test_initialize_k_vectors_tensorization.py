"""
Test suite for _initialize_k_vectors tensorization.

This module tests the unified tensorized implementation of _initialize_k_vectors
and validates that it produces mathematically equivalent results for both
single and batched source inputs.

Following TDD principles: tests are written first to define expected behavior,
then implementation is created to satisfy the tests.
"""

import numpy as np
import pytest
import torch

from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm, Precision
from torchrdit.utils import create_material


class TestMathematicalEquivalence:
    """Test mathematical equivalence between current implementations."""

    def setup_method(self):
        """Set up test fixture with a basic solver configuration."""
        self.device = "cpu"  # Use CPU for deterministic testing
        self.wavelengths = np.array([1.55, 1.31, 1.0])  # Multiple wavelengths
        self.tolerance_abs = 1e-10  # Absolute tolerance for comparison
        self.tolerance_rel = 1e-6  # Relative tolerance for comparison

        # Create a solver with known configuration
        self.solver = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.DOUBLE,  # Use double precision for accuracy
            lam0=self.wavelengths,
            rdim=[256, 256],
            kdim=[3, 3],
            device=self.device,
        )

        # Add basic materials
        air = create_material(name="air", permittivity=1.0)
        silicon = create_material(name="Si", permittivity=11.9)
        self.solver.add_materials([air, silicon])

        # Add a simple layer structure
        self.solver.add_layer(material_name="Si", thickness=torch.tensor(0.5))

    def test_single_source_both_methods(self):
        """Test that both methods give identical results for single source."""
        # Create a single source
        source = self.solver.add_source(theta=0.1, phi=0.2, pte=1.0, ptm=0.0)

        # Set up solver for single source (calls _pre_solve)
        self.solver.src = source
        self.solver._pre_solve()

        # Call the single-source method
        kx_single, ky_single, kz_ref_single, kz_trn_single = (
            self.solver._initialize_k_vectors()
        )

        # Set up solver for batched source with single element (calls _pre_solve_batched)
        sources = [source]
        self.solver._pre_solve(sources)

        # Call the unified method (now handles batched automatically)
        kx_batched, ky_batched, kz_ref_batched, kz_trn_batched = (
            self.solver._initialize_k_vectors()
        )

        # Extract single source results from batched
        kx_batched_single = kx_batched[0]
        ky_batched_single = ky_batched[0]
        kz_ref_batched_single = kz_ref_batched[0]
        kz_trn_batched_single = kz_trn_batched[0]

        # Compare results
        self._assert_tensors_equal(kx_single, kx_batched_single, "kx_0")
        self._assert_tensors_equal(ky_single, ky_batched_single, "ky_0")
        self._assert_tensors_equal(kz_ref_single, kz_ref_batched_single, "kz_ref_0")
        self._assert_tensors_equal(kz_trn_single, kz_trn_batched_single, "kz_trn_0")

    def test_batched_vs_sequential(self):
        """Test that batched processing gives same results as sequential."""
        # Create multiple sources with different angles
        sources = [
            self.solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0),
            self.solver.add_source(theta=0.1, phi=0.2, pte=0.8, ptm=0.6),
            self.solver.add_source(theta=0.3, phi=0.1, pte=0.0, ptm=1.0),
        ]

        # Process sources sequentially
        sequential_results = []
        for source in sources:
            self.solver.src = source
            self.solver._pre_solve()
            result = self.solver._initialize_k_vectors()
            sequential_results.append(result)

        # Process sources in batch
        self.solver._pre_solve(sources)
        kx_batched, ky_batched, kz_ref_batched, kz_trn_batched = (
            self.solver._initialize_k_vectors()
        )

        # Compare each source's results
        for i, (kx_seq, ky_seq, kz_ref_seq, kz_trn_seq) in enumerate(
            sequential_results
        ):
            self._assert_tensors_equal(kx_seq, kx_batched[i], f"kx_0 source {i}")
            self._assert_tensors_equal(ky_seq, ky_batched[i], f"ky_0 source {i}")
            self._assert_tensors_equal(
                kz_ref_seq, kz_ref_batched[i], f"kz_ref_0 source {i}"
            )
            self._assert_tensors_equal(
                kz_trn_seq, kz_trn_batched[i], f"kz_trn_0 source {i}"
            )

    def test_multiple_frequencies(self):
        """Test with multiple frequencies to ensure frequency dimension is handled correctly."""
        # Use solver with multiple wavelengths (already configured in setup)
        source = self.solver.add_source(theta=0.2, phi=0.3, pte=1.0, ptm=0.0)

        # Single source method
        self.solver.src = source
        self.solver._pre_solve()
        kx_single, ky_single, kz_ref_single, kz_trn_single = (
            self.solver._initialize_k_vectors()
        )

        # Batched method with single source
        self.solver._pre_solve([source])
        kx_batched, ky_batched, kz_ref_batched, kz_trn_batched = (
            self.solver._initialize_k_vectors()
        )

        # Verify shapes
        n_freqs = len(self.wavelengths)
        kdim = self.solver.kdim

        # Single source shapes: (n_freqs, kdim[0], kdim[1])
        assert kx_single.shape == (n_freqs, kdim[0], kdim[1])
        assert ky_single.shape == (n_freqs, kdim[0], kdim[1])

        # Batched shapes: (1, n_freqs, kdim[0], kdim[1])
        assert kx_batched.shape == (1, n_freqs, kdim[0], kdim[1])
        assert ky_batched.shape == (1, n_freqs, kdim[0], kdim[1])

        # Compare values
        self._assert_tensors_equal(kx_single, kx_batched[0], "kx_0 multi-freq")
        self._assert_tensors_equal(ky_single, ky_batched[0], "ky_0 multi-freq")

    def _assert_tensors_equal(self, tensor1, tensor2, name):
        """Assert that two tensors are equal within tolerance."""
        # Check shapes match
        assert tensor1.shape == tensor2.shape, (
            f"{name}: Shape mismatch {tensor1.shape} vs {tensor2.shape}"
        )

        # Check values are close
        diff_abs = torch.abs(tensor1 - tensor2)
        max_diff_abs = diff_abs.max().item()

        # Relative difference (avoid division by zero)
        tensor1_abs = torch.abs(tensor1)
        tensor2_abs = torch.abs(tensor2)
        max_magnitude = torch.max(tensor1_abs, tensor2_abs)
        diff_rel = diff_abs / (max_magnitude + 1e-12)  # Add small epsilon
        max_diff_rel = diff_rel.max().item()

        # Assert within tolerances
        assert max_diff_abs < self.tolerance_abs, (
            f"{name}: Absolute difference {max_diff_abs:.2e} exceeds tolerance {self.tolerance_abs:.2e}"
        )

        assert max_diff_rel < self.tolerance_rel, (
            f"{name}: Relative difference {max_diff_rel:.2e} exceeds tolerance {self.tolerance_rel:.2e}"
        )


class TestEdgeCases:
    """Test edge cases that could cause numerical issues."""

    def setup_method(self):
        """Set up test fixture."""
        self.device = "cpu"
        self.solver = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.DOUBLE,
            lam0=np.array([1.55]),
            rdim=[256, 256],
            kdim=[5, 5],  # Use larger kdim for better testing
            device=self.device,
        )

        # Add materials
        air = create_material(name="air", permittivity=1.0)
        self.solver.add_materials([air])
        self.solver.add_layer(material_name="air", thickness=torch.tensor(1.0))

    def test_zero_incidence_angle(self):
        """Test normal incidence (θ=0, φ=0)."""
        source = self.solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)

        # Test both methods work without errors
        self.solver.src = source
        self.solver._pre_solve()
        kx_single, ky_single, kz_ref_single, kz_trn_single = (
            self.solver._initialize_k_vectors()
        )

        self.solver._pre_solve([source])
        kx_batched, ky_batched, kz_ref_batched, kz_trn_batched = (
            self.solver._initialize_k_vectors()
        )

        # Check that results are finite (no NaN or Inf)
        assert torch.isfinite(kx_single).all(), "kx_single contains non-finite values"
        assert torch.isfinite(ky_single).all(), "ky_single contains non-finite values"
        assert torch.isfinite(kx_batched).all(), "kx_batched contains non-finite values"
        assert torch.isfinite(ky_batched).all(), "ky_batched contains non-finite values"

    def test_grazing_incidence(self):
        """Test near-grazing incidence (θ ≈ π/2)."""
        # Use angle close to but not exactly 90 degrees to avoid singularities
        theta_grazing = np.pi / 2 - 0.01  # 89.43 degrees
        source = self.solver.add_source(theta=theta_grazing, phi=0.0, pte=1.0, ptm=0.0)

        # Test both methods work without errors
        self.solver.src = source
        self.solver._pre_solve()
        kx_single, ky_single, kz_ref_single, kz_trn_single = (
            self.solver._initialize_k_vectors()
        )

        self.solver._pre_solve([source])
        kx_batched, ky_batched, kz_ref_batched, kz_trn_batched = (
            self.solver._initialize_k_vectors()
        )

        # Check that results are finite
        assert torch.isfinite(kx_single).all(), (
            "kx_single contains non-finite values at grazing incidence"
        )
        assert torch.isfinite(ky_single).all(), (
            "ky_single contains non-finite values at grazing incidence"
        )

    def test_complex_refractive_index(self):
        """Test with materials having complex refractive indices."""
        # Create material with complex permittivity (lossy material)
        lossy_material = create_material(name="lossy", permittivity=3.0 + 0.1j)
        self.solver.add_materials([lossy_material])
        self.solver.add_layer(material_name="lossy", thickness=torch.tensor(0.5))

        source = self.solver.add_source(theta=0.2, phi=0.1, pte=1.0, ptm=0.0)

        # Test both methods work with complex materials
        self.solver.src = source
        self.solver._pre_solve()
        kx_single, ky_single, kz_ref_single, kz_trn_single = (
            self.solver._initialize_k_vectors()
        )

        self.solver._pre_solve([source])
        kx_batched, ky_batched, kz_ref_batched, kz_trn_batched = (
            self.solver._initialize_k_vectors()
        )

        # Results should be complex but finite
        assert torch.isfinite(kx_single).all(), (
            "kx_single non-finite with complex material"
        )
        assert torch.isfinite(ky_single).all(), (
            "ky_single non-finite with complex material"
        )
        assert torch.isfinite(kz_ref_single).all(), (
            "kz_ref_single non-finite with complex material"
        )
        assert torch.isfinite(kz_trn_single).all(), (
            "kz_trn_single non-finite with complex material"
        )

    def test_numerical_relaxation(self):
        """Test that numerical relaxation is applied consistently."""
        # Create a source that will likely have zero k-components
        source = self.solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)

        # Test single source
        self.solver.src = source
        self.solver._pre_solve()
        kx_single, ky_single, _, _ = self.solver._initialize_k_vectors()

        # Test batched source
        self.solver._pre_solve([source])
        kx_batched, ky_batched, _, _ = self.solver._initialize_k_vectors()

        # Check that relaxation was applied (no exact zeros should remain after relaxation)
        epsilon = 1e-6  # Expected relaxation value

        # For normal incidence, the zero-order term should be close to epsilon
        # (after relaxation is applied)
        zero_order_idx = self.solver.kdim[0] // 2, self.solver.kdim[1] // 2

        kx_zero_single = kx_single[0, zero_order_idx[0], zero_order_idx[1]]
        ky_zero_single = ky_single[0, zero_order_idx[0], zero_order_idx[1]]

        kx_zero_batched = kx_batched[0, 0, zero_order_idx[0], zero_order_idx[1]]
        ky_zero_batched = ky_batched[0, 0, zero_order_idx[0], zero_order_idx[1]]

        # Both should be close to epsilon (not exactly zero)
        assert abs(kx_zero_single.real - epsilon) < 1e-8, (
            f"kx relaxation not applied correctly: {kx_zero_single}"
        )
        assert abs(ky_zero_single.real - epsilon) < 1e-8, (
            f"ky relaxation not applied correctly: {ky_zero_single}"
        )

        # And both methods should give same result
        assert abs(kx_zero_single - kx_zero_batched) < 1e-12, (
            "kx relaxation differs between methods"
        )
        assert abs(ky_zero_single - ky_zero_batched) < 1e-12, (
            "ky relaxation differs between methods"
        )


class TestGradientPreservation:
    """Test that gradient computation works correctly through all operations."""

    def setup_method(self):
        """Set up test fixture with gradient-enabled parameters."""
        self.device = "cpu"
        self.solver = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.DOUBLE,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[3, 3],
            device=self.device,
        )

        # Add materials
        air = create_material(name="air", permittivity=1.0)
        self.solver.add_materials([air])
        self.solver.add_layer(material_name="air", thickness=torch.tensor(1.0))

    def test_gradient_flow_single_source(self):
        """Test gradient flow through single source k-vector calculation."""
        # Create source parameters with gradients enabled
        theta = torch.tensor(0.1, requires_grad=True, dtype=torch.float64)
        phi = torch.tensor(0.2, requires_grad=True, dtype=torch.float64)

        # Manually set up kinc with gradients (simulating _pre_solve behavior)
        self.solver._setup_reciprocal_space()

        # Compute kinc with gradients (matching _pre_solve pattern)
        refractive_1 = torch.sqrt(self.solver.ur1 * self.solver.er1).real
        if refractive_1.dim() == 0:  # non-dispersive
            refractive_1 = refractive_1.unsqueeze(0).expand(self.solver.n_freqs)

        # Create frequency-expanded angles
        if theta.dim() == 0:
            theta_exp = theta.unsqueeze(0).expand(self.solver.n_freqs)
            phi_exp = phi.unsqueeze(0).expand(self.solver.n_freqs)
        else:
            theta_exp = theta
            phi_exp = phi

        # Compute kinc components
        kinc_x = refractive_1 * torch.sin(theta_exp) * torch.cos(phi_exp)
        kinc_y = refractive_1 * torch.sin(theta_exp) * torch.sin(phi_exp)
        kinc_z = refractive_1 * torch.cos(theta_exp)

        kinc = torch.stack([kinc_x, kinc_y, kinc_z], dim=1)  # Shape: (n_freqs, 3)

        self.solver.kinc = kinc

        # Compute k-vectors
        kx, ky, kz_ref, kz_trn = self.solver._initialize_k_vectors()

        # Compute scalar loss from results (use real part for autograd compatibility)
        loss = (kx.real**2 + ky.real**2).sum()

        # Backpropagate
        loss.backward()

        # Check that gradients exist and are finite
        assert theta.grad is not None, "No gradient for theta"
        assert phi.grad is not None, "No gradient for phi"
        assert torch.isfinite(theta.grad), f"theta gradient is not finite: {theta.grad}"
        assert torch.isfinite(phi.grad), f"phi gradient is not finite: {phi.grad}"

    def test_gradient_flow_batched_source(self):
        """Test gradient flow through batched source k-vector calculation."""
        # Create multiple source parameters with gradients
        theta_batch = torch.tensor(
            [0.1, 0.2, 0.3], requires_grad=True, dtype=torch.float64
        )
        phi_batch = torch.tensor(
            [0.0, 0.1, 0.2], requires_grad=True, dtype=torch.float64
        )

        # Set up reciprocal space
        self.solver._setup_reciprocal_space()

        # Compute batched kinc with gradients (matching _pre_solve pattern)
        refractive_1 = torch.sqrt(self.solver.ur1 * self.solver.er1).real
        if refractive_1.dim() == 0:  # non-dispersive
            refractive_1 = refractive_1.unsqueeze(0).expand(self.solver.n_freqs)

        # Expand batch angles to match frequencies
        # Shape: (n_sources,) -> (n_sources, n_freqs)
        theta_exp = theta_batch.unsqueeze(1).expand(-1, self.solver.n_freqs)
        phi_exp = phi_batch.unsqueeze(1).expand(-1, self.solver.n_freqs)

        # Compute kinc components with broadcasting
        kinc_x = refractive_1 * torch.sin(theta_exp) * torch.cos(phi_exp)
        kinc_y = refractive_1 * torch.sin(theta_exp) * torch.sin(phi_exp)
        kinc_z = refractive_1 * torch.cos(theta_exp)

        kinc = torch.stack(
            [kinc_x, kinc_y, kinc_z], dim=2
        )  # Shape: (n_sources, n_freqs, 3)
        self.solver.kinc = kinc

        # Compute k-vectors
        kx, ky, kz_ref, kz_trn = self.solver._initialize_k_vectors()

        # Compute scalar loss (use real part for autograd compatibility)
        loss = (kx.real**2 + ky.real**2).sum()

        # Backpropagate
        loss.backward()

        # Check gradients
        assert theta_batch.grad is not None, "No gradient for theta_batch"
        assert phi_batch.grad is not None, "No gradient for phi_batch"
        assert torch.isfinite(theta_batch.grad).all(), (
            f"theta_batch gradients not finite: {theta_batch.grad}"
        )
        assert torch.isfinite(phi_batch.grad).all(), (
            f"phi_batch gradients not finite: {phi_batch.grad}"
        )

    def test_gradient_consistency(self):
        """Test that gradients are consistent between single and batched methods."""
        # Use same angle for both tests
        theta_val = 0.15
        phi_val = 0.25

        # Single source gradient test
        theta_single = torch.tensor(theta_val, requires_grad=True, dtype=torch.float64)
        phi_single = torch.tensor(phi_val, requires_grad=True, dtype=torch.float64)

        self.solver._setup_reciprocal_space()
        refractive_1 = torch.sqrt(self.solver.ur1 * self.solver.er1).real
        if refractive_1.dim() == 0:  # non-dispersive
            refractive_1 = refractive_1.unsqueeze(0).expand(self.solver.n_freqs)

        # Single source pattern
        theta_exp = theta_single.unsqueeze(0).expand(self.solver.n_freqs)
        phi_exp = phi_single.unsqueeze(0).expand(self.solver.n_freqs)

        kinc_x = refractive_1 * torch.sin(theta_exp) * torch.cos(phi_exp)
        kinc_y = refractive_1 * torch.sin(theta_exp) * torch.sin(phi_exp)
        kinc_z = refractive_1 * torch.cos(theta_exp)

        kinc_single = torch.stack([kinc_x, kinc_y, kinc_z], dim=1)

        self.solver.kinc = kinc_single
        kx_single, ky_single, _, _ = self.solver._initialize_k_vectors()
        loss_single = (kx_single.real**2 + ky_single.real**2).sum()
        loss_single.backward()

        grad_theta_single = theta_single.grad.clone()
        grad_phi_single = phi_single.grad.clone()

        # Batched source gradient test (single element batch)
        theta_batch = torch.tensor([theta_val], requires_grad=True, dtype=torch.float64)
        phi_batch = torch.tensor([phi_val], requires_grad=True, dtype=torch.float64)

        # Batched source pattern
        if refractive_1.dim() == 0:  # non-dispersive
            refractive_1 = refractive_1.unsqueeze(0).expand(self.solver.n_freqs)

        # Expand batch angles to match frequencies
        theta_exp = theta_batch.unsqueeze(1).expand(-1, self.solver.n_freqs)
        phi_exp = phi_batch.unsqueeze(1).expand(-1, self.solver.n_freqs)

        kinc_x = refractive_1 * torch.sin(theta_exp) * torch.cos(phi_exp)
        kinc_y = refractive_1 * torch.sin(theta_exp) * torch.sin(phi_exp)
        kinc_z = refractive_1 * torch.cos(theta_exp)

        kinc_batched = torch.stack(
            [kinc_x, kinc_y, kinc_z], dim=2
        )  # Shape: (1, n_freqs, 3)

        self.solver.kinc = kinc_batched
        kx_batched, ky_batched, _, _ = self.solver._initialize_k_vectors()
        loss_batched = (kx_batched.real**2 + ky_batched.real**2).sum()
        loss_batched.backward()

        grad_theta_batched = theta_batch.grad[0]
        grad_phi_batched = phi_batch.grad[0]

        # Compare gradients
        grad_diff_theta = abs(grad_theta_single - grad_theta_batched)
        grad_diff_phi = abs(grad_phi_single - grad_phi_batched)

        assert grad_diff_theta < 1e-10, f"Theta gradient differs: {grad_diff_theta}"
        assert grad_diff_phi < 1e-10, f"Phi gradient differs: {grad_diff_phi}"


class TestHelperFunctions:
    """Test helper functions used by k-vector initialization."""

    def setup_method(self):
        """Set up test fixture."""
        self.device = "cpu"
        self.solver = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.DOUBLE,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[3, 3],
            device=self.device,
        )

    def test_apply_numerical_relaxation_equivalence(self):
        """Test equivalence between relaxation methods."""
        # Create test tensors
        kx_single = torch.zeros((1, 3, 3), dtype=torch.complex128)
        ky_single = torch.zeros((1, 3, 3), dtype=torch.complex128)

        kx_batched = torch.zeros((2, 1, 3, 3), dtype=torch.complex128)
        ky_batched = torch.zeros((2, 1, 3, 3), dtype=torch.complex128)

        epsilon = 1e-6

        # Apply relaxation methods
        kx_single_relaxed, ky_single_relaxed = self.solver._apply_numerical_relaxation(
            kx_single, ky_single, epsilon
        )

        kx_batched_relaxed, ky_batched_relaxed = (
            self.solver._apply_numerical_relaxation(kx_batched, ky_batched, epsilon)
        )

        # Check that zero values were replaced with epsilon
        assert torch.allclose(
            kx_single_relaxed.real, torch.full_like(kx_single_relaxed.real, epsilon)
        )
        assert torch.allclose(
            ky_single_relaxed.real, torch.full_like(ky_single_relaxed.real, epsilon)
        )
        assert torch.allclose(
            kx_batched_relaxed.real, torch.full_like(kx_batched_relaxed.real, epsilon)
        )
        assert torch.allclose(
            ky_batched_relaxed.real, torch.full_like(ky_batched_relaxed.real, epsilon)
        )

    def test_calculate_kz_region_equivalence(self):
        """Test equivalence between kz calculation methods."""
        # Create test k-vectors
        kx_single = torch.randn((1, 3, 3), dtype=torch.complex128) * 0.1
        ky_single = torch.randn((1, 3, 3), dtype=torch.complex128) * 0.1

        kx_batched = kx_single.unsqueeze(0)  # Add batch dimension
        ky_batched = ky_single.unsqueeze(0)

        # Test non-dispersive case
        ur, er = torch.tensor(1.0), torch.tensor(2.25)  # Convert to tensors

        kz_single = self.solver._calculate_kz_region(
            ur, er, kx_single, ky_single, is_dispersive=False
        )
        kz_batched = self.solver._calculate_kz_region(
            ur, er, kx_batched, ky_batched, is_dispersive=False
        )

        # Compare results
        assert torch.allclose(kz_single, kz_batched[0], atol=1e-12), (
            "kz calculation differs between methods"
        )

        # Test dispersive case
        ur_disp = torch.tensor([1.0], dtype=torch.complex128)
        er_disp = torch.tensor([2.25], dtype=torch.complex128)

        kz_single_disp = self.solver._calculate_kz_region(
            ur_disp, er_disp, kx_single, ky_single, is_dispersive=True
        )
        kz_batched_disp = self.solver._calculate_kz_region(
            ur_disp, er_disp, kx_batched, ky_batched, is_dispersive=True
        )

        assert torch.allclose(kz_single_disp, kz_batched_disp[0], atol=1e-12), (
            "Dispersive kz calculation differs"
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
