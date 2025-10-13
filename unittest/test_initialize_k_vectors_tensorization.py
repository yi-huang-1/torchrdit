"""
Test suite for _initialize_k_vectors tensorization.

This module tests the unified tensorized implementation of _initialize_k_vectors
and validates that it produces mathematically equivalent results for both
single and batched source inputs.

Following TDD principles: tests are written first to define expected behavior,
then implementation is created to satisfy the tests.
"""

import numpy as np
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
        """Test near-grazing incidence (θ ≈ π/2) yields finite outputs."""
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
        assert torch.isfinite(kx_batched).all(), (
            "kx_batched contains non-finite values at grazing incidence"
        )
        assert torch.isfinite(ky_batched).all(), (
            "ky_batched contains non-finite values at grazing incidence"
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
        """Zero-order terms are relaxed (non-zero) and consistent across flows."""
        # Create a source that will likely have zero k-components
        source = self.solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)

        # Test single source
        self.solver.src = source
        self.solver._pre_solve()
        kx_single, ky_single, _, _ = self.solver._initialize_k_vectors()

        # Test batched source
        self.solver._pre_solve([source])
        kx_batched, ky_batched, _, _ = self.solver._initialize_k_vectors()

        # Check that relaxation was applied (no exact zeros should remain)
        zero_order_idx = self.solver.kdim[0] // 2, self.solver.kdim[1] // 2

        kx_zero_single = kx_single[0, zero_order_idx[0], zero_order_idx[1]]
        ky_zero_single = ky_single[0, zero_order_idx[0], zero_order_idx[1]]

        kx_zero_batched = kx_batched[0, 0, zero_order_idx[0], zero_order_idx[1]]
        ky_zero_batched = ky_batched[0, 0, zero_order_idx[0], zero_order_idx[1]]

        # Both should be non-zero but small in magnitude and finite
        assert torch.isfinite(kx_zero_single).item(), "kx_zero_single is not finite"
        assert torch.isfinite(ky_zero_single).item(), "ky_zero_single is not finite"
        assert torch.isfinite(kx_zero_batched).item(), "kx_zero_batched is not finite"
        assert torch.isfinite(ky_zero_batched).item(), "ky_zero_batched is not finite"

        assert abs(kx_zero_single.real) > 0.0 and abs(kx_zero_single.real) < 1e-3, (
            f"kx relaxation not applied (value={kx_zero_single})"
        )
        assert abs(ky_zero_single.real) > 0.0 and abs(ky_zero_single.real) < 1e-3, (
            f"ky relaxation not applied (value={ky_zero_single})"
        )

        # And both flows should agree tightly
        assert abs((kx_zero_single - kx_zero_batched).real) < 1e-10, (
            "kx relaxation differs between flows"
        )
        assert abs((ky_zero_single - ky_zero_batched).real) < 1e-10, (
            "ky relaxation differs between flows"
        )


class TestAutogradSmoke:
    """Smoke test: autograd flows via public add_source/_pre_solve path (single source)."""

    def setup_method(self):
        self.device = "cpu"
        self.solver = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.DOUBLE,
            lam0=np.array([1.55, 1.31]),
            rdim=[64, 64],
            kdim=[3, 3],
            device=self.device,
        )

        air = create_material(name="air", permittivity=1.0)
        self.solver.add_materials([air])
        self.solver.add_layer(material_name="air", thickness=torch.tensor(1.0))

    def test_single_source_autograd_public_api(self):
        # Parameters with gradients
        theta = torch.tensor(0.12, requires_grad=True, dtype=torch.float64)
        phi = torch.tensor(0.34, requires_grad=True, dtype=torch.float64)

        # Public API path: add_source + _pre_solve without manual kinc
        source = self.solver.add_source(theta=theta, phi=phi, pte=1.0, ptm=0.0)
        self.solver.src = source
        self.solver._pre_solve()

        # Initialize k-vectors and build a simple scalar loss
        kx, ky, kz_ref, kz_trn = self.solver._initialize_k_vectors()
        loss = (kx.real**2 + ky.real**2).sum()
        loss.backward()

        # Gradients should reach theta/phi and be finite
        assert theta.grad is not None, "No gradient for theta via public API"
        assert phi.grad is not None, "No gradient for phi via public API"
        assert torch.isfinite(theta.grad).item(), f"theta gradient not finite: {theta.grad}"
        assert torch.isfinite(phi.grad).item(), f"phi gradient not finite: {phi.grad}"

    def test_batched_sources_autograd_public_api(self):
        """Batched autograd via public API should preserve gradients for angles.

        This currently fails because _pre_solve builds batched theta/phi using
        torch.tensor([...]) which detaches from input tensors. This test
        documents the desired behavior and will pass after fixing _pre_solve.
        """
        # Create batched angle tensors with gradients
        theta_list = [
            torch.tensor(0.10, requires_grad=True, dtype=torch.float64),
            torch.tensor(0.20, requires_grad=True, dtype=torch.float64),
            torch.tensor(0.30, requires_grad=True, dtype=torch.float64),
        ]
        phi_list = [
            torch.tensor(0.00, requires_grad=True, dtype=torch.float64),
            torch.tensor(0.10, requires_grad=True, dtype=torch.float64),
            torch.tensor(0.20, requires_grad=True, dtype=torch.float64),
        ]

        # Build sources via public API
        sources = [
            self.solver.add_source(theta=t, phi=p, pte=1.0, ptm=0.0)
            for t, p in zip(theta_list, phi_list)
        ]

        # Pre-solve and initialize k-vectors
        self.solver._pre_solve(sources)
        kx, ky, kz_ref, kz_trn = self.solver._initialize_k_vectors()

        # Define scalar loss and backpropagate
        loss = (kx.real**2 + ky.real**2).sum()
        loss.backward()

        # Expect gradients to flow to each angle tensor
        for t, p in zip(theta_list, phi_list):
            assert t.grad is not None, "No gradient for a batched theta via public API"
            assert p.grad is not None, "No gradient for a batched phi via public API"
            assert torch.isfinite(t.grad).item(), f"theta grad not finite: {t.grad}"
            assert torch.isfinite(p.grad).item(), f"phi grad not finite: {p.grad}"
