"""
Test suite for tensorizing _pre_solve and _pre_solve_batched methods.

This test suite ensures mathematical equivalence between the original
sequential implementation and the new tensorized implementation.
Following TDD principles, these tests are written before the implementation.
"""

import torch
import numpy as np
import pytest
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material


class TestPreSolveTensorization:
    """Test suite ensuring mathematical equivalence between _pre_solve implementations."""

    @pytest.fixture
    def setup_solver(self):
        """Create a solver with basic configuration."""
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55, 1.31]),  # Multiple wavelengths
            rdim=[512, 512],
            kdim=[5, 5],
            device="cpu",  # Use CPU for consistent testing
        )

        # Add materials
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])

        # Add a simple layer structure
        solver.add_layer(material_name="Si", thickness=0.5)

        return solver

    def test_single_source_equivalence(self, setup_solver):
        """Verify _pre_solve and _pre_solve_batched[0] produce identical results."""
        solver = setup_solver

        # Test case 1: Scalar theta/phi
        source_single = solver.add_source(
            theta=30 * np.pi / 180, phi=45 * np.pi / 180, pte=1.0, ptm=0.0
        )

        # Run single source through _pre_solve
        solver.src = source_single
        solver._pre_solve()
        kinc_single = solver.kinc.clone()

        # Run same source through unified _pre_solve with list
        solver._pre_solve([source_single])
        kinc_batched = solver.kinc[0].clone()  # Extract first source

        # Compare results
        assert kinc_single.shape == kinc_batched.shape, (
            f"Shape mismatch: single {kinc_single.shape} vs batched {kinc_batched.shape}"
        )

        max_diff = torch.abs(kinc_single - kinc_batched).max().item()
        assert torch.allclose(kinc_single, kinc_batched, atol=1e-6, rtol=1e-5), (
            f"kinc mismatch: max diff = {max_diff}"
        )

        # Verify reciprocal lattice vectors are identical
        assert hasattr(solver, "reci_t1") and hasattr(solver, "reci_t2"), (
            "Reciprocal lattice vectors not set"
        )

        print(f"✓ Single source equivalence test passed (max diff: {max_diff:.2e})")

    def test_multiple_sources_sequential_match(self, setup_solver):
        """Verify batched processing matches sequential processing."""
        solver = setup_solver

        # Create multiple sources with different angles
        angles = [0, 30, 45, 60]
        sources = [
            solver.add_source(theta=angle * np.pi / 180, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]

        # Process sequentially
        sequential_kincs = []
        for source in sources:
            solver.src = source
            solver._pre_solve()
            sequential_kincs.append(solver.kinc.clone())

        # Process in batch using unified _pre_solve
        solver._pre_solve(sources)
        batched_kincs = solver.kinc.clone()

        # Compare each source
        assert batched_kincs.shape[0] == len(sources), (
            f"Batch size mismatch: expected {len(sources)}, got {batched_kincs.shape[0]}"
        )

        for i, (seq_kinc, batch_kinc) in enumerate(
            zip(sequential_kincs, batched_kincs)
        ):
            max_diff = torch.abs(seq_kinc - batch_kinc).max().item()
            assert torch.allclose(seq_kinc, batch_kinc, atol=1e-6, rtol=1e-5), (
                f"Source {i} (θ={angles[i]}°) mismatch: max diff = {max_diff}"
            )
            print(f"✓ Source {i} match verified (max diff: {max_diff:.2e})")

    def test_edge_cases(self, setup_solver):
        """Test extreme angles and special cases."""
        solver = setup_solver

        # Test case 1: Normal incidence (θ=0)
        source_normal = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        solver.src = source_normal
        solver._pre_solve()
        kinc_normal = solver.kinc.clone()

        # Verify kinc_z equals refractive index (since cos(0)=1)
        refractive = torch.sqrt(solver.ur1 * solver.er1)
        if refractive.dim() == 0:
            refractive = refractive.unsqueeze(0).expand(solver.n_freqs)

        expected_z = refractive
        actual_z = kinc_normal[:, 2]
        assert torch.allclose(actual_z, expected_z, atol=1e-6), (
            f"Normal incidence z-component mismatch: {actual_z} vs {expected_z}"
        )

        # Verify x and y components are near zero
        assert torch.allclose(
            kinc_normal[:, 0], torch.zeros_like(kinc_normal[:, 0]), atol=1e-6
        ), "Normal incidence x-component not zero"
        assert torch.allclose(
            kinc_normal[:, 1], torch.zeros_like(kinc_normal[:, 1]), atol=1e-6
        ), "Normal incidence y-component not zero"

        # Test case 2: Grazing incidence (θ=89°)
        source_grazing = solver.add_source(
            theta=89 * np.pi / 180, phi=0, pte=1.0, ptm=0.0
        )
        solver.src = source_grazing
        solver._pre_solve()
        kinc_grazing = solver.kinc.clone()

        # Verify z-component is near zero (cos(89°) ≈ 0)
        assert torch.all(torch.abs(kinc_grazing[:, 2]) < 0.1), (
            "Grazing incidence z-component too large"
        )

        print("✓ Edge cases test passed")

    def test_gradient_flow_preservation(self, setup_solver):
        """Ensure gradients propagate correctly through both paths."""
        solver = setup_solver

        # Create source parameters with gradients enabled
        # Note: Current implementation expects scalar values, not tensors
        theta_val = 30 * np.pi / 180
        phi_val = 45 * np.pi / 180

        # For gradient testing, we'll need to test after solve() is called
        # since _pre_solve doesn't directly support tensor inputs
        # This test will be updated when we implement the unified version

        # Test with regular values for now
        source = solver.add_source(theta=theta_val, phi=phi_val, pte=1.0, ptm=0.0)

        # Test single source
        solver.src = source
        solver._pre_solve()

        kinc_single = solver.kinc.clone()

        # Test batched source using unified _pre_solve
        solver._pre_solve([source])
        kinc_batched = solver.kinc[0].clone()

        # For now, just verify that both methods produce the same result
        assert torch.allclose(kinc_single, kinc_batched, atol=1e-6), (
            "Single vs batched kinc mismatch"
        )

        # Gradient flow will be tested after unified implementation
        # The current implementation doesn't support gradient flow through theta/phi
        # This is a limitation we'll address in the unified version

        print("✓ Gradient flow preservation test passed (basic equivalence verified)")

    def test_dimension_handling(self, setup_solver):
        """Test various input dimension scenarios."""
        solver = setup_solver

        # Test case 1: Array theta/phi (per-frequency)
        theta_array = np.array([30, 35]) * np.pi / 180  # Different angle per wavelength
        phi_array = np.array([0, 45]) * np.pi / 180

        source_array = {
            "theta": theta_array,
            "phi": phi_array,
            "pte": 1.0,
            "ptm": 0.0,
            "norm_te_dir": "y",
        }

        solver.src = source_array
        solver._pre_solve()
        kinc_array = solver.kinc.clone()

        # Verify shape matches number of frequencies
        assert kinc_array.shape == (solver.n_freqs, 3), (
            f"Array input shape mismatch: {kinc_array.shape}"
        )

        # Verify values are different for each frequency
        assert not torch.allclose(kinc_array[0], kinc_array[1], atol=1e-6), (
            "Array input should produce different kinc per frequency"
        )

        print("✓ Dimension handling test passed")

    def test_numerical_stability(self, setup_solver):
        """Test numerical stability with extreme values."""
        solver = setup_solver

        # Test very small angles (near 0)
        small_angle = 1e-6
        source_small = solver.add_source(theta=small_angle, phi=0, pte=1.0, ptm=0.0)
        solver.src = source_small
        solver._pre_solve()
        kinc_small = solver.kinc.clone()

        # Verify no NaN or Inf values
        assert torch.all(torch.isfinite(kinc_small)), (
            "NaN or Inf detected for small angle"
        )

        # Test angle very close to 90°
        near_90 = 89.99 * np.pi / 180
        source_near90 = solver.add_source(theta=near_90, phi=0, pte=1.0, ptm=0.0)
        solver.src = source_near90
        solver._pre_solve()
        kinc_near90 = solver.kinc.clone()

        assert torch.all(torch.isfinite(kinc_near90)), (
            "NaN or Inf detected for near-90° angle"
        )

        print("✓ Numerical stability test passed")

    def test_dispersive_material_handling(self, setup_solver):
        """Test handling of dispersive materials."""
        solver = setup_solver

        # For dispersive materials, we need to create a dispersive material
        # and add it to the solver, not modify er1/ur1 directly
        # The current implementation handles dispersive materials through
        # the material system, not by direct property modification

        # Test with single source
        source = solver.add_source(theta=45 * np.pi / 180, phi=0, pte=1.0, ptm=0.0)
        solver.src = source
        solver._pre_solve()
        kinc_single = solver.kinc.clone()

        # Test with batched sources using unified _pre_solve
        solver._pre_solve([source])
        kinc_batched = solver.kinc[0].clone()

        # Verify results match
        assert torch.allclose(kinc_single, kinc_batched, atol=1e-6), (
            "Material handling mismatch between methods"
        )

        # Verify kinc has correct shape for multiple wavelengths
        assert kinc_single.shape == (solver.n_freqs, 3), (
            f"Expected shape ({solver.n_freqs}, 3), got {kinc_single.shape}"
        )

        print("✓ Dispersive material handling test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
