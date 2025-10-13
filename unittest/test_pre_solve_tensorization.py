"""
Unit tests for tensorized _pre_solve.

Focus:
- Single vs batched equivalence (multi-source sequential vs batched)
- Edge angles (normal and grazing) and dimension handling
- Real gradient flow through theta/phi in both single and batched paths

Notes:
- Removed self-referential or placeholder tests that did not validate distinct behavior.
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

    # Removed single-source self-comparison; multi-source test below subsumes it.

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
            # Assertions above ensure equivalence.

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

        # Assertions above ensure edge-case behavior.

    def test_gradients_theta_phi_single_and_batched(self, setup_solver):
        """Real gradient test: kinc should depend on theta/phi in both paths."""
        solver = setup_solver

        # Single-source path with differentiable theta/phi
        theta = torch.tensor(0.7, dtype=solver.tfloat, requires_grad=True)
        phi = torch.tensor(0.3, dtype=solver.tfloat, requires_grad=True)
        src = solver.add_source(theta=theta, phi=phi, pte=1.0, ptm=0.0)

        solver.src = src
        solver._pre_solve()
        kinc = solver.kinc
        loss = (kinc.real ** 2).sum()
        loss.backward()

        assert theta.grad is not None and torch.isfinite(theta.grad)
        assert phi.grad is not None and torch.isfinite(phi.grad)

        # Batched path with two independent differentiable angle pairs
        theta_b = [
            torch.tensor(0.5, dtype=solver.tfloat, requires_grad=True),
            torch.tensor(1.0, dtype=solver.tfloat, requires_grad=True),
        ]
        phi_b = [
            torch.tensor(0.2, dtype=solver.tfloat, requires_grad=True),
            torch.tensor(0.8, dtype=solver.tfloat, requires_grad=True),
        ]
        sources = [
            solver.add_source(theta=theta_b[0], phi=phi_b[0], pte=1.0, ptm=0.0),
            solver.add_source(theta=theta_b[1], phi=phi_b[1], pte=1.0, ptm=0.0),
        ]

        solver._pre_solve(sources)
        kinc_b = solver.kinc
        loss_b = (kinc_b.real ** 2).sum()
        loss_b.backward()

        for v in theta_b + phi_b:
            assert v.grad is not None and torch.isfinite(v.grad)

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

        # Assertions above ensure per-frequency inputs are handled correctly.

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

        # Assertions above ensure no NaN/Inf at extremes.

    # Removed dispersive-material test; it didn't introduce dispersion and duplicated equivalence checks.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
