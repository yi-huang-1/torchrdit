"""
Tests for tensor-level batched kinc calculation.

These tests verify that the solver's native batched path in `_pre_solve` matches
the sequential path, produces correct shapes, is independent of polarization,
and supports autograd. The tests avoid re-implementing kinc logic in test code.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm


class TestBatchedKincCalculation:
    """Test batched kinc calculation for tensor-level source processing."""

    def setup_solver(self, n_freqs=3):
        """Create a basic solver for testing."""
        # Create materials
        mat_air = create_material(name="air", permittivity=1.0)
        mat_si = create_material(name="silicon", permittivity=11.7)

        # Create solver
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.linspace(0.8, 1.2, n_freqs),
            rdim=[256, 256],
            kdim=[3, 3],
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.0, 1.0]]),
            device="cpu",
            is_use_FFF=False,
        )

        # Add materials and layers
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("air")
        solver.update_trn_material("air")
        solver.add_layer(material_name="silicon", thickness=torch.tensor(0.5), is_homogeneous=True)

        return solver

    def compute_kinc_sequential(self, solver, sources):
        """Compute kinc for each source using the solver sequentially."""
        kinc_list = []
        for src in sources:
            solver._pre_solve(src)
            kinc_list.append(solver.kinc.clone())
        return kinc_list

    def test_single_source_matches_original(self):
        """Test that batched computation with single source matches original."""
        solver = self.setup_solver(n_freqs=3)
        source = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}

        # Sequential (original)
        solver._pre_solve(source)
        kinc_original = solver.kinc.clone()

        # Batched with single source via solver path
        solver._pre_solve([source])
        kinc_batched = solver.kinc.clone()

        assert kinc_batched.shape == (1, 3, 3), f"Expected shape (1, 3, 3), got {kinc_batched.shape}"
        torch.testing.assert_close(kinc_batched[0], kinc_original, rtol=1e-6, atol=1e-8)

    def test_multiple_sources_dimension(self):
        """Test dimensions with multiple sources."""
        solver = self.setup_solver(n_freqs=3)
        
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": np.pi/4, "pte": 0.7, "ptm": 0.3},
            {"theta": 0.3, "phi": np.pi/2, "pte": 0.0, "ptm": 1.0},
        ]
        
        solver._pre_solve(sources)
        kinc_batched = solver.kinc

        assert kinc_batched.shape == (4, 3, 3), f"Expected shape (4, 3, 3), got {kinc_batched.shape}"
        assert kinc_batched.dim() == 3
        assert kinc_batched.size(0) == len(sources)
        assert kinc_batched.size(1) == solver.n_freqs
        assert kinc_batched.size(2) == 3

    def test_batched_matches_sequential(self):
        """Test that batched kinc matches sequential computation."""
        solver = self.setup_solver(n_freqs=5)
        
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/6, "phi": 0.0, "pte": 1.0, "ptm": 0.0},  # 30 degrees
            {"theta": np.pi/4, "phi": np.pi/4, "pte": 0.5, "ptm": 0.5},  # 45 degrees
            {"theta": np.pi/3, "phi": np.pi/2, "pte": 0.0, "ptm": 1.0},  # 60 degrees
        ]
        
        # Sequential computation (solver path)
        kinc_sequential = self.compute_kinc_sequential(solver, sources)

        # Batched computation (solver path)
        solver._pre_solve(sources)
        kinc_batched = solver.kinc
        
        # Compare each source
        for i, kinc_seq in enumerate(kinc_sequential):
            # NaN values are bugs - kinc should never contain NaN
            assert not torch.isnan(kinc_batched[i]).any(), \
                f"NaN values found in batched kinc for source {i} - this is a bug!"
            assert not torch.isnan(kinc_seq).any(), \
                f"NaN values found in sequential kinc for source {i} - this is a bug!"
            
            torch.testing.assert_close(
                kinc_batched[i], 
                kinc_seq, 
                rtol=1e-6, 
                atol=1e-8,
                msg=f"Mismatch for source {i}"
            )

    def test_edge_cases(self):
        """Test edge cases for kinc calculation."""
        solver = self.setup_solver(n_freqs=2)
        
        # Edge cases
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},  # Normal incidence
            {"theta": 1e-10, "phi": 0.0, "pte": 1.0, "ptm": 0.0},  # Near-normal
            {"theta": np.pi/2 - 1e-6, "phi": 0.0, "pte": 1.0, "ptm": 0.0},  # Near-grazing
        ]
        
        solver._pre_solve(sources)
        kinc_batched = solver.kinc
        
        # Check normal incidence (theta=0)
        # kinc should be [0, 0, 1] * refractive_index
        ur1 = solver.ur1
        er1 = solver.er1
        if not isinstance(ur1, torch.Tensor):
            ur1 = torch.tensor(ur1, dtype=solver.tcomplex, device=solver.device)
        if not isinstance(er1, torch.Tensor):
            er1 = torch.tensor(er1, dtype=solver.tcomplex, device=solver.device)
        refractive_1 = torch.sqrt(ur1 * er1)
        if refractive_1.dim() == 0:
            refractive_1 = refractive_1.unsqueeze(0).expand(solver.n_freqs)
        # Create expected values for all frequencies
        expected_normal = torch.zeros(solver.n_freqs, 3, dtype=solver.tfloat, device=solver.device)
        expected_normal[:, 2] = refractive_1.real
        torch.testing.assert_close(kinc_batched[0].real, expected_normal, rtol=1e-6, atol=1e-8)
        
        # Check near-grazing (theta ≈ π/2)
        # kinc_z should be very small
        assert torch.abs(kinc_batched[2, :, 2]).max() < 1e-5, "kinc_z should be near zero for grazing incidence"

    def test_different_wavelengths(self):
        """Test kinc calculation with different wavelength configurations."""
        # Single wavelength
        solver1 = self.setup_solver(n_freqs=1)
        sources = [
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": 0.1, "pte": 0.0, "ptm": 1.0},
        ]
        solver1._pre_solve(sources)
        kinc1 = solver1.kinc
        assert kinc1.shape == (2, 1, 3)

        # Many wavelengths
        solver2 = self.setup_solver(n_freqs=10)
        solver2._pre_solve(sources)
        kinc2 = solver2.kinc
        assert kinc2.shape == (2, 10, 3)

    def test_polarization_independence(self):
        """Test that kinc is independent of polarization (pte, ptm)."""
        solver = self.setup_solver(n_freqs=3)

        theta, phi = 0.2, 0.1
        sources_different_pol = [
            {"theta": theta, "phi": phi, "pte": 1.0, "ptm": 0.0},
            {"theta": theta, "phi": phi, "pte": 0.0, "ptm": 1.0},
            {"theta": theta, "phi": phi, "pte": 0.7, "ptm": 0.3},
        ]

        solver._pre_solve(sources_different_pol)
        kinc_batched = solver.kinc

        # All should be identical since kinc doesn't depend on polarization
        torch.testing.assert_close(kinc_batched[0], kinc_batched[1], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(kinc_batched[0], kinc_batched[2], rtol=1e-10, atol=1e-12)

    def test_gradient_flow(self):
        """Test that gradients can flow through batched kinc calculation."""
        solver = self.setup_solver(n_freqs=3)

        # Create sources with gradient-enabled angles (preserve tensors in dicts)
        theta_values = torch.tensor([0.1, 0.2, 0.3], dtype=solver.tfloat, requires_grad=True)
        phi_values = torch.tensor([0.0, 0.1, 0.2], dtype=solver.tfloat, requires_grad=True)

        sources = [
            {"theta": theta_values[i], "phi": phi_values[i], "pte": 1.0, "ptm": 0.0}
            for i in range(3)
        ]

        # Use solver's batched path, which preserves autograd
        solver._pre_solve(sources)
        kinc_batched = solver.kinc

        loss = kinc_batched.real.sum()
        loss.backward()

        assert theta_values.grad is not None, "Gradients should flow to theta"
        assert phi_values.grad is not None, "Gradients should flow to phi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
