"""
Tests for tensor-level batched polarization calculation.

These tests drive the solver's native `_calculate_polarization` via `_pre_solve`
for single and batched sources, avoiding any reimplementation of polarization
math in the tests.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm


class TestBatchedPolarization:
    """Test batched polarization calculation for tensor-level source processing."""

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

    def compute_polarization_sequential(self, solver, sources):
        """Compute polarization per source through solver's implementation."""
        out = []
        for src in sources:
            solver._pre_solve(src)
            pol = solver._calculate_polarization(sources=src)
            out.append({
                'ate': pol['ate'],
                'atm': pol['atm'],
                'pol_vec': pol['pol_vec'],
            })
        return out

    def test_single_source_matches_original(self):
        """Test that batched computation with single source matches original."""
        solver = self.setup_solver(n_freqs=3)

        source = {"theta": 0.1, "phi": 0.2, "pte": 0.7, "ptm": 0.3}

        # Sequential (solver path)
        pol_seq = self.compute_polarization_sequential(solver, [source])[0]

        # Batched with single source (solver path)
        solver._pre_solve([source])
        pol_b = solver._calculate_polarization(sources=[source])

        torch.testing.assert_close(pol_b['ate'][0], pol_seq['ate'], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(pol_b['atm'][0], pol_seq['atm'], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(pol_b['pol_vec'][0], pol_seq['pol_vec'], rtol=1e-6, atol=1e-8)

    def test_multiple_sources_dimensions(self):
        """Test dimensions with multiple sources."""
        solver = self.setup_solver(n_freqs=3)

        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": np.pi/4, "pte": 0.7, "ptm": 0.3},
            {"theta": 0.3, "phi": np.pi/2, "pte": 0.0, "ptm": 1.0},
        ]

        solver._pre_solve(sources)
        pol_b = solver._calculate_polarization(sources=sources)

        # Check dimensions: (n_sources, n_freqs, 3)
        assert pol_b['ate'].shape == (4, 3, 3), "ate shape mismatch"
        assert pol_b['atm'].shape == (4, 3, 3), "atm shape mismatch"
        assert pol_b['pol_vec'].shape == (4, 3, 3), "pol_vec shape mismatch"

    def test_normal_incidence_edge_case(self):
        """Test polarization for normal incidence (theta ≈ 0)."""
        solver = self.setup_solver(n_freqs=2)

        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},        # Exact zero
            {"theta": 1e-10, "phi": 0.0, "pte": 1.0, "ptm": 0.0},      # Near zero
            {"theta": 1e-7, "phi": np.pi/4, "pte": 0.5, "ptm": 0.5},   # Very small angle
        ]

        solver._pre_solve(sources)
        pol_b = solver._calculate_polarization(sources=sources)

        # For normal incidence, ate ~ [0, 1, 0] and atm ~ [1, 0, 0]
        expected_ate = torch.tensor([0.0, 1.0, 0.0], dtype=solver.tcomplex, device=solver.device)
        expected_atm = torch.tensor([1.0, 0.0, 0.0], dtype=solver.tcomplex, device=solver.device)
        ate_target = expected_ate[None, :].expand(solver.n_freqs, -1)
        atm_target = expected_atm[None, :].expand(solver.n_freqs, -1)

        for i in range(len(sources)):
            torch.testing.assert_close(pol_b['ate'][i], ate_target, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(pol_b['atm'][i], atm_target, rtol=1e-5, atol=1e-5)

    def test_grazing_incidence_edge_case(self):
        """Test polarization for near-grazing incidence."""
        solver = self.setup_solver(n_freqs=2)

        # Near grazing angles
        sources = [
            {"theta": np.pi/2 - 1e-3, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/2 - 1e-2, "phi": np.pi/4, "pte": 0.0, "ptm": 1.0},
            {"theta": np.pi/2 - 0.1, "phi": np.pi/2, "pte": 0.5, "ptm": 0.5},
        ]

        solver._pre_solve(sources)
        pol_b = solver._calculate_polarization(sources=sources)

        # Verify no NaN or inf values
        assert not torch.isnan(pol_b['ate']).any()
        assert not torch.isnan(pol_b['atm']).any()
        assert not torch.isnan(pol_b['pol_vec']).any()
        assert torch.isfinite(pol_b['ate']).all()
        assert torch.isfinite(pol_b['atm']).all()
        assert torch.isfinite(pol_b['pol_vec']).all()

    def test_batched_matches_sequential(self):
        """Test that batched polarization matches sequential computation."""
        solver = self.setup_solver(n_freqs=5)

        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},           # Normal, pure TE
            {"theta": np.pi/6, "phi": 0.0, "pte": 1.0, "ptm": 0.0},       # 30 deg, pure TE
            {"theta": np.pi/4, "phi": np.pi/4, "pte": 0.5, "ptm": 0.5},   # 45 deg, mixed
            {"theta": np.pi/3, "phi": np.pi/2, "pte": 0.0, "ptm": 1.0},   # 60 deg, pure TM
            {"theta": 1e-8, "phi": np.pi/3, "pte": 0.3, "ptm": 0.7},      # Near normal, mixed
        ]

        # Sequential computation (solver path)
        pol_seq = self.compute_polarization_sequential(solver, sources)

        # Batched computation (solver path)
        solver._pre_solve(sources)
        pol_b = solver._calculate_polarization(sources=sources)

        # Compare each source
        for i in range(len(sources)):
            assert not torch.isnan(pol_b['ate'][i]).any()
            assert not torch.isnan(pol_seq[i]['ate']).any()

            assert not torch.isnan(pol_b['atm'][i]).any()
            assert not torch.isnan(pol_seq[i]['atm']).any()

            assert not torch.isnan(pol_b['pol_vec'][i]).any()
            assert not torch.isnan(pol_seq[i]['pol_vec']).any()

            torch.testing.assert_close(pol_b['ate'][i], pol_seq[i]['ate'], rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(pol_b['atm'][i], pol_seq[i]['atm'], rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(pol_b['pol_vec'][i], pol_seq[i]['pol_vec'], rtol=1e-6, atol=1e-8)

    def test_polarization_normalization(self):
        """Test that polarization vectors are properly normalized."""
        solver = self.setup_solver(n_freqs=3)

        sources = [
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": np.pi/4, "pte": 0.0, "ptm": 1.0},
            {"theta": 0.3, "phi": np.pi/2, "pte": 0.7071, "ptm": 0.7071},  # Equal mix
        ]

        solver._pre_solve(sources)
        pol_b = solver._calculate_polarization(sources=sources)

        # Check orthogonality and unit-norm across frequencies
        for i in range(len(sources)):
            ate = pol_b['ate'][i]
            atm = pol_b['atm'][i]
            dot_product = (ate * torch.conj(atm)).real.sum(dim=-1)
            assert torch.all(torch.abs(dot_product) < 1e-5), f"ate and atm not orthogonal for source {i}"

            ate_norm = torch.norm(ate, dim=-1)
            atm_norm = torch.norm(atm, dim=-1)
            assert torch.all(torch.abs(ate_norm - 1.0) < 1e-5), f"ate not unit normalized for source {i}"
            assert torch.all(torch.abs(atm_norm - 1.0) < 1e-5), f"atm not unit normalized for source {i}"

    def test_gradient_flow(self):
        """Test that gradients can flow through batched polarization calculation."""
        solver = self.setup_solver(n_freqs=3)

        # Create sources with gradient-enabled parameters
        theta_values = torch.tensor([0.1, 0.2, 0.3], dtype=solver.tfloat, requires_grad=True)
        phi_values = torch.tensor([0.0, 0.1, 0.2], dtype=solver.tfloat, requires_grad=True)
        pte_values = torch.tensor([1.0, 0.5, 0.0], dtype=solver.tfloat, requires_grad=True)
        ptm_values = torch.tensor([0.0, 0.5, 1.0], dtype=solver.tfloat, requires_grad=True)

        sources = [
            {"theta": theta_values[i], "phi": phi_values[i], "pte": pte_values[i], "ptm": ptm_values[i]}
            for i in range(3)
        ]

        solver._pre_solve(sources)
        pol_b = solver._calculate_polarization(sources=sources)

        loss = pol_b['pol_vec'].real.sum()
        loss.backward()

        assert theta_values.grad is not None, "Gradients should flow to theta"
        assert phi_values.grad is not None, "Gradients should flow to phi"
        assert pte_values.grad is not None, "Gradients should flow to pte"
        assert ptm_values.grad is not None, "Gradients should flow to ptm"

    def test_mixed_normal_oblique_sources(self):
        """Test mixed batch with both normal and oblique incidence."""
        solver = self.setup_solver(n_freqs=3)

        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},        # Normal
            {"theta": 0.3, "phi": 0.5, "pte": 0.6, "ptm": 0.4},        # Oblique
            {"theta": 5e-7, "phi": 1.0, "pte": 0.0, "ptm": 1.0},       # Near normal
            {"theta": 0.8, "phi": 2.0, "pte": 0.3, "ptm": 0.7},        # Oblique
            {"theta": 0.0, "phi": 3.0, "pte": 0.5, "ptm": 0.5},        # Normal
        ]

        pol_seq = self.compute_polarization_sequential(solver, sources)
        solver._pre_solve(sources)
        pol_b = solver._calculate_polarization(sources=sources)

        for i in range(len(sources)):
            torch.testing.assert_close(pol_b['pol_vec'][i], pol_seq[i]['pol_vec'], rtol=1e-6, atol=1e-8,
                                       msg=f"Mismatch for source {i} (theta={sources[i]['theta']})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
