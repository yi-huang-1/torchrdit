"""
Tests for tensor-level batched matrix setup in the solver.

These tests exercise the solver's own batched matrix setup via
`_initialize_k_vectors` and `_setup_common_matrices` to ensure batched results
match sequential results and shapes are correct. They avoid re-implementing
matrix logic in the test and do not test generic PyTorch behavior.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm


class TestBatchedMatrices:
    """Test batched matrix operations for tensor-level source processing."""

    def setup_solver(self, n_freqs=3, kdim=(3, 3)):
        """Create a basic solver for testing."""
        # Create materials
        mat_air = create_material(name="air", permittivity=1.0)
        mat_si = create_material(name="silicon", permittivity=11.7)

        # Create solver
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.linspace(0.8, 1.2, n_freqs),
            rdim=[256, 256],
            kdim=list(kdim),
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

    def _compute_matrices_sequential(self, solver, sources):
        """Compute matrices per source using solver sequential path."""
        mats = []
        for src in sources:
            solver._pre_solve(src)
            kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
            matrices = solver._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
            mats.append(matrices)
        return mats

    def test_setup_common_matrices_single_matches_batched(self):
        """Single-source matrices equal the first slice of batched matrices."""
        solver = self.setup_solver(n_freqs=3, kdim=(5, 5))
        source = {"theta": 0.1, "phi": 0.2, "pte": 1.0, "ptm": 0.0}

        # Sequential
        solver._pre_solve(source)
        kx_s, ky_s, kz_ref_s, kz_trn_s = solver._initialize_k_vectors()
        mats_single = solver._setup_common_matrices(kx_s, ky_s, kz_ref_s, kz_trn_s)

        # Batched with single source
        solver._pre_solve([source])
        kx_b, ky_b, kz_ref_b, kz_trn_b = solver._initialize_k_vectors()
        mats_batched = solver._setup_common_matrices(kx_b, ky_b, kz_ref_b, kz_trn_b)

        # Compare a subset of key matrices
        keys = [
            "mat_kx", "mat_ky", "mat_kz_ref", "mat_kz_trn",
            "mat_kx_kx", "mat_ky_ky", "mat_kx_ky", "mat_ky_kx",
            "mat_w0", "mat_v0",
        ]
        for k in keys:
            torch.testing.assert_close(mats_batched[k][0], mats_single[k], rtol=1e-5, atol=1e-6)

    def test_setup_common_matrices_multi_sources_consistency(self):
        """Batched matrices match sequential per-source results for multiple sources."""
        solver = self.setup_solver(n_freqs=4, kdim=(3, 3))
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/6, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/4, "phi": np.pi/4, "pte": 0.5, "ptm": 0.5},
            {"theta": np.pi/3, "phi": np.pi/2, "pte": 0.0, "ptm": 1.0},
        ]

        mats_seq = self._compute_matrices_sequential(solver, sources)

        solver._pre_solve(sources)
        kx_b, ky_b, kz_ref_b, kz_trn_b = solver._initialize_k_vectors()
        mats_b = solver._setup_common_matrices(kx_b, ky_b, kz_ref_b, kz_trn_b)

        # Sanity: shapes
        n_sources = len(sources)
        n_freqs = solver.n_freqs
        n_harm = solver.kdim[0] * solver.kdim[1]
        assert mats_b["mat_kx"].shape == (n_sources, n_freqs, n_harm)
        assert mats_b["mat_w0"].shape == (n_sources, n_freqs, 2 * n_harm, 2 * n_harm)

        # No NaNs and per-source equality
        for i in range(n_sources):
            for k, v_seq in mats_seq[i].items():
                v_b = mats_b[k][i]
                assert not torch.isnan(v_b).any(), f"NaN in batched matrix {k} for source {i}"
                assert not torch.isnan(v_seq).any(), f"NaN in sequential matrix {k} for source {i}"
                torch.testing.assert_close(v_b, v_seq, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
