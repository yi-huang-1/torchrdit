"""
Tests for tensor-level batched k-vector initialization.

These tests verify that the solver's own batched implementation in
`_pre_solve` + `_initialize_k_vectors` matches the sequential path and
produces correct shapes. The tests avoid re‑implementing solver logic in the
test (no shadow implementations), and do not depend on other test files.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm


class TestBatchedKVectors:
    """Test batched k-vector initialization for tensor-level source processing."""

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

    def compute_k_vectors_sequential(self, solver, sources):
        """Compute k-vectors for each source using the solver sequentially."""
        k_vectors_list = []
        for src in sources:
            solver._pre_solve(src)
            kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
            k_vectors_list.append({
                "kx_0": kx_0.clone(),
                "ky_0": ky_0.clone(),
                "kz_ref_0": kz_ref_0.clone(),
                "kz_trn_0": kz_trn_0.clone(),
            })
        return k_vectors_list

    def test_single_source_matches_original(self):
        """Test that batched computation with single source matches original."""
        solver = self.setup_solver(n_freqs=3, kdim=(5, 5))
        
        source = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        
        # Sequential (original)
        solver._pre_solve(source)
        kx_0_orig, ky_0_orig, kz_ref_0_orig, kz_trn_0_orig = solver._initialize_k_vectors()
        
        # Batched with single source using solver's batched path
        solver._pre_solve([source])
        kx_b, ky_b, kz_ref_b, kz_trn_b = solver._initialize_k_vectors()
        
        # Compare
        torch.testing.assert_close(kx_b[0], kx_0_orig, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ky_b[0], ky_0_orig, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(kz_ref_b[0], kz_ref_0_orig, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(kz_trn_b[0], kz_trn_0_orig, rtol=1e-5, atol=1e-6)

    def test_multiple_sources_dimensions(self):
        """Test dimensions with multiple sources."""
        solver = self.setup_solver(n_freqs=3, kdim=(5, 5))
        n_sources = 4
        
        # Create simple set of sources and run batched path
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
            for _ in range(n_sources)
        ]
        solver._pre_solve(sources)
        kx_0, ky_0, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        
        # Check dimensions
        expected_shape = (n_sources, 3, 5, 5)  # (n_sources, n_freqs, kdim[0], kdim[1])
        assert kx_0.shape == expected_shape, "kx_0 shape mismatch"
        assert ky_0.shape == expected_shape, "ky_0 shape mismatch"
        assert kz_ref_0.shape == expected_shape, "kz_ref_0 shape mismatch"
        assert kz_trn_0.shape == expected_shape, "kz_trn_0 shape mismatch"

    def test_batched_matches_sequential(self):
        """Test that batched k-vectors match sequential computation."""
        solver = self.setup_solver(n_freqs=5, kdim=(7, 7))
        
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/6, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/4, "phi": np.pi/4, "pte": 0.5, "ptm": 0.5},
            {"theta": np.pi/3, "phi": np.pi/2, "pte": 0.0, "ptm": 1.0},
        ]
        
        # Sequential computation (solver path)
        k_vectors_sequential = self.compute_k_vectors_sequential(solver, sources)

        # Batched computation (solver path)
        solver._pre_solve(sources)
        kx_b, ky_b, kz_ref_b, kz_trn_b = solver._initialize_k_vectors()
        
        # Compare each source
        for i in range(len(sources)):
            # NaN values are bugs - k-vectors should never contain NaN
            assert not torch.isnan(kx_b[i]).any(), \
                f"NaN values found in batched kx_0 for source {i} - this is a bug!"
            assert not torch.isnan(k_vectors_sequential[i]['kx_0']).any(), \
                f"NaN values found in sequential kx_0 for source {i} - this is a bug!"
            
            assert not torch.isnan(ky_b[i]).any(), \
                f"NaN values found in batched ky_0 for source {i} - this is a bug!"
            assert not torch.isnan(k_vectors_sequential[i]['ky_0']).any(), \
                f"NaN values found in sequential ky_0 for source {i} - this is a bug!"
                
            assert not torch.isnan(kz_ref_b[i]).any(), \
                f"NaN values found in batched kz_ref_0 for source {i} - this is a bug!"
            assert not torch.isnan(k_vectors_sequential[i]['kz_ref_0']).any(), \
                f"NaN values found in sequential kz_ref_0 for source {i} - this is a bug!"
                
            assert not torch.isnan(kz_trn_b[i]).any(), \
                f"NaN values found in batched kz_trn_0 for source {i} - this is a bug!"
            assert not torch.isnan(k_vectors_sequential[i]['kz_trn_0']).any(), \
                f"NaN values found in sequential kz_trn_0 for source {i} - this is a bug!"
            
            torch.testing.assert_close(
                kx_b[i], 
                k_vectors_sequential[i]['kx_0'], 
                rtol=1e-5, atol=1e-6,
                msg=f"kx_0 mismatch for source {i}"
            )
            torch.testing.assert_close(
                ky_b[i], 
                k_vectors_sequential[i]['ky_0'], 
                rtol=1e-5, atol=1e-6,
                msg=f"ky_0 mismatch for source {i}"
            )
            torch.testing.assert_close(
                kz_ref_b[i], 
                k_vectors_sequential[i]['kz_ref_0'], 
                rtol=1e-5, atol=1e-6,
                msg=f"kz_ref_0 mismatch for source {i}"
            )
            torch.testing.assert_close(
                kz_trn_b[i], 
                k_vectors_sequential[i]['kz_trn_0'], 
                rtol=1e-5, atol=1e-6,
                msg=f"kz_trn_0 mismatch for source {i}"
            )

    def test_normal_incidence(self):
        """Test k-vectors for normal incidence."""
        solver = self.setup_solver(n_freqs=2, kdim=(3, 3))
        
        # Normal incidence
        source = {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        # Use solver batched path with a single source to get shape (1, ...)
        solver._pre_solve([source])
        kx_0, ky_0, _, _ = solver._initialize_k_vectors()
        
        # For normal incidence, kx and ky at center should be zero
        center_idx = solver.kdim[0] // 2, solver.kdim[1] // 2
        # Numerical relaxation in solver may set exact zeros to small epsilon; allow tolerance
        assert torch.max(torch.abs(kx_0[0, :, center_idx[0], center_idx[1]])).item() <= 1e-5
        assert torch.max(torch.abs(ky_0[0, :, center_idx[0], center_idx[1]])).item() <= 1e-5

    def test_oblique_incidence(self):
        """Test k-vectors for oblique incidence."""
        solver = self.setup_solver(n_freqs=3, kdim=(5, 5))
        
        # Oblique incidence
        theta = np.pi/4  # 45 degrees
        phi = np.pi/6    # 30 degrees
        source = {"theta": theta, "phi": phi, "pte": 1.0, "ptm": 0.0}
        
        solver._pre_solve([source])
        kx_0, ky_0, _, _ = solver._initialize_k_vectors()
        
        # Check that k-vectors are not symmetric for oblique incidence
        kx_0 = kx_0[0]
        ky_0 = ky_0[0]
        
        # Should have non-zero values at center
        center_idx = solver.kdim[0] // 2, solver.kdim[1] // 2
        assert not torch.allclose(kx_0[:, center_idx[0], center_idx[1]], torch.zeros(3, dtype=solver.tcomplex))
        assert not torch.allclose(ky_0[:, center_idx[0], center_idx[1]], torch.zeros(3, dtype=solver.tcomplex))

    def test_branch_cut_handling(self):
        """Test proper handling of branch cuts in kz calculation."""
        solver = self.setup_solver(n_freqs=2, kdim=(3, 3))
        
        # Create cases with various incident angles
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},           # Normal incidence
            {"theta": np.pi/6, "phi": 0.0, "pte": 1.0, "ptm": 0.0},       # 30 degrees
            {"theta": np.pi/4, "phi": 0.0, "pte": 1.0, "ptm": 0.0},       # 45 degrees
            {"theta": np.pi/3, "phi": 0.0, "pte": 1.0, "ptm": 0.0},       # 60 degrees
        ]
        
        # Compare solver batched vs sequential (branch cut handled internally)
        k_vectors_sequential = self.compute_k_vectors_sequential(solver, sources)
        solver._pre_solve(sources)
        _, _, kz_ref_b, kz_trn_b = solver._initialize_k_vectors()
        
        # Compare batched vs sequential for branch cut handling
        for i in range(len(sources)):
            # The solver's implementation handles branch cuts correctly
            # We just need to verify batched matches sequential
            torch.testing.assert_close(kz_ref_b[i], k_vectors_sequential[i]['kz_ref_0'], rtol=1e-5, atol=1e-6,
                                       msg=f"kz_ref mismatch for source {i}")
            torch.testing.assert_close(kz_trn_b[i], k_vectors_sequential[i]['kz_trn_0'], rtol=1e-5, atol=1e-6,
                                       msg=f"kz_trn mismatch for source {i}")

    def test_different_materials(self):
        """Test k-vectors with different reflection/transmission materials."""
        # Create solver with different materials
        mat_air = create_material(name="air", permittivity=1.0)
        mat_si = create_material(name="silicon", permittivity=11.7)
        mat_sio2 = create_material(name="sio2", permittivity=2.25)
        
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.0]),
            rdim=[256, 256],
            kdim=[3, 3],
            device="cpu",
        )
        
        solver.add_materials([mat_air, mat_si, mat_sio2])
        solver.update_ref_material("silicon")  # Different from air
        solver.update_trn_material("sio2")     # Different from air
        solver.add_layer(material_name="air", thickness=torch.tensor(0.5), is_homogeneous=True)
        
        source = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        solver._pre_solve(source)
        _, _, kz_ref_0, kz_trn_0 = solver._initialize_k_vectors()
        
        # kz values should be different for ref and trn regions
        assert not torch.allclose(kz_ref_0, kz_trn_0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
