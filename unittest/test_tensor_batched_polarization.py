"""
Test suite for tensor-level batched polarization calculation.

This test file validates that batched polarization vector computation matches sequential
results and properly handles edge cases like near-normal and near-grazing incidence.
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
        """Compute polarization vectors for each source sequentially (current implementation)."""
        polarization_list = []
        
        for src in sources:
            solver.src = src
            solver._pre_solve()
            
            # Extract polarization computation from solver
            # This is typically done in _calculate_fields_and_efficiencies
            theta = torch.tensor(src["theta"], dtype=solver.tfloat, device=solver.device)
            phi = torch.tensor(src["phi"], dtype=solver.tfloat, device=solver.device)
            pte = torch.tensor(src["pte"], dtype=solver.tfloat, device=solver.device)
            ptm = torch.tensor(src["ptm"], dtype=solver.tfloat, device=solver.device)
            
            # Calculate TE and TM vectors
            if abs(theta) < 1e-6:  # Normal incidence
                ate = torch.tensor([0.0, 1.0, 0.0], dtype=solver.tfloat, device=solver.device)
                atm = torch.tensor([1.0, 0.0, 0.0], dtype=solver.tfloat, device=solver.device)
            else:
                ate = torch.tensor([
                    -torch.sin(phi),
                    torch.cos(phi),
                    0.0
                ], dtype=solver.tfloat, device=solver.device)
                
                atm = torch.tensor([
                    torch.cos(theta) * torch.cos(phi),
                    torch.cos(theta) * torch.sin(phi),
                    -torch.sin(theta)
                ], dtype=solver.tfloat, device=solver.device)
            
            # Combine with polarization weights
            polarization = pte * ate + ptm * atm
            
            polarization_list.append({
                'ate': ate.clone(),
                'atm': atm.clone(),
                'polarization': polarization.clone()
            })
        
        return polarization_list

    def compute_polarization_batched(self, solver, sources):
        """Compute polarization vectors for all sources in a batched manner (target implementation)."""
        n_sources = len(sources)
        device = solver.device
        tfloat = solver.tfloat
        
        # Stack source parameters
        theta_batch = torch.stack([
            torch.tensor(src["theta"], dtype=tfloat, device=device) 
            for src in sources
        ])  # Shape: (n_sources,)
        
        phi_batch = torch.stack([
            torch.tensor(src["phi"], dtype=tfloat, device=device)
            for src in sources
        ])  # Shape: (n_sources,)
        
        pte_batch = torch.stack([
            torch.tensor(src["pte"], dtype=tfloat, device=device)
            for src in sources
        ])  # Shape: (n_sources,)
        
        ptm_batch = torch.stack([
            torch.tensor(src["ptm"], dtype=tfloat, device=device)
            for src in sources
        ])  # Shape: (n_sources,)
        
        # Initialize ate and atm tensors
        ate_batch = torch.zeros(n_sources, 3, dtype=tfloat, device=device)
        atm_batch = torch.zeros(n_sources, 3, dtype=tfloat, device=device)
        
        # Create mask for normal incidence
        normal_mask = torch.abs(theta_batch) < 1e-6
        
        # Handle normal incidence cases
        ate_batch[normal_mask, 1] = 1.0  # [0, 1, 0]
        atm_batch[normal_mask, 0] = 1.0  # [1, 0, 0]
        
        # Handle oblique incidence cases
        oblique_mask = ~normal_mask
        
        # TE vector for oblique incidence
        ate_batch[oblique_mask, 0] = -torch.sin(phi_batch[oblique_mask])
        ate_batch[oblique_mask, 1] = torch.cos(phi_batch[oblique_mask])
        ate_batch[oblique_mask, 2] = 0.0
        
        # TM vector for oblique incidence
        atm_batch[oblique_mask, 0] = torch.cos(theta_batch[oblique_mask]) * torch.cos(phi_batch[oblique_mask])
        atm_batch[oblique_mask, 1] = torch.cos(theta_batch[oblique_mask]) * torch.sin(phi_batch[oblique_mask])
        atm_batch[oblique_mask, 2] = -torch.sin(theta_batch[oblique_mask])
        
        # Combine with polarization weights
        # Shape: (n_sources, 3)
        polarization_batch = pte_batch[:, None] * ate_batch + ptm_batch[:, None] * atm_batch
        
        return {
            'ate': ate_batch,
            'atm': atm_batch,
            'polarization': polarization_batch
        }

    def test_single_source_matches_original(self):
        """Test that batched computation with single source matches original."""
        solver = self.setup_solver(n_freqs=3)
        
        source = {"theta": 0.1, "phi": 0.2, "pte": 0.7, "ptm": 0.3}
        
        # Sequential (original)
        pol_sequential = self.compute_polarization_sequential(solver, [source])
        
        # Batched with single source
        pol_batched = self.compute_polarization_batched(solver, [source])
        
        # Should match exactly
        torch.testing.assert_close(pol_batched['ate'][0], pol_sequential[0]['ate'], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(pol_batched['atm'][0], pol_sequential[0]['atm'], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(pol_batched['polarization'][0], pol_sequential[0]['polarization'], rtol=1e-6, atol=1e-8)

    def test_multiple_sources_dimensions(self):
        """Test dimensions with multiple sources."""
        solver = self.setup_solver(n_freqs=3)
        
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": np.pi/4, "pte": 0.7, "ptm": 0.3},
            {"theta": 0.3, "phi": np.pi/2, "pte": 0.0, "ptm": 1.0},
        ]
        
        pol_batched = self.compute_polarization_batched(solver, sources)
        
        # Check dimensions
        assert pol_batched['ate'].shape == (4, 3), "ate shape mismatch"
        assert pol_batched['atm'].shape == (4, 3), "atm shape mismatch"
        assert pol_batched['polarization'].shape == (4, 3), "polarization shape mismatch"

    def test_normal_incidence_edge_case(self):
        """Test polarization for normal incidence (theta ≈ 0)."""
        solver = self.setup_solver(n_freqs=2)
        
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},        # Exact zero
            {"theta": 1e-10, "phi": 0.0, "pte": 1.0, "ptm": 0.0},      # Near zero
            {"theta": 1e-7, "phi": np.pi/4, "pte": 0.5, "ptm": 0.5},   # Very small angle
        ]
        
        pol_batched = self.compute_polarization_batched(solver, sources)
        
        # For normal incidence, ate should be [0, 1, 0] and atm should be [1, 0, 0]
        expected_ate = torch.tensor([0.0, 1.0, 0.0], dtype=solver.tfloat, device=solver.device)
        expected_atm = torch.tensor([1.0, 0.0, 0.0], dtype=solver.tfloat, device=solver.device)
        
        for i in range(len(sources)):
            torch.testing.assert_close(pol_batched['ate'][i], expected_ate, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(pol_batched['atm'][i], expected_atm, rtol=1e-6, atol=1e-8)

    def test_grazing_incidence_edge_case(self):
        """Test polarization for near-grazing incidence."""
        solver = self.setup_solver(n_freqs=2)
        
        # Near grazing angles
        sources = [
            {"theta": np.pi/2 - 1e-3, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/2 - 1e-2, "phi": np.pi/4, "pte": 0.0, "ptm": 1.0},
            {"theta": np.pi/2 - 0.1, "phi": np.pi/2, "pte": 0.5, "ptm": 0.5},
        ]
        
        pol_batched = self.compute_polarization_batched(solver, sources)
        
        # Verify no NaN or inf values
        assert not torch.isnan(pol_batched['ate']).any()
        assert not torch.isnan(pol_batched['atm']).any()
        assert not torch.isnan(pol_batched['polarization']).any()
        assert torch.isfinite(pol_batched['ate']).all()
        assert torch.isfinite(pol_batched['atm']).all()
        assert torch.isfinite(pol_batched['polarization']).all()

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
        
        # Sequential computation
        pol_sequential = self.compute_polarization_sequential(solver, sources)
        
        # Batched computation
        pol_batched = self.compute_polarization_batched(solver, sources)
        
        # Compare each source
        for i in range(len(sources)):
            # NaN values are bugs - polarization should never contain NaN
            assert not torch.isnan(pol_batched['ate'][i]).any(), \
                f"NaN values found in batched ate for source {i} - this is a bug!"
            assert not torch.isnan(pol_sequential[i]['ate']).any(), \
                f"NaN values found in sequential ate for source {i} - this is a bug!"
            
            assert not torch.isnan(pol_batched['atm'][i]).any(), \
                f"NaN values found in batched atm for source {i} - this is a bug!"
            assert not torch.isnan(pol_sequential[i]['atm']).any(), \
                f"NaN values found in sequential atm for source {i} - this is a bug!"
                
            assert not torch.isnan(pol_batched['polarization'][i]).any(), \
                f"NaN values found in batched polarization for source {i} - this is a bug!"
            assert not torch.isnan(pol_sequential[i]['polarization']).any(), \
                f"NaN values found in sequential polarization for source {i} - this is a bug!"
            
            torch.testing.assert_close(
                pol_batched['ate'][i], 
                pol_sequential[i]['ate'], 
                rtol=1e-6, atol=1e-8,
                msg=f"ate mismatch for source {i}"
            )
            torch.testing.assert_close(
                pol_batched['atm'][i], 
                pol_sequential[i]['atm'], 
                rtol=1e-6, atol=1e-8,
                msg=f"atm mismatch for source {i}"
            )
            torch.testing.assert_close(
                pol_batched['polarization'][i], 
                pol_sequential[i]['polarization'], 
                rtol=1e-6, atol=1e-8,
                msg=f"polarization mismatch for source {i}"
            )

    def test_polarization_normalization(self):
        """Test that polarization vectors are properly normalized."""
        solver = self.setup_solver(n_freqs=3)
        
        sources = [
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": np.pi/4, "pte": 0.0, "ptm": 1.0},
            {"theta": 0.3, "phi": np.pi/2, "pte": 0.7071, "ptm": 0.7071},  # Equal mix
        ]
        
        pol_batched = self.compute_polarization_batched(solver, sources)
        
        # Check orthogonality of ate and atm
        for i in range(len(sources)):
            dot_product = torch.dot(pol_batched['ate'][i], pol_batched['atm'][i])
            assert torch.abs(dot_product) < 1e-6, f"ate and atm not orthogonal for source {i}"
            
            # Check unit norm for ate and atm
            ate_norm = torch.norm(pol_batched['ate'][i])
            atm_norm = torch.norm(pol_batched['atm'][i])
            assert torch.abs(ate_norm - 1.0) < 1e-6, f"ate not unit normalized for source {i}"
            assert torch.abs(atm_norm - 1.0) < 1e-6, f"atm not unit normalized for source {i}"

    def test_gradient_flow(self):
        """Test that gradients can flow through batched polarization calculation."""
        solver = self.setup_solver(n_freqs=3)
        
        # Create sources with gradient-enabled angles
        theta_values = torch.tensor([0.1, 0.2, 0.3], dtype=solver.tfloat, requires_grad=True)
        phi_values = torch.tensor([0.0, 0.1, 0.2], dtype=solver.tfloat, requires_grad=True)
        pte_values = torch.tensor([1.0, 0.5, 0.0], dtype=solver.tfloat, requires_grad=True)
        ptm_values = torch.tensor([0.0, 0.5, 1.0], dtype=solver.tfloat, requires_grad=True)
        
        # Compute polarization with gradients
        ate_batch = torch.zeros(3, 3, dtype=solver.tfloat, device=solver.device)
        atm_batch = torch.zeros(3, 3, dtype=solver.tfloat, device=solver.device)
        
        # Normal incidence mask
        normal_mask = torch.abs(theta_values) < 1e-6
        oblique_mask = ~normal_mask
        
        # All should be oblique in this test
        ate_batch[oblique_mask, 0] = -torch.sin(phi_values[oblique_mask])
        ate_batch[oblique_mask, 1] = torch.cos(phi_values[oblique_mask])
        atm_batch[oblique_mask, 0] = torch.cos(theta_values[oblique_mask]) * torch.cos(phi_values[oblique_mask])
        atm_batch[oblique_mask, 1] = torch.cos(theta_values[oblique_mask]) * torch.sin(phi_values[oblique_mask])
        atm_batch[oblique_mask, 2] = -torch.sin(theta_values[oblique_mask])
        
        polarization_batch = pte_values[:, None] * ate_batch + ptm_values[:, None] * atm_batch
        
        # Compute a loss and check gradients
        loss = polarization_batch.sum()
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
        
        pol_sequential = self.compute_polarization_sequential(solver, sources)
        pol_batched = self.compute_polarization_batched(solver, sources)
        
        # Verify all match
        for i in range(len(sources)):
            torch.testing.assert_close(
                pol_batched['polarization'][i], 
                pol_sequential[i]['polarization'], 
                rtol=1e-6, atol=1e-8,
                msg=f"Mismatch for source {i} (theta={sources[i]['theta']})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])