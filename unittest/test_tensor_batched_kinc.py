"""
Test suite for tensor-level batched kinc calculation.

This test file validates that the batched kinc computation matches sequential
results and follows the expected dimension flow for tensor-level source batching.
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
        """Compute kinc for each source sequentially (current implementation)."""
        kinc_list = []
        
        for src in sources:
            solver.src = src
            solver._pre_solve()
            kinc_list.append(solver.kinc.clone())
        
        return kinc_list

    def compute_kinc_batched(self, solver, sources):
        """Compute kinc for all sources in a batched manner (target implementation)."""
        n_sources = len(sources)
        n_freqs = solver.n_freqs
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
        
        # Calculate refractive index
        # Handle both dispersive (with frequency dimension) and non-dispersive materials
        ur1 = solver.ur1
        er1 = solver.er1
        
        # Ensure we have tensors
        if not isinstance(ur1, torch.Tensor):
            ur1 = torch.tensor(ur1, dtype=solver.tcomplex, device=device)
        if not isinstance(er1, torch.Tensor):
            er1 = torch.tensor(er1, dtype=solver.tcomplex, device=device)
        
        # Calculate refractive index
        refractive_1 = torch.sqrt(ur1 * er1)
        
        # Handle scalar case (non-dispersive material)
        if refractive_1.dim() == 0:
            # Expand to frequency dimension
            refractive_1 = refractive_1.unsqueeze(0).expand(n_freqs)
        
        # Expand dimensions for broadcasting
        # theta: (n_sources,) -> (n_sources, 1) -> (n_sources, n_freqs)
        theta_expanded = theta_batch[:, None].expand(-1, n_freqs)
        phi_expanded = phi_batch[:, None].expand(-1, n_freqs)
        
        # Compute kinc components
        # Shape: (n_sources, n_freqs)
        kinc_x = torch.sin(theta_expanded) * torch.cos(phi_expanded)
        kinc_y = torch.sin(theta_expanded) * torch.sin(phi_expanded)
        kinc_z = torch.cos(theta_expanded)
        
        # Stack and multiply by refractive index
        # kinc shape: (n_sources, n_freqs, 3)
        kinc_batched = refractive_1[None, :, None] * torch.stack([kinc_x, kinc_y, kinc_z], dim=2)
        
        return kinc_batched

    def test_single_source_matches_original(self):
        """Test that batched computation with single source matches original."""
        solver = self.setup_solver(n_freqs=3)
        
        source = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        
        # Sequential (original)
        solver.src = source
        solver._pre_solve()
        kinc_original = solver.kinc.clone()
        
        # Batched with single source
        kinc_batched = self.compute_kinc_batched(solver, [source])
        
        # Should match exactly
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
        
        kinc_batched = self.compute_kinc_batched(solver, sources)
        
        # Check dimensions
        assert kinc_batched.shape == (4, 3, 3), f"Expected shape (4, 3, 3), got {kinc_batched.shape}"
        
        # Verify each component
        assert kinc_batched.dim() == 3
        assert kinc_batched.size(0) == len(sources)  # n_sources
        assert kinc_batched.size(1) == solver.n_freqs  # n_freqs
        assert kinc_batched.size(2) == 3  # x, y, z components

    def test_batched_matches_sequential(self):
        """Test that batched kinc matches sequential computation."""
        solver = self.setup_solver(n_freqs=5)
        
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/6, "phi": 0.0, "pte": 1.0, "ptm": 0.0},  # 30 degrees
            {"theta": np.pi/4, "phi": np.pi/4, "pte": 0.5, "ptm": 0.5},  # 45 degrees
            {"theta": np.pi/3, "phi": np.pi/2, "pte": 0.0, "ptm": 1.0},  # 60 degrees
        ]
        
        # Sequential computation
        kinc_sequential = self.compute_kinc_sequential(solver, sources)
        
        # Batched computation
        kinc_batched = self.compute_kinc_batched(solver, sources)
        
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
        
        kinc_batched = self.compute_kinc_batched(solver, sources)
        
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
        expected_normal[:, 2] = refractive_1.real  # Only real part for comparison
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
        kinc1 = self.compute_kinc_batched(solver1, sources)
        assert kinc1.shape == (2, 1, 3)
        
        # Many wavelengths
        solver2 = self.setup_solver(n_freqs=10)
        kinc2 = self.compute_kinc_batched(solver2, sources)
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
        
        kinc_batched = self.compute_kinc_batched(solver, sources_different_pol)
        
        # All should be identical since kinc doesn't depend on polarization
        torch.testing.assert_close(kinc_batched[0], kinc_batched[1], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(kinc_batched[0], kinc_batched[2], rtol=1e-10, atol=1e-12)

    def test_gradient_flow(self):
        """Test that gradients can flow through batched kinc calculation."""
        solver = self.setup_solver(n_freqs=3)
        
        # Create sources with gradient-enabled angles
        theta_values = torch.tensor([0.1, 0.2, 0.3], dtype=solver.tfloat, requires_grad=True)
        phi_values = torch.tensor([0.0, 0.1, 0.2], dtype=solver.tfloat, requires_grad=True)
        
        sources = [
            {"theta": theta_values[i].item(), "phi": phi_values[i].item(), "pte": 1.0, "ptm": 0.0}
            for i in range(3)
        ]
        
        # Compute with gradients
        theta_batch = theta_values
        phi_batch = phi_values
        
        # Handle refractive index calculation
        ur1 = solver.ur1
        er1 = solver.er1
        if not isinstance(ur1, torch.Tensor):
            ur1 = torch.tensor(ur1, dtype=solver.tcomplex, device=solver.device)
        if not isinstance(er1, torch.Tensor):
            er1 = torch.tensor(er1, dtype=solver.tcomplex, device=solver.device)
        refractive_1 = torch.sqrt(ur1 * er1)
        if refractive_1.dim() == 0:
            refractive_1 = refractive_1.unsqueeze(0).expand(solver.n_freqs)
            
        theta_expanded = theta_batch[:, None].expand(-1, solver.n_freqs)
        phi_expanded = phi_batch[:, None].expand(-1, solver.n_freqs)
        
        kinc_x = torch.sin(theta_expanded) * torch.cos(phi_expanded)
        kinc_y = torch.sin(theta_expanded) * torch.sin(phi_expanded)
        kinc_z = torch.cos(theta_expanded)
        
        kinc_batched = refractive_1[None, :, None] * torch.stack([kinc_x, kinc_y, kinc_z], dim=2)
        
        # Compute a loss and check gradients (use real part for backward)
        loss = kinc_batched.real.sum()
        loss.backward()
        
        assert theta_values.grad is not None, "Gradients should flow to theta"
        assert phi_values.grad is not None, "Gradients should flow to phi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])