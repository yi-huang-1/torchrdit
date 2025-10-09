"""
Tests for differentiable numerical stability utilities used by the solver.

These tests directly exercise the project’s implementations (softplus-based
protections) rather than re-implementing math inline, and keep one
physics-based integration check.
"""

import pytest
import torch
import numpy as np
from torchrdit.solver import (
    create_solver,
    softplus_protect_kz,
    softplus_clamp_min,
)
from torchrdit.constants import Algorithm, Precision


class TestDifferentiableStability:
    """Unit tests for differentiable stability helpers."""
    
    @pytest.fixture
    def basic_solver(self):
        """Create a basic solver for testing."""
        # Create solver with RCWA algorithm
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[512, 512],
            kdim=[13, 13],
            device="cpu",
            precision=Precision.DOUBLE
        )
        return solver
    
    def test_gradient_flow_through_softplus_protection(self, basic_solver):
        """Gradients flow through the library's kz protection."""
        kz = torch.tensor([0.0001, 0.001, 0.01, 0.1], dtype=torch.complex128, requires_grad=True)
        kz_protected = softplus_protect_kz(kz, min_kz=1e-3, beta=100)

        loss = torch.sum(torch.abs(kz_protected))
        loss.backward()

        assert kz.grad is not None
        assert torch.all(torch.abs(kz.grad) > 0)
    
    def test_gradient_flow_through_softplus_clamp(self):
        """Gradient flows through the library's softplus clamp."""
        x = torch.tensor([0.0001, 0.0005, 0.001, 0.005, 0.01], dtype=torch.float64, requires_grad=True)
        x_clamped = softplus_clamp_min(x, min_val=1e-3, beta=100)

        loss = torch.sum(x_clamped**2)
        loss.backward()

        assert x.grad is not None
        assert torch.all(x.grad != 0)
    
    def test_numerical_stability_small_kz(self, basic_solver):
        """Protection keeps tiny kz finite and above threshold."""
        kz_values = torch.tensor([1e-16, 1e-12, 1e-8, 1e-6, 1e-4], dtype=torch.complex128)
        kz_protected = softplus_protect_kz(kz_values, min_kz=1e-3, beta=100)

        assert torch.all(torch.isfinite(kz_protected))
        assert torch.all(torch.abs(kz_protected) >= 1e-3 * 0.9)
    
    def test_grazing_incidence_stability(self, basic_solver):
        """kinc_z protection is finite and differentiable near grazing."""
        theta = np.deg2rad(89.99)
        kinc_z = torch.tensor([np.cos(theta)], dtype=torch.float64, requires_grad=True)

        kinc_z_protected = softplus_clamp_min(kinc_z, min_val=1e-3, beta=100)
        assert kinc_z_protected > 1e-3 * 0.9
        assert torch.isfinite(kinc_z_protected)

        loss = kinc_z_protected**2
        loss.backward()
        assert kinc_z.grad is not None
        assert torch.isfinite(kinc_z.grad)
    
    def test_comparison_with_torch_where(self, basic_solver):
        """Softplus protection approximates torch.where when beta is large."""
        kz_values = torch.tensor([0.0001, 0.0005, 0.001, 0.002, 0.01], dtype=torch.complex128)
        min_kz = 1e-3
        beta = 1000

        kz_abs = torch.abs(kz_values)
        kz_where = torch.where(
            kz_abs < min_kz,
            kz_values + min_kz * torch.sign(torch.real(kz_values) + 1e-16),
            kz_values,
        )
        kz_softplus = softplus_protect_kz(kz_values, min_kz=min_kz, beta=beta)

        diff = torch.abs(kz_where - kz_softplus)
        assert torch.all(diff < 1e-3), f"Differences should be small, got max diff: {torch.max(diff)}"
    
    def test_energy_conservation_with_protection(self, basic_solver):
        """Integration sanity: energy conservation with protection enabled."""
        # Create a simple source
        source = basic_solver.add_source(
            theta=np.deg2rad(30),
            phi=0,
            pte=1,
            ptm=0
        )
        
        # Solve with the differentiable protection enabled
        result = basic_solver.solve(source)
        
        # Get reflection and transmission
        R = result.reflection[0]
        T = result.transmission[0]
        
        # Verify energy conservation
        assert np.isclose(R + T, 1.0, atol=1e-5), f"Energy not conserved: R+T = {R+T}"
    
    def test_beta_parameter_sensitivity(self):
        """Higher beta makes clamp approach the hard min."""
        x = torch.tensor([0.0005], dtype=torch.float64)
        min_val = 1e-3

        beta_values = [10, 100, 1000, 10000]
        results = [softplus_clamp_min(x, min_val=min_val, beta=b).item() for b in beta_values]

        for i in range(len(results) - 1):
            dist_current = abs(results[i] - min_val)
            dist_next = abs(results[i + 1] - min_val)
            assert dist_next <= dist_current, f"Results should converge to min_val={min_val}: {results}"

        assert abs(results[-1] - min_val) < 1e-6, (
            f"High beta result {results[-1]} should be close to min_val {min_val}"
        )
