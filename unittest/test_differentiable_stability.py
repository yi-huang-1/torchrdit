"""
Test suite for differentiable numerical stability alternatives.

This module tests the replacement of non-differentiable torch.where operations
with smooth, differentiable alternatives that maintain gradient flow while
ensuring numerical stability.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from torchrdit.solver import create_solver, get_solver_builder
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator
import time


class TestDifferentiableStability:
    """Test suite for differentiable stability implementations."""
    
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
        """Test that gradients flow through softplus protection."""
        # Create a tensor that requires gradient
        kz = torch.tensor([0.0001, 0.001, 0.01, 0.1], dtype=torch.complex128, requires_grad=True)
        min_kz = 1e-3
        beta = 100
        
        # Apply softplus protection
        kz_abs = torch.abs(kz)
        protection = F.softplus(min_kz - kz_abs, beta=beta)
        kz_protected = kz + protection * torch.sign(torch.real(kz) + 1e-16)
        
        # Create a simple loss
        loss = torch.sum(torch.abs(kz_protected))
        loss.backward()
        
        # Check that gradients exist and are non-zero
        assert kz.grad is not None, "Gradient should exist"
        assert torch.all(torch.abs(kz.grad) > 0), "All gradients should be non-zero"
        
        # Check gradient continuity - gradients should be smooth
        grad_values = kz.grad.detach().numpy()
        grad_diff = np.diff(np.real(grad_values))
        assert np.all(np.abs(grad_diff) < 10), "Gradients should be relatively smooth"
    
    def test_gradient_flow_through_softplus_clamp(self):
        """Test gradient flow through softplus minimum clamping."""
        # Test real values (like kinc_z_real)
        x = torch.tensor([0.0001, 0.0005, 0.001, 0.005, 0.01], dtype=torch.float64, requires_grad=True)
        min_val = 1e-3
        beta = 100
        
        # Apply softplus clamping
        x_clamped = x + F.softplus(min_val - x, beta=beta)
        
        # Create loss
        loss = torch.sum(x_clamped ** 2)
        loss.backward()
        
        # Verify gradients exist
        assert x.grad is not None, "Gradient should exist"
        assert torch.all(x.grad != 0), "All gradients should be non-zero"
    
    def test_numerical_stability_small_kz(self, basic_solver):
        """Test numerical stability with very small kz values."""
        # Create kz values that would cause numerical issues
        kz_values = torch.tensor([1e-16, 1e-12, 1e-8, 1e-6, 1e-4], dtype=torch.complex128)
        min_kz = 1e-3
        beta = 100
        
        # Apply softplus protection
        kz_abs = torch.abs(kz_values)
        protection = F.softplus(min_kz - kz_abs, beta=beta)
        kz_protected = kz_values + protection * torch.sign(torch.real(kz_values) + 1e-16)
        
        # Verify no NaN or Inf values
        assert torch.all(torch.isfinite(kz_protected)), "Protected values should be finite"
        assert torch.all(torch.abs(kz_protected) >= min_kz * 0.9), "Protected values should be above threshold"
    
    def test_grazing_incidence_stability(self, basic_solver):
        """Test solver stability at extreme grazing incidence with differentiable protection."""
        # Test at extreme angle
        theta_deg = 89.99
        theta = np.deg2rad(theta_deg)
        
        # Simulate kinc_z calculation
        kinc_z = torch.tensor([np.cos(theta)], dtype=torch.float64, requires_grad=True)
        min_kinc_z = 1e-3
        beta = 100
        
        # Apply softplus protection
        kinc_z_protected = kinc_z + F.softplus(min_kinc_z - kinc_z, beta=beta)
        
        # Verify protection
        assert kinc_z_protected > min_kinc_z * 0.9, "kinc_z should be protected"
        assert torch.isfinite(kinc_z_protected), "Protected value should be finite"
        
        # Test gradient flow
        loss = kinc_z_protected ** 2
        loss.backward()
        assert kinc_z.grad is not None, "Gradient should exist"
        assert torch.isfinite(kinc_z.grad), "Gradient should be finite"
    
    def test_comparison_with_torch_where(self, basic_solver):
        """Compare softplus results with torch.where implementation."""
        # Test values around the threshold
        kz_values = torch.tensor([0.0001, 0.0005, 0.001, 0.002, 0.01], dtype=torch.complex128)
        min_kz = 1e-3
        beta = 1000  # High beta for sharp transition
        
        # torch.where implementation
        kz_abs = torch.abs(kz_values)
        kz_where = torch.where(
            kz_abs < min_kz,
            kz_values + min_kz * torch.sign(torch.real(kz_values) + 1e-16),
            kz_values
        )
        
        # softplus implementation
        protection = F.softplus(min_kz - kz_abs, beta=beta)
        kz_softplus = kz_values + protection * torch.sign(torch.real(kz_values) + 1e-16)
        
        # Compare results - should be very close with high beta
        diff = torch.abs(kz_where - kz_softplus)
        # Softplus is not exactly equal to hard threshold, so allow larger tolerance
        assert torch.all(diff < 1e-3), f"Differences should be small, got max diff: {torch.max(diff)}"
    
    def test_energy_conservation_with_protection(self, basic_solver):
        """Test that energy conservation is maintained with differentiable protection."""
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
        """Test sensitivity to beta parameter in softplus."""
        x = torch.tensor([0.0005], dtype=torch.float64)
        min_val = 1e-3
        
        # Test different beta values
        beta_values = [10, 100, 1000, 10000]
        results = []
        
        for beta in beta_values:
            # This is the softplus_clamp_min function behavior
            x_protected = x + F.softplus(min_val - x, beta=beta)
            results.append(x_protected.item())
        
        # With higher beta, the softplus function becomes sharper
        # The result should converge towards min_val as beta increases
        # since x < min_val in this test case
        for i in range(len(results)-1):
            # Each result should be closer to min_val than the previous
            dist_current = abs(results[i] - min_val)
            dist_next = abs(results[i+1] - min_val)
            assert dist_next <= dist_current, \
                f"Results should converge to min_val={min_val}: {results}"
        
        # With very high beta, result should be very close to min_val
        # since x < min_val and softplus is clamping it
        assert abs(results[-1] - min_val) < 1e-6, \
            f"High beta result {results[-1]} should be close to min_val {min_val}"
    
    def test_complex_division_stability(self):
        """Test complex-safe division for preventing 1/kz issues."""
        # Test small complex kz values
        kz = torch.tensor([1e-16 + 1e-16j, 1e-8 + 1e-8j, 1e-4 + 1e-4j], dtype=torch.complex128)
        epsilon = 1e-12
        
        # Complex-safe division: kz* / (|kz|^2 + epsilon^2)
        kz_conj = torch.conj(kz)
        kz_abs_sq = torch.real(kz * kz_conj)
        safe_reciprocal = kz_conj / (kz_abs_sq + epsilon**2)
        
        # Verify no infinities
        assert torch.all(torch.isfinite(safe_reciprocal)), "Safe reciprocal should be finite"
        
        # Verify approximate correctness for larger values
        direct_reciprocal = 1 / kz[2]  # Only safe for larger value
        safe_result = safe_reciprocal[2]
        relative_error = torch.abs((safe_result - direct_reciprocal) / direct_reciprocal)
        assert relative_error < 1e-6, f"Safe division should be accurate for larger values"
    
    def test_performance_comparison(self, basic_solver):
        """Compare performance of softplus vs torch.where."""
        # Create large tensor for performance testing
        size = 10000
        kz = torch.rand(size, dtype=torch.complex128) * 0.01
        min_kz = 1e-3
        beta = 100
        
        # Time torch.where
        start = time.time()
        for _ in range(100):
            kz_abs = torch.abs(kz)
            kz_where = torch.where(
                kz_abs < min_kz,
                kz + min_kz * torch.sign(torch.real(kz) + 1e-16),
                kz
            )
        where_time = time.time() - start
        
        # Time softplus
        start = time.time()
        for _ in range(100):
            kz_abs = torch.abs(kz)
            protection = F.softplus(min_kz - kz_abs, beta=beta)
            kz_softplus = kz + protection * torch.sign(torch.real(kz) + 1e-16)
        softplus_time = time.time() - start
        
        # Softplus is expected to be somewhat slower due to exponential operations
        # Allow up to 3x slower which is acceptable for differentiability benefits
        performance_ratio = softplus_time / where_time
        assert performance_ratio < 3.0, f"Softplus is {performance_ratio:.2f}x slower than torch.where"
        print(f"Performance ratio (softplus/where): {performance_ratio:.2f}")


class TestPhysicsBasedRegularization:
    """Test physics-based regularization approach."""
    
    def test_complex_regularization_kz_calculation(self):
        """Test adding small imaginary component to kz calculation."""
        # Simulate kz calculation with regularization
        k0 = 2 * np.pi / 1.55  # Wave number at λ=1.55
        kx = k0 * 0.7  # Some x-component
        ky = 0  # No y-component
        
        # Test case where kz would be near zero (critical angle)
        k_parallel_sq = kx**2 + ky**2
        
        # Standard calculation would fail near k0^2 = k_parallel^2
        epsilon = torch.finfo(torch.float64).eps
        
        # Physics-based regularization
        kz_reg = torch.sqrt(torch.tensor(k0**2 - k_parallel_sq + 1j * epsilon, dtype=torch.complex128))
        
        # Verify result is well-behaved
        assert torch.isfinite(kz_reg), "Regularized kz should be finite"
        assert torch.abs(kz_reg) > 0, "Regularized kz should be non-zero"
        
        # For propagating modes, imaginary part should be negligible
        if k0**2 > k_parallel_sq:
            assert torch.abs(torch.imag(kz_reg)) < 1e-10, "Imaginary part should be negligible for propagating modes"
    
    def test_evanescent_mode_handling(self):
        """Test handling of evanescent modes with complex regularization."""
        k0 = 2 * np.pi / 1.55
        
        # Create case where mode is evanescent (k_parallel > k0)
        kx = k0 * 1.5  # Larger than k0
        ky = 0
        
        epsilon = torch.finfo(torch.float64).eps
        k_parallel_sq = kx**2 + ky**2
        
        # With regularization
        kz_reg = torch.sqrt(torch.tensor(k0**2 - k_parallel_sq + 1j * epsilon, dtype=torch.complex128))
        
        # For evanescent modes, kz should be primarily imaginary
        assert torch.abs(torch.imag(kz_reg)) > torch.abs(torch.real(kz_reg)), \
            "Evanescent mode should have dominant imaginary component"
        assert torch.isfinite(kz_reg), "Result should be finite"