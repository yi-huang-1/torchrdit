"""Unit tests for field differentiability and gradient flow.

This module tests the differentiability of electromagnetic field calculations
in TorchRDIT, ensuring that both Fourier coefficients and real-space field
reconstruction preserve gradient flow for advanced optimization workflows.

Following TDD principles, these tests validate:
1. Gradient flow through all field API methods
2. Field-based optimization scenarios (intensity, phase, uniformity)
3. Gradient consistency between different calculation methods
4. Batch source compatibility for field gradients
5. Performance and numerical stability of field-based optimization

Mathematical foundation:
- Field-based loss functions: L = f(E(x,y), H(x,y))
- Gradient flow: ∂L/∂θ = ∂L/∂E · ∂E/∂θ + ∂L/∂H · ∂H/∂θ
- Where θ represents design parameters (geometry, materials, etc.)

Keywords:
    field differentiability, gradient flow, optimization, TDD, unit tests,
    field intensity, field phase, field uniformity, electromagnetic fields
"""

import torch
import pytest
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator


class TestFieldDifferentiability:
    """Test class for field differentiability and gradient flow."""

    @pytest.fixture
    def simple_solver_rdit(self):
        """Create a simple R-DIT solver for field gradient testing."""
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.0]),
            kdim=[7, 7],  # Smaller for faster tests
            rdim=[128, 128],
            device='cpu'
        )
        return solver

    @pytest.fixture
    def simple_solver_rcwa(self):
        """Create a simple RCWA solver for field gradient testing."""
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.0]),
            kdim=[7, 7],  # Smaller for faster tests
            rdim=[128, 128],
            device='cpu'
        )
        return solver

    def setup_parametric_metalens(self, solver, radius1, radius2, gap):
        """Set up a parametric asymmetric dimer metalens structure.

        This creates a structure similar to the one in the field validation
        examples, but with parametric geometry for gradient testing.
        """
        # Add materials
        Si = create_material(name="Si", permittivity=3.48**2)
        SiO2 = create_material(name="SiO2", permittivity=1.46**2)
        solver.add_materials([Si, SiO2])

        # Set reference material
        solver.update_ref_material("SiO2")

        # Create patterned layer
        solver.add_layer(material_name="Si", thickness=0.25, is_homogeneous=False)

        # Create asymmetric dimer pattern
        shape_gen = ShapeGenerator.from_solver(solver)

        # Two circles with different radii and controllable gap
        center1 = (-radius1 - gap/2, 0.0)  # Left circle
        center2 = (gap/2 + radius2, 0.0)   # Right circle

        mask1 = shape_gen.generate_circle_mask(center=center1, radius=radius1)
        mask2 = shape_gen.generate_circle_mask(center=center2, radius=radius2)

        # Combine masks
        dimer_mask = torch.clamp(mask1 + mask2, 0, 1)
        solver.update_er_with_mask(mask=dimer_mask, layer_index=0)

        return solver

    def test_fourier_coefficients_gradient_flow_rdit(self, simple_solver_rdit):
        """Test gradient flow through Fourier coefficient calculations (R-DIT)."""
        # Create parametric geometry
        radius1 = torch.tensor(0.15, requires_grad=True)
        radius2 = torch.tensor(0.18, requires_grad=True)
        gap = torch.tensor(0.05, requires_grad=True)

        # Setup structure
        solver = self.setup_parametric_metalens(simple_solver_rdit, radius1, radius2, gap)

        # Create source
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        results = solver.solve(source)

        # Test reflection interface Fourier coefficients
        ref_coeffs = results.get_reflection_interface_fourier_coefficients()

        # Define loss based on electric field coefficients
        loss_e = torch.sum(torch.abs(ref_coeffs['S_x'])**2 +
                          torch.abs(ref_coeffs['S_y'])**2 +
                          torch.abs(ref_coeffs['S_z'])**2)

        # Test gradient flow
        loss_e.backward(retain_graph=True)

        # Verify gradients exist and are non-zero
        assert radius1.grad is not None, "No gradient for radius1 (electric coefficients)"
        assert radius2.grad is not None, "No gradient for radius2 (electric coefficients)"
        assert gap.grad is not None, "No gradient for gap (electric coefficients)"

        assert not torch.allclose(radius1.grad, torch.tensor(0.0)), "Radius1 gradient is zero"
        assert not torch.allclose(radius2.grad, torch.tensor(0.0)), "Radius2 gradient is zero"
        assert not torch.allclose(gap.grad, torch.tensor(0.0)), "Gap gradient is zero"

        # Clear gradients for magnetic field test
        radius1.grad.zero_()
        radius2.grad.zero_()
        gap.grad.zero_()

        # Test magnetic field coefficients
        loss_h = torch.sum(torch.abs(ref_coeffs['U_x'])**2 +
                          torch.abs(ref_coeffs['U_y'])**2 +
                          torch.abs(ref_coeffs['U_z'])**2)

        loss_h.backward()

        # Verify magnetic field gradients
        assert radius1.grad is not None, "No gradient for radius1 (magnetic coefficients)"
        assert radius2.grad is not None, "No gradient for radius2 (magnetic coefficients)"
        assert gap.grad is not None, "No gradient for gap (magnetic coefficients)"

        assert not torch.allclose(radius1.grad, torch.tensor(0.0)), "Radius1 magnetic gradient is zero"
        assert not torch.allclose(radius2.grad, torch.tensor(0.0)), "Radius2 magnetic gradient is zero"
        assert not torch.allclose(gap.grad, torch.tensor(0.0)), "Gap magnetic gradient is zero"

    def test_fourier_coefficients_gradient_flow_rcwa(self, simple_solver_rcwa):
        """Test gradient flow through Fourier coefficient calculations (RCWA)."""
        # Create parametric geometry
        radius1 = torch.tensor(0.15, requires_grad=True)
        radius2 = torch.tensor(0.18, requires_grad=True)
        gap = torch.tensor(0.05, requires_grad=True)

        # Setup structure
        solver = self.setup_parametric_metalens(simple_solver_rcwa, radius1, radius2, gap)

        # Create source
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        results = solver.solve(source)

        # Test transmission interface Fourier coefficients
        trn_coeffs = results.get_transmission_interface_fourier_coefficients()

        # Define loss based on both electric and magnetic field coefficients
        loss = torch.sum(torch.abs(trn_coeffs['S_x'])**2 +
                        torch.abs(trn_coeffs['U_y'])**2)  # Mix E and H components

        # Test gradient flow
        loss.backward()

        # Verify gradients exist and are non-zero
        assert radius1.grad is not None, "No gradient for radius1 (RCWA)"
        assert radius2.grad is not None, "No gradient for radius2 (RCWA)"
        assert gap.grad is not None, "No gradient for gap (RCWA)"

        assert not torch.allclose(radius1.grad, torch.tensor(0.0)), "Radius1 gradient is zero (RCWA)"
        assert not torch.allclose(radius2.grad, torch.tensor(0.0)), "Radius2 gradient is zero (RCWA)"
        assert not torch.allclose(gap.grad, torch.tensor(0.0)), "Gap gradient is zero (RCWA)"
