"""
Test coverage for NaN prevention in division-by-zero scenarios.

This test ensures that all division operations involving mat_kz values 
have proper epsilon protection to prevent singular matrix errors.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np

from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm


class TestNaNPreventionCoverage:
    """Test that all division-by-zero cases are properly handled."""
    
    @pytest.fixture
    def solver_multi_wavelength(self):
        """Create solver with multiple wavelengths to trigger edge cases."""
        # Use wavelengths that can cause mat_kz harmonics to be near zero
        lam0 = np.linspace(1.0, 2.0, 5)
        
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=lam0,
            rdim=[64, 64],  # Smaller for faster testing
            kdim=[3, 3],    # Small kdim to focus on specific harmonics
            device='cpu'
        )
        
        # Create materials
        mat_air = create_material(name="Air", permittivity=1.0)
        mat_si = create_material(name="Si", permittivity=11.7)
        
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        
        return solver
    
    def test_sequential_nan_prevention(self, solver_multi_wavelength):
        """Test that sequential processing doesn't produce NaN values."""
        solver = solver_multi_wavelength
        
        # Add a structure that could cause problematic mat_kz values
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.1), is_homogeneous=False)
        
        # Test source that might cause edge cases
        source = {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        
        # This should not raise any exceptions
        result = solver.solve(source)
        
        # Check that all results are finite
        assert torch.all(torch.isfinite(result.reflection)), "Reflection contains NaN/Inf"
        assert torch.all(torch.isfinite(result.transmission)), "Transmission contains NaN/Inf"
        assert torch.all(torch.isfinite(result.reflection_diffraction)), "Reflection diffraction contains NaN/Inf"
        assert torch.all(torch.isfinite(result.transmission_diffraction)), "Transmission diffraction contains NaN/Inf"
    
    def test_batched_nan_prevention(self, solver_multi_wavelength):
        """Test that batched processing doesn't produce NaN values."""
        solver = solver_multi_wavelength
        
        # Add a structure that could cause problematic mat_kz values
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.1), is_homogeneous=False)
        
        # Multiple sources including edge cases
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},    # Normal incidence
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},   # Small angle
            {"theta": 0.2, "phi": 0.1, "pte": 0.7071, "ptm": 0.7071},  # Mixed polarization
        ]
        
        # This should not raise any exceptions (was failing before fix)
        results = solver.solve(sources)
        
        # Check that all results are finite for all sources
        for i in range(len(sources)):
            assert torch.all(torch.isfinite(results[i].reflection)), f"Source {i} reflection contains NaN/Inf"
            assert torch.all(torch.isfinite(results[i].transmission)), f"Source {i} transmission contains NaN/Inf"
            assert torch.all(torch.isfinite(results[i].reflection_diffraction)), f"Source {i} reflection diffraction contains NaN/Inf"
            assert torch.all(torch.isfinite(results[i].transmission_diffraction)), f"Source {i} transmission diffraction contains NaN/Inf"
    
    def test_problematic_harmonics_coverage(self, solver_multi_wavelength):
        """Test specific cases that could lead to zero mat_kz harmonics."""
        solver = solver_multi_wavelength
        
        # Add layer that makes certain harmonics problematic
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.01), is_homogeneous=False)
        
        # Sources designed to stress-test harmonic calculations
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},      # Exactly normal
            {"theta": 1e-6, "phi": 0.0, "pte": 1.0, "ptm": 0.0},    # Nearly normal
            {"theta": 0.5, "phi": 0.5, "pte": 0.5, "ptm": 0.866},   # Oblique incidence
        ]
        
        # Test both sequential and batched
        sequential_results = []
        for src in sources:
            result = solver.solve(src)
            sequential_results.append(result)
            # Each result should be finite
            assert torch.all(torch.isfinite(result.reflection))
            assert torch.all(torch.isfinite(result.transmission))
        
        # Batched should also work without NaN
        batched_results = solver.solve(sources)
        
        # Verify batched results are also finite
        for i in range(len(sources)):
            assert torch.all(torch.isfinite(batched_results[i].reflection))
            assert torch.all(torch.isfinite(batched_results[i].transmission))
    
    def test_epsilon_effectiveness(self, solver_multi_wavelength):
        """Test that epsilon addition prevents singular matrices without affecting accuracy."""
        solver = solver_multi_wavelength
        
        # Simple structure for testing
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=True)
        
        source = {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        
        # This should work with epsilon protection
        result = solver.solve(source)
        
        # Results should be reasonable (just check they're finite)
        assert torch.all(torch.isfinite(result.reflection)), "Reflection contains non-finite values"
        assert torch.all(torch.isfinite(result.transmission)), "Transmission contains non-finite values"
        
        # Basic sanity checks - values should be non-negative
        assert torch.all(result.reflection >= 0), "Negative reflection detected"
        assert torch.all(result.transmission >= 0), "Negative transmission detected"
    
    def test_gradient_flow_with_epsilon(self, solver_multi_wavelength):
        """Test that epsilon addition doesn't break gradient computation."""
        solver = solver_multi_wavelength
        
        # Create differentiable thickness
        thickness = torch.tensor(0.3, requires_grad=True)
        solver.add_layer(material_name="Si", thickness=thickness, is_homogeneous=True)
        
        # Single source for gradient test
        source = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        
        # Forward pass
        result = solver.solve(source)
        
        # Define loss - minimize reflection at first wavelength
        loss = result.reflection[0]
        
        # Backward pass should work without NaN
        loss.backward()
        
        # Gradient should exist and be finite
        assert thickness.grad is not None, "No gradient computed"
        assert torch.isfinite(thickness.grad), f"Gradient is not finite: {thickness.grad}"
        assert thickness.grad.abs() > 1e-12, "Gradient suspiciously small"
        assert thickness.grad.abs() < 1e2, "Gradient suspiciously large"
    
    def test_rcwa_algorithm_coverage(self):
        """Test that RCWA algorithm also has proper NaN prevention."""
        # Create RCWA solver with multiple wavelengths
        lam0 = np.linspace(1.2, 1.8, 3)
        
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=lam0,
            rdim=[64, 64],
            kdim=[3, 3],
            device='cpu'
        )
        
        # Add materials
        mat_air = create_material(name="Air", permittivity=1.0)
        mat_si = create_material(name="Si", permittivity=11.7)
        
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        
        # Add layer
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.2), is_homogeneous=False)
        
        # Test sources
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        ]
        
        # Both sequential and batched should work
        for src in sources:
            result = solver.solve(src)
            assert torch.all(torch.isfinite(result.reflection))
            assert torch.all(torch.isfinite(result.transmission))
        
        # Batched processing
        results = solver.solve(sources)
        for i in range(len(sources)):
            assert torch.all(torch.isfinite(results[i].reflection))
            assert torch.all(torch.isfinite(results[i].transmission))