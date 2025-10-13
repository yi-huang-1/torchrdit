"""Focused NaN-prevention coverage for solver paths (R-DIT and RCWA).

Exercises real solver code paths that apply epsilon/softplus protections
around mat_kz and kinc_z without duplicating similar checks elsewhere.
"""

import pytest
import torch
import numpy as np

from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm


class TestNaNPreventionCoverage:
    """Ensure epsilon protections prevent NaNs/Inf in outputs and grads."""

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

    def test_nan_prevention_rdit_sequential_and_batched(self, solver_multi_wavelength):
        """R-DIT: outputs remain finite for sequential and batched sources."""
        solver = solver_multi_wavelength
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.1), is_homogeneous=False)

        # Representative sources: normal, small angle, mixed polarization
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 1e-4, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.25, "phi": 0.1, "pte": 0.6, "ptm": 0.8},
        ]

        # Sequential
        for src in sources:
            res = solver.solve(src)
            assert torch.isfinite(res.reflection).all()
            assert torch.isfinite(res.transmission).all()

        # Batched
        batched = solver.solve(sources)
        for i in range(len(sources)):
            assert torch.isfinite(batched[i].reflection).all()
            assert torch.isfinite(batched[i].transmission).all()
    
    def test_gradient_flow_with_epsilon(self, solver_multi_wavelength):
        """R-DIT: finite gradients despite epsilon protections."""
        solver = solver_multi_wavelength
        thickness = torch.tensor(0.3, requires_grad=True)
        solver.add_layer(material_name="Si", thickness=thickness, is_homogeneous=True)

        res = solver.solve({"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0})
        loss = res.reflection[0]
        loss.backward()

        assert thickness.grad is not None
        assert torch.isfinite(thickness.grad)
    
    def test_nan_prevention_rcwa_sequential_and_batched(self):
        """RCWA: outputs remain finite for sequential and batched sources."""
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
        
        # Sequential
        for src in sources:
            result = solver.solve(src)
            assert torch.isfinite(result.reflection).all()
            assert torch.isfinite(result.transmission).all()

        # Batched
        results = solver.solve(sources)
        for i in range(len(sources)):
            assert torch.isfinite(results[i].reflection).all()
            assert torch.isfinite(results[i].transmission).all()
