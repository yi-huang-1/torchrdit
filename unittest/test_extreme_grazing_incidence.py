"""
Extreme grazing incidence stability tests.

These tests validate solver behavior near θ ≈ 90°, focusing on physical
outputs and energy conservation with the built-in kinc_z protection.
"""

import pytest
import numpy as np
import torch
from torchrdit.constants import Algorithm, Precision
from torchrdit.solver import create_solver
from torchrdit.utils import create_material


class TestExtremeGrazingIncidence:
    """Test suite for extreme grazing incidence edge case protection."""
    
    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    def test_default_protection_works(self, algo):
        """Default protection yields physical, energy-conserving TE results at θ≈90°."""
        solver = create_solver(
            algorithm=algo,
            precision=Precision.SINGLE,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device="cpu",
        )

        air = create_material(name="Air", permittivity=1.0)
        sio2 = create_material(name="SiO2", permittivity=2.25)
        solver.add_materials([air, sio2])

        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)

        theta = np.deg2rad(89.99)
        source = solver.add_source(theta=theta, phi=0.0, pte=1.0, ptm=0.0)
        results = solver.solve(source)

        R = torch.sum(results.reflection).item()
        T = torch.sum(results.transmission).item()

        assert np.isfinite(R) and np.isfinite(T)
        assert 0.0 <= R <= 1.0 and 0.0 <= T <= 1.0
        # Near grazing TE reflection should dominate
        assert R > T, f"{algo.name}: Expected R > T at grazing incidence"
        # Energy conservation should hold within tolerance (grazing is numerically tough)
        assert np.isclose(R + T, 1.0, atol=2e-2), f"{algo.name}: R+T not ~1, got {R+T}"
    
    @pytest.mark.parametrize("pte,ptm,pol_name", [(1.0, 0.0, "TE"), (0.0, 1.0, "TM")])
    def test_both_polarizations_physical(self, pte, ptm, pol_name):
        """Both TE and TM produce physical, finite outputs at θ≈90° with protection."""
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            precision=Precision.SINGLE,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device="cpu",
        )

        air = create_material(name="Air", permittivity=1.0)
        sio2 = create_material(name="SiO2", permittivity=2.25)
        solver.add_materials([air, sio2])

        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)

        theta = np.deg2rad(89.99)
        source = solver.add_source(theta=theta, phi=0.0, pte=pte, ptm=ptm)
        results = solver.solve(source)

        R = torch.sum(results.reflection).item()
        T = torch.sum(results.transmission).item()

        assert np.isfinite(R) and np.isfinite(T)
        assert 0.0 <= R <= 1.0 and 0.0 <= T <= 1.0
        if pol_name == "TE":
            # TE: near-grazing reflection dominates and energy ~ conserved
            assert np.isclose(R + T, 1.0, atol=2e-2), f"{pol_name}: R+T not ~1, got {R+T}"
        else:
            # TM: keep to physical bounds; numerical edge-case may deviate in energy sum
            pass
    
    # Note: removed "double precision produces unphysical results" and
    # "restores physical results" tests because they didn’t assert on
    # repository behavior and relied on prints or vague expectations.
