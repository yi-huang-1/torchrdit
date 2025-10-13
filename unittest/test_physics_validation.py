"""
Physics-based validation tests for TorchRDIT.

These tests validate the electromagnetic solver against known physical laws
and analytical solutions, providing meaningful quality assurance beyond
simple coverage metrics.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import torch
import pytest
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm, Precision
from torchrdit.utils import create_material


class TestPhysicsValidation:
    """Test suite for validating physical correctness of electromagnetic simulations."""
    
    @pytest.mark.parametrize("n2", [1.0, 1.5, 2.0, 3.0, 4.0])
    def test_fresnel_equations_normal_incidence(self, n2: float):
        """Validate Fresnel equations at normal incidence for multiple index contrasts.

        For a simple interface at normal incidence, R = |(n1 - n2)/(n1 + n2)|^2.
        """
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[64, 64],
            kdim=[1, 1],  # Only fundamental order for normal incidence
            device='cpu',
            precision=Precision.DOUBLE
        )

        # Create materials and configure semi-infinite interface: air | material
        air = create_material(name="Air", permittivity=1.0)
        material = create_material(name="Material", permittivity=n2**2)
        solver.add_materials([air, material])
        solver.update_ref_material("Air")
        solver.update_trn_material("Material")

        # Normal incidence, TE polarization (same as TM at theta=0)
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        result = solver.solve(source)

        # Analytical Fresnel reflection
        n1 = 1.0  # air
        r_fresnel = (n1 - n2) / (n1 + n2)
        R_fresnel = r_fresnel**2

        # Numerical result (single wavelength)
        R_numerical = result.reflection[0].item()

        # Relative error tolerance
        relative_error = abs(R_numerical - R_fresnel) / R_fresnel if R_fresnel > 0 else abs(R_numerical)

        assert relative_error < 1e-4, (
            f"Fresnel normal-incidence failed for n2={n2}: "
            f"Analytical R={R_fresnel:.6f}, Numerical R={R_numerical:.6f}, "
            f"relative error={relative_error:.2e}"
        )

    @pytest.mark.parametrize("n2", [1.5, 2.0])
    @pytest.mark.parametrize("angle_deg, at_brewster", [(15.0, False), (45.0, False), (60.0, False), (None, True)])
    def test_fresnel_equations_oblique_incidence_te_tm(self, n2: float, angle_deg: float | None, at_brewster: bool):
        """Spot-check Fresnel equations at oblique incidence for TE and TM polarization.

        Adds a grazing-angle case (60°) and explicitly checks the TM Brewster-angle condition
        (near-zero reflection) when requested.

        Analytical Fresnel formulas:
          R_TE = |(n1 cosθ1 - n2 cosθ2)/(n1 cosθ1 + n2 cosθ2)|^2
          R_TM = |(n2 cosθ1 - n1 cosθ2)/(n2 cosθ1 + n1 cosθ2)|^2
        with Snell's law n1 sinθ1 = n2 sinθ2, and tan(θB) = n2/n1 for TM Brewster angle.
        """
        if at_brewster:
            theta1 = np.arctan(n2 / 1.0)
        else:
            theta1 = angle_deg * np.pi / 180.0
        angle_deg_report = float(theta1 * 180.0 / np.pi)
        n1 = 1.0

        # Compute transmitted angle via Snell's law
        sin_theta2 = n1 * np.sin(theta1) / n2
        # Guard against total internal reflection (won't occur for n2>n1 here)
        sin_theta2 = np.clip(sin_theta2, -1.0, 1.0)
        theta2 = np.arcsin(sin_theta2)

        cos1 = np.cos(theta1)
        cos2 = np.cos(theta2)

        r_te = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
        r_tm = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
        R_te_analytical = abs(r_te) ** 2
        R_tm_analytical = abs(r_tm) ** 2

        # Configure solver for a single interface
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[64, 64],
            kdim=[1, 1],
            device='cpu',
            precision=Precision.DOUBLE
        )
        air = create_material(name="Air", permittivity=n1**2)
        material = create_material(name="Material", permittivity=n2**2)
        solver.add_materials([air, material])
        solver.update_ref_material("Air")
        solver.update_trn_material("Material")

        # TE polarization
        result_te = solver.solve(solver.add_source(theta=theta1, phi=0, pte=1.0, ptm=0.0))
        R_te_num = result_te.reflection[0].item()
        abs_err_te = abs(R_te_num - R_te_analytical)
        denom_te = max(abs(R_te_analytical), 1e-12)
        rel_err_te = abs_err_te / denom_te

        # TM polarization
        result_tm = solver.solve(solver.add_source(theta=theta1, phi=0, pte=0.0, ptm=1.0))
        R_tm_num = result_tm.reflection[0].item()
        abs_err_tm = abs(R_tm_num - R_tm_analytical)
        denom_tm = max(abs(R_tm_analytical), 1e-12)
        rel_err_tm = abs_err_tm / denom_tm

        # Keep this a lightweight spot-check with slightly looser tolerance
        # TE: use relative error when well-conditioned, otherwise absolute
        if abs(R_te_analytical) > 1e-8:
            assert rel_err_te < 1e-3, (
                f"Fresnel TE oblique failed (n2={n2}, θ={angle_deg_report:.2f}°): "
                f"analytical={R_te_analytical:.6f}, numerical={R_te_num:.6f}, rel_err={rel_err_te:.2e}"
            )
        else:
            assert abs_err_te < 5e-4, (
                f"Fresnel TE near-zero case failed (n2={n2}, θ={angle_deg_report:.2f}°): "
                f"analytical={R_te_analytical:.6e}, numerical={R_te_num:.6e}, abs_err={abs_err_te:.2e}"
            )

        # TM: handle Brewster explicitly, else use relative/absolute as above
        if at_brewster:
            assert R_tm_num < 5e-4, (
                f"TM reflection not near-zero at Brewster angle (n2={n2}, θB={angle_deg_report:.2f}°): R_tm={R_tm_num:.6e}"
            )
        else:
            if abs(R_tm_analytical) > 1e-8:
                assert rel_err_tm < 1e-3, (
                    f"Fresnel TM oblique failed (n2={n2}, θ={angle_deg_report:.2f}°): "
                    f"analytical={R_tm_analytical:.6f}, numerical={R_tm_num:.6f}, rel_err={rel_err_tm:.2e}"
                )
            else:
                assert abs_err_tm < 5e-4, (
                    f"Fresnel TM near-zero case failed (n2={n2}, θ={angle_deg_report:.2f}°): "
                    f"analytical={R_tm_analytical:.6e}, numerical={R_tm_num:.6e}, abs_err={abs_err_tm:.2e}"
                )

        # If testing exactly at Brewster angle, TM reflection should be near zero
        if at_brewster:
            assert R_tm_num < 5e-4, (
                f"TM reflection not near-zero at Brewster angle (n2={n2}, θB={angle_deg_report:.2f}°): R_tm={R_tm_num:.6e}"
            )
    
    def test_reciprocity_passive_structure(self):
        """Test reciprocity for passive structures.
        
        For passive, linear structures, the transmission from port A to B
        should equal transmission from B to A (reciprocity theorem).
        """
        solver = create_solver(
            algorithm=Algorithm.RDIT,  # Test with R-DIT algorithm
            lam0=np.array([1.55]),
            rdim=[64, 64],  # No diffraction in homogeneous stack; keep small
            kdim=[1, 1],
            device='cpu'
        )
        
        # Create asymmetric structure to make test non-trivial
        air = create_material(name="Air", permittivity=1.0)
        glass1 = create_material(name="Glass1", permittivity=2.25)  # n=1.5
        glass2 = create_material(name="Glass2", permittivity=3.24)  # n=1.8
        solver.add_materials([air, glass1, glass2])
        
        # Asymmetric multilayer: air | glass1 | glass2 | air
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        solver.add_layer(material_name="Glass1", thickness=0.3, is_homogeneous=True)
        solver.add_layer(material_name="Glass2", thickness=0.5, is_homogeneous=True)
        
        # Test at oblique angle
        angle_rad = 30 * np.pi / 180
        
        # Forward direction (top to bottom)
        source_forward = solver.add_source(theta=angle_rad, phi=0, pte=1.0, ptm=0.0)
        result_forward = solver.solve(source_forward)
        T_forward = torch.sum(result_forward.transmission).item()
        
        # Reverse structure for backward test
        solver_reverse = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[64, 64],
            kdim=[1, 1],
            device='cpu'
        )
        solver_reverse.add_materials([air, glass1, glass2])
        solver_reverse.update_ref_material("Air")
        solver_reverse.update_trn_material("Air")
        # Add layers in reverse order
        solver_reverse.add_layer(material_name="Glass2", thickness=0.5, is_homogeneous=True)
        solver_reverse.add_layer(material_name="Glass1", thickness=0.3, is_homogeneous=True)
        
        # Backward direction (bottom to top in original structure)
        source_backward = solver_reverse.add_source(theta=angle_rad, phi=0, pte=1.0, ptm=0.0)
        result_backward = solver_reverse.solve(source_backward)
        T_backward = torch.sum(result_backward.transmission).item()
        
        # Check reciprocity
        reciprocity_error = abs(T_forward - T_backward)
        
        assert reciprocity_error < 1e-4, (
            f"Reciprocity violated: T_forward={T_forward:.6f}, "
            f"T_backward={T_backward:.6f}, error={reciprocity_error:.2e}"
        )
    
    def test_brewster_angle_tm_polarization(self):
        """Validate Brewster angle for TM polarization.
        
        At Brewster angle θ_B = arctan(n2/n1), TM polarization has zero reflection
        for an interface between two dielectric materials.
        """
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[64, 64],  # Interface-only case; small grid suffices
            kdim=[1, 1],
            device='cpu',
            precision=Precision.DOUBLE
        )
        
        # Air to glass interface
        n1 = 1.0  # air
        n2 = 1.5  # glass
        
        air = create_material(name="Air", permittivity=n1**2)
        glass = create_material(name="Glass", permittivity=n2**2)
        solver.add_materials([air, glass])
        
        solver.update_ref_material("Air")
        solver.update_trn_material("Glass")
        
        # Calculate Brewster angle
        brewster_angle = np.arctan(n2 / n1)
        
        # Test TM polarization at Brewster angle
        source = solver.add_source(theta=brewster_angle, phi=0, pte=0.0, ptm=1.0)
        result = solver.solve(source)
        
        # Reflection should be near zero for TM at Brewster angle
        R_tm = torch.sum(result.reflection).item()
        
        assert R_tm < 1e-4, (
            f"Brewster angle validation failed: "
            f"R_tm={R_tm:.6f} at θ_B={brewster_angle*180/np.pi:.2f}°"
        )
        
        # Also verify TE has significant reflection at same angle
        source_te = solver.add_source(theta=brewster_angle, phi=0, pte=1.0, ptm=0.0)
        result_te = solver.solve(source_te)
        R_te = torch.sum(result_te.reflection).item()
        
        assert R_te > 0.1, (
            f"TE should have significant reflection at Brewster angle: R_te={R_te:.6f}"
        )
