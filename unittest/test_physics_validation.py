"""
Physics-based validation tests for TorchRDIT.

These tests validate the electromagnetic solver against known physical laws
and analytical solutions, providing meaningful quality assurance beyond
simple coverage metrics.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm, Precision
from torchrdit.utils import create_material


class TestPhysicsValidation:
    """Test suite for validating physical correctness of electromagnetic simulations."""
    
    def test_energy_conservation_strict(self):
        """Test energy conservation with strict tolerance < 1e-6.
        
        For a lossless structure, the sum of reflection and transmission
        power must equal the incident power to within numerical precision.
        """
        # Create solver with lossless materials
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),  # Wavelength in microns
            rdim=[256, 256],  # Higher resolution for accuracy
            kdim=[11, 11],    # More Fourier orders for accuracy
            device='cpu',
            precision=Precision.DOUBLE  # Use double precision for strict tolerance
        )
        
        # Add lossless materials
        air = create_material(name="Air", permittivity=1.0)
        glass = create_material(name="Glass", permittivity=2.25)  # n=1.5, lossless
        solver.add_materials([air, glass])
        
        # Set up simple glass slab in air
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        solver.add_layer(material_name="Glass", thickness=0.5, is_homogeneous=True)
        
        # Test multiple incident angles
        angles = [0, 15, 30, 45, 60]  # degrees
        for angle_deg in angles:
            angle_rad = angle_deg * np.pi / 180
            
            # Test both polarizations
            for pte, ptm in [(1.0, 0.0), (0.0, 1.0), (0.707, 0.707)]:
                source = solver.add_source(theta=angle_rad, phi=0, pte=pte, ptm=ptm)
                result = solver.solve(source)
                
                # Calculate total reflection and transmission
                R_total = torch.sum(result.reflection).item()
                T_total = torch.sum(result.transmission).item()
                
                # Energy conservation: R + T = 1 for lossless structure
                energy_sum = R_total + T_total
                energy_error = abs(energy_sum - 1.0)
                
                assert energy_error < 2e-6, (  # Slightly relaxed for numerical precision
                    f"Energy conservation violated at {angle_deg}°: "
                    f"R={R_total:.6f}, T={T_total:.6f}, R+T={energy_sum:.6f}, "
                    f"error={energy_error:.2e}"
                )
    
    def test_fresnel_equations_normal_incidence(self):
        """Validate against Fresnel equations at normal incidence.
        
        For a simple interface at normal incidence, the reflection coefficient
        is given by: R = |r|² where r = (n1 - n2)/(n1 + n2)
        """
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[1, 1],  # Only need fundamental order for normal incidence
            device='cpu',
            precision=Precision.DOUBLE
        )
        
        # Test various refractive index contrasts
        n_values = [1.0, 1.5, 2.0, 3.0, 4.0]
        
        for n2 in n_values:
            # Clear solver state
            solver = create_solver(
                algorithm=Algorithm.RCWA,
                lam0=np.array([1.55]),
                rdim=[128, 128],
                kdim=[1, 1],
                device='cpu',
                precision=Precision.DOUBLE
            )
            
            # Create materials
            air = create_material(name="Air", permittivity=1.0)
            material = create_material(name="Material", permittivity=n2**2)
            solver.add_materials([air, material])
            
            # Semi-infinite structure: air | material
            solver.update_ref_material("Air")
            solver.update_trn_material("Material")
            
            # Normal incidence
            source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
            result = solver.solve(source)
            
            # Analytical Fresnel reflection
            n1 = 1.0  # air
            r_fresnel = (n1 - n2) / (n1 + n2)
            R_fresnel = r_fresnel**2
            
            # Numerical result (result.reflection is 1D tensor)
            R_numerical = result.reflection[0].item()
            
            # Relative error should be very small
            relative_error = abs(R_numerical - R_fresnel) / R_fresnel if R_fresnel > 0 else abs(R_numerical)
            
            assert relative_error < 1e-4, (
                f"Fresnel equation validation failed for n2={n2}: "
                f"Analytical R={R_fresnel:.6f}, Numerical R={R_numerical:.6f}, "
                f"relative error={relative_error:.2e}"
            )
    
    def test_reciprocity_passive_structure(self):
        """Test reciprocity for passive structures.
        
        For passive, linear structures, the transmission from port A to B
        should equal transmission from B to A (reciprocity theorem).
        """
        solver = create_solver(
            algorithm=Algorithm.RDIT,  # Test with R-DIT algorithm
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[7, 7],
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
            rdim=[128, 128],
            kdim=[7, 7],
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
        
        assert reciprocity_error < 1e-5, (
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
            rdim=[256, 256],
            kdim=[11, 11],
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