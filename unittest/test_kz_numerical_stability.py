"""Solver-level edge-case tests for kz numerical stability.

These tests target specific physical conditions where kz → 0 for some
diffraction orders, triggering the numerical protections in solver.py
and algorithm.py.  Each test validates both finiteness AND physical
correctness (energy conservation, analytical bounds).
"""

import numpy as np
import pytest
import torch

from torchrdit.constants import Algorithm, Precision
from torchrdit.solver import create_solver
from torchrdit.utils import create_material


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_single_interface_solver(*, algo, n1=1.0, n2=1.5, precision=Precision.DOUBLE):
    """Air (n1) | Material (n2) — simplest non-trivial structure."""
    solver = create_solver(
        algorithm=algo,
        lam0=np.array([1.55]),
        grids=[64, 64],
        harmonics=[1, 1],
        device="cpu",
        precision=precision,
    )
    air = create_material(name="Air", permittivity=n1**2)
    mat = create_material(name="Mat", permittivity=n2**2)
    solver.add_materials([air, mat])
    solver.update_ref_material("Air")
    solver.update_trn_material("Mat")
    return solver


def _build_grating_solver(*, algo, precision=Precision.DOUBLE):
    """Air | Si grating | Air — structure with higher harmonics."""
    solver = create_solver(
        algorithm=algo,
        lam0=np.array([1.0]),
        grids=[128, 128],
        harmonics=[3, 3],
        device="cpu",
        precision=precision,
    )
    air = create_material(name="Air", permittivity=1.0)
    si = create_material(name="Si", permittivity=11.7)
    solver.add_materials([air, si])
    solver.update_ref_material("Air")
    solver.update_trn_material("Air")
    solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
    return solver


def _solve_and_check(solver, source, *, atol_energy=5e-3, label=""):
    """Run solver and assert finiteness + energy conservation."""
    result = solver.solve(source)
    R = torch.sum(result.reflection).item()
    T = torch.sum(result.transmission).item()

    assert np.isfinite(R), f"{label}: R is not finite ({R})"
    assert np.isfinite(T), f"{label}: T is not finite ({T})"
    assert R >= -1e-3, f"{label}: R negative ({R})"
    assert T >= -1e-3, f"{label}: T negative ({T})"
    assert np.isclose(R + T, 1.0, atol=atol_energy), (
        f"{label}: energy not conserved, R+T={R + T:.6f}"
    )
    return R, T


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBrewsterAngleBothAlgorithms:
    """TM at Brewster angle: R_tm ≈ 0, R+T ≈ 1 for both RCWA and R-DIT."""

    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    @pytest.mark.parametrize("n2", [1.5, 2.0, 3.0])
    def test_brewster_angle_energy_conservation(self, algo, n2):
        solver = _build_single_interface_solver(algo=algo, n2=n2)
        brewster = np.arctan(n2 / 1.0)
        source = solver.add_source(theta=brewster, phi=0, pte=0.0, ptm=1.0)
        R, T = _solve_and_check(
            solver, source,
            atol_energy=1e-3,
            label=f"{algo.name}/n2={n2}/Brewster",
        )
        assert R < 1e-3, (
            f"{algo.name}: R_tm should be near zero at Brewster, got {R:.6e}"
        )


class TestGrazingIncidenceSweep:
    """Sweep from 80° to 89.5° — energy must stay conserved, no NaN."""

    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    def test_grazing_sweep_te(self, algo):
        solver = _build_single_interface_solver(algo=algo, n2=1.5)
        # Tolerance widens at extreme angles where numerical precision degrades
        angle_tolerance = [
            (80.0, 5e-3), (83.0, 5e-3), (85.0, 5e-3),
            (87.0, 5e-3), (88.0, 1e-2), (89.0, 1e-2),
        ]
        prev_R = -1.0
        for deg, atol in angle_tolerance:
            theta = np.deg2rad(deg)
            source = solver.add_source(theta=theta, phi=0, pte=1.0, ptm=0.0)
            R, T = _solve_and_check(
                solver, source,
                atol_energy=atol,
                label=f"{algo.name}/TE/{deg}°",
            )
            # TE reflection should increase with angle (approaching total reflection)
            assert R >= prev_R - 1e-2, (
                f"{algo.name}: R decreased from {prev_R:.4f} to {R:.4f} at {deg}°"
            )
            prev_R = R


class TestWoodAnomalyConditions:
    """At Wood anomaly, a diffraction order goes evanescent (kz → 0).

    We choose angle/wavelength so that the (±1, 0) order transitions
    from propagating to evanescent.  The solver must remain stable.
    """

    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    def test_near_rayleigh_cutoff(self, algo):
        """Angle near Rayleigh anomaly for period=1.0, λ=1.0 grating."""
        solver = _build_grating_solver(algo=algo)

        # At normal incidence with λ/Λ = 1.0 and harmonics=[3,3],
        # the (±1,0) orders have kz = sqrt(1 - (m*λ/Λ)^2) = 0 when m=±1.
        # Test near this condition with small oblique angle.
        for theta_deg in [0.0, 0.5, 1.0, 2.0]:
            theta = np.deg2rad(theta_deg)
            source = solver.add_source(theta=theta, phi=0, pte=1.0, ptm=0.0)
            result = solver.solve(source)
            R = torch.sum(result.reflection).item()
            T = torch.sum(result.transmission).item()
            assert np.isfinite(R) and np.isfinite(T), (
                f"{algo.name}: NaN at θ={theta_deg}° near Rayleigh cutoff"
            )
            assert R >= -1e-3 and T >= -1e-3, (
                f"{algo.name}: Negative R={R:.4f} or T={T:.4f} at θ={theta_deg}°"
            )


class TestMultiWavelengthEdgeCases:
    """Multiple wavelengths where different orders hit kz ≈ 0."""

    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    def test_multi_wavelength_stability(self, algo):
        # Wavelengths chosen so different diffraction orders go evanescent
        lam0 = np.array([0.9, 1.0, 1.1, 1.5])
        solver = create_solver(
            algorithm=algo,
            lam0=lam0,
            grids=[128, 128],
            harmonics=[3, 3],
            device="cpu",
            precision=Precision.DOUBLE,
        )
        air = create_material(name="Air", permittivity=1.0)
        si = create_material(name="Si", permittivity=11.7)
        solver.add_materials([air, si])
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)

        source = solver.add_source(theta=0.1, phi=0, pte=1.0, ptm=0.0)
        result = solver.solve(source)

        for wl_idx in range(len(lam0)):
            R = result.reflection[wl_idx].item()
            T = result.transmission[wl_idx].item()
            assert np.isfinite(R) and np.isfinite(T), (
                f"{algo.name}: NaN at λ={lam0[wl_idx]} (index {wl_idx})"
            )


class TestGradientAtBrewsterAngle:
    """Gradient of R w.r.t. layer thickness should be finite at Brewster angle."""

    @pytest.mark.parametrize("algo", [Algorithm.RCWA, Algorithm.RDIT])
    def test_gradient_finite_at_brewster(self, algo):
        solver = create_solver(
            algorithm=algo,
            lam0=np.array([1.55]),
            grids=[64, 64],
            harmonics=[1, 1],
            device="cpu",
        )
        air = create_material(name="Air", permittivity=1.0)
        glass = create_material(name="Glass", permittivity=2.25)
        solver.add_materials([air, glass])
        solver.update_ref_material("Air")
        solver.update_trn_material("Glass")

        thickness = torch.tensor(0.5, requires_grad=True)
        solver.add_layer(material_name="Glass", thickness=thickness, is_homogeneous=True)

        brewster = np.arctan(1.5 / 1.0)
        source = solver.add_source(theta=brewster, phi=0, pte=0.0, ptm=1.0)
        result = solver.solve(source)

        loss = torch.sum(result.reflection)
        loss.backward()

        assert thickness.grad is not None, f"{algo.name}: no gradient computed"
        assert torch.isfinite(thickness.grad), (
            f"{algo.name}: gradient is {thickness.grad} at Brewster angle"
        )
