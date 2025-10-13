"""
Unit tests for unified polarization tensorization logic.

Scope:
- Verify mathematical equivalence between single-source and batched paths
- Cover representative angles (normal/oblique) and polarizations (TE/TM/mixed/circular)
- Exercise multi-source batching vs sequential processing
- Include edge angles (near-0 and grazing) for numerical robustness

Notes:
- Tests intentionally avoid self-referential comparisons (same function vs itself)
  that do not validate distinct code paths.
"""

import torch
import numpy as np
import pytest
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material


class TestPolarizationTensorization:
    """Test suite ensuring mathematical equivalence between polarization implementations."""

    @pytest.fixture
    def setup_solver(self):
        """Create a solver with basic configuration for testing."""
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array(
                [1.55, 1.31, 1.064]
            ),  # Multiple wavelengths for comprehensive testing
            rdim=[512, 512],
            kdim=[5, 5],
            device="cpu",  # Use CPU for consistent testing
            debug_batching=True,  # Enable debug output for development
        )

        # Add materials
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])

        # Add a simple layer structure
        solver.add_layer(material_name="Si", thickness=0.5)

        return solver

    def _extract_single_polarization_data(self, solver, source):
        """Extract polarization data using unified method for single source."""
        # Set source and prepare solver state
        solver.src = source
        solver._pre_solve()  # Sets up kinc

        # Use the unified method without arguments for single source
        return solver._calculate_polarization()

    def _extract_batched_polarization_data(self, solver, sources):
        """Extract polarization data using unified method for batched sources."""
        # Set up solver state for batched sources
        solver._pre_solve(sources)  # Sets up batched kinc

        # Use the unified method with sources argument for batched
        return solver._calculate_polarization(sources)

    # Removed unified helper (was only used for self-comparisons of the same method).

    def test_single_vs_batched_normal_incidence(self, setup_solver):
        """Test mathematical equivalence for normal incidence (θ≈0)."""
        solver = setup_solver

        # Create normal incidence source
        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)

        # Extract data using single-source logic
        single_data = self._extract_single_polarization_data(solver, source)

        # Extract data using batched logic
        batched_data = self._extract_batched_polarization_data(solver, [source])

        # Compare ate vectors
        ate_single = single_data["ate"]
        ate_batched = batched_data["ate"][0]  # Extract first source

        max_diff_ate = torch.abs(ate_single - ate_batched).max().item()
        assert torch.allclose(ate_single, ate_batched, atol=1e-6, rtol=1e-5), (
            f"ate mismatch for normal incidence: max diff = {max_diff_ate}"
        )

        # Compare atm vectors
        atm_single = single_data["atm"]
        atm_batched = batched_data["atm"][0]

        max_diff_atm = torch.abs(atm_single - atm_batched).max().item()
        assert torch.allclose(atm_single, atm_batched, atol=1e-6, rtol=1e-5), (
            f"atm mismatch for normal incidence: max diff = {max_diff_atm}"
        )

        # Compare polarization vectors
        pol_single = single_data["pol_vec"]
        pol_batched = batched_data["pol_vec"][0]

        max_diff_pol = torch.abs(pol_single - pol_batched).max().item()
        assert torch.allclose(pol_single, pol_batched, atol=1e-6, rtol=1e-5), (
            f"pol_vec mismatch for normal incidence: max diff = {max_diff_pol}"
        )

        # Compare electric field source vectors
        esrc_single = single_data["esrc"]
        esrc_batched = batched_data["esrc"][0]

        max_diff_esrc = torch.abs(esrc_single - esrc_batched).max().item()
        assert torch.allclose(esrc_single, esrc_batched, atol=1e-6, rtol=1e-5), (
            f"esrc mismatch for normal incidence: max diff = {max_diff_esrc}"
        )

        # No printouts in unit tests; assertions above suffice.

    def test_single_vs_batched_oblique_incidence(self, setup_solver):
        """Test mathematical equivalence for oblique incidence angles."""
        solver = setup_solver

        # Test various oblique angles
        test_angles = [15, 30, 45, 60]  # degrees

        for angle_deg in test_angles:
            angle_rad = angle_deg * np.pi / 180

            # Test both phi=0 and phi=45 for comprehensive coverage
            for phi_deg in [0, 45]:
                phi_rad = phi_deg * np.pi / 180

                source = solver.add_source(
                    theta=angle_rad, phi=phi_rad, pte=0.7071, ptm=0.7071
                )

                # Extract data using both methods
                single_data = self._extract_single_polarization_data(solver, source)
                batched_data = self._extract_batched_polarization_data(solver, [source])

                # Compare all outputs
                for key in ["ate", "atm", "pol_vec", "esrc"]:
                    single_val = single_data[key]
                    batched_val = batched_data[key][0]

                    max_diff = torch.abs(single_val - batched_val).max().item()
                    assert torch.allclose(
                        single_val, batched_val, atol=1e-6, rtol=1e-5
                    ), (
                        f"{key} mismatch for θ={angle_deg}°, φ={phi_deg}°: max diff = {max_diff}"
                    )

                # Assertions above ensure correctness across angles and azimuths.

    def test_edge_cases(self, setup_solver):
        """Test extreme angles and special cases."""
        solver = setup_solver

        # Test case 1: Very small angle (near 0)
        small_angle = 1e-6
        source_small = solver.add_source(theta=small_angle, phi=0, pte=1.0, ptm=0.0)

        single_data = self._extract_single_polarization_data(solver, source_small)
        batched_data = self._extract_batched_polarization_data(solver, [source_small])

        # Should behave like normal incidence
        for key in ["ate", "atm", "pol_vec", "esrc"]:
            single_val = single_data[key]
            batched_val = batched_data[key][0]

            assert torch.allclose(single_val, batched_val, atol=1e-6), (
                f"{key} mismatch for very small angle"
            )

            # Check for NaN/Inf
            assert torch.all(torch.isfinite(single_val)), f"NaN/Inf in single {key}"
            assert torch.all(torch.isfinite(batched_val)), f"NaN/Inf in batched {key}"

        # Test case 2: Grazing incidence (89.9°)
        grazing_angle = 89.9 * np.pi / 180
        source_grazing = solver.add_source(theta=grazing_angle, phi=0, pte=1.0, ptm=0.0)

        single_data = self._extract_single_polarization_data(solver, source_grazing)
        batched_data = self._extract_batched_polarization_data(solver, [source_grazing])

        for key in ["ate", "atm", "pol_vec", "esrc"]:
            single_val = single_data[key]
            batched_val = batched_data[key][0]

            assert torch.allclose(single_val, batched_val, atol=1e-5), (
                f"{key} mismatch for grazing incidence"
            )

            # Check for NaN/Inf
            assert torch.all(torch.isfinite(single_val)), f"NaN/Inf in single {key}"
            assert torch.all(torch.isfinite(batched_val)), f"NaN/Inf in batched {key}"

        # Assertions above ensure numerical stability at edge angles.

    def test_polarization_combinations(self, setup_solver):
        """Test various polarization combinations including circular polarization."""
        solver = setup_solver

        # Test cases: (pte, ptm, description)
        polarization_cases = [
            (1.0, 0.0, "pure TE"),
            (0.0, 1.0, "pure TM"),
            (0.7071, 0.7071, "45° linear"),
            (1.0, 1.0j, "right circular"),  # 90° phase shift
            (1.0, -1.0j, "left circular"),  # -90° phase shift
            (0.8, 0.6j, "elliptical"),
        ]

        angle = 30 * np.pi / 180  # 30° incidence

        for pte, ptm, description in polarization_cases:
            source = solver.add_source(theta=angle, phi=0, pte=pte, ptm=ptm)

            single_data = self._extract_single_polarization_data(solver, source)
            batched_data = self._extract_batched_polarization_data(solver, [source])

            for key in ["ate", "atm", "pol_vec", "esrc"]:
                single_val = single_data[key]
                batched_val = batched_data[key][0]

                max_diff = torch.abs(single_val - batched_val).max().item()
                assert torch.allclose(single_val, batched_val, atol=1e-6, rtol=1e-5), (
                    f"{key} mismatch for {description}: max diff = {max_diff}"
                )

            # Ensures various polarization states are handled consistently.
    # Removed tests that were self-referential or redundant:
    # - Multi-frequency equality: already covered implicitly since solver has >1 wavelengths
    # - Gradient preservation: current implementation does not preserve autograd for pte/ptm
    # - Delta indexing: unified path uses a single implementation; no distinct behavior to compare

    def test_multiple_sources_sequential_equivalence(self, setup_solver):
        """Verify batched processing matches sequential processing."""
        solver = setup_solver

        # Create multiple sources with different parameters
        sources = [
            solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0),  # Normal
            solver.add_source(
                theta=30 * np.pi / 180, phi=0, pte=1.0, ptm=0.0
            ),  # Oblique TE
            solver.add_source(
                theta=45 * np.pi / 180, phi=45 * np.pi / 180, pte=0.0, ptm=1.0
            ),  # Oblique TM
            solver.add_source(
                theta=60 * np.pi / 180, phi=0, pte=0.7071, ptm=0.7071
            ),  # Mixed
        ]

        # Process sequentially using single-source logic
        sequential_data = []
        for source in sources:
            data = self._extract_single_polarization_data(solver, source)
            sequential_data.append(data)

        # Process in batch
        batched_data = self._extract_batched_polarization_data(solver, sources)

        # Compare each source
        for i, (seq_data, source) in enumerate(zip(sequential_data, sources)):
            for key in ["ate", "atm", "pol_vec", "esrc"]:
                seq_val = seq_data[key]
                batch_val = batched_data[key][i]

                max_diff = torch.abs(seq_val - batch_val).max().item()
                assert torch.allclose(seq_val, batch_val, atol=1e-6, rtol=1e-5), (
                    f"Source {i} {key} mismatch: max diff = {max_diff}"
                )

            # Ensures batched path matches sequential evaluation per-source.

    def test_gradients_wrt_pte_ptm(self, setup_solver):
        """Gradients should flow from esrc back to complex pte/ptm amplitudes."""
        solver = setup_solver

        # Single source with complex amplitudes requiring grad
        pte = torch.tensor(0.7 + 0.2j, dtype=solver.tcomplex, requires_grad=True)
        ptm = torch.tensor(0.3 - 0.5j, dtype=solver.tcomplex, requires_grad=True)
        source = solver.add_source(theta=0.5, phi=0.25, pte=pte, ptm=ptm)

        single_data = self._extract_single_polarization_data(solver, source)
        esrc = single_data["esrc"]

        loss = (esrc.abs() ** 2).sum()
        loss.backward()

        assert pte.grad is not None and torch.isfinite(pte.grad.real) and torch.isfinite(pte.grad.imag)
        assert ptm.grad is not None and torch.isfinite(ptm.grad.real) and torch.isfinite(ptm.grad.imag)

        # Batched case with two sources
        pte_b = [
            torch.tensor(1.0 + 0.0j, dtype=solver.tcomplex, requires_grad=True),
            torch.tensor(0.6 - 0.4j, dtype=solver.tcomplex, requires_grad=True),
        ]
        ptm_b = [
            torch.tensor(0.2 + 0.8j, dtype=solver.tcomplex, requires_grad=True),
            torch.tensor(0.4 + 0.1j, dtype=solver.tcomplex, requires_grad=True),
        ]
        sources = [
            solver.add_source(theta=0.4, phi=0.1, pte=pte_b[0], ptm=ptm_b[0]),
            solver.add_source(theta=0.9, phi=0.6, pte=pte_b[1], ptm=ptm_b[1]),
        ]

        batched_data = self._extract_batched_polarization_data(solver, sources)
        loss_b = (batched_data["esrc"].abs() ** 2).sum()
        loss_b.backward()

        for v in pte_b + ptm_b:
            assert v.grad is not None
            assert torch.isfinite(v.grad.real) and torch.isfinite(v.grad.imag)

    def test_gradients_wrt_angles_theta_phi(self, setup_solver):
        """Gradients should flow from esrc back to theta/phi via kinc and polarization."""
        solver = setup_solver

        # Use oblique angles to avoid normal-incidence special-case branch
        theta = torch.tensor(0.7, dtype=solver.tfloat, requires_grad=True)
        phi = torch.tensor(0.3, dtype=solver.tfloat, requires_grad=True)
        source = solver.add_source(theta=theta, phi=phi, pte=1.0, ptm=0.5j)

        # Single-source path
        single_data = self._extract_single_polarization_data(solver, source)
        loss = (single_data["esrc"].abs() ** 2).sum()
        loss.backward()

        assert theta.grad is not None and torch.isfinite(theta.grad)
        assert phi.grad is not None and torch.isfinite(phi.grad)

        # Batched path with two independent angle variables
        theta_b = [
            torch.tensor(0.5, dtype=solver.tfloat, requires_grad=True),
            torch.tensor(1.0, dtype=solver.tfloat, requires_grad=True),
        ]
        phi_b = [
            torch.tensor(0.2, dtype=solver.tfloat, requires_grad=True),
            torch.tensor(0.8, dtype=solver.tfloat, requires_grad=True),
        ]
        sources = [
            solver.add_source(theta=theta_b[0], phi=phi_b[0], pte=0.8, ptm=0.2),
            solver.add_source(theta=theta_b[1], phi=phi_b[1], pte=0.6, ptm=0.4j),
        ]

        batched_data = self._extract_batched_polarization_data(solver, sources)
        loss_b = (batched_data["esrc"].abs() ** 2).sum()
        loss_b.backward()

        for v in theta_b + phi_b:
            assert v.grad is not None and torch.isfinite(v.grad)
    # Removed self-comparison of unified method vs itself; provides no signal.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
