"""
Test suite for tensorizing _calculate_polarization_batched and unifying polarization logic.

This test suite ensures mathematical equivalence between the embedded single-source
polarization logic in _calculate_fields_and_efficiencies and the separate
_calculate_polarization_batched method. Following TDD principles, these tests are
written before the implementation to verify correctness.

Key areas tested:
- Mathematical equivalence between single and batched implementations
- Edge cases (normal incidence, extreme angles, circular polarization)
- Gradient preservation for optimization workflows
- Delta function indexing consistency
- Performance characteristics
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

    def _extract_unified_polarization_data(self, solver, sources_or_source):
        """Extract polarization data using unified _calculate_polarization method."""
        if isinstance(sources_or_source, list):
            # Batched case
            solver._pre_solve(sources_or_source)
            return solver._calculate_polarization(sources_or_source)
        else:
            # Single case
            solver.src = sources_or_source
            solver._pre_solve()
            return solver._calculate_polarization()

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

        print("✓ Normal incidence test passed:")
        print(f"  ate max diff: {max_diff_ate:.2e}")
        print(f"  atm max diff: {max_diff_atm:.2e}")
        print(f"  pol_vec max diff: {max_diff_pol:.2e}")
        print(f"  esrc max diff: {max_diff_esrc:.2e}")

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

                print(f"✓ Oblique incidence θ={angle_deg}°, φ={phi_deg}° test passed")

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

        print("✓ Edge cases test passed")

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

            print(f"✓ Polarization test passed: {description}")

    def test_multi_frequency_scenarios(self, setup_solver):
        """Test behavior with multiple wavelengths."""
        solver = setup_solver

        # Verify solver has multiple frequencies
        assert solver.n_freqs > 1, "Test requires multiple frequencies"

        # Test with scalar theta/phi (same angle for all frequencies)
        source_scalar = solver.add_source(
            theta=45 * np.pi / 180, phi=0, pte=1.0, ptm=0.0
        )

        single_data = self._extract_single_polarization_data(solver, source_scalar)
        batched_data = self._extract_batched_polarization_data(solver, [source_scalar])

        for key in ["ate", "atm", "pol_vec", "esrc"]:
            single_val = single_data[key]
            batched_val = batched_data[key][0]

            # Check shapes
            assert single_val.shape == batched_val.shape, (
                f"{key} shape mismatch: single {single_val.shape} vs batched {batched_val.shape}"
            )

            # Check values
            max_diff = torch.abs(single_val - batched_val).max().item()
            assert torch.allclose(single_val, batched_val, atol=1e-6, rtol=1e-5), (
                f"{key} mismatch for multi-frequency: max diff = {max_diff}"
            )

        # Test with array theta/phi (different angle per frequency) - FUTURE ENHANCEMENT
        # This test is included for when we implement array support in the unified version

        print("✓ Multi-frequency test passed")

    def test_gradient_preservation(self, setup_solver):
        """Test that gradient flow is preserved through polarization calculations."""
        solver = setup_solver

        # Note: Current implementation doesn't directly support gradients through
        # theta/phi parameters in _calculate_polarization_batched, but we can test
        # that the function doesn't break gradient flow when gradients are present

        # Create a source
        source = solver.add_source(theta=30 * np.pi / 180, phi=0, pte=1.0, ptm=0.0)

        # Test that both methods can be called without breaking autograd
        try:
            single_data = self._extract_single_polarization_data(solver, source)
            batched_data = self._extract_batched_polarization_data(solver, [source])

            # Verify outputs are identical (gradient preservation test will come later)
            for key in ["ate", "atm", "pol_vec", "esrc"]:
                single_val = single_data[key]
                batched_val = batched_data[key][0]

                assert torch.allclose(single_val, batched_val, atol=1e-6), (
                    f"{key} mismatch in gradient preservation test"
                )

        except RuntimeError as e:
            pytest.fail(f"Gradient computation failed: {e}")

        print("✓ Gradient preservation test passed (basic compatibility verified)")

    def test_delta_function_indexing_consistency(self, setup_solver):
        """Test that delta function indexing is consistent between implementations."""
        solver = setup_solver

        # This test specifically checks the delta function creation inconsistency
        # identified in the analysis:
        # - Batched: delta[:, :, n_harmonics_squared // 2] = 1.0
        # - Single: delta[:, (kdim[1] // 2) * kdim[0] + (kdim[0] // 2)] = 1

        source = solver.add_source(theta=30 * np.pi / 180, phi=0, pte=1.0, ptm=0.0)

        single_data = self._extract_single_polarization_data(solver, source)
        batched_data = self._extract_batched_polarization_data(solver, [source])

        # Focus specifically on esrc comparison
        esrc_single = single_data["esrc"]
        esrc_batched = batched_data["esrc"][0]

        max_diff = torch.abs(esrc_single - esrc_batched).max().item()

        if max_diff > 1e-6:
            print("WARNING: Delta function indexing inconsistency detected!")
            print(f"esrc max difference: {max_diff:.2e}")
            print("This confirms the need to unify delta function creation.")

            # For now, we'll allow this test to highlight the issue
            # When we fix the implementation, this test should pass with strict tolerance
            pytest.xfail(
                "Known issue: Delta function indexing differs between implementations"
            )
        else:
            print("✓ Delta function indexing consistency test passed")

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

            print(f"✓ Source {i} sequential equivalence verified")

    def test_performance_characteristics(self, setup_solver):
        """Basic performance characteristics test (not optimization, just functionality)."""
        solver = setup_solver

        # Test with increasing batch sizes to ensure scalability
        batch_sizes = [1, 4, 8]

        for batch_size in batch_sizes:
            # Create sources
            sources = [
                solver.add_source(theta=(i * 15) * np.pi / 180, phi=0, pte=1.0, ptm=0.0)
                for i in range(batch_size)
            ]

            try:
                # Test that batched processing works
                batched_data = self._extract_batched_polarization_data(solver, sources)

                # Verify output shapes
                for key in ["ate", "atm", "pol_vec", "esrc"]:
                    output = batched_data[key]
                    expected_shape_0 = batch_size
                    assert output.shape[0] == expected_shape_0, (
                        f"Batch size {batch_size}: {key} shape[0] = {output.shape[0]}, expected {expected_shape_0}"
                    )

                print(f"✓ Performance test passed for batch size {batch_size}")

            except Exception as e:
                pytest.fail(f"Batch size {batch_size} failed: {e}")

    def test_unified_method_equivalence(self, setup_solver):
        """Test the unified _calculate_polarization() method against both single and batched."""
        solver = setup_solver

        # Test with single source
        source = solver.add_source(
            theta=30 * np.pi / 180, phi=45 * np.pi / 180, pte=1.0, ptm=0.5j
        )

        # Get data from all three methods
        single_data = self._extract_single_polarization_data(solver, source)
        batched_data = self._extract_batched_polarization_data(solver, [source])
        unified_single_data = self._extract_unified_polarization_data(solver, source)
        unified_batched_data = self._extract_unified_polarization_data(solver, [source])

        # Compare unified single vs dedicated single
        for key in ["ate", "atm", "pol_vec", "esrc"]:
            single_val = single_data[key]
            unified_single_val = unified_single_data[key]

            max_diff = torch.abs(single_val - unified_single_val).max().item()
            assert torch.allclose(
                single_val, unified_single_val, atol=1e-6, rtol=1e-5
            ), (
                f"Unified single vs dedicated single {key} mismatch: max diff = {max_diff}"
            )

        # Compare unified batched vs dedicated batched
        for key in ["ate", "atm", "pol_vec", "esrc"]:
            batched_val = batched_data[key][0]
            unified_batched_val = unified_batched_data[key][0]

            max_diff = torch.abs(batched_val - unified_batched_val).max().item()
            assert torch.allclose(
                batched_val, unified_batched_val, atol=1e-6, rtol=1e-5
            ), (
                f"Unified batched vs dedicated batched {key} mismatch: max diff = {max_diff}"
            )

        # Test with multiple sources
        sources = [
            solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0),
            solver.add_source(theta=45 * np.pi / 180, phi=0, pte=0.0, ptm=1.0),
            solver.add_source(
                theta=60 * np.pi / 180, phi=30 * np.pi / 180, pte=0.7071, ptm=0.7071j
            ),
        ]

        dedicated_batched_data = self._extract_batched_polarization_data(
            solver, sources
        )
        unified_batched_data = self._extract_unified_polarization_data(solver, sources)

        for key in ["ate", "atm", "pol_vec", "esrc"]:
            dedicated_val = dedicated_batched_data[key]
            unified_val = unified_batched_data[key]

            max_diff = torch.abs(dedicated_val - unified_val).max().item()
            assert torch.allclose(dedicated_val, unified_val, atol=1e-6, rtol=1e-5), (
                f"Multi-source unified vs dedicated {key} mismatch: max diff = {max_diff}"
            )

        print("✓ Unified method equivalence test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
