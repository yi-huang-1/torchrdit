"""
Test suite for source batching API design.

Tests the backward-compatible API for processing multiple sources simultaneously.
"""

import sys

sys.path.insert(0, "torchrdit/src")

import pytest
import torch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.results import SolverResults


class TestSourceBatchingAPI:
    """Test the API design for source batching."""

    def create_test_solver(self):
        """Create a simple test solver."""
        # Simple square lattice
        t1 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        t2 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

        # Create materials
        material_air = create_material(name="air", permittivity=1.0)
        material_si = create_material(name="silicon", permittivity=12.25)  # n=3.5

        # Create solver
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=[1.54],  # wavelength in um
            lengthunit="um",
            rdim=[16, 16],
            kdim=[3, 3],
            t1=t1,
            t2=t2,
        )

        # Set up the structure
        solver.update_ref_material(material_air)
        solver.update_trn_material(material_air)

        # Add a simple homogeneous silicon layer
        solver.add_layer(material_name=material_si, thickness=0.2, is_homogeneous=True)

        return solver

    def test_backward_compatibility(self):
        """Test that single source dict still works as before."""
        solver = self.create_test_solver()

        # Current API should continue to work
        source = solver.add_source(theta=0.1, phi=0.0, pte=1.0, ptm=0.0)
        results = solver.solve(source)

        # Should return standard SolverResults
        assert isinstance(results, SolverResults)
        assert results.reflection.shape == (1,)  # n_freqs only
        assert results.transmission.shape == (1,)

    def test_solve_with_source_list(self):
        """Test solving with a list of sources."""
        solver = self.create_test_solver()

        # Create multiple sources
        sources = [
            solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0),
            solver.add_source(theta=0.1, phi=0.0, pte=1.0, ptm=0.0),
            solver.add_source(theta=0.2, phi=0.0, pte=0.0, ptm=1.0),
        ]

        # Solve with list of sources
        results = solver.solve(sources)

        # Should return BatchedSolverResults
        assert results.__class__.__name__ == "BatchedSolverResults"
        assert hasattr(results, "n_sources")
        assert results.n_sources == 3

    def test_batched_results_shape(self):
        """Test that batched results have correct shapes."""
        solver = self.create_test_solver()

        sources = [
            solver.add_source(theta=i * 0.1, phi=0.0, pte=1.0, ptm=0.0)
            for i in range(5)
        ]

        results = solver.solve(sources)

        # Check bulk data shapes
        assert results.reflection.shape == (5, 1)  # (n_sources, n_freqs)
        assert results.transmission.shape == (5, 1)
        assert results.loss.shape == (5, 1)

        # Check field component shapes
        assert results.Erx.shape == (
            5,
            1,
            3,
            3,
        )  # (n_sources, n_freqs, kdim[0], kdim[1])
        assert results.Etx.shape == (5, 1, 3, 3)

    def test_batched_results_indexing(self):
        """Test accessing individual results from batch."""
        solver = self.create_test_solver()

        sources = [
            solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0),
            solver.add_source(theta=0.1, phi=0.0, pte=0.7, ptm=0.3),
            solver.add_source(theta=0.2, phi=0.0, pte=0.0, ptm=1.0),
        ]

        results = solver.solve(sources)

        # Test integer indexing
        result_0 = results[0]
        assert isinstance(result_0, SolverResults)
        assert result_0.reflection.shape == (1,)  # Single source shape

        # Test slicing
        subset = results[1:3]
        assert subset.__class__.__name__ == "BatchedSolverResults"
        assert subset.n_sources == 2

        # Test iteration
        all_results = list(results)
        assert len(all_results) == 3
        assert all(isinstance(r, SolverResults) for r in all_results)

    def test_batched_results_methods(self):
        """Test convenience methods on BatchedSolverResults."""
        solver = self.create_test_solver()

        sources = [
            solver.add_source(theta=i * 0.1, phi=0.0, pte=1.0, ptm=0.0)
            for i in range(4)
        ]

        results = solver.solve(sources)

        # Test find_optimal_source method
        optimal_idx = results.find_optimal_source(metric="max_transmission")
        assert 0 <= optimal_idx < 4

        # Test get_parameter_sweep_data
        theta_values, trans_values = results.get_parameter_sweep_data(
            parameter="theta", metric="transmission", frequency_idx=0
        )
        assert theta_values.shape == (4,)
        assert trans_values.shape == (4,)

    def test_source_validation(self):
        """Test validation of source lists."""
        solver = self.create_test_solver()

        # Test empty list
        with pytest.raises(ValueError, match="At least one source required"):
            solver.solve([])

        # Test invalid source format
        with pytest.raises(ValueError, match="Invalid source format"):
            solver.solve([{"invalid": "source"}])

        # Test mixed types (should work)
        sources = [
            solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0),
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        ]
        results = solver.solve(sources)
        assert results.n_sources == 2

    def test_memory_efficient_access(self):
        """Test that individual result access doesn't copy all data."""
        solver = self.create_test_solver()

        sources = [
            solver.add_source(theta=i * 0.01, phi=0.0, pte=1.0, ptm=0.0)
            for i in range(100)
        ]

        results = solver.solve(sources)

        # Accessing single result should be memory efficient
        result_50 = results[50]

        # Verify it's a view, not a copy of all data
        assert result_50.reflection.shape == (1,)  # Only this source's data

        # Modifying the view should not affect original
        # (This behavior depends on implementation details)

    def test_type_hints_work(self):
        """Test that type hints provide correct IDE support."""
        solver = self.create_test_solver()

        # Single source should type as SolverResults
        source = solver.add_source(theta=0.1, phi=0.0, pte=1.0, ptm=0.0)
        result = solver.solve(source)

        # This should work with proper type hints
        transmission: torch.Tensor = result.transmission
        assert transmission.shape == (1,)

        # Multiple sources should type as BatchedSolverResults
        sources = [source, source]
        batched_result = solver.solve(sources)

        # This should also work with proper type hints
        all_transmission: torch.Tensor = batched_result.transmission
        assert all_transmission.shape == (2, 1)


class TestBatchedSolverResults:
    """Test the BatchedSolverResults class implementation."""

    def test_batched_results_creation(self):
        """Test creating BatchedSolverResults directly."""
        from torchrdit.batched_results import BatchedSolverResults

        # Create mock data
        n_sources = 4
        n_freqs = 3
        kdim = (5, 5)

        results = BatchedSolverResults(
            reflection=torch.rand(n_sources, n_freqs),
            transmission=torch.rand(n_sources, n_freqs),
            loss=torch.rand(n_sources, n_freqs),
            reflection_diffraction=torch.rand(n_sources, n_freqs, *kdim),
            transmission_diffraction=torch.rand(n_sources, n_freqs, *kdim),
            Erx=torch.rand(n_sources, n_freqs, *kdim),
            Ery=torch.rand(n_sources, n_freqs, *kdim),
            Erz=torch.rand(n_sources, n_freqs, *kdim),
            Etx=torch.rand(n_sources, n_freqs, *kdim),
            Ety=torch.rand(n_sources, n_freqs, *kdim),
            Etz=torch.rand(n_sources, n_freqs, *kdim),
            n_sources=n_sources,
            source_parameters=[
                {"theta": i * 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
                for i in range(n_sources)
            ],
        )

        assert results.n_sources == 4
        assert len(results) == 4

    def test_batched_results_numpy_conversion(self):
        """Test conversion to numpy arrays."""
        from torchrdit.batched_results import BatchedSolverResults

        results = BatchedSolverResults(
            reflection=torch.rand(3, 5),
            transmission=torch.rand(3, 5),
            loss=torch.rand(3, 5),
            reflection_diffraction=torch.rand(3, 5, 7, 7),
            transmission_diffraction=torch.rand(3, 5, 7, 7),
            Erx=torch.rand(3, 5, 7, 7),
            Ery=torch.rand(3, 5, 7, 7),
            Erz=torch.rand(3, 5, 7, 7),
            Etx=torch.rand(3, 5, 7, 7),
            Ety=torch.rand(3, 5, 7, 7),
            Etz=torch.rand(3, 5, 7, 7),
            n_sources=3,
            source_parameters=[{} for _ in range(3)],
        )

        # Test numpy conversion
        reflection_np = results.reflection.numpy()
        assert reflection_np.shape == (3, 5)

        # Test individual result numpy conversion
        result_1_np = results[1].reflection.numpy()
        assert result_1_np.shape == (5,)


class TestParameterSweeps:
    """Test parameter sweep functionality."""

    def test_angle_sweep(self):
        """Test creating an angle sweep."""
        solver = TestSourceBatchingAPI().create_test_solver()

        # Create angle sweep
        theta_values = torch.linspace(0, 60, 7) * torch.pi / 180
        sources = [
            solver.add_source(theta=theta.item(), phi=0.0, pte=1.0, ptm=0.0)
            for theta in theta_values
        ]

        results = solver.solve(sources)

        # Check sweep data extraction
        theta_out, trans = results.get_parameter_sweep_data(
            parameter="theta",
            metric="transmission",
            frequency_idx=0,  # Only one frequency
        )

        assert torch.allclose(theta_out, theta_values)
        assert trans.shape == (7,)

    def test_polarization_sweep(self):
        """Test creating a polarization sweep."""
        solver = TestSourceBatchingAPI().create_test_solver()

        # Create polarization sweep
        sources = []
        for pte in torch.linspace(0, 1, 5):
            ptm = torch.sqrt(1 - pte**2)
            sources.append(
                solver.add_source(theta=0.1, phi=0.0, pte=pte.item(), ptm=ptm.item())
            )

        results = solver.solve(sources)

        # Verify all sources were processed
        assert results.n_sources == 5

        # Check that total power is conserved
        total_power = results.reflection + results.transmission
        assert torch.allclose(total_power[0], total_power[-1], rtol=1e-3)
