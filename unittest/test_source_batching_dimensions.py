"""
Test suite for documenting and verifying dimension flow in source batching implementation.

This test file serves two purposes:
1. Document the current (single source) dimension flow
2. Define expected behavior for batched sources
"""

import sys

sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm
from torchrdit.batched_results import BatchedSolverResults


class TestCurrentDimensionFlow:
    """Document and verify current single-source dimension flow."""

    def setup_solver(self, n_freqs=3, kdim=(3, 3)):
        """Create a basic solver for testing."""
        # Create materials
        mat_air = create_material(name="air", permittivity=1.0)
        mat_si = create_material(name="silicon", permittivity=11.7)

        # Create solver
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.linspace(0.8, 2.0, n_freqs),
            rdim=[512, 512],
            kdim=list(kdim),
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.0, 1.0]]),
            device="cpu",
            is_use_FFF=False,
        )

        # Add materials and layers
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("air")
        solver.update_trn_material("air")
        solver.add_layer(material_name="silicon", thickness=torch.tensor(0.5), is_homogeneous=False)

        return solver

    def test_pre_solve_dimensions(self):
        """Test dimensions with single source."""
        solver = self.setup_solver(n_freqs=3)
        
        # Single source
        source = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        result = solver.solve(source)
        
        # Document current dimensions
        assert result.reflection.shape == (3,), "reflection should be (n_freqs,)"
        assert result.transmission.shape == (3,), "transmission should be (n_freqs,)"

    def test_k_vector_dimensions(self):
        """Test dimensions in results with single source."""
        solver = self.setup_solver(n_freqs=3, kdim=(5, 5))
        
        source = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        result = solver.solve(source)

        # Document diffraction order dimensions
        assert result.reflection_diffraction.shape == (3, 5, 5), (
            "reflection_diffraction should be (n_freqs, kdim[0], kdim[1])"
        )
        assert result.transmission_diffraction.shape == (3, 5, 5), (
            "transmission_diffraction should be (n_freqs, kdim[0], kdim[1])"
        )

    def test_polarization_dimensions(self):
        """Test dimensions with different polarizations."""
        solver = self.setup_solver(n_freqs=3)
        
        # TE polarization
        source_te = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        result_te = solver.solve(source_te)
        
        # TM polarization
        source_tm = {"theta": 0.1, "phi": 0.0, "pte": 0.0, "ptm": 1.0}
        result_tm = solver.solve(source_tm)
        
        # Results should have same dimensions
        assert result_te.reflection.shape == result_tm.reflection.shape

    def test_result_dimensions(self):
        """Test dimensions of solver results."""
        solver = self.setup_solver(n_freqs=3, kdim=(3, 3))
        source = {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        results = solver.solve(source)

        # Document result dimensions
        assert results.reflection.shape == (3,), "reflection should be (n_freqs,)"
        assert results.transmission.shape == (3,), "transmission should be (n_freqs,)"
        assert hasattr(results, 'reflection_diffraction')
        assert hasattr(results, 'transmission_diffraction')


class TestBatchedSourceDimensions:
    """Test expected dimension flow with batched sources."""

    def create_batched_solver(self, n_freqs=3, kdim=(3, 3)):
        """Create solver for batched testing."""
        # Create materials
        mat_air = create_material(name="air", permittivity=1.0)
        mat_si = create_material(name="silicon", permittivity=11.7)

        # Create solver  
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.linspace(0.8, 2.0, n_freqs),
            rdim=[512, 512],
            kdim=list(kdim),
            device="cpu",
        )

        # Add materials and layers
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("air")
        solver.update_trn_material("air")
        solver.add_layer(material_name="silicon", thickness=torch.tensor(0.5), is_homogeneous=False)

        return solver

    def test_batched_pre_solve_dimensions(self):
        """Test pre_solve dimensions with batched sources."""
        solver = self.create_batched_solver(n_freqs=3)

        # Batched sources
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": 0.0, "pte": 0.0, "ptm": 1.0},
        ]

        results = solver.solve(sources)
        
        # Verify batched results
        assert isinstance(results, BatchedSolverResults)
        assert len(results) == 3
        
        # Check stacked dimensions
        assert results.reflection.shape == (3, 3), "stacked reflection should be (n_sources, n_freqs)"
        assert results.transmission.shape == (3, 3), "stacked transmission should be (n_sources, n_freqs)"

    def test_batched_k_vector_dimensions(self):
        """Test k-vector dimensions with batched sources."""
        solver = self.create_batched_solver(n_freqs=3, kdim=(5, 5))

        sources = [
            {"theta": i * 0.05, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
            for i in range(4)
        ]

        results = solver.solve(sources)
        
        # Check individual result dimensions
        for i in range(4):
            assert results[i].reflection_diffraction.shape == (3, 5, 5)
            assert results[i].transmission_diffraction.shape == (3, 5, 5)

    def test_batched_result_dimensions(self):
        """Test result dimensions with batched sources."""
        solver = self.create_batched_solver(n_freqs=5, kdim=(3, 3))

        sources = [
            {"theta": theta, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
            for theta in [0.0, 0.1, 0.2]
        ]

        results = solver.solve(sources)

        # Test stacked results
        assert results.reflection.shape == (3, 5), "reflection should be (n_sources, n_freqs)"
        assert results.transmission.shape == (3, 5), "transmission should be (n_sources, n_freqs)"
        assert results.reflection_diffraction.shape == (3, 5, 3, 3), (
            "reflection_diffraction should be (n_sources, n_freqs, kdim[0], kdim[1])"
        )
        assert results.transmission_diffraction.shape == (3, 5, 3, 3), (
            "transmission_diffraction should be (n_sources, n_freqs, kdim[0], kdim[1])"
        )

    def test_backward_compatibility(self):
        """Test that single source still works with expected dimensions."""
        solver = self.create_batched_solver(n_freqs=3, kdim=(3, 3))

        # Single source (backward compatibility)
        source = {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
        result = solver.solve(source)

        # Should return SolverResults, not BatchedSolverResults
        assert not isinstance(result, BatchedSolverResults)
        assert result.reflection.shape == (3,), "single source should maintain (n_freqs,) shape"
        assert result.transmission.shape == (3,), "single source should maintain (n_freqs,) shape"

    def test_dimension_consistency_with_batched_sources(self):
        """Test that dimensions are consistent across different batch sizes."""
        solver = self.create_batched_solver(n_freqs=4, kdim=(5, 5))

        # Test different batch sizes
        for batch_size in [1, 5, 10]:
            sources = [
                {"theta": i * 0.01, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
                for i in range(batch_size)
            ]

            if batch_size == 1:
                # Single source list should still return BatchedSolverResults
                results = solver.solve(sources)
                assert isinstance(results, BatchedSolverResults)
                assert len(results) == 1
            else:
                results = solver.solve(sources)
                assert isinstance(results, BatchedSolverResults)
                assert len(results) == batch_size

            # Check dimensions
            assert results.reflection.shape == (batch_size, 4)
            assert results.transmission.shape == (batch_size, 4)
            assert results.reflection_diffraction.shape == (batch_size, 4, 5, 5)
            assert results.transmission_diffraction.shape == (batch_size, 4, 5, 5)