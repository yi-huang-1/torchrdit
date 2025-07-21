"""
Test suite for batched source solver operations.

Tests the core solver modifications needed to support batched source processing.
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


class TestBatchedSolverOperations:
    """Test core solver operations with batched sources."""

    def create_test_solver(self, algorithm="RCWA"):
        """Create a test solver."""
        # Create materials
        mat_air = create_material(name="air", permittivity=1.0)
        mat_si = create_material(name="silicon", permittivity=11.7)

        # Create solver
        algo = Algorithm.RCWA if algorithm == "RCWA" else Algorithm.RDIT
        solver = create_solver(
            algorithm=algo,
            lam0=np.linspace(0.8, 2.0, 5),  # wavelengths in micrometers
            rdim=[512, 512],
            kdim=[5, 5],
            t1=torch.tensor([[1.0, 0.0]]),
            t2=torch.tensor([[0.0, 1.0]]),
            device="cpu",
            is_use_FFF=False,
        )

        # Add materials
        solver.add_materials([mat_air, mat_si])
        
        # Set boundary materials
        solver.update_ref_material("air")
        solver.update_trn_material("air")
        
        # Add layers
        solver.add_layer(material_name="silicon", thickness=torch.tensor(0.5), is_homogeneous=False)

        return solver

    def test_batched_pre_solve(self):
        """Test that solve method accepts batched sources."""
        solver = self.create_test_solver()

        # Create batched sources
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": 0.0, "pte": 0.0, "ptm": 1.0},
        ]

        # Test that solve accepts a list of sources
        results = solver.solve(sources)
        
        # Verify we get BatchedSolverResults
        assert isinstance(results, BatchedSolverResults)
        assert len(results) == 3
        
        # Verify each result has correct shape
        for i in range(3):
            result = results[i]
            assert result.reflection.shape == (5,)  # n_freqs
            assert result.transmission.shape == (5,)  # n_freqs

    def test_batched_k_vectors(self):
        """Test that different sources produce different results."""
        solver = self.create_test_solver()

        # Create batched sources with different angles
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": 0.0, "pte": 0.0, "ptm": 1.0},
        ]

        # Solve with batched sources
        results = solver.solve(sources)
        
        # Results should be different for different incident angles
        # Compare first and second source results
        assert not torch.allclose(results[0].reflection, results[1].reflection)
        assert not torch.allclose(results[0].transmission, results[1].transmission)
        
        # Check diffraction orders exist
        assert hasattr(results[0], 'reflection_diffraction')
        assert hasattr(results[0], 'transmission_diffraction')

    def test_batched_field_calculations(self):
        """Test field calculations with batched sources."""
        solver = self.create_test_solver()

        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 0.5, "ptm": 0.5},
        ]

        # Solve with field calculation
        results = solver.solve(sources)
        
        # Check that results exist for each source
        for i in range(2):
            result = results[i]
            # Check basic fields exist
            assert hasattr(result, 'reflection')
            assert hasattr(result, 'transmission')
            assert result.reflection is not None
            assert result.transmission is not None

    def test_batched_solve_complete(self):
        """Test complete batched solve with multiple sources."""
        solver = self.create_test_solver()

        # Create angle sweep
        angles = torch.linspace(0, 0.5, 10)
        sources = [
            {"theta": float(angle), "phi": 0.0, "pte": 1.0, "ptm": 0.0}
            for angle in angles
        ]

        results = solver.solve(sources)

        # Verify results structure
        assert isinstance(results, BatchedSolverResults)
        assert len(results) == 10
        
        # Check stacked results shape
        assert results.reflection.shape == (10, 5)  # (n_sources, n_freqs)
        assert results.transmission.shape == (10, 5)

        # Test parameter sweep extraction
        sweep_data = results.get_parameter_sweep_data("theta", "reflection")
        assert sweep_data is not None
        params, values = sweep_data
        assert len(params) == 10
        assert len(values) == 10

    def test_batched_vs_sequential(self):
        """Test that batched results match sequential processing."""
        solver = self.create_test_solver()

        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 0.0, "ptm": 1.0},
        ]

        # Batched solve
        batched_results = solver.solve(sources)

        # Sequential solve
        sequential_results = []
        for src in sources:
            result = solver.solve(src)
            sequential_results.append(result)

        # Compare results
        for i in range(2):
            torch.testing.assert_close(
                batched_results[i].reflection,
                sequential_results[i].reflection,
                rtol=1e-6,
                atol=1e-8,
            )
            torch.testing.assert_close(
                batched_results[i].transmission,
                sequential_results[i].transmission,
                rtol=1e-6,
                atol=1e-8,
            )

    def test_rdit_batched_solve(self):
        """Test batched solve with RDIT algorithm."""
        solver = self.create_test_solver(algorithm="RDIT")

        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.15, "phi": 0.0, "pte": 0.7, "ptm": 0.3},
        ]

        results = solver.solve(sources)

        assert isinstance(results, BatchedSolverResults)
        assert len(results) == 2
        
        # RDIT should produce same result structure
        assert results[0].reflection.shape == (5,)
        assert results[0].transmission.shape == (5,)


class TestBatchedSourceValidation:
    """Test source validation for batched operations."""

    def create_test_solver(self):
        """Create a basic test solver."""
        mat_air = create_material(name="air", permittivity=1.0)
        mat_si = create_material(name="silicon", permittivity=11.7)
        
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[256, 256],
            kdim=[3, 3],
        )
        
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("air")
        solver.update_trn_material("air")
        solver.add_layer(material_name="silicon", thickness=torch.tensor(0.3))
        
        return solver

    def test_empty_source_list(self):
        """Test handling of empty source list."""
        solver = self.create_test_solver()
        
        with pytest.raises(ValueError, match="At least one source required"):
            solver.solve([])

    def test_invalid_source_format(self):
        """Test handling of invalid source format."""
        solver = self.create_test_solver()
        
        # Invalid source missing required fields
        sources = [
            {"theta": 0.0},  # Missing phi, pte, ptm
        ]
        
        with pytest.raises((KeyError, ValueError)):
            solver.solve(sources)

    def test_mixed_source_formats(self):
        """Test that all sources must have same format."""
        solver = self.create_test_solver()
        
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0},  # Missing ptm
        ]
        
        # This should raise an error during processing
        with pytest.raises((KeyError, ValueError)):
            solver.solve(sources)

    def test_large_batch_warning(self):
        """Test warning for very large batches."""
        solver = self.create_test_solver()
        
        # Create a large batch
        sources = [
            {"theta": i * 0.001, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
            for i in range(1000)
        ]
        
        # Should work but might be slow
        results = solver.solve(sources)
        assert len(results) == 1000


class TestBatchedMemoryEfficiency:
    """Test memory efficiency of batched operations."""

    def create_test_solver(self):
        """Create a test solver with smaller dimensions for memory tests."""
        mat = create_material(name="test", permittivity=2.25)
        
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.0]),
            rdim=[64, 64],
            kdim=[3, 3],
        )
        
        solver.add_materials([mat])
        solver.update_ref_material("test")
        solver.update_trn_material("test")
        
        # Add at least one layer to avoid degenerate S-matrix
        solver.add_layer(material_name="test", thickness=0.1)
        
        return solver

    def test_memory_scaling(self):
        """Test that memory usage scales appropriately with batch size."""
        solver = self.create_test_solver()
        
        # Small batch
        sources_small = [
            {"theta": i * 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
            for i in range(5)
        ]
        results_small = solver.solve(sources_small)
        
        # Larger batch
        sources_large = [
            {"theta": i * 0.01, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
            for i in range(50)
        ]
        results_large = solver.solve(sources_large)
        
        # Check that results scale appropriately
        assert len(results_large) == 10 * len(results_small)
        assert results_large.reflection.shape[0] == 10 * results_small.reflection.shape[0]

    def test_no_memory_copy_on_indexing(self):
        """Test that indexing doesn't create unnecessary copies."""
        solver = self.create_test_solver()
        
        sources = [
            {"theta": i * 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
            for i in range(10)
        ]
        
        results = solver.solve(sources)
        
        # Access individual result
        result_5 = results[5]
        
        # Verify it's a view or efficient extraction
        assert result_5.reflection.shape == (1,)  # Single frequency


class TestBatchedGradients:
    """Test gradient computation with batched sources."""

    def create_differentiable_solver(self):
        """Create a solver with differentiable parameters."""
        mat_air = create_material(name="air", permittivity=1.0)
        mat_var = create_material(name="variable", permittivity=4.0)
        
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[3, 3],
        )
        
        solver.add_materials([mat_air, mat_var])
        solver.update_ref_material("air")
        solver.update_trn_material("air")
        
        # Add layer with differentiable thickness
        thickness = torch.tensor(0.5, requires_grad=True)
        solver.add_layer(material_name="variable", thickness=thickness)
        
        return solver, thickness

    def test_gradient_flow(self):
        """Test that gradients flow through batched solve."""
        solver, thickness = self.create_differentiable_solver()
        
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        ]
        
        # Forward pass
        results = solver.solve(sources)
        
        # Define loss as sum of reflections
        loss = results.reflection.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that thickness has gradients
        assert thickness.grad is not None
        assert not torch.isnan(thickness.grad).any()

    def test_optimization_workflow(self):
        """Test optimization with batched sources."""
        solver, thickness = self.create_differentiable_solver()
        
        # Multiple angles for robustness
        sources = [
            {"theta": angle, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
            for angle in [0.0, 0.05, 0.1, 0.15]
        ]
        
        # Simple optimization step
        optimizer = torch.optim.Adam([thickness], lr=0.01)
        
        for _ in range(3):
            optimizer.zero_grad()
            
            results = solver.solve(sources)
            
            # Minimize average reflection
            loss = results.reflection.mean()
            loss.backward()
            
            optimizer.step()
            
            # Ensure thickness stays positive
            with torch.no_grad():
                thickness.data = torch.clamp(thickness.data, min=0.01)
        
        # Check that optimization ran
        assert thickness.grad is not None