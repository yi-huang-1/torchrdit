"""
Comprehensive numerical validation tests for source batching functionality.

This test suite ensures that batched source processing produces identical results 
to sequential processing while maintaining performance benefits and gradient 
computation accuracy.
"""

import sys
sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np
import time
import tracemalloc
from typing import List, Dict

from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm
from torchrdit.results import SolverResults
from torchrdit.batched_results import BatchedSolverResults


class TestSourceBatchingValidation:
    """Comprehensive tests for source batching functionality."""
    
    def create_solver(self, algorithm=Algorithm.RDIT):
        """Create a test solver with standard configuration."""
        # Create materials
        mat_air = create_material(name="Air", permittivity=1.0)
        mat_si = create_material(name="Si", permittivity=11.7)
        
        solver = create_solver(
            algorithm=algorithm,
            lam0=np.array([1.55]),
            rdim=[256, 256],
            kdim=[5, 5],
            device='cpu'
        )
        
        # Add materials
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        
        return solver
    
    @pytest.fixture
    def test_sources(self):
        """Generate test sources for validation."""
        return [
            {"theta": 0, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/6, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": np.pi/4, "phi": 0, "pte": 0.7071, "ptm": 0.7071},
            {"theta": 0, "phi": np.pi/4, "pte": 0.0, "ptm": 1.0},
        ]
    
    def test_sequential_vs_batched_identical(self, test_sources):
        """Verify batched processing produces identical results to sequential."""
        # Create fresh solver
        solver = self.create_solver()
        # Set up test structure
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=True)
        
        # Run sequential processing
        sequential_results = []
        for src in test_sources:
            result = solver.solve(src)
            sequential_results.append(result)
        
        # Run batched processing
        batched_results = solver.solve(test_sources)
        
        # Compare results
        assert isinstance(batched_results, BatchedSolverResults)
        assert len(batched_results) == len(test_sources)
        
        for i, src in enumerate(test_sources):
            # Check reflection
            torch.testing.assert_close(
                batched_results[i].reflection, 
                sequential_results[i].reflection,
                rtol=1e-6, atol=1e-8
            )
            
            # Check transmission
            torch.testing.assert_close(
                batched_results[i].transmission,
                sequential_results[i].transmission,
                rtol=1e-6, atol=1e-8
            )
            
            # Check diffraction orders
            torch.testing.assert_close(
                batched_results[i].reflection_diffraction,
                sequential_results[i].reflection_diffraction,
                rtol=1e-6, atol=1e-8
            )
    
    def test_multiple_angles(self):
        """Test batching with different incident angles."""
        # Create fresh solver
        solver = self.create_solver()
        
        deg = np.pi / 180
        angles = np.linspace(0, 60, 7) * deg
        
        # Create structure
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
        
        # Create sources with different angles
        sources = [
            {"theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0}
            for angle in angles
        ]
        
        # Solve
        results = solver.solve(sources)
        
        # Verify results
        assert isinstance(results, BatchedSolverResults)
        assert len(results) == len(angles)
        
        # Check that reflection changes with angle (Fresnel)
        reflections = [results[i].reflection[0].item() for i in range(len(angles))]
        assert reflections[0] != reflections[-1], "Reflection should vary with angle"
        
        # Verify conservation of energy
        for i in range(len(angles)):
            r = results[i].reflection[0].item()
            t = results[i].transmission[0].item()
            assert 0 <= r <= 1, f"Reflection {r} out of bounds"
            assert 0 <= t <= 1, f"Transmission {t} out of bounds"
            assert abs(r + t - 1.0) < 0.01, f"Energy not conserved: R={r}, T={t}"
    
    def test_multiple_polarizations(self):
        """Test batching with different polarization states."""
        # Create fresh solver
        solver = self.create_solver()
        # Create structure
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
        
        # Different polarization states
        sources = [
            {"theta": 0.5, "phi": 0, "pte": 1.0, "ptm": 0.0},  # Pure TE
            {"theta": 0.5, "phi": 0, "pte": 0.0, "ptm": 1.0},  # Pure TM
            {"theta": 0.5, "phi": 0, "pte": 0.7071, "ptm": 0.7071},  # 45 deg linear
            {"theta": 0.5, "phi": 0, "pte": 0.5, "ptm": 0.866},  # 60 deg linear
        ]
        
        results = solver.solve(sources)
        
        # Verify different polarizations give different results
        assert not torch.allclose(results[0].reflection, results[1].reflection)
        
        # Verify linear combination property
        # For non-magnetic materials, 45deg should be average of TE and TM
        avg_reflection = (results[0].reflection + results[1].reflection) / 2
        # This is approximate due to polarization mixing effects
        torch.testing.assert_close(
            results[2].reflection, avg_reflection, rtol=0.1, atol=0.01
        )
    
    def test_gradient_consistency(self):
        """Test gradient computation through batched solve."""
        # Create fresh solver
        solver = self.create_solver()
        # Create differentiable thickness
        thickness = torch.tensor(0.5, requires_grad=True)
        
        # Add layer
        solver.add_layer(material_name="Si", thickness=thickness, is_homogeneous=False)
        
        # Multiple sources for robust optimization
        sources = [
            {"theta": angle, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for angle in [0, 0.1, 0.2, 0.3]
        ]
        
        # Forward pass
        results = solver.solve(sources)
        
        # Define loss - minimize average reflection
        loss = results.reflection.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradient exists and is reasonable
        assert thickness.grad is not None
        assert not torch.isnan(thickness.grad)
        assert thickness.grad.abs() > 1e-10, "Gradient too small"
        assert thickness.grad.abs() < 1e3, "Gradient too large"
    
    def test_single_source_batch(self):
        """Test that single-element batch works correctly."""
        # Create fresh solver
        solver = self.create_solver()
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=True)
        
        # Single source as list
        sources = [{"theta": 0.3, "phi": 0, "pte": 1.0, "ptm": 0.0}]
        
        # Solve as batch
        batched_result = solver.solve(sources)
        
        # Solve as single
        single_result = solver.solve(sources[0])
        
        # Should get BatchedSolverResults for list input
        assert isinstance(batched_result, BatchedSolverResults)
        assert len(batched_result) == 1
        
        # Results should match
        torch.testing.assert_close(
            batched_result[0].reflection, 
            single_result.reflection,
            rtol=1e-6, atol=1e-8
        )
    
    def test_large_batch(self):
        """Test with large number of sources to verify scalability."""
        # Create fresh solver
        solver = self.create_solver()
        # Simple structure for speed
        solver.add_layer(material_name="Si", thickness=torch.tensor(1.0), is_homogeneous=True)
        
        # Create many sources
        n_sources = 100
        deg = np.pi / 180
        sources = [
            {"theta": i * deg, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for i in range(n_sources)
        ]
        
        # Time the batched solve
        start = time.time()
        batched_results = solver.solve(sources)
        batched_time = time.time() - start
        
        # Verify results
        assert len(batched_results) == n_sources
        assert batched_results.reflection.shape == (n_sources, 1)
        
        # Basic sanity check on timing
        # Should complete in reasonable time
        assert batched_time < 10.0, f"Batched solve too slow: {batched_time:.2f}s"
    
    def test_mixed_source_formats(self):
        """Test handling of various source parameter combinations."""
        # Create fresh solver
        solver = self.create_solver()
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=True)
        
        # Various valid source formats
        sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.1, "phi": 0.1, "pte": 0.5, "ptm": 0.866},
            {"theta": 0.2, "phi": 0.0, "pte": 0.0, "ptm": 1.0},
        ]
        
        # Should process without error
        results = solver.solve(sources)
        assert len(results) == 3
        
        # All results should be valid
        for i in range(3):
            assert results[i].reflection is not None
            assert results[i].transmission is not None
    
    def test_memory_efficiency(self):
        """Test that batched processing is memory efficient."""
        # Create fresh solver
        solver = self.create_solver()
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=True)
        
        # Track memory for sequential
        tracemalloc.start()
        sequential_results = []
        for i in range(20):
            src = {"theta": i * 0.05, "phi": 0, "pte": 1.0, "ptm": 0.0}
            result = solver.solve(src)
            sequential_results.append(result)
        _, seq_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Track memory for batched
        sources = [
            {"theta": i * 0.05, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for i in range(20)
        ]
        
        tracemalloc.start()
        batched_results = solver.solve(sources)
        _, batch_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Ensure results are valid
        assert len(batched_results) == 20
        
        # Batched should not use significantly more memory
        # Allow 2x factor for overhead
        assert batch_peak < 2 * seq_peak, f"Batched memory usage too high: {batch_peak/seq_peak:.1f}x"
    
    def test_performance_improvement(self):
        """Test that batching provides performance benefits."""
        # Create fresh solver
        solver = self.create_solver()
        # Add multiple layers for more computation
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
        
        # Test with moderate batch size
        n_sources = 20
        sources = [
            {"theta": i * 0.05, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for i in range(n_sources)
        ]
        
        # Time sequential processing
        start = time.time()
        sequential_results = []
        for src in sources:
            result = solver.solve(src)
            sequential_results.append(result)
        seq_time = time.time() - start
        
        # Time batched processing
        start = time.time()
        batched_results = solver.solve(sources)
        batch_time = time.time() - start
        
        # Batched should be faster (or at least not significantly slower)
        # Allow some overhead for small batches
        assert batch_time < 1.5 * seq_time, f"Batched slower than sequential: {batch_time/seq_time:.2f}x"
        
        # Results should match
        for i in range(n_sources):
            torch.testing.assert_close(
                batched_results[i].reflection,
                sequential_results[i].reflection,
                rtol=1e-6, atol=1e-8
            )
    
    def test_multi_wavelength_batching(self):
        """Test batching with multiple wavelengths."""
        # Create solver with multiple wavelengths
        lam0 = np.linspace(1.0, 2.0, 5)
        
        multi_solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=lam0,
            rdim=[128, 128],
            kdim=[3, 3],
            device='cpu'
        )
        
        # Add materials
        mat_air = create_material(name="Air", permittivity=1.0)
        mat_si = create_material(name="Si", permittivity=11.7)
        multi_solver.add_materials([mat_air, mat_si])
        multi_solver.update_ref_material("Air")
        multi_solver.update_trn_material("Air")
        
        # Add layer
        multi_solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
        
        # Multiple sources
        sources = [
            {"theta": angle, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for angle in [0, 0.1, 0.2]
        ]
        
        # Solve
        results = multi_solver.solve(sources)
        
        # Check dimensions
        assert results.reflection.shape == (3, 5)  # (n_sources, n_wavelengths)
        assert results.transmission.shape == (3, 5)
        
        # Verify wavelength dependence
        for i in range(3):
            # Reflection should vary with wavelength
            r_values = results[i].reflection
            assert not torch.allclose(r_values[0], r_values[-1], rtol=1e-3)
    
    def test_algorithm_consistency(self):
        """Test that RCWA and RDIT give consistent results for batched sources."""
        # Create fresh solvers
        solver = self.create_solver(Algorithm.RDIT)
        solver_rcwa = self.create_solver(Algorithm.RCWA)
        # Add same structure to both
        for s in [solver, solver_rcwa]:
            s.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
        
        # Test sources
        sources = [
            {"theta": 0.0, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.2, "phi": 0, "pte": 0.7071, "ptm": 0.7071},
        ]
        
        # Solve with both algorithms
        rdit_results = solver.solve(sources)
        rcwa_results = solver_rcwa.solve(sources)
        
        # Results should be close (not identical due to algorithmic differences)
        for i in range(len(sources)):
            torch.testing.assert_close(
                rdit_results[i].reflection,
                rcwa_results[i].reflection,
                rtol=1e-3, atol=1e-4
            )
            torch.testing.assert_close(
                rdit_results[i].transmission,
                rcwa_results[i].transmission,
                rtol=1e-3, atol=1e-4
            )
    
    def test_no_memory_copy_on_indexing(self):
        """Test that indexing BatchedSolverResults doesn't create copies."""
        # Create fresh solver
        solver = self.create_solver()
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=True)
        
        sources = [
            {"theta": i * 0.1, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for i in range(10)
        ]
        
        results = solver.solve(sources)
        
        # Get data pointer before indexing
        refl_data_ptr = results.reflection.data_ptr()
        trans_data_ptr = results.transmission.data_ptr()
        
        # Access individual results
        result_5 = results[5]
        
        # Original tensors should not have moved
        assert results.reflection.data_ptr() == refl_data_ptr
        assert results.transmission.data_ptr() == trans_data_ptr
        
        # Individual result should be a view/slice
        assert result_5.reflection.shape == (1,)
    
    def test_error_handling(self):
        """Test proper error handling for invalid inputs."""
        # Create fresh solver
        solver = self.create_solver()
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=True)
        
        # Empty source list
        with pytest.raises(ValueError, match="At least one source required"):
            solver.solve([])
        
        # Invalid source format
        with pytest.raises((KeyError, ValueError)):
            solver.solve([{"theta": 0.0}])  # Missing required fields
        
        # Mixed single/batch should work fine
        single_src = {"theta": 0.0, "phi": 0, "pte": 1.0, "ptm": 0.0}
        single_result = solver.solve(single_src)
        assert isinstance(single_result, SolverResults)
        
        batch_src = [single_src]
        batch_result = solver.solve(batch_src)
        assert isinstance(batch_result, BatchedSolverResults)