"""
Numerical validation of batched source processing.

Focus: batched results must be numerically identical to sequential solves.
Secondary checks (angles/polarizations, wavelengths) are validated via
batched==sequential rather than loose sanity or performance metrics.
"""

import pytest
import torch
import numpy as np

from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm
from torchrdit.results import SolverResults


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
            rdim=[64, 64],
            kdim=[3, 3],
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
        assert isinstance(batched_results, SolverResults)
        assert batched_results.is_batched
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
    
    def test_angle_sweep_batched_equals_sequential(self):
        """Angle sweep: batched results must match per-angle sequential solves."""
        solver = self.create_solver()
        deg = np.pi / 180
        angles = np.linspace(0, 60, 5) * deg
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
        sources = [
            {"theta": float(angle), "phi": 0, "pte": 1.0, "ptm": 0.0}
            for angle in angles
        ]
        # Sequential
        seq = [solver.solve(src) for src in sources]
        # Batched
        bat = solver.solve(sources)
        assert isinstance(bat, SolverResults) and bat.is_batched and len(bat) == len(sources)
        for i in range(len(sources)):
            torch.testing.assert_close(bat[i].reflection, seq[i].reflection, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(bat[i].transmission, seq[i].transmission, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(bat[i].reflection_diffraction, seq[i].reflection_diffraction, rtol=1e-6, atol=1e-8)
    
    def test_polarizations_batched_equals_sequential(self):
        """Polarization sweep: batched must match sequential."""
        solver = self.create_solver()
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
        sources = [
            {"theta": 0.5, "phi": 0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.5, "phi": 0, "pte": 0.0, "ptm": 1.0},
            {"theta": 0.5, "phi": 0, "pte": 0.7071, "ptm": 0.7071},
        ]
        seq = [solver.solve(src) for src in sources]
        bat = solver.solve(sources)
        for i in range(len(sources)):
            torch.testing.assert_close(bat[i].reflection, seq[i].reflection, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(bat[i].transmission, seq[i].transmission, rtol=1e-6, atol=1e-8)
    
    # Removed gradient smoke check (covered by dedicated gradient tests).
    
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
        
        # Should get unified SolverResults with batching support for list input
        assert isinstance(batched_result, SolverResults)
        assert batched_result.is_batched
        assert len(batched_result) == 1
        
        # Results should match
        torch.testing.assert_close(
            batched_result[0].reflection, 
            single_result.reflection,
            rtol=1e-6, atol=1e-8
        )
    
    # Removed large-batch timing test (brittle and not about correctness).
    
    # Removed mixed format smoke test (covered by API tests).
    
    # Removed memory-efficiency/timing tests (non-deterministic, not correctness-focused).
    
    # Removed performance timing (brittle) — correctness already validated above.
    
    def test_multi_wavelength_batched_equals_sequential(self):
        """Multi-wavelength: batched must equal sequential for each source."""
        lam0 = np.linspace(1.0, 2.0, 5)
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=lam0,
            rdim=[64, 64],
            kdim=[3, 3],
            device='cpu'
        )
        mat_air = create_material(name="Air", permittivity=1.0)
        mat_si = create_material(name="Si", permittivity=11.7)
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("Air")
        solver.update_trn_material("Air")
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.5), is_homogeneous=False)
        sources = [
            {"theta": angle, "phi": 0, "pte": 1.0, "ptm": 0.0}
            for angle in [0, 0.1, 0.2]
        ]
        seq = [solver.solve(src) for src in sources]
        bat = solver.solve(sources)
        assert bat.reflection.shape == (3, len(lam0))
        for i in range(len(sources)):
            torch.testing.assert_close(bat[i].reflection, seq[i].reflection, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(bat[i].transmission, seq[i].transmission, rtol=1e-6, atol=1e-8)
    
    # Removed cross-algorithm consistency (not batching-specific).
    
    # Removed memory copy checks via data_ptr (brittle, implementation detail).
    
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
        
        # Mixed single vs. batch behaviors are covered in API tests.
