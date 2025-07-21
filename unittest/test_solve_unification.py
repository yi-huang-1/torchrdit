"""
Test suite for unified solve method implementation.

This test suite captures the current behavior of both single and batched source
processing to ensure that unifying _solve_batched and solve methods preserves
all existing functionality and error handling.
"""

import sys

sys.path.insert(0, "torchrdit/src")

import pytest
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.batched_results import BatchedSolverResults
from torchrdit.results import SolverResults
from torchrdit.utils import create_material


class TestSolveUnification:
    """Test unified solve method handles both single and batched sources."""

    def create_test_solver(self, algorithm="RCWA"):
        """Create a standardized test solver."""
        # Create materials
        mat_air = create_material(name="air", permittivity=1.0)
        mat_si = create_material(name="silicon", permittivity=11.7)

        # Create solver
        algo = Algorithm.RCWA if algorithm == "RCWA" else Algorithm.RDIT
        solver = create_solver(
            algorithm=algo,
            lam0=np.array([1.0, 1.5]),  # Multiple wavelengths
            rdim=[64, 64],  # Smaller for faster tests
            kdim=[3, 3],
            device="cpu",
        )

        # Add materials and layers
        solver.add_materials([mat_air, mat_si])
        solver.update_ref_material("air")
        solver.update_trn_material("air")
        solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2), is_homogeneous=False)

        return solver

    def test_single_source_unchanged(self):
        """Verify single source behavior remains identical."""
        solver = self.create_test_solver()
        
        # Create single source
        source = solver.add_source(theta=0.1, phi=0.0, pte=1.0, ptm=0.0)
        
        # Test that dict input returns SolverResults
        result = solver.solve(source)
        
        assert isinstance(result, SolverResults), f"Expected SolverResults, got {type(result)}"
        assert result.reflection.shape == (2,), f"Expected shape (2,), got {result.reflection.shape}"
        assert result.transmission.shape == (2,), f"Expected shape (2,), got {result.transmission.shape}"
        
        # Verify fields are accessible
        assert hasattr(result, 'reflection_field')
        assert hasattr(result, 'transmission_field')

    def test_batched_sources_unchanged(self):
        """Verify batched source behavior remains identical."""
        solver = self.create_test_solver()
        
        # Create batched sources
        sources = [
            solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0),
            solver.add_source(theta=0.1, phi=0.0, pte=1.0, ptm=0.0),
            solver.add_source(theta=0.2, phi=0.0, pte=0.0, ptm=1.0),
        ]
        
        # Test that list input returns BatchedSolverResults
        results = solver.solve(sources)
        
        assert isinstance(results, BatchedSolverResults), f"Expected BatchedSolverResults, got {type(results)}"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert results.reflection.shape == (3, 2), f"Expected shape (3, 2), got {results.reflection.shape}"
        assert results.transmission.shape == (3, 2), f"Expected shape (3, 2), got {results.transmission.shape}"
        
        # Verify individual results are accessible
        for i in range(3):
            result_i = results[i]
            assert isinstance(result_i, SolverResults)
            assert result_i.reflection.shape == (2,)

    def test_input_validation_preserved(self):
        """Ensure all validation logic is preserved."""
        solver = self.create_test_solver()
        
        # Test empty list raises ValueError
        with pytest.raises(ValueError, match="At least one source required"):
            solver.solve([])
        
        # Test invalid source format raises ValueError with clear message
        invalid_sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},  # Valid
            {"invalid": "source"},  # Missing required keys
        ]
        
        with pytest.raises(ValueError, match="Invalid source format"):
            solver.solve(invalid_sources)
        
        # Test non-dict in source list
        mixed_sources = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            "not_a_dict",
        ]
        
        with pytest.raises(ValueError, match="Invalid source format"):
            solver.solve(mixed_sources)

    def test_input_type_validation(self):
        """Test that non-dict/non-list input raises appropriate errors."""
        solver = self.create_test_solver()
        
        # Test invalid input types - capturing current behavior exactly
        
        # String is iterable, so goes to validation and fails there
        with pytest.raises(ValueError, match="Invalid source format"):
            solver.solve("string")
        
        # Integer should go to _solve_batched and fail iteration
        with pytest.raises(TypeError, match="'int' object is not iterable"):
            solver.solve(42)
        
        # None is falsy, so hits the "At least one source required" check first
        with pytest.raises(ValueError, match="At least one source required"):
            solver.solve(None)
        
        # Tensor causes specific RuntimeError in current implementation
        with pytest.raises(RuntimeError, match="Boolean value of Tensor"):
            solver.solve(torch.tensor([1, 2, 3]))

    def test_single_element_list_returns_batched(self):
        """Single source in list should return BatchedSolverResults."""
        solver = self.create_test_solver()
        
        # Create single source in a list
        sources = [solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)]
        
        result = solver.solve(sources)
        
        # Should return BatchedSolverResults, not SolverResults
        assert isinstance(result, BatchedSolverResults), f"Expected BatchedSolverResults, got {type(result)}"
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert result.reflection.shape == (1, 2), f"Expected shape (1, 2), got {result.reflection.shape}"

    def test_current_solve_batched_behavior(self):
        """Test current _solve_batched method behavior before unification."""
        solver = self.create_test_solver()
        
        sources = [
            solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0),
            solver.add_source(theta=0.1, phi=0.0, pte=1.0, ptm=0.0),
        ]
        
        # Test current behavior through solve() method
        result = solver.solve(sources)
        
        # Verify it calls _solve_batched internally (currently)
        assert isinstance(result, BatchedSolverResults)
        assert len(result) == 2
        
        # Verify results are reasonable
        assert torch.all(result.reflection >= 0)
        assert torch.all(result.transmission >= 0)
        assert torch.all(result.reflection + result.transmission <= 1.01)  # Energy conservation

    def test_algorithm_independence(self):
        """Test that unification works for both RCWA and RDIT algorithms."""
        algorithms = ["RCWA", "RDIT"]
        
        for algo in algorithms:
            solver = self.create_test_solver(algorithm=algo)
            
            # Single source
            source = solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
            single_result = solver.solve(source)
            assert isinstance(single_result, SolverResults)
            
            # Batched sources
            sources = [
                solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0),
                solver.add_source(theta=0.1, phi=0.0, pte=1.0, ptm=0.0),
            ]
            batch_result = solver.solve(sources)
            assert isinstance(batch_result, BatchedSolverResults)

    def test_error_message_specificity(self):
        """Verify error messages are specific and helpful."""
        solver = self.create_test_solver()
        
        # Test specific error for missing theta
        invalid_source = {"phi": 0.0, "pte": 1.0, "ptm": 0.0}  # Missing theta
        
        with pytest.raises(ValueError) as exc_info:
            solver.solve([invalid_source])
        
        # Should mention "theta" in error message
        assert "theta" in str(exc_info.value) or "Invalid source format" in str(exc_info.value)

    def test_maintains_source_state(self):
        """Verify that solver.src is properly maintained."""
        solver = self.create_test_solver()
        
        # Single source should set self.src
        source = solver.add_source(theta=0.1, phi=0.0, pte=1.0, ptm=0.0)
        result = solver.solve(source)
        
        # Verify self.src is set correctly
        assert hasattr(solver, 'src')
        assert solver.src == source
        
        # Batched sources should also work (though src state is less important)
        sources = [
            solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0),
            solver.add_source(theta=0.1, phi=0.0, pte=1.0, ptm=0.0),
        ]
        batch_result = solver.solve(sources)
        assert isinstance(batch_result, BatchedSolverResults)

    def test_kwargs_passthrough(self):
        """Verify that kwargs are properly passed through to _solve_structure."""
        solver = self.create_test_solver()
        
        source = solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
        
        # Test with various kwargs (even if not used, should not raise errors)
        result = solver.solve(source, compute_fields=True)
        assert isinstance(result, SolverResults)
        
        # Test batched with kwargs
        sources = [source]
        batch_result = solver.solve(sources, compute_fields=True)
        assert isinstance(batch_result, BatchedSolverResults)