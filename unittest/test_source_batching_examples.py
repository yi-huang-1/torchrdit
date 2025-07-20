"""Test cases for source batching examples from docstrings.

This module tests the examples provided in the docstrings of solver.py and
__init__.py to ensure they work correctly and demonstrate proper usage
of the source batching feature.
"""

import unittest
import numpy as np
import torch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.batched_results import BatchedSolverResults


class TestSourceBatchingDocExamples(unittest.TestCase):
    """Test source batching examples from module and function docstrings."""

    def test_init_module_example(self):
        """Test the source batching example from __init__.py module docstring."""
        # This is the example from the module docstring
        # Create solver
        solver = create_solver(algorithm=Algorithm.RDIT, rdim=[512, 512], kdim=[7, 7])
        
        # Add materials (required for the solver to work)
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        
        # Add a layer to have a structure
        solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
        
        # Create simple pattern
        mask = torch.zeros(512, 512)
        mask[200:312, 200:312] = 1.0
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Create multiple sources for angle sweep
        deg = np.pi / 180
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in np.linspace(0, 60, 13) * deg
        ]
        
        # Batch solve - returns BatchedSolverResults
        results = solver.solve(sources)
        
        # Verify results
        self.assertIsInstance(results, BatchedSolverResults)
        self.assertEqual(results.n_sources, 13)
        
        # Access results
        trans_all = results.transmission[:, 0]
        self.assertEqual(trans_all.shape[0], 13)
        
        # Find optimal source
        best_idx = results.find_optimal_source('max_transmission')
        self.assertIsInstance(best_idx, int)
        self.assertTrue(0 <= best_idx < 13)
        
        # Verify we can access the best angle
        best_angle = sources[best_idx]['theta'] * 180/np.pi
        self.assertIsInstance(best_angle, (float, np.floating))

    def test_create_solver_batching_example(self):
        """Test the source batching example from create_solver docstring."""
        # Create solver and set up structure
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[512, 512],
            kdim=[7, 7],
            device='cpu'  # Use CPU for testing
        )
        
        # Add materials and layers
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        solver.add_layer(material_name="Si", thickness=0.6, is_homogeneous=False)
        
        # Create grating pattern
        mask = torch.zeros(512, 512)
        period = int(512 * 0.8 / 1.0)  # Period in pixels
        duty_cycle = 0.5
        for i in range(0, 512, period):
            mask[:, i:i+int(period*duty_cycle)] = 1.0
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Create multiple sources for angle sweep
        deg = np.pi / 180
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in np.linspace(0, 60, 13) * deg
        ]
        
        # Batch solve - much faster than sequential processing
        results = solver.solve(sources)  # Returns BatchedSolverResults
        
        # Verify results type
        self.assertIsInstance(results, BatchedSolverResults)
        self.assertEqual(results.n_sources, 13)
        
        # Analyze results
        best_idx = results.find_optimal_source('max_transmission')
        self.assertIsInstance(best_idx, int)
        self.assertTrue(0 <= best_idx < 13)
        
        # Verify angle access
        best_angle = sources[best_idx]['theta'] * 180/np.pi
        self.assertTrue(0 <= best_angle <= 60)

    def test_batched_results_module_example(self):
        """Test the example from batched_results.py module docstring."""
        # Create a solver first
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[256, 256],
            kdim=[5, 5]
        )
        
        # Add materials and structure
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
        
        # Simple structure
        mask = torch.ones(256, 256) * 0.5
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        deg = np.pi / 180
        
        # Example from docstring
        # Solve with multiple sources
        sources = [
            solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0),
            solver.add_source(theta=30*deg, phi=0, pte=1.0, ptm=0.0),
            solver.add_source(theta=45*deg, phi=0, pte=0.7, ptm=0.3)
        ]
        batched_results = solver.solve(sources)
        
        # Access bulk results
        all_transmission = batched_results.transmission  # Shape: (3, n_freqs)
        self.assertEqual(all_transmission.shape[0], 3)
        self.assertEqual(all_transmission.shape[1], 1)  # Single wavelength
        
        # Access individual results
        result_0 = batched_results[0]  # Returns SolverResults for first source
        from torchrdit.results import SolverResults
        self.assertIsInstance(result_0, SolverResults)
        
        # Find optimal source
        best_idx = batched_results.find_optimal_source('max_transmission')
        self.assertIsInstance(best_idx, int)
        self.assertTrue(0 <= best_idx < 3)

    def test_batched_results_class_example(self):
        """Test the example from BatchedSolverResults class docstring."""
        # Create solver
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[256, 256],
            kdim=[5, 5]
        )
        
        # Add materials and structure
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        solver.add_layer(material_name="Si", thickness=0.3, is_homogeneous=False)
        
        # Create a pattern
        mask = torch.zeros(256, 256)
        mask[100:156, 100:156] = 1.0
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Create sources
        deg = np.pi / 180
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in np.array([0, 30, 45, 60]) * deg
        ]
        
        # Solve
        batched_results = solver.solve(sources)
        
        # Examples from docstring
        # Access bulk results
        all_trans = batched_results.transmission  # (n_sources, n_freqs)
        self.assertEqual(all_trans.shape, (4, 1))
        
        # Get results for specific source
        source_2 = batched_results[2]  # Returns SolverResults
        from torchrdit.results import SolverResults
        self.assertIsInstance(source_2, SolverResults)
        
        # Iterate over all sources
        count = 0
        for i, result in enumerate(batched_results):
            trans_val = result.transmission[0].item()
            self.assertIsInstance(trans_val, float)
            self.assertTrue(0 <= trans_val <= 1)
            count += 1
        self.assertEqual(count, 4)
        
        # Find best performing source
        best_idx = batched_results.find_optimal_source('max_transmission')
        self.assertIsInstance(best_idx, int)
        self.assertTrue(0 <= best_idx < 4)

    def test_energy_conservation(self):
        """Test that batched results satisfy energy conservation."""
        # Create solver with simple structure
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[256, 256],
            kdim=[5, 5]
        )
        
        # Add materials
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        
        # Add homogeneous layer (no scattering)
        solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=True)
        
        # Create sources at different angles
        deg = np.pi / 180
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in np.linspace(0, 45, 5) * deg
        ]
        
        # Solve
        results = solver.solve(sources)
        
        # Check energy conservation for each source
        for i in range(results.n_sources):
            R = results.reflection[i, 0].item()
            T = results.transmission[i, 0].item()
            loss = results.loss[i, 0].item()
            
            # Energy should be conserved (R + T + loss ≈ 1)
            total = R + T + loss
            self.assertAlmostEqual(total, 1.0, places=5, 
                                 msg=f"Energy not conserved for source {i}: R+T+loss={total}")
            
            # For lossless materials, loss should be near zero
            self.assertLess(loss, 1e-5, 
                          msg=f"Unexpected loss for lossless material: {loss}")


if __name__ == '__main__':
    unittest.main()