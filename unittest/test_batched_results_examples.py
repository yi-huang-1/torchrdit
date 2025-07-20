"""Test cases for BatchedSolverResults examples from docstrings.

This module tests the examples provided in the docstrings of batched_results.py
to ensure they work correctly and demonstrate proper usage of the BatchedSolverResults
class and its methods.
"""

import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
from unittest.mock import patch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.batched_results import BatchedSolverResults
from torchrdit.results import SolverResults


class TestBatchedResultsDocExamples(unittest.TestCase):
    """Test BatchedSolverResults examples from docstrings."""

    def setUp(self):
        """Set up a common solver and results for testing."""
        # Create solver
        self.solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[256, 256],
            kdim=[5, 5]
        )
        
        # Add materials and structure
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        self.solver.add_materials([si, air])
        self.solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
        
        # Create simple pattern
        mask = torch.zeros(256, 256)
        mask[100:156, 100:156] = 1.0
        self.solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Create sources for testing
        deg = np.pi / 180
        self.sources = [
            self.solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in np.array([0, 20, 40, 60]) * deg
        ]
        
        # Get batched results
        self.results = self.solver.solve(self.sources)

    def test_find_optimal_source_example(self):
        """Test the find_optimal_source method examples from docstring."""
        # Example 1: Find source with highest transmission
        best_idx = self.results.find_optimal_source('max_transmission')
        best_result = self.results[best_idx]
        
        # Verify
        self.assertIsInstance(best_idx, int)
        self.assertTrue(0 <= best_idx < len(self.sources))
        self.assertIsInstance(best_result, SolverResults)
        
        # Verify it's actually the maximum
        all_trans = self.results.transmission.mean(dim=1)
        max_trans_idx = torch.argmax(all_trans).item()
        self.assertEqual(best_idx, max_trans_idx)
        
        # Example 2: Find source with lowest reflection at specific frequency
        best_idx = self.results.find_optimal_source('min_reflection', frequency_idx=0)
        
        # Verify
        self.assertIsInstance(best_idx, int)
        self.assertTrue(0 <= best_idx < len(self.sources))
        
        # Verify it's actually the minimum
        reflections_at_freq0 = self.results.reflection[:, 0]
        min_refl_idx = torch.argmin(reflections_at_freq0).item()
        self.assertEqual(best_idx, min_refl_idx)

    def test_get_parameter_sweep_data_example(self):
        """Test the get_parameter_sweep_data method examples from docstring."""
        # Example: Get data for angle sweep
        angles, trans = self.results.get_parameter_sweep_data('theta', 'transmission')
        
        # Verify return types and shapes
        self.assertIsInstance(angles, torch.Tensor)
        self.assertIsInstance(trans, torch.Tensor)
        self.assertEqual(angles.shape[0], len(self.sources))
        self.assertEqual(trans.shape[0], len(self.sources))
        
        # Verify angle values are correct
        expected_angles = torch.tensor([src['theta'] for src in self.sources])
        torch.testing.assert_close(angles, expected_angles)
        
        # Verify transmission values match
        expected_trans = self.results.transmission[:, 0]
        torch.testing.assert_close(trans, expected_trans)
        
        # Test with matplotlib mock
        with patch('matplotlib.pyplot.plot') as mock_plot:
            # Example plotting code from docstring
            plt.plot(angles * 180/np.pi, trans)
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Transmission')
            
            # Verify plot was called
            mock_plot.assert_called_once()
            args = mock_plot.call_args[0]
            np.testing.assert_array_almost_equal(
                args[0], angles.numpy() * 180/np.pi
            )
            np.testing.assert_array_almost_equal(
                args[1], trans.numpy()
            )

    def test_all_metrics_in_find_optimal_source(self):
        """Test all available metrics in find_optimal_source."""
        # Test max_transmission
        idx = self.results.find_optimal_source('max_transmission')
        self.assertIsInstance(idx, int)
        self.assertTrue(0 <= idx < len(self.sources))
        
        # Test min_reflection
        idx = self.results.find_optimal_source('min_reflection')
        self.assertIsInstance(idx, int)
        self.assertTrue(0 <= idx < len(self.sources))
        
        # Test max_efficiency
        idx = self.results.find_optimal_source('max_efficiency')
        self.assertIsInstance(idx, int)
        self.assertTrue(0 <= idx < len(self.sources))
        
        # Test with frequency index
        idx = self.results.find_optimal_source('max_transmission', frequency_idx=0)
        self.assertIsInstance(idx, int)
        self.assertTrue(0 <= idx < len(self.sources))
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            self.results.find_optimal_source('invalid_metric')

    def test_all_metrics_in_get_parameter_sweep_data(self):
        """Test all available metrics in get_parameter_sweep_data."""
        # Test transmission
        params, values = self.results.get_parameter_sweep_data('theta', 'transmission')
        self.assertEqual(values.shape[0], len(self.sources))
        
        # Test reflection
        params, values = self.results.get_parameter_sweep_data('theta', 'reflection')
        self.assertEqual(values.shape[0], len(self.sources))
        
        # Test loss
        params, values = self.results.get_parameter_sweep_data('theta', 'loss')
        self.assertEqual(values.shape[0], len(self.sources))
        
        # Test different parameters
        for param in ['theta', 'phi', 'pte', 'ptm']:
            params, values = self.results.get_parameter_sweep_data(param, 'transmission')
            self.assertEqual(params.shape[0], len(self.sources))
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            self.results.get_parameter_sweep_data('theta', 'invalid_metric')

    def test_iteration_and_list_conversion(self):
        """Test iteration and list conversion methods."""
        # Test __iter__
        count = 0
        for i, result in enumerate(self.results):
            self.assertIsInstance(result, SolverResults)
            self.assertEqual(result.transmission.shape[0], 1)  # Single frequency
            count += 1
        self.assertEqual(count, len(self.sources))
        
        # Test as_list property
        results_list = self.results.as_list
        self.assertIsInstance(results_list, list)
        self.assertEqual(len(results_list), len(self.sources))
        for result in results_list:
            self.assertIsInstance(result, SolverResults)
        
        # Test get_source_result
        for i in range(len(self.sources)):
            result = self.results.get_source_result(i)
            self.assertIsInstance(result, SolverResults)
            # Should be same as indexing
            result_idx = self.results[i]
            torch.testing.assert_close(result.transmission, result_idx.transmission)

    def test_slicing_batched_results(self):
        """Test slicing functionality of BatchedSolverResults."""
        # Test single element slicing
        subset = self.results[1:3]
        self.assertIsInstance(subset, BatchedSolverResults)
        self.assertEqual(subset.n_sources, 2)
        self.assertEqual(len(subset.source_parameters), 2)
        
        # Test negative indexing in slice
        subset = self.results[-2:]
        self.assertIsInstance(subset, BatchedSolverResults)
        self.assertEqual(subset.n_sources, 2)
        
        # Test step in slice
        subset = self.results[::2]
        self.assertIsInstance(subset, BatchedSolverResults)
        self.assertEqual(subset.n_sources, 2)
        
        # Test empty slice
        with self.assertRaises(ValueError):
            _ = self.results[2:2]

    def test_multi_frequency_batched_results(self):
        """Test BatchedSolverResults with multiple frequencies."""
        # Create solver with multiple wavelengths
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.31, 1.55, 1.80]),
            rdim=[256, 256],
            kdim=[5, 5]
        )
        
        # Add materials and structure
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        solver.add_layer(material_name="Si", thickness=0.3, is_homogeneous=True)
        
        # Create sources
        deg = np.pi / 180
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in np.array([0, 30, 60]) * deg
        ]
        
        # Solve
        results = solver.solve(sources)
        
        # Verify shape
        self.assertEqual(results.transmission.shape, (3, 3))  # 3 sources, 3 frequencies
        
        # Test find_optimal_source with specific frequency
        for freq_idx in range(3):
            best_idx = results.find_optimal_source('max_transmission', frequency_idx=freq_idx)
            self.assertTrue(0 <= best_idx < 3)
        
        # Test get_parameter_sweep_data with different frequencies
        for freq_idx in range(3):
            angles, trans = results.get_parameter_sweep_data('theta', 'transmission', frequency_idx=freq_idx)
            self.assertEqual(trans.shape[0], 3)

    def test_polarization_sweep(self):
        """Test BatchedSolverResults with polarization sweep."""
        # Create sources with different polarizations
        sources = [
            self.solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0),  # TE
            self.solver.add_source(theta=0, phi=0, pte=0.0, ptm=1.0),  # TM
            self.solver.add_source(theta=0, phi=0, pte=0.707, ptm=0.707),  # 45°
        ]
        
        # Solve
        results = self.solver.solve(sources)
        
        # Get polarization data
        pte_values, trans_te = results.get_parameter_sweep_data('pte', 'transmission')
        ptm_values, trans_tm = results.get_parameter_sweep_data('ptm', 'transmission')
        
        # Verify polarization values
        expected_pte = torch.tensor([1.0, 0.0, 0.707])
        expected_ptm = torch.tensor([0.0, 1.0, 0.707])
        torch.testing.assert_close(pte_values, expected_pte, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ptm_values, expected_ptm, rtol=1e-3, atol=1e-3)
        
        # Find best polarization
        best_idx = results.find_optimal_source('max_transmission')
        self.assertTrue(0 <= best_idx < 3)


if __name__ == '__main__':
    unittest.main()