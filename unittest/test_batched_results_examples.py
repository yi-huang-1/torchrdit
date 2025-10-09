"""Test cases for unified SolverResults with batching support.

This module tests the batching functionality of the unified SolverResults class,
ensuring it works correctly for all methods that previously existed in the 
now-removed BatchedSolverResults class.
"""

import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
from unittest.mock import patch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.results import SolverResults


class TestUnifiedResultsBatchingExamples(unittest.TestCase):
    """Test unified SolverResults batching functionality."""

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
        
        # Get unified results with batching support  
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
        
        # Also verify frequency-specific max transmission
        best_idx_freq = self.results.find_optimal_source('max_transmission', frequency_idx=0)
        self.assertEqual(best_idx_freq, torch.argmax(self.results.transmission[:, 0]).item())
        
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
        self.assertEqual(idx, torch.argmax(self.results.transmission.mean(dim=1)).item())
        
        # Test min_reflection
        idx = self.results.find_optimal_source('min_reflection')
        self.assertIsInstance(idx, int)
        self.assertTrue(0 <= idx < len(self.sources))
        self.assertEqual(idx, torch.argmin(self.results.reflection.mean(dim=1)).item())
        
        # Test max_efficiency
        idx = self.results.find_optimal_source('max_efficiency')
        self.assertIsInstance(idx, int)
        self.assertTrue(0 <= idx < len(self.sources))
        eff_mean = (self.results.transmission + self.results.reflection).mean(dim=1)
        self.assertEqual(idx, torch.argmax(eff_mean).item())
        
        # Test with frequency index
        idx = self.results.find_optimal_source('max_transmission', frequency_idx=0)
        self.assertIsInstance(idx, int)
        self.assertTrue(0 <= idx < len(self.sources))
        self.assertEqual(idx, torch.argmax(self.results.transmission[:, 0]).item())
        # Frequency-specific checks for other metrics
        idx = self.results.find_optimal_source('min_reflection', frequency_idx=0)
        self.assertEqual(idx, torch.argmin(self.results.reflection[:, 0]).item())
        idx = self.results.find_optimal_source('max_efficiency', frequency_idx=0)
        self.assertEqual(idx, torch.argmax(self.results.transmission[:, 0] + self.results.reflection[:, 0]).item())
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            self.results.find_optimal_source('invalid_metric')

    def test_all_metrics_in_get_parameter_sweep_data(self):
        """Test all available metrics in get_parameter_sweep_data."""
        # Test transmission
        params, values = self.results.get_parameter_sweep_data('theta', 'transmission')
        self.assertEqual(values.shape[0], len(self.sources))
        torch.testing.assert_close(values, self.results.transmission[:, 0])
        
        # Test reflection
        params, values = self.results.get_parameter_sweep_data('theta', 'reflection')
        self.assertEqual(values.shape[0], len(self.sources))
        torch.testing.assert_close(values, self.results.reflection[:, 0])
        
        # Test loss
        params, values = self.results.get_parameter_sweep_data('theta', 'loss')
        self.assertEqual(values.shape[0], len(self.sources))
        torch.testing.assert_close(
            self.results.transmission[:, 0] + self.results.reflection[:, 0] + values,
            torch.ones_like(values),
            rtol=1e-4,
            atol=1e-4,
        )
        
        # Test different parameters
        for param in ['theta', 'phi', 'pte', 'ptm']:
            params, values = self.results.get_parameter_sweep_data(param, 'transmission')
            self.assertEqual(params.shape[0], len(self.sources))
        # Verify actual parameter arrays match sources as well
        expected_phi = torch.tensor([src['phi'] for src in self.sources])
        phi_vals, _ = self.results.get_parameter_sweep_data('phi', 'transmission')
        torch.testing.assert_close(phi_vals, expected_phi)
        expected_pte = torch.tensor([src['pte'] for src in self.sources])
        pte_vals, _ = self.results.get_parameter_sweep_data('pte', 'transmission')
        torch.testing.assert_close(pte_vals, expected_pte)
        expected_ptm = torch.tensor([src['ptm'] for src in self.sources])
        ptm_vals, _ = self.results.get_parameter_sweep_data('ptm', 'transmission')
        torch.testing.assert_close(ptm_vals, expected_ptm)
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            self.results.get_parameter_sweep_data('theta', 'invalid_metric')
        # Test invalid parameter name raises KeyError
        with self.assertRaises(KeyError):
            self.results.get_parameter_sweep_data('nonexistent_param', 'transmission')

    def test_iteration_and_list_conversion(self):
        """Test iteration and list conversion methods."""
        # Test __iter__
        count = 0
        for i, result in enumerate(self.results):
            self.assertIsInstance(result, SolverResults)
            self.assertEqual(result.transmission.shape[0], 1)  # Single frequency
            self.assertFalse(result.is_batched)
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
        """Test slicing functionality of unified SolverResults with batching."""
        # Test single element slicing
        subset = self.results[1:3]
        self.assertIsInstance(subset, SolverResults)
        self.assertTrue(subset.is_batched)  # Should be batched
        self.assertEqual(subset.n_sources, 2)
        self.assertEqual(len(subset.source_parameters), 2)
        
        # Test negative indexing in slice
        subset = self.results[-2:]
        self.assertIsInstance(subset, SolverResults)
        self.assertTrue(subset.is_batched)  # Should be batched
        self.assertEqual(subset.n_sources, 2)
        
        # Test step in slice
        subset = self.results[::2]
        self.assertIsInstance(subset, SolverResults)
        self.assertTrue(subset.is_batched)  # Should be batched
        self.assertEqual(subset.n_sources, 2)
        
        # Test empty slice
        with self.assertRaises(ValueError):
            _ = self.results[2:2]
        
        # Test out-of-range integer index raises
        with self.assertRaises(IndexError):
            _ = self.results[999]
        
        # Test negative integer index equals last element
        last_pos = self.results[len(self.sources) - 1]
        last_neg = self.results[-1]
        torch.testing.assert_close(last_pos.transmission, last_neg.transmission)

    def test_multi_frequency_batched_results(self):
        """Test unified SolverResults with multiple frequencies."""
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
            self.assertEqual(best_idx, torch.argmax(results.transmission[:, freq_idx]).item())
        
        # Test get_parameter_sweep_data with different frequencies
        for freq_idx in range(3):
            angles, trans = results.get_parameter_sweep_data('theta', 'transmission', frequency_idx=freq_idx)
            self.assertEqual(trans.shape[0], 3)
            torch.testing.assert_close(trans, results.transmission[:, freq_idx])
        
        # Invalid frequency index should raise IndexError
        with self.assertRaises(IndexError):
            _ = results.get_parameter_sweep_data('theta', 'transmission', frequency_idx=999)
        with self.assertRaises(IndexError):
            _ = results.find_optimal_source('max_transmission', frequency_idx=999)

    def test_polarization_sweep(self):
        """Test unified SolverResults with polarization sweep."""
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
        self.assertEqual(best_idx, torch.argmax(results.transmission.mean(dim=1)).item())

    def test_single_source_error_paths(self):
        """Methods restricted to batched results should raise on single-source."""
        single_source = self.solver.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
        single_results = self.solver.solve(single_source)
        with self.assertRaises(ValueError):
            _ = single_results.find_optimal_source('max_transmission')
        with self.assertRaises(ValueError):
            _ = single_results.get_parameter_sweep_data('theta', 'transmission')


if __name__ == '__main__':
    unittest.main()
