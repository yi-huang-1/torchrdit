import unittest
from unittest.mock import patch, MagicMock
import io
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from torchrdit.viz import (
    plot2d, plot_layer, display_fitted_permittivity
)
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.solver import create_solver


class TestVizDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in viz.py.
    
    This test suite ensures that the examples provided in the docstrings of
    viz.py work as expected. It tests examples for:
    1. plot2d function for visualizing 2D field data
    2. plot_layer function for visualizing material distributions
    3. display_fitted_permittivity function for visualizing dispersive material fitting
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Prepare common test data
        self.test_wavelength = np.array([1.55])
        
        # Create a simple 2D field distribution for testing
        x = np.linspace(-1, 1, 32)
        y = np.linspace(-1, 1, 32)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Create a sinusoidal field pattern
        self.field_data = np.sin(5 * self.X) * np.cos(5 * self.Y) + 1j * np.cos(5 * self.X) * np.sin(5 * self.Y)
        
        # Convert to torch tensor for compatibility
        self.field_tensor = torch.tensor(self.field_data)
        
        # Create a simple outline mask
        self.outline_mask = np.zeros_like(self.field_data, dtype=np.float32)
        self.outline_mask[8:24, 8:24] = 1.0
    
    def test_plot2d_example(self):
        """Test the plot2d function example from docstring."""
        # Suppress actual plotting during the test
        plt.close('all')
        with patch('matplotlib.pyplot.show'):
            # Create data and layout similar to the example
            x_grid, y_grid = self.X, self.Y
            e_field = self.field_tensor
            
            # Create a figure and plot as in the example
            fig, ax = plt.subplots(figsize=(8, 6))
            result_ax = plot2d(
                data=e_field,
                layout=(x_grid, y_grid),
                func='real',
                cmap='RdBu_r',
                title='Electric Field (Ex) Distribution',
                labels=('x (μm)', 'y (μm)')
            )
            
            # Verify that plot2d returns a matplotlib axes object
            self.assertIsInstance(result_ax, Axes)
            
            # Verify that the title and labels are set correctly
            self.assertEqual(result_ax.get_title(), 'Electric Field (Ex) Distribution')
            self.assertEqual(result_ax.get_xlabel(), 'x (μm)')
            self.assertEqual(result_ax.get_ylabel(), 'y (μm)')
            
            # Verify that a colorbar was added
            self.assertEqual(len(result_ax.figure.axes), 2)  # Main axis + colorbar axis
            
            # Close the figure to avoid memory leaks
            plt.close(fig)
    
    def test_plot2d_with_outline(self):
        """Test the plot2d function with an outline mask."""
        plt.close('all')
        with patch('matplotlib.pyplot.show'):
            # Test with outline mask
            fig, ax = plt.subplots(figsize=(8, 6))
            result_ax = plot2d(
                data=self.field_tensor,
                layout=(self.X, self.Y),
                func='abs',
                outline=self.outline_mask,
                fig_ax=ax,
                cmap='magma',
                title='Field with Outline',
                labels=('x', 'y')
            )
            
            # Verify that the result axis is the same as the input axis
            self.assertEqual(result_ax, ax)
            
            # Verify that outline affects the plot (check for contour collection)
            has_contour = False
            for child in ax.get_children():
                if 'QuadContourSet' in str(type(child)):
                    has_contour = True
                    break
            
            self.assertTrue(has_contour, "No contour found in plot with outline")
            
            # Close the figure
            plt.close(fig)
    
    def test_plot2d_different_functions(self):
        """Test plot2d with different processing functions."""
        plt.close('all')
        with patch('matplotlib.pyplot.show'):
            functions = ['abs', 'real', 'imag']
            
            for func in functions:
                fig, ax = plt.subplots()
                result_ax = plot2d(
                    data=self.field_tensor,
                    layout=(self.X, self.Y),
                    func=func,
                    cmap='viridis'
                )
                
                # Check that the function works with different processing functions
                self.assertIsInstance(result_ax, Axes)
                
                # Close the figure
                plt.close(fig)
    
    def test_plot_layer_example(self):
        """Test the plot_layer function example from docstring."""
        plt.close('all')
        with patch('matplotlib.pyplot.show'):
            # Create a simulation with multiple layers
            solver = create_solver(
                algorithm=Algorithm.RDIT,
                lam0=np.array([1.55]),
                rdim=[32, 32],  # Small dimensions for testing
                kdim=[3, 3]     # Small dimensions for testing
            )
            
            # Mock the get_layout method to return our test grid
            solver.get_layout = MagicMock(return_value=(self.X, self.Y))
            
            # Create example permittivity data for the second layer
            test_permittivity = np.ones((32, 32), dtype=complex) * 2.1
            test_permittivity[10:20, 10:20] = 3.0  # Add some pattern
            test_permittivity_tensor = torch.tensor(test_permittivity)
            
            # Properly mock the layer structure for plot_layer function
            mock_layer0 = MagicMock()
            mock_layer0.is_dispersive = False
            mock_layer0.ermat = torch.ones((32, 32), dtype=torch.complex64) * 11.7
            mock_layer0.material_name = 'Si'
            
            mock_layer1 = MagicMock()
            mock_layer1.is_dispersive = False
            mock_layer1.ermat = test_permittivity_tensor
            mock_layer1.material_name = 'sio2'
            
            # Set up the layer_manager mock
            solver.layer_manager = MagicMock()
            solver.layer_manager.layers = [mock_layer0, mock_layer1]
            solver.layer_manager.nlayer = 2
            
            # Plot the permittivity distribution of the second layer
            fig, ax = plt.subplots(figsize=(8, 6))
            result_ax = plot_layer(
                solver, 
                layer_index=1, 
                fig_ax=ax,
                title='SiO2 Layer', 
                labels=('x (μm)', 'y (μm)')
            )
            
            # Verify that plot_layer returns a matplotlib axes object
            self.assertIsInstance(result_ax, Axes)
            
            # Verify that the title and labels are set correctly
            self.assertEqual(result_ax.get_title(), 'SiO2 Layer')
            self.assertEqual(result_ax.get_xlabel(), 'x (μm)')
            self.assertEqual(result_ax.get_ylabel(), 'y (μm)')
            
            # Close the figure
            plt.close(fig)
    
    def test_display_fitted_permittivity_example(self):
        """Test the display_fitted_permittivity function example from docstring."""
        plt.close('all')
        
        # Create a simulation
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.3, 1.4, 1.5, 1.6]),
            rdim=[32, 32],
            kdim=[3, 3]
        )
        
        # Mock a dispersive layer
        mock_layer = MagicMock()
        mock_layer.is_dispersive = True
        mock_layer.material_name = 'gold'
        
        solver.layer_manager = MagicMock()
        solver.layer_manager.layers = [mock_layer]
        solver.layer_manager.nlayer = 1
        
        # Mock the material library
        mock_gold_material = MagicMock()
        wls = np.linspace(1.0, 2.0, 10)
        data_eps1 = np.linspace(1.0, 2.0, 10)
        data_eps2 = np.linspace(0.1, 0.5, 10)
        
        # Create mock polynomial functions
        class MockPolynomial:
            def __call__(self, x):
                return np.ones_like(x) * 1.5
        
        mock_gold_material.fitted_data = {
            'wavelengths': wls,
            'data_eps1': data_eps1,
            'data_eps2': data_eps2,
            'fitted_eps1': MockPolynomial(),
            'fitted_eps2': MockPolynomial()
        }
        
        # Mock the material library
        solver._matlib = {'gold': mock_gold_material}
        solver._lam0 = solver.lam0  # Needed for the function
        solver._lenunit = 'um'  # Set the length unit
        
        # Create figure and axes for the test
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # Call the actual function
        result = display_fitted_permittivity(solver, axes)
        
        # Verify that the function returns a tuple of axes
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        # Check that axes were returned
        self.assertEqual(result[0], axes[0])
        self.assertEqual(result[1], axes[1])
        
        # Close the figure
        plt.close(fig)
    
    def test_no_dispersive_material(self):
        """Test behavior when no dispersive materials are present."""
        plt.close('all')
        # Redirect stdout to capture printed messages
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Create a simulation with no dispersive materials
            solver = create_solver(algorithm=Algorithm.RDIT)
            
            # Add a non-dispersive material
            silicon = create_material(name="Si", permittivity=11.7)
            solver.add_materials([silicon])
            solver.add_layer(material_name='Si', thickness=torch.tensor(0.2))
            
            # Make sure all layers are marked as non-dispersive
            for layer in solver.layer_manager.layers:
                layer.is_dispersive = False
            
            # Test the function
            result = display_fitted_permittivity(solver)
            
            # Verify that the function returns None
            self.assertIsNone(result)
            
            # Verify that the function printed the correct message
            self.assertEqual(captured_output.getvalue().strip(), "No dispersive material loaded.")
        finally:
            # Reset stdout
            sys.stdout = sys.__stdout__


if __name__ == '__main__':
    unittest.main() 