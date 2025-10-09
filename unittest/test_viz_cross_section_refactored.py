"""
Comprehensive tests for cross-section visualization functionality in viz.py.

This test module validates the plot_cross_section function and its helper functions
following TDD principles. Tests cover basic functionality, edge cases, error handling,
and integration with the TorchRDIT solver system.
"""

import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
import os

# Add the src directory to the path for importing torchrdit
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torchrdit.viz import plot_cross_section, plot2d, plot_layer, display_fitted_permittivity


class MockLayer:
    """Mock layer class for testing."""
    
    def __init__(self, thickness=0.1, material_name="silicon", is_homogeneous=True, 
                 is_dispersive=False, ermat=None, bg_material=None):
        self.thickness = torch.tensor(thickness) if isinstance(thickness, (int, float)) else thickness
        self.material_name = material_name
        self.is_homogeneous = is_homogeneous
        self.is_dispersive = is_dispersive
        self.ermat = ermat
        if bg_material:
            self.bg_material = bg_material


class MockLayerManager:
    """Mock layer manager class for testing."""
    
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.ref_material_name = "air"  # Default reflection material
        self.trn_material_name = "air"  # Default transmission material
        self.nlayer = len(self.layers)


class MockSolver:
    """Mock solver class for testing."""
    
    def __init__(self, layers=None, layout_size=(64, 64), layout_extent=1.0):
        self.layer_manager = MockLayerManager(layers)
        self.layout_size = layout_size
        self.layout_extent = layout_extent
        self._lenunit = "um"
    
    def get_layout(self):
        """Mock get_layout method."""
        nx, ny = self.layout_size
        x = torch.linspace(-self.layout_extent/2, self.layout_extent/2, nx)
        y = torch.linspace(-self.layout_extent/2, self.layout_extent/2, ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return X, Y


class TestPlotCrossSection(unittest.TestCase):
    """Comprehensive test cases for plot_cross_section function."""
    
    def setUp(self):
        """Set up test fixtures."""
        plt.ioff()  # Turn off interactive plotting for tests
        
        # Create basic test solver with homogeneous layers
        layers = [
            MockLayer(thickness=0.1, material_name="silicon"),
            MockLayer(thickness=0.2, material_name="sio2"), 
            MockLayer(thickness=0.15, material_name="air")
        ]
        self.mock_solver = MockSolver(layers=layers)
        
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')  # Close all matplotlib figures
    
    def test_basic_xz_cross_section(self):
        """Test basic XZ cross-section visualization."""
        fig, ax = plt.subplots()
        
        result_ax = plot_cross_section(
            self.mock_solver, 
            plane='xz', 
            fig_ax=ax,
            title='Test XZ Cross-Section'
        )
        
        # Check that function returns the axis
        self.assertIs(result_ax, ax)
        
        # Check that labels are set correctly  
        self.assertEqual(ax.get_xlabel(), 'X (μm)')
        self.assertEqual(ax.get_ylabel(), 'Z (μm)')
        
        # Check title is set
        self.assertEqual(ax.get_title(), 'Test XZ Cross-Section')
        
        # Check aspect ratio is equal
        self.assertEqual(ax.get_aspect(), 1.0)
    
    def test_basic_yz_cross_section(self):
        """Test basic YZ cross-section visualization."""
        fig, ax = plt.subplots()
        
        result_ax = plot_cross_section(
            self.mock_solver,
            plane='yz',
            fig_ax=ax,
            title='Test YZ Cross-Section'
        )
        
        # Check that function returns the axis
        self.assertIs(result_ax, ax)
        
        # Check that labels are set correctly
        self.assertEqual(ax.get_xlabel(), 'Y (μm)')
        self.assertEqual(ax.get_ylabel(), 'Z (μm)')
        
        # Check title is set
        self.assertEqual(ax.get_title(), 'Test YZ Cross-Section')
    
    def test_invalid_plane_parameter(self):
        """Test error handling for invalid plane parameter."""
        with self.assertRaises(ValueError) as context:
            plot_cross_section(self.mock_solver, plane='xy')
        
        self.assertIn("Invalid plane 'xy'", str(context.exception))
        self.assertIn("Must be 'xz' or 'yz'", str(context.exception))
    
    def test_missing_layer_manager(self):
        """Test error handling when solver lacks layer_manager."""
        mock_solver_no_manager = Mock()
        del mock_solver_no_manager.layer_manager  # Remove layer_manager attribute
        
        with self.assertRaises(AttributeError) as context:
            plot_cross_section(mock_solver_no_manager)
        
        self.assertIn("layer_manager", str(context.exception))
    
    def test_missing_get_layout_method(self):
        """Test error handling when solver lacks get_layout method."""
        mock_solver_no_layout = Mock()
        mock_solver_no_layout.layer_manager = MockLayerManager()
        del mock_solver_no_layout.get_layout  # Remove get_layout method
        
        with self.assertRaises(AttributeError) as context:
            plot_cross_section(mock_solver_no_layout)
        
        self.assertIn("get_layout", str(context.exception))
    
    def test_custom_labels(self):
        """Test custom axis labels."""
        fig, ax = plt.subplots()
        custom_labels = ('Custom X', 'Custom Z')
        
        plot_cross_section(
            self.mock_solver,
            plane='xz',
            fig_ax=ax,
            labels=custom_labels
        )
        
        self.assertEqual(ax.get_xlabel(), custom_labels[0])
        self.assertEqual(ax.get_ylabel(), custom_labels[1])
    
    
    
    def test_auto_figure_creation(self):
        """Test automatic figure creation when fig_ax is None."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            # Configure mock_ax to have required methods and return values
            mock_ax.get_xlim.return_value = (0, 1)
            mock_ax.set_xlim = Mock()
            mock_ax.set_ylim = Mock()
            mock_ax.set_xticks = Mock()
            mock_ax.set_yticks = Mock()
            mock_ax.set_xlabel = Mock()
            mock_ax.set_ylabel = Mock()
            mock_ax.set_title = Mock()
            mock_ax.set_aspect = Mock()
            mock_ax.add_patch = Mock()
            mock_ax.axhline = Mock()
            mock_ax.legend = Mock()
            mock_ax.text = Mock()
            mock_ax.get_yaxis_transform = Mock()
            
            result_ax = plot_cross_section(self.mock_solver, plane='xz')
            
            # Check that subplots was called and axis was returned
            mock_subplots.assert_called_once()
            self.assertIs(result_ax, mock_ax)
    
    def test_custom_material_colormap(self):
        """Test custom material colormap."""
        fig, ax = plt.subplots()
        custom_colors = {
            'silicon': '#FF0000',  # Red
            'sio2': '#00FF00',     # Green
            'air': '#0000FF'       # Blue
        }
        
        plot_cross_section(
            self.mock_solver,
            plane='xz',
            fig_ax=ax,
            material_colormap=custom_colors
        )
        
        # Function should complete without errors
        self.assertIsNotNone(ax.get_title())
    
    def test_patterned_layers(self):
        """Test visualization of patterned (non-homogeneous) layers."""
        # Create pattern data
        pattern_data = torch.ones(16, 16) * 2.13  # SiO2 background
        pattern_data[4:12, 4:12] = 12.11  # Silicon pattern in center
        
        patterned_layer = MockLayer(
            thickness=0.2,
            material_name="silicon",
            is_homogeneous=False,
            ermat=pattern_data,
            bg_material="sio2"
        )
        
        solver = MockSolver(layers=[patterned_layer], layout_size=(16, 16))
        
        fig, ax = plt.subplots()
        plot_cross_section(solver, plane='xz', fig_ax=ax)
        
        # Function should complete without errors
        self.assertIsNotNone(ax.get_title())
    
    def test_dispersive_materials(self):
        """Test visualization with dispersive materials."""
        # Create frequency-dependent permittivity data
        freq_data = torch.stack([
            torch.ones(8, 8) * 2.1,   # Frequency 1
            torch.ones(8, 8) * 2.5,   # Frequency 2  
            torch.ones(8, 8) * 3.0    # Frequency 3
        ])
        
        dispersive_layer = MockLayer(
            thickness=0.1,
            material_name="dispersive_material",
            is_homogeneous=False,
            is_dispersive=True,
            ermat=freq_data
        )
        
        solver = MockSolver(layers=[dispersive_layer], layout_size=(8, 8))
        
        fig, ax = plt.subplots()
        plot_cross_section(solver, plane='xz', fig_ax=ax, frequency_index=1)
        
        # Function should complete without errors
        self.assertIsNotNone(ax.get_title())
    
    def test_show_semi_infinite_option(self):
        """Test semi-infinite region visualization option."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # With semi-infinite regions
        plot_cross_section(self.mock_solver, plane='xz', fig_ax=ax1, show_semi_infinite=True)
        
        # Without semi-infinite regions
        plot_cross_section(self.mock_solver, plane='xz', fig_ax=ax2, show_semi_infinite=False)
        
        # Both should succeed without errors
        self.assertIsNotNone(ax1.get_title())
        self.assertIsNotNone(ax2.get_title())
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Keep a minimal edge case to ensure simple stacks render
        single_layer = [MockLayer(thickness=0.1, material_name="test")]
        solver = MockSolver(layers=single_layer, layout_size=(4, 4))
        fig, ax = plt.subplots()
        plot_cross_section(solver, fig_ax=ax)
        self.assertIsNotNone(ax.get_title())
    
    def test_complex_layer_stack(self):
        """Test complex multi-layer stack with different material types."""
        # Create complex permittivity pattern
        ermat_pattern = torch.ones(32, 32) * 2.1316  # SiO2 background
        ermat_pattern[8:24, 8:24] = 12.1104  # Silicon square
        
        # Create dispersive permittivity data
        ermat_dispersive = torch.stack([
            torch.ones(32, 32) * (1.5 + 0.1j),  # Gold at freq 1
            torch.ones(32, 32) * (2.0 + 0.2j),  # Gold at freq 2
        ])
        
        layers = [
            MockLayer(thickness=0.1, material_name="substrate", is_homogeneous=True),
            MockLayer(thickness=0.2, material_name="patterned", is_homogeneous=False, ermat=ermat_pattern),
            MockLayer(thickness=0.15, material_name="gold", is_homogeneous=False, 
                     is_dispersive=True, ermat=ermat_dispersive),
            MockLayer(thickness=0.05, material_name="air", is_homogeneous=True)
        ]
        
        solver = MockSolver(layers=layers, layout_size=(32, 32))
        
        # Test both planes with various options
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Basic XZ and YZ
        plot_cross_section(solver, plane='xz', fig_ax=ax1, title='XZ Cross-Section')
        plot_cross_section(solver, plane='yz', fig_ax=ax2, title='YZ Cross-Section')
        
        # With custom options
        custom_colors = {'substrate': 'brown', 'patterned': 'blue', 'gold': 'gold', 'air': 'lightblue'}
        plot_cross_section(solver, plane='xz', fig_ax=ax3, material_colormap=custom_colors,
                          layer_boundaries=False, z_scale=2.0)
        
        # Dispersive materials
        plot_cross_section(solver, plane='yz', fig_ax=ax4, frequency_index=1, slice_position=0.1)
        
        # All should complete without errors
        for ax in [ax1, ax2, ax3, ax4]:
            self.assertIsNotNone(ax.get_title())


class TestPlot2D(unittest.TestCase):
    """Test cases for plot2d function."""
    
    def setUp(self):
        """Set up test fixtures."""
        plt.ioff()  # Turn off interactive plotting
        
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
    
    def test_basic_plot2d(self):
        """Test basic 2D plotting functionality."""
        # Create test data
        x = torch.linspace(-1, 1, 10)
        y = torch.linspace(-1, 1, 8)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        data = torch.sin(X) * torch.cos(Y)
        layout = (X, Y)
        
        fig, ax = plt.subplots()
        result_ax = plot2d(data, layout, fig_ax=ax, title='Test 2D Plot')
        
        self.assertIs(result_ax, ax)
        self.assertEqual(ax.get_title(), 'Test 2D Plot')
        self.assertEqual(ax.get_xlabel(), 'x')
        self.assertEqual(ax.get_ylabel(), 'y')
        self.assertEqual(ax.get_aspect(), 1.0)  # 'equal' aspect returns 1.0
    
    def test_plot2d_with_torch_tensor(self):
        """Test plot2d with PyTorch tensor input."""
        data = torch.ones(5, 5) * (1 + 1j)
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        layout = (X, Y)
        
        fig, ax = plt.subplots()
        result_ax = plot2d(data, layout, func='real', fig_ax=ax)
        
        self.assertIs(result_ax, ax)
    
    def test_plot2d_functions(self):
        """Test different function transformations."""
        data = torch.ones(4, 4) * (2 + 3j)
        x = torch.linspace(0, 1, 4)
        y = torch.linspace(0, 1, 4)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        layout = (X, Y)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Test different functions
        for ax, func in zip(axes, ['abs', 'real', 'imag', 'invalid']):
            plot2d(data, layout, func=func, fig_ax=ax, title=f'func={func}')
        
        # All should succeed (invalid defaults to abs)
        for ax in axes:
            self.assertIsNotNone(ax.get_title())
    
    def test_plot2d_with_outline(self):
        """Test plot2d with outline contour."""
        data = torch.ones(6, 6)
        outline = torch.zeros(6, 6)
        outline[2:4, 2:4] = 1  # Create a small square outline
        
        x = torch.linspace(0, 1, 6)
        y = torch.linspace(0, 1, 6)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        layout = (X, Y)
        
        fig, ax = plt.subplots()
        result_ax = plot2d(data, layout, outline=outline, fig_ax=ax, outline_alpha=0.7)
        
        self.assertIs(result_ax, ax)
    
    def test_plot2d_no_colorbar(self):
        """Test plot2d without colorbar."""
        data = torch.rand(5, 5)
        x = torch.linspace(0, 1, 5)
        y = torch.linspace(0, 1, 5)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        layout = (X, Y)
        
        fig, ax = plt.subplots()
        result_ax = plot2d(data, layout, fig_ax=ax, cbar=False)
        
        self.assertIs(result_ax, ax)
    
    def test_plot2d_auto_figure(self):
        """Test automatic figure creation."""
        data = torch.rand(3, 3)
        x = torch.linspace(0, 1, 3)
        y = torch.linspace(0, 1, 3)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        layout = (X, Y)
        
        result_ax = plot2d(data, layout)
        
        self.assertIsNotNone(result_ax)


class TestPlotLayer(unittest.TestCase):
    """Test cases for plot_layer function."""
    
    def setUp(self):
        """Set up test fixtures."""
        plt.ioff()  # Turn off interactive plotting
        
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
    
    def test_plot_layer_non_dispersive(self):
        """Test plotting non-dispersive layer."""
        # Create mock simulation with non-dispersive layer
        ermat = torch.ones(8, 8) * 2.25  # Silicon permittivity
        layer = MockLayer(ermat=ermat, is_dispersive=False)
        sim = MockSolver(layers=[layer], layout_size=(8, 8))
        
        fig, ax = plt.subplots()
        result_ax = plot_layer(sim, layer_index=0, fig_ax=ax, title='Non-dispersive Layer')
        
        self.assertIs(result_ax, ax)
        self.assertEqual(ax.get_title(), 'Non-dispersive Layer')
    
    def test_plot_layer_dispersive(self):
        """Test plotting dispersive layer."""
        # Create frequency-dependent permittivity
        ermat = torch.stack([
            torch.ones(6, 6) * 1.5,  # freq 0
            torch.ones(6, 6) * 2.0,  # freq 1
            torch.ones(6, 6) * 2.5   # freq 2
        ])
        layer = MockLayer(ermat=ermat, is_dispersive=True)
        sim = MockSolver(layers=[layer], layout_size=(6, 6))
        
        fig, ax = plt.subplots()
        result_ax = plot_layer(sim, layer_index=0, frequency_index=1, fig_ax=ax)
        
        self.assertIs(result_ax, ax)


class TestDisplayFunctions(unittest.TestCase):
    """Test cases for display_fitted_permittivity function."""
    
    def setUp(self):
        """Set up test fixtures."""
        plt.ioff()  # Turn off interactive plotting
        
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
    
    def test_display_fitted_permittivity_no_dispersive(self):
        """Test display_fitted_permittivity with no dispersive materials."""
        # Create solver with only non-dispersive layers
        layers = [MockLayer(is_dispersive=False, material_name="silicon")]
        sim = MockSolver(layers=layers)
        
        # Redirect stdout to capture print statement
        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output
        
        result = display_fitted_permittivity(sim)
        
        sys.stdout = sys.__stdout__  # Reset stdout
        
        self.assertIsNone(result)
        self.assertIn("No dispersive material loaded", captured_output.getvalue())
    
    def test_display_fitted_permittivity_with_dispersive(self):
        """Test display_fitted_permittivity with dispersive materials."""
        # Create mock simulation with dispersive material
        class MockMatLib:
            def __getitem__(self, key):
                return MockFittedData()
        
        class MockFittedData:
            fitted_data = {
                'wavelengths': np.linspace(0.4, 0.8, 10),
                'data_eps1': np.ones(10) * 2.0,
                'data_eps2': np.ones(10) * 0.1,
                'fitted_eps1': lambda x: np.ones_like(x) * 2.0,
                'fitted_eps2': lambda x: np.ones_like(x) * 0.1
            }
        
        layer = MockLayer(is_dispersive=True, material_name="gold")
        sim = MockSolver(layers=[layer])
        sim._matlib = MockMatLib()
        sim._lam0 = np.linspace(0.4, 0.8, 5)
        sim._lenunit = "um"
        
        fig, axes = plt.subplots(2, 1)
        result = display_fitted_permittivity(sim, fig_ax=axes)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # Returns tuple of two axes
        ax1, ax2 = result
        self.assertIsNotNone(ax1.get_title())
        self.assertEqual(ax1.get_xlabel(), "Wavelength [um]")
    


if __name__ == '__main__':
    unittest.main()
