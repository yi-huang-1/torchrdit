"""Test suite for get_coord function refactoring in GDS module.

This test file follows TDD principles - tests are written before implementation.
The tests expect a refactored structure with separate functions for Cartesian
and non-Cartesian coordinate systems.
"""

import unittest
import torch
import numpy as np
from torchrdit.gds import _get_cartesian_coord, _get_non_cartesian_coord, get_coord


class TestGetCoordRefactoring(unittest.TestCase):
    """Tests for refactored get_coord functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple Cartesian grid
        self.cart_X, self.cart_Y = torch.meshgrid(
            torch.linspace(-1, 1, 5),
            torch.linspace(-1, 1, 5),
            indexing='xy'
        )
        
        # Create non-Cartesian (hexagonal) grid
        a = 1.150
        t1 = torch.tensor([a/2, -a*np.sqrt(3)/2], dtype=torch.float32)
        t2 = torch.tensor([a/2, a*np.sqrt(3)/2], dtype=torch.float32)
        
        vec_p = torch.linspace(-0.5, 0.5, 5)
        vec_q = torch.linspace(-0.5, 0.5, 5)
        mesh_q, mesh_p = torch.meshgrid(vec_q, vec_p, indexing="xy")
        
        self.non_cart_X = mesh_p * t1[0] + mesh_q * t2[0]
        self.non_cart_Y = mesh_p * t1[1] + mesh_q * t2[1]
        
    def test_cartesian_coord_in_bounds(self):
        """Test _get_cartesian_coord with in-bounds indices."""
        x, y = _get_cartesian_coord(2, 3, self.cart_X, self.cart_Y)
        
        # Should return exact values from grid
        expected_x = self.cart_X[2, 3].item()
        expected_y = self.cart_Y[2, 3].item()
        
        self.assertAlmostEqual(x, expected_x, places=6)
        self.assertAlmostEqual(y, expected_y, places=6)
        
    def test_cartesian_coord_out_of_bounds(self):
        """Test _get_cartesian_coord with out-of-bounds indices."""
        # Test extrapolation beyond grid
        x, y = _get_cartesian_coord(10, 8, self.cart_X, self.cart_Y)
        
        # For Cartesian grid, should extrapolate linearly
        # Grid spacing is 0.5 in both directions
        dx = (self.cart_X[0, 1] - self.cart_X[0, 0]).item()
        dy = (self.cart_Y[1, 0] - self.cart_Y[0, 0]).item()
        
        expected_x = self.cart_X[0, 0].item() + 8 * dx
        expected_y = self.cart_Y[0, 0].item() + 10 * dy
        
        self.assertAlmostEqual(x, expected_x, places=6)
        self.assertAlmostEqual(y, expected_y, places=6)
        
    def test_non_cartesian_coord_in_bounds(self):
        """Test _get_non_cartesian_coord with in-bounds indices."""
        x, y = _get_non_cartesian_coord(2, 3, self.non_cart_X, self.non_cart_Y)
        
        # Should return exact values from grid
        expected_x = self.non_cart_X[2, 3].item()
        expected_y = self.non_cart_Y[2, 3].item()
        
        self.assertAlmostEqual(x, expected_x, places=6)
        self.assertAlmostEqual(y, expected_y, places=6)
        
    def test_non_cartesian_coord_out_of_bounds(self):
        """Test _get_non_cartesian_coord with out-of-bounds indices."""
        # Test extrapolation using parametric approach
        x, y = _get_non_cartesian_coord(10, 8, self.non_cart_X, self.non_cart_Y)
        
        # Should use parametric transformation
        n_rows, n_cols = self.non_cart_X.shape
        p = -0.5 + 1.0 * 10 / (n_rows - 1)
        q = -0.5 + 1.0 * 8 / (n_cols - 1)
        
        # Calculate expected based on lattice vectors
        t1_x = self.non_cart_X[0, 1].item() - self.non_cart_X[0, 0].item()
        t1_y = self.non_cart_Y[0, 1].item() - self.non_cart_Y[0, 0].item()
        t2_x = self.non_cart_X[1, 0].item() - self.non_cart_X[0, 0].item()
        t2_y = self.non_cart_Y[1, 0].item() - self.non_cart_Y[0, 0].item()
        
        # Scale by grid size
        t1_x *= (n_cols - 1)
        t1_y *= (n_cols - 1)
        t2_x *= (n_rows - 1)
        t2_y *= (n_rows - 1)
        
        expected_x = p * t2_x + q * t1_x
        expected_y = p * t2_y + q * t1_y
        
        self.assertAlmostEqual(x, expected_x, places=5)
        self.assertAlmostEqual(y, expected_y, places=5)
        
    def test_get_coord_dispatcher_cartesian(self):
        """Test get_coord correctly dispatches to Cartesian handler."""
        # get_coord should detect Cartesian grid and use appropriate function
        x, y = get_coord(2, 3, self.cart_X, self.cart_Y)
        x_expected, y_expected = _get_cartesian_coord(2, 3, self.cart_X, self.cart_Y)
        
        self.assertEqual(x, x_expected)
        self.assertEqual(y, y_expected)
        
    def test_get_coord_dispatcher_non_cartesian(self):
        """Test get_coord correctly dispatches to non-Cartesian handler."""
        # get_coord should detect non-Cartesian grid and use appropriate function
        x, y = get_coord(2, 3, self.non_cart_X, self.non_cart_Y)
        x_expected, y_expected = _get_non_cartesian_coord(2, 3, self.non_cart_X, self.non_cart_Y)
        
        self.assertEqual(x, x_expected)
        self.assertEqual(y, y_expected)
        
    def test_edge_case_single_element_grid(self):
        """Test handling of edge case with 1x1 grid."""
        X = torch.tensor([[1.5]])
        Y = torch.tensor([[2.5]])
        
        # Should return the single value for any index
        x, y = get_coord(0, 0, X, Y)
        self.assertEqual(x, 1.5)
        self.assertEqual(y, 2.5)
        
        # Even for out of bounds
        x, y = get_coord(5, 10, X, Y)
        self.assertEqual(x, 1.5)
        self.assertEqual(y, 2.5)
        
    def test_float_indices_handling(self):
        """Test that float indices are properly converted to integers."""
        # This can happen from boundary smoothing operations
        x, y = _get_cartesian_coord(2.7, 3.2, self.cart_X, self.cart_Y)
        
        # Should be same as integer indices
        x_int, y_int = _get_cartesian_coord(2, 3, self.cart_X, self.cart_Y)
        
        self.assertEqual(x, x_int)
        self.assertEqual(y, y_int)
        
    def test_negative_indices(self):
        """Test handling of negative indices."""
        # Should extrapolate correctly
        x, y = _get_cartesian_coord(-2, -3, self.cart_X, self.cart_Y)
        
        dx = (self.cart_X[0, 1] - self.cart_X[0, 0]).item()
        dy = (self.cart_Y[1, 0] - self.cart_Y[0, 0]).item()
        
        expected_x = self.cart_X[0, 0].item() + (-3) * dx
        expected_y = self.cart_Y[0, 0].item() + (-2) * dy
        
        self.assertAlmostEqual(x, expected_x, places=6)
        self.assertAlmostEqual(y, expected_y, places=6)


if __name__ == '__main__':
    unittest.main()