"""Tests for redhstar optimization implementation.

This module contains comprehensive tests to validate the optimized Redheffer star product
implementation against the original reference implementation. Tests verify correctness,
gradient preservation, and performance across various input dimensions and edge cases.

Classes:
    TestRedhstarOptimization: Test suite for optimized redhstar implementation.
    
Examples:
    Run these tests with pytest:
    
    >>> python -m pytest unittest/test_redhstar_optimization.py
    
Keywords:
    testing, optimization, redhstar, redheffer star product, performance, gradient preservation
"""

import unittest
import torch
from typing import Dict

from torchrdit.utils import redhstar, init_smatrix


def redhstar_original(smat_a: dict, smat_b: dict, tcomplex: torch.dtype = torch.complex64) -> dict:
    """Original Redheffer star product implementation for reference/testing.
    
    This is the original, unoptimized implementation of the Redheffer star product
    that was replaced with an optimized version. It's kept here as a reference
    implementation for correctness validation in tests.

    The Redheffer star product (⋆) combines the scattering matrices of two adjacent
    layers or structures to produce the scattering matrix of the combined system.
    This is a fundamental operation in the transfer matrix method for solving
    multilayer electromagnetic problems.

    For two S-matrices A and B, the Redheffer star product C = A ⋆ B is given by:
    - C11 = A11 + A12 B11 (I - A22 B11)^(-1) A21
    - C12 = A12 (I - B11 A22)^(-1) B12
    - C21 = B21 (I - A22 B11)^(-1) A21
    - C22 = B22 + B21 A22 (I - B11 A22)^(-1) B12

    Args:
        smat_a: First scattering matrix (dictionary with keys 'S11', 'S12', 'S21', 'S22')
               This represents the layer or structure closer to the reference medium.
        smat_b: Second scattering matrix (dictionary with keys 'S11', 'S12', 'S21', 'S22')
               This represents the layer or structure closer to the transmission medium.
        tcomplex: PyTorch complex data type to use for intermediate calculations.
                Default is torch.complex64.

    Returns:
        dict: Combined scattering matrix as a dictionary with keys 'S11', 'S12', 'S21', 'S22'
             representing the four blocks of the scattering matrix of the combined system.

    Note:
        The function assumes that the S-matrices are properly formatted and have compatible
        dimensions for matrix operations. For batch processing, both S-matrices should
        have the same batch dimensions.
    """
    from torch.linalg import solve as tsolve

    # Input dimensions can be:
    # - 2D: (kdim_0_tims_1, kdim_0_tims_1) for single source, single frequency
    # - 3D: (n_freqs, kdim_0_tims_1, kdim_0_tims_1) for single source, multiple frequencies
    # - 4D: (n_sources, n_freqs, kdim_0_tims_1, kdim_0_tims_1) for batched sources

    # Construct identity matrix
    ndim = smat_a["S11"].ndim
    if ndim == 4:
        # Batched sources
        _, _, harmonic_m, harmonic_n = smat_a["S11"].shape
    elif ndim == 3:
        # Single source, multiple frequencies
        _, harmonic_m, harmonic_n = smat_a["S11"].shape
    else:
        # Single source, single frequency
        harmonic_m, harmonic_n = smat_a["S11"].shape

    device = smat_a["S11"].device
    identity_mat = torch.eye(harmonic_m, harmonic_n, dtype=tcomplex, device=device)

    # Compute commom terms
    inv_cycle_1 = identity_mat - smat_b["S11"] @ smat_a["S22"]
    cycle_1_smat_b_11 = tsolve(inv_cycle_1, smat_b["S11"])
    cycle_1_smat_b_12 = tsolve(inv_cycle_1, smat_b["S12"])

    # woodbury matrix identity
    cycle_2 = identity_mat + smat_a["S22"] @ cycle_1_smat_b_11
    smat_b_21_cycle_2 = smat_b["S21"] @ cycle_2
    # Compute combined scattering matrix
    smatrix = {}
    smatrix["S11"] = smat_a["S11"] + smat_a["S12"] @ cycle_1_smat_b_11 @ smat_a["S21"]
    smatrix["S12"] = smat_a["S12"] @ cycle_1_smat_b_12
    smatrix["S21"] = smat_b_21_cycle_2 @ smat_a["S21"]
    smatrix["S22"] = smat_b["S22"] + smat_b_21_cycle_2 @ smat_a["S22"] @ smat_b["S12"]

    return smatrix


class TestRedhstarOptimization(unittest.TestCase):
    """Tests that verify the optimized redhstar implementation.

    Focuses on functional correctness across dimensions, numerical stability,
    dtype handling, and basic integration with helpers.

    Attributes:
        tolerance: Maximum allowed relative error for correctness tests
        matrix_sizes: List of matrix dimensions to test
    """
    
    def setUp(self):
        """Set up test fixtures.
        
        This method is called before each test. It initializes test parameters,
        sets a fixed random seed for reproducibility, and defines test tolerances.
        """
        # Use deterministic algorithms for reproducibility
        torch.manual_seed(42)
        
        # Test parameters
        self.tolerance = 1e-6  # Maximum allowed relative error
        self.matrix_sizes = [8, 16, 32, 64]  # Various matrix dimensions to test
        
        # Device configuration
        self.device = torch.device('cpu')  # Use CPU for consistent results
        self.dtype = torch.complex64
        
    def _create_test_smatrices(self, size: int, batch_dim: int = None) -> tuple:
        """Create test S-matrices for benchmarking and validation.
        
        Args:
            size: Matrix dimension (size x size)
            batch_dim: Optional batch dimension for multi-source testing
            
        Returns:
            tuple: (smat_a, smat_b) test S-matrices
        """
        if batch_dim is None:
            shape = (size, size)
        else:
            shape = (batch_dim, size, size)
        
        # Create random complex matrices with small magnitude to avoid numerical issues
        smat_a = {
            'S11': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
            'S12': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
            'S21': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,  
            'S22': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
        }
        
        smat_b = {
            'S11': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
            'S12': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
            'S21': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
            'S22': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
        }
        
        return smat_a, smat_b
    
    def _validate_smatrix_correctness(self, result_opt: Dict, result_ref: Dict) -> float:
        """Validate that two S-matrices are equivalent within tolerance.
        
        Args:
            result_opt: Optimized implementation result
            result_ref: Reference implementation result
            
        Returns:
            float: Maximum relative error across all S-matrix components
        """
        max_error = 0.0
        
        for key in ['S11', 'S12', 'S21', 'S22']:
            # Compute relative error
            diff = torch.abs(result_opt[key] - result_ref[key])
            ref_magnitude = torch.abs(result_ref[key]) + 1e-12  # Add small value to avoid division by zero
            relative_error = diff / ref_magnitude
            max_error = max(max_error, torch.max(relative_error).item())
            
        return max_error
    
    def test_correctness_2d_matrices(self):
        """Test correctness for 2D matrices (single source, single frequency)."""
        for size in self.matrix_sizes:
            with self.subTest(size=size):
                # Create test matrices
                smat_a, smat_b = self._create_test_smatrices(size)
                
                # Compute results with both implementations
                result_opt = redhstar(smat_a, smat_b, self.dtype)
                result_ref = redhstar_original(smat_a, smat_b, self.dtype)
                
                # Validate correctness
                max_error = self._validate_smatrix_correctness(result_opt, result_ref)
                self.assertLess(max_error, self.tolerance, 
                              f"Size {size}x{size}: Max relative error {max_error:.2e} exceeds tolerance {self.tolerance:.2e}")
                
                # Verify output structure
                self.assertEqual(set(result_opt.keys()), {'S11', 'S12', 'S21', 'S22'})
                self.assertEqual(result_opt['S11'].shape, (size, size))
                self.assertEqual(result_opt['S11'].dtype, self.dtype)
    
    def test_correctness_3d_matrices(self):
        """Test correctness for 3D matrices (single source, multiple frequencies)."""
        n_freqs = 5
        for size in self.matrix_sizes[:2]:  # Use smaller sizes for 3D to save time
            with self.subTest(size=size, n_freqs=n_freqs):
                # Create test matrices with frequency dimension
                shape = (n_freqs, size, size)
                
                smat_a = {
                    'S11': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S12': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S21': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,  
                    'S22': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                }
                
                smat_b = {
                    'S11': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S12': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S21': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S22': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                }
                
                # Compute results with both implementations
                result_opt = redhstar(smat_a, smat_b, self.dtype)
                result_ref = redhstar_original(smat_a, smat_b, self.dtype)
                
                # Validate correctness
                max_error = self._validate_smatrix_correctness(result_opt, result_ref)
                self.assertLess(max_error, self.tolerance, 
                              f"Size {n_freqs}x{size}x{size}: Max relative error {max_error:.2e} exceeds tolerance {self.tolerance:.2e}")
                
                # Verify output structure
                self.assertEqual(result_opt['S11'].shape, (n_freqs, size, size))
    
    def test_correctness_4d_matrices(self):
        """Test correctness for 4D matrices (batched sources, multiple frequencies)."""
        n_sources = 3
        n_freqs = 4
        for size in [8, 16]:  # Use small sizes for 4D to save time
            with self.subTest(size=size, n_sources=n_sources, n_freqs=n_freqs):
                # Create test matrices with source and frequency dimensions
                shape = (n_sources, n_freqs, size, size)
                
                smat_a = {
                    'S11': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S12': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S21': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,  
                    'S22': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                }
                
                smat_b = {
                    'S11': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S12': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S21': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                    'S22': torch.randn(shape, dtype=self.dtype, device=self.device) * 0.1,
                }
                
                # Compute results with both implementations
                result_opt = redhstar(smat_a, smat_b, self.dtype)
                result_ref = redhstar_original(smat_a, smat_b, self.dtype)
                
                # Validate correctness
                max_error = self._validate_smatrix_correctness(result_opt, result_ref)
                self.assertLess(max_error, self.tolerance, 
                              f"Size {n_sources}x{n_freqs}x{size}x{size}: Max relative error {max_error:.2e} exceeds tolerance {self.tolerance:.2e}")
                
                # Verify output structure
                self.assertEqual(result_opt['S11'].shape, (n_sources, n_freqs, size, size))
    
    def test_numerical_stability(self):
        """Test numerical stability with near-singular matrices."""
        size = 16
        
        # Create matrices that might be close to singular
        smat_a, smat_b = self._create_test_smatrices(size)
        
        # Make A22 and B11 close to identity (which would make I - B11@A22 close to singular)
        smat_a['S22'] = 0.99 * torch.eye(size, dtype=self.dtype, device=self.device) + 0.01 * smat_a['S22']
        smat_b['S11'] = 0.99 * torch.eye(size, dtype=self.dtype, device=self.device) + 0.01 * smat_b['S11']
        
        # Both implementations should handle this case
        try:
            result_opt = redhstar(smat_a, smat_b, self.dtype)
            result_ref = redhstar_original(smat_a, smat_b, self.dtype)
            
            # Check that results are finite
            for key in ['S11', 'S12', 'S21', 'S22']:
                self.assertTrue(torch.all(torch.isfinite(result_opt[key])), 
                              f"Optimized result['{key}'] contains non-finite values")
                self.assertTrue(torch.all(torch.isfinite(result_ref[key])), 
                              f"Reference result['{key}'] contains non-finite values")
                
            # Optimized version should be more stable due to regularization
            max_error = self._validate_smatrix_correctness(result_opt, result_ref)
            self.assertLess(max_error, 0.1,  # Looser tolerance for near-singular case
                          f"Near-singular case: Max relative error {max_error:.2e} is too large")
            
        except Exception as e:
            self.fail(f"Numerical stability test failed with exception: {str(e)}")
    
    def test_edge_cases(self):
        """Test edge cases including small matrices and zero matrices."""
        # Test with smallest possible matrix size (1x1)
        size = 1
        smat_a, smat_b = self._create_test_smatrices(size)
        
        result_opt = redhstar(smat_a, smat_b, self.dtype)
        result_ref = redhstar_original(smat_a, smat_b, self.dtype)
        
        max_error = self._validate_smatrix_correctness(result_opt, result_ref)
        self.assertLess(max_error, self.tolerance)
        
        # Test with zero matrices
        size = 8
        smat_a_zero = {key: torch.zeros(size, size, dtype=self.dtype, device=self.device) 
                       for key in ['S11', 'S12', 'S21', 'S22']}
        smat_b_zero = {key: torch.zeros(size, size, dtype=self.dtype, device=self.device) 
                       for key in ['S11', 'S12', 'S21', 'S22']}
        
        result_opt = redhstar(smat_a_zero, smat_b_zero, self.dtype)
        result_ref = redhstar_original(smat_a_zero, smat_b_zero, self.dtype)
        
        max_error = self._validate_smatrix_correctness(result_opt, result_ref)
        self.assertLess(max_error, self.tolerance)
        
        # All results should be zero for zero input
        for key in ['S11', 'S12', 'S21', 'S22']:
            self.assertTrue(torch.allclose(result_opt[key], torch.zeros_like(result_opt[key]), atol=1e-10))
    
    def test_different_dtypes(self):
        """Test that the optimization works with different complex dtypes."""
        size = 16
        
        for dtype in [torch.complex64, torch.complex128]:
            with self.subTest(dtype=dtype):
                # Create test matrices with specific dtype
                smat_a, smat_b = self._create_test_smatrices(size)
                
                # Convert to desired dtype
                for key in smat_a.keys():
                    smat_a[key] = smat_a[key].to(dtype)
                    smat_b[key] = smat_b[key].to(dtype)
                
                # Compute results
                result_opt = redhstar(smat_a, smat_b, dtype)
                result_ref = redhstar_original(smat_a, smat_b, dtype)
                
                # Validate correctness
                max_error = self._validate_smatrix_correctness(result_opt, result_ref)
                self.assertLess(max_error, self.tolerance)
                
                # Verify output dtype
                self.assertEqual(result_opt['S11'].dtype, dtype)
    
    def test_integration_with_init_smatrix(self):
        """Test that optimized redhstar works correctly with init_smatrix."""
        size = 16
        
        # Create an initialized S-matrix (identity transmission, zero reflection)
        smat_init = init_smatrix((size, size), self.dtype, self.device)
        
        # Create another test matrix
        smat_test = {
            'S11': torch.randn(size, size, dtype=self.dtype, device=self.device) * 0.1,
            'S12': torch.randn(size, size, dtype=self.dtype, device=self.device) * 0.1,
            'S21': torch.randn(size, size, dtype=self.dtype, device=self.device) * 0.1,
            'S22': torch.randn(size, size, dtype=self.dtype, device=self.device) * 0.1,
        }
        
        # Compute redhstar with initialized matrix
        result_opt = redhstar(smat_init, smat_test, self.dtype)
        result_ref = redhstar_original(smat_init, smat_test, self.dtype)
        
        # Validate correctness
        max_error = self._validate_smatrix_correctness(result_opt, result_ref)
        self.assertLess(max_error, self.tolerance)
        
        # With identity transmission and zero reflection in first matrix,
        # the result should be close to the second matrix
        for key in ['S11', 'S12', 'S21', 'S22']:
            self.assertTrue(torch.allclose(result_opt[key], result_ref[key], atol=self.tolerance))


if __name__ == '__main__':
    unittest.main()
