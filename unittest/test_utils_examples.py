"""Tests for utility functions examples in torchrdit/utils.py.

This module contains tests that verify the functionality shown in docstring examples
in the utils.py module. It ensures that the examples provided in the documentation
work as expected and serve as reliable references for users.

Classes:
    TestUtilsDocExamples: Test suite for utils.py docstring examples.
    
Examples:
    Run these tests with pytest:
    
    >>> python -m pytest unittest/test_utils_examples.py
    
Keywords:
    testing, docstring examples, utils, blur, projection, topology optimization, diagonal
"""

import unittest
import torch
import numpy as np

from torchrdit.utils import (
    _create_blur_kernel, 
    operator_blur, 
    operator_proj, 
    blur_filter,
    to_diag_util
)


class TestUtilsDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in utils.py.
    
    This test suite ensures that the examples provided in the docstrings of
    utils.py work as expected. The tests focus on the following utility functions:
    1. _create_blur_kernel: Creates normalized circular convolution kernels
    2. operator_blur: Blurs tensors using 2D convolution
    3. operator_proj: Projects values toward binary (0 or 1) values
    4. blur_filter: Combines blurring and projection for topology optimization
    5. to_diag_util: Converts vectors to diagonal matrices
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    
    Attributes:
        rho_2d: 4D tensor used for testing 2D blur operations.
        rho_3d: 4D tensor with multiple batches for testing batch operations.
        vec_2d: Vector used to test diagonal matrix creation.
        vec_3d: Multi-batch vector used to test batch diagonal matrix creation.
        kdim: Dimensions in Fourier space for diagonal matrix functions.
    
    Examples:
        >>> # Example of running a specific test
        >>> unittest.TextTestRunner().run(
        ...     unittest.TestLoader().loadTestsFromName(
        ...         'test_utils_examples.TestUtilsDocExamples.test_operator_blur_2d'
        ...     )
        ... )
    
    Keywords:
        unittest, blur, projection, filter, diagonal matrix, topology optimization
    """
    
    def setUp(self):
        """Set up test fixtures.
        
        This method is called before each test. It initializes test tensors and
        sets a fixed random seed for reproducibility.
        """
        # Use deterministic algorithms for reproducibility
        torch.manual_seed(0)
        
        # Create some test tensors
        self.rho_2d = torch.rand(1, 1, 5, 5)  # Format: [batch, channel, height, width] for operator_blur
        self.rho_3d = torch.rand(2, 1, 5, 5)  # 4D tensor with batch dimension and channel
        self.vec_2d = torch.randn(5)  # Vector to convert to diagonal matrix
        self.vec_3d = torch.randn(2, 5)  # Batched vector
        self.kdim = [1, 5]  # k dimensions for to_diag_util
    
    def test_create_blur_kernel(self):
        """Test the _create_blur_kernel function example.
        
        This test verifies that the circular blur kernel is created correctly with
        the expected dimensions, is normalized, and is symmetric.
        """
        # Create a blur kernel with radius 2
        kernel = _create_blur_kernel(2)
        
        # Verify kernel properties
        self.assertIsInstance(kernel, torch.Tensor)
        self.assertEqual(kernel.shape, (5, 5))  # 2*radius+1 = 5
        
        # Kernel should be normalized (sum to 1)
        self.assertAlmostEqual(kernel.sum().item(), 1.0, places=6)
        
        # Kernel should be symmetric
        self.assertTrue(torch.allclose(kernel, kernel.transpose(0, 1)))
        
        # Test with different radius
        kernel_r3 = _create_blur_kernel(3)
        self.assertEqual(kernel_r3.shape, (7, 7))  # 2*radius+1 = 7
        self.assertAlmostEqual(kernel_r3.sum().item(), 1.0, places=6)
    
    def test_operator_blur_2d(self):
        """Test the operator_blur function with 2D tensor example.
        
        This test verifies that the blur operation works correctly on a 2D tensor,
        maintaining the input shape while reducing variance.
        """
        # Get initial tensor with format [batch, channel, height, width]
        rho = self.rho_2d.clone()
        
        # Apply blur operator
        rho_blurred = operator_blur(rho, radius=1)
        
        # Verify properties
        self.assertEqual(rho_blurred.shape, rho.shape)
        self.assertFalse(torch.equal(rho_blurred, rho))  # Should be different
        
        # Check that the blurring reduces variance
        var_original = rho.var()
        var_blurred = rho_blurred.var()
        self.assertLessEqual(var_blurred, var_original * 1.1)  # Allow some tolerance
    
    def test_operator_blur_3d(self):
        """Test the operator_blur function with 3D tensor (batch) example.
        
        This test verifies that the blur operation works correctly on a tensor with
        multiple batches, maintaining the input shape while reducing variance for each batch.
        """
        # Get initial tensor with format [batch, channel, height, width]
        rho = self.rho_3d.clone()
        
        # Apply blur operator
        rho_blurred = operator_blur(rho, radius=1)
        
        # Verify properties
        self.assertEqual(rho_blurred.shape, rho.shape)
        
        # Check each batch independently
        for i in range(rho.shape[0]):
            # Blurring should retain or reduce variability
            var_original = rho[i].var()
            var_blurred = rho_blurred[i].var()
            self.assertLessEqual(var_blurred, var_original * 1.1)  # Allow some tolerance
    
    def test_operator_blur_multiple(self):
        """Test applying operator_blur multiple times example.
        
        This test verifies that applying the blur operation multiple times results
        in more blurring (greater variance reduction) than a single application.
        """
        # Get initial tensor with format [batch, channel, height, width]
        rho = self.rho_2d.clone()
        
        # Apply blur operator with different numbers of applications
        rho_blur_1 = operator_blur(rho, radius=1, num_blur=1)
        rho_blur_3 = operator_blur(rho, radius=1, num_blur=3)
        
        # Verify properties
        self.assertEqual(rho_blur_3.shape, rho.shape)
        
        # More applications should result in more blurring (lower variance)
        var_original = rho.var()
        var_blur_1 = rho_blur_1.var()
        var_blur_3 = rho_blur_3.var()
        
        # Check that multiple applications create more smoothing
        self.assertLessEqual(var_blur_3, var_blur_1 * 1.05)  # Allow some tolerance
    
    def test_operator_proj(self):
        """Test the operator_proj function example.
        
        This test verifies that the projection operation pushes values toward 0 or 1,
        with higher beta values resulting in more binary results.
        """
        # Create a test tensor with values between 0 and 1
        rho = torch.rand(5, 5)
        
        # Apply projection with different beta values
        beta_low = 1
        beta_high = 10
        
        rho_proj_low = operator_proj(rho, beta=beta_low)
        rho_proj_high = operator_proj(rho, beta=beta_high)
        
        # Verify properties
        self.assertEqual(rho_proj_low.shape, rho.shape)
        self.assertEqual(rho_proj_high.shape, rho.shape)
        
        # Higher beta should make projection more binary
        # (more values closer to 0 or 1)
        distance_to_binary_low = torch.minimum(rho_proj_low, 1 - rho_proj_low).mean()
        distance_to_binary_high = torch.minimum(rho_proj_high, 1 - rho_proj_high).mean()
        
        self.assertLess(distance_to_binary_high, distance_to_binary_low)
        
        # Should keep values in [0,1] range
        self.assertTrue(torch.all(rho_proj_high >= 0))
        self.assertTrue(torch.all(rho_proj_high <= 1))
    
    def test_operator_proj_threshold(self):
        """Test the operator_proj function with custom threshold example.
        
        This test verifies that the projection operation uses the eta parameter as
        a threshold, with higher eta values resulting in fewer values projected toward 1.
        It also tests that multiple applications of the projection result in more
        binary outcomes.
        """
        # Create a tensor with values between 0 and 1
        rho = torch.rand(5, 5)
        
        # Default threshold (eta) is 0.5
        rho_proj_default = operator_proj(rho, beta=5)
        
        # Use a custom threshold of 0.7
        rho_proj_custom = operator_proj(rho, eta=0.7, beta=5)
        
        # With higher threshold, fewer values should be projected to 1
        count_ones_default = (rho_proj_default > 0.9).sum()
        count_ones_custom = (rho_proj_custom > 0.9).sum()
        
        self.assertLessEqual(count_ones_custom, count_ones_default)
        
        # Test multiple projections 
        rho_proj_multiple = operator_proj(rho, beta=5, num_proj=3)
        
        # Multiple projections should make result more binary
        distance_single = torch.minimum(rho_proj_default, 1 - rho_proj_default).mean()
        distance_multiple = torch.minimum(rho_proj_multiple, 1 - rho_proj_multiple).mean()
        
        self.assertLessEqual(distance_multiple, distance_single)
    
    def test_blur_filter(self):
        """Test the blur_filter function example for topology optimization.
        
        This test verifies that the combined blur and projection filter works correctly,
        with higher beta values resulting in more binary results after filtering.
        It also tests custom parameters for eta and num_proj.
        """
        # Create a test tensor with format [batch, channel, height, width]
        rho = torch.rand(1, 1, 5, 5)
        
        # Apply blur filter with different settings
        # Low filtering
        rho_filtered_low = blur_filter(rho, radius=1, beta=1)
        
        # High filtering (strong blurring, strong projection)
        rho_filtered_high = blur_filter(rho, radius=2, beta=10)
        
        # Verify properties
        self.assertEqual(rho_filtered_low.shape, rho.shape)
        self.assertEqual(rho_filtered_high.shape, rho.shape)
        
        # Higher beta should make the result more binary
        distance_to_binary_low = torch.minimum(rho_filtered_low, 1 - rho_filtered_low).mean()
        distance_to_binary_high = torch.minimum(rho_filtered_high, 1 - rho_filtered_high).mean()
        
        self.assertLessEqual(distance_to_binary_high, distance_to_binary_low)
        
        # Test with custom eta and num_proj parameters
        rho_custom = blur_filter(rho, radius=1, beta=5, eta=0.7, num_proj=2)
        self.assertEqual(rho_custom.shape, rho.shape)
    
    def test_to_diag_util_single(self):
        """Test the to_diag_util function for single vector example.
        
        This test verifies that a vector is correctly converted to a diagonal matrix,
        with the vector values along the diagonal and zeros elsewhere.
        """
        # Get vector
        vec = self.vec_2d.clone()
        
        # Convert to diagonal matrix
        diag_mat = to_diag_util(vec, kdim=self.kdim)
        
        # Verify properties
        self.assertEqual(diag_mat.shape, (5, 5))
        
        # Check that diagonal elements match the vector
        for i in range(vec.shape[0]):
            self.assertEqual(diag_mat[i, i].item(), vec[i].item())
        
        # Off-diagonal elements should be zero
        off_diag_mask = ~torch.eye(vec.shape[0], dtype=bool)
        self.assertTrue(torch.all(diag_mat[off_diag_mask] == 0))
    
    def test_to_diag_util_batch(self):
        """Test the to_diag_util function for batched vectors example.
        
        This test verifies that batched vectors are correctly converted to batched
        diagonal matrices, with each vector's values along the diagonal of its
        corresponding matrix and zeros elsewhere.
        """
        # Get batched vector
        vec = self.vec_3d.clone()
        
        # Convert to diagonal matrix
        diag_mat = to_diag_util(vec, kdim=self.kdim)
        
        # Verify properties
        self.assertEqual(diag_mat.shape, (2, 5, 5))
        
        # Check each batch
        for b in range(vec.shape[0]):
            # Check diagonal elements
            for i in range(vec.shape[1]):
                self.assertEqual(diag_mat[b, i, i].item(), vec[b, i].item())
            
            # Off-diagonal elements should be zero
            off_diag_mask = ~torch.eye(vec.shape[1], dtype=bool)
            self.assertTrue(torch.all(diag_mat[b, off_diag_mask] == 0))
    
    def test_to_diag_util_complex(self):
        """Test the to_diag_util function with complex values example.
        
        This test verifies that complex vectors are correctly converted to complex
        diagonal matrices, preserving both real and imaginary components.
        """
        # Create a complex vector
        vec_real = torch.randn(5)
        vec_imag = torch.randn(5)
        vec_complex = torch.complex(vec_real, vec_imag)
        
        # Convert to diagonal matrix
        diag_mat = to_diag_util(vec_complex, kdim=self.kdim)
        
        # Verify properties
        self.assertEqual(diag_mat.shape, (5, 5))
        self.assertTrue(torch.is_complex(diag_mat))
        
        # Check diagonal elements
        for i in range(vec_complex.shape[0]):
            self.assertEqual(diag_mat[i, i].real.item(), vec_complex[i].real.item())
            self.assertEqual(diag_mat[i, i].imag.item(), vec_complex[i].imag.item())
    
    def test_to_diag_util_dual_polarization(self):
        """Test the to_diag_util function for dual polarization example.
        
        This test verifies that vectors with length 2*n_harmonics are correctly
        identified as dual polarization inputs and converted to appropriately sized
        diagonal matrices.
        """
        # Create a test vector with 2*n_harmonics elements (dual polarization)
        vec = torch.randn(10)  # 2 * n_harmonics (n_harmonics = 5)
        
        # Convert to diagonal matrix - will automatically detect dual polarization
        diag_mat = to_diag_util(vec, kdim=self.kdim)
        
        # Verify shape (should match the vector length for dual polarization)
        self.assertEqual(diag_mat.shape, (10, 10))
        
        # Check diagonal elements
        for i in range(vec.shape[0]):
            self.assertEqual(diag_mat[i, i].item(), vec[i].item())
        
        # With large number of harmonics
        large_kdim = [3, 3]  # 9 harmonics
        large_vec = torch.randn(18)  # 2 * (3*3) = 18 for dual polarization
        
        large_diag = to_diag_util(large_vec, kdim=large_kdim)
        self.assertEqual(large_diag.shape, (18, 18))


if __name__ == '__main__':
    unittest.main() 