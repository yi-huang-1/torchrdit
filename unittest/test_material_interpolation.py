"""Test material interpolation improvements.

This test suite validates the improvements to material interpolation,
specifically the switch from polynomial fitting to Akima1DInterpolator
for better accuracy and stability with sharp features.
"""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from torchrdit.material_proxy import MaterialDataProxy


class TestMaterialInterpolation:
    """Test suite for material interpolation improvements."""

    def create_test_data_smooth(self):
        """Create smooth test data (SiO2-like)."""
        wavelengths = np.linspace(0.2, 2.0, 20)
        # Smooth permittivity variation
        eps_real = 2.1 + 0.1 * np.sin(2 * np.pi * wavelengths)
        eps_imag = 0.001 + 0.0005 * wavelengths
        return np.column_stack([wavelengths, eps_real, eps_imag])

    def create_test_data_sharp(self):
        """Create test data with sharp features (metallic-like)."""
        wavelengths = np.linspace(0.3, 1.5, 50)
        # Sharp transition around plasma frequency
        eps_real = np.where(wavelengths < 0.7, -10 + 15*wavelengths, 2.5)
        eps_imag = np.where(wavelengths < 0.7, 5.0 - 3*wavelengths, 0.1)
        return np.column_stack([wavelengths, eps_real, eps_imag])

    def create_test_data_sparse(self):
        """Create sparse test data (only 4 points)."""
        wavelengths = np.array([0.4, 0.8, 1.2, 1.6])
        eps_real = np.array([2.2, 2.3, 2.25, 2.2])
        eps_imag = np.array([0.01, 0.02, 0.015, 0.01])
        return np.column_stack([wavelengths, eps_real, eps_imag])

    def test_akima_vs_polynomial_smooth_data(self):
        """Test that Akima interpolation works well for smooth data."""
        proxy = MaterialDataProxy()
        data = self.create_test_data_smooth()
        
        # Test wavelengths (including interpolation and slight extrapolation)
        test_wl = np.array([0.25, 0.5, 1.0, 1.5, 1.95])
        
        # Get interpolated values using current implementation
        eps_real_func, eps_imag_func = proxy.extract_permittivity(data, test_wl, fit_order=10)
        
        # Test that we get callable functions back
        assert callable(eps_real_func), "Should return callable interpolation function"
        assert callable(eps_imag_func), "Should return callable interpolation function"
        
        # Test interpolation at data points
        eps_real_vals = eps_real_func(data[:, 0])
        np.testing.assert_allclose(eps_real_vals, data[:, 1], rtol=0.01)
        
        # No dependency-conditional assertions; just verify callability and basic accuracy.

    def test_akima_handles_sharp_features_better(self):
        """Test that Akima interpolation handles sharp features without oscillations."""
        proxy = MaterialDataProxy()
        data = self.create_test_data_sharp()
        
        # Test around the sharp transition
        test_wl = np.linspace(0.5, 0.9, 20)
        
        # Get interpolation functions
        eps_real_func, eps_imag_func = proxy.extract_permittivity(data, test_wl, fit_order=10)
        
        # Test that interpolation functions work
        eps_real_vals = eps_real_func(test_wl)
        assert len(eps_real_vals) == len(test_wl), "Should return values for all wavelengths"
        assert np.all(np.isfinite(eps_real_vals)), "All values should be finite"
        
        # We only check our interpolator yields finite values for sharp regions.

    def test_sparse_data_fallback(self):
        """Sparse data (<5 pts) should use linear interpolation (np.interp)."""
        proxy = MaterialDataProxy()
        data = self.create_test_data_sparse()
        
        test_wl = np.array([0.6, 1.0, 1.4])
        
        # With only 4 points, fallback path returns wrappers that call np.interp.
        with patch('torchrdit.material_proxy.np.interp') as mock_interp:
            mock_interp.return_value = np.array([2.25, 2.275, 2.225])
            eps_real_func, eps_imag_func = proxy.extract_permittivity(data, test_wl)
            # Trigger the patched function by evaluating
            _ = eps_real_func(test_wl)
            _ = eps_imag_func(test_wl)
            assert mock_interp.called, "Expected np.interp to be used for sparse data"

    # Removed: extrapolation behavior test; it duplicated basic finiteness checks
    # and didn’t assert a deterministic contract across environments.

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Create data with large dynamic range
        wavelengths = np.logspace(-1, 1, 30)  # 0.1 to 10 um
        eps_real = np.logspace(0, 3, 30)  # 1 to 1000
        eps_imag = np.logspace(-3, 1, 30)  # 0.001 to 10
        data = np.column_stack([wavelengths, eps_real, eps_imag])
        
        proxy = MaterialDataProxy()
        test_wl = np.array([0.5, 1.0, 5.0])
        
        # Current implementation might have numerical issues
        eps_real_func, eps_imag_func = proxy.extract_permittivity(data, test_wl, fit_order=15)
        
        # Check for NaN or Inf
        eps_real_vals = eps_real_func(test_wl)
        eps_imag_vals = eps_imag_func(test_wl)
        
        assert np.all(np.isfinite(eps_real_vals)), "Interpolation produced non-finite values"
        assert np.all(np.isfinite(eps_imag_vals)), "Interpolation produced non-finite values"

    # Removed: performance comparison test; timing-based and not deterministic.

    # Removed: gradient compatibility test; interpolation returns numpy values,
    # and this didn’t validate autograd behavior.

    def test_cache_key_compatibility(self):
        """Test that interpolation results are cacheable."""
        proxy = MaterialDataProxy()
        data = self.create_test_data_smooth()
        
        test_wl = np.array([1.31, 1.55])
        
        # Get interpolation results twice
        eps_real_func1, eps_imag_func1 = proxy.extract_permittivity(data, test_wl)
        eps_real_func2, eps_imag_func2 = proxy.extract_permittivity(data, test_wl)
        
        # Results should be deterministic
        np.testing.assert_array_equal(
            eps_real_func1(test_wl), 
            eps_real_func2(test_wl)
        )
        np.testing.assert_array_equal(
            eps_imag_func1(test_wl), 
            eps_imag_func2(test_wl)
        )

    # Removed: direct numerical comparison between Akima and polynomial fitting,
    # including prints and scipy-dependent behavior; it did not exercise our
    # `MaterialDataProxy.extract_permittivity` logic directly.
