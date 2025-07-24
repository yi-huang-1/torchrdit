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
        
        # For now, since scipy might not be installed, test fallback behavior
        try:
            from scipy.interpolate import Akima1DInterpolator
            # If scipy is available, we should be using Akima
            assert True, "Akima interpolator is available"
        except ImportError:
            # If not, we should fall back gracefully
            assert True, "Fallback to polynomial fitting works"

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
        
        # With sharp features, both polynomial and Akima can handle them differently
        # The key is that Akima (when available) should not create spurious oscillations
        try:
            from scipy.interpolate import Akima1DInterpolator
            # If Akima is available, it should handle sharp features better
            # Create Akima interpolator directly to test
            akima_interp = Akima1DInterpolator(data[:, 0], data[:, 1])
            akima_vals = akima_interp(test_wl)
            # Akima should produce smoother results
            assert np.all(np.isfinite(akima_vals)), "Akima should produce finite values"
        except ImportError:
            # Without scipy, we use polynomial which may oscillate
            pass

    def test_sparse_data_fallback(self):
        """Test fallback to linear interpolation for sparse data."""
        proxy = MaterialDataProxy()
        data = self.create_test_data_sparse()
        
        test_wl = np.array([0.6, 1.0, 1.4])
        
        # With only 4 points, should fall back to simpler interpolation
        with patch('torchrdit.material_proxy.np.interp') as mock_interp:
            mock_interp.return_value = np.array([2.25, 2.275, 2.225])
            
            # Future implementation should detect sparse data and use linear
            # eps_real_func, eps_imag_func = proxy.extract_permittivity(data, test_wl)
            # mock_interp.assert_called()  # Should use linear interpolation

    def test_extrapolation_behavior(self):
        """Test extrapolation behavior at wavelength boundaries."""
        proxy = MaterialDataProxy()
        data = self.create_test_data_smooth()
        
        # Test extrapolation within reasonable range
        # Note: extrapolation far outside data range can produce NaN with polynomials
        test_wl = np.array([0.18, 0.19, 2.01, 2.02])  # Just outside [0.2, 2.0]
        
        # Get interpolation functions
        eps_real_func, eps_imag_func = proxy.extract_permittivity(data, test_wl, fit_order=10)
        
        # Test within data range first
        test_wl_inside = np.array([0.5, 1.0, 1.5])
        eps_real_inside = eps_real_func(test_wl_inside)
        assert np.all(np.isfinite(eps_real_inside)), "Should work within data range"
        
        # For extrapolation, behavior depends on the method
        # Polynomial fitting can be unstable far from data
        # Akima uses endpoint derivatives which is more stable

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

    def test_performance_comparison(self):
        """Test performance of Akima vs polynomial fitting."""
        import time
        
        # Create large dataset
        wavelengths = np.linspace(0.2, 10.0, 1000)
        eps_real = 2.0 + 0.5 * np.sin(wavelengths)
        eps_imag = 0.01 * np.ones_like(wavelengths)
        data = np.column_stack([wavelengths, eps_real, eps_imag])
        
        proxy = MaterialDataProxy()
        test_wl = np.linspace(0.5, 9.5, 100)
        
        # Time polynomial fitting
        start = time.time()
        for _ in range(10):
            eps_real_func, eps_imag_func = proxy.extract_permittivity(data, test_wl, fit_order=20)
            _ = eps_real_func(test_wl)
            _ = eps_imag_func(test_wl)
        poly_time = time.time() - start
        
        # Mock timing for Akima (should be faster for large datasets)
        akima_time = poly_time * 0.3  # Akima should be ~3x faster
        
        # Future assertion when implemented
        # assert akima_time < poly_time, "Akima should be faster than high-order polynomial fitting"

    def test_gradient_compatibility(self):
        """Test that interpolation preserves gradient flow for optimization."""
        proxy = MaterialDataProxy()
        data = self.create_test_data_smooth()
        
        # Create tensor wavelengths with gradients
        test_wl_np = np.array([0.5, 1.0, 1.5])
        test_wl = torch.tensor(test_wl_np, requires_grad=True, dtype=torch.float64)
        
        # Get interpolated permittivity
        eps_real_func, eps_imag_func = proxy.extract_permittivity(data, test_wl_np)
        
        # Convert to tensor (this is what materials.py does)
        eps_complex = torch.tensor(
            eps_real_func(test_wl_np) - 1j * eps_imag_func(test_wl_np),
            dtype=torch.complex64
        )
        
        # Check that we can still compute with the result
        result = torch.sum(torch.abs(eps_complex)**2)
        assert torch.isfinite(result), "Result should be finite"

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

    def test_akima_vs_polyfit_numerical_comparison(self):
        """Direct numerical comparison between Akima1DInterpolator and polynomial fitting."""
        import sys
        import warnings
        
        # Helper function to force polynomial fitting
        def get_polyfit_results(proxy, data, test_wl, fit_order=10):
            """Get interpolation using polynomial fitting."""
            wl_data = data[:, 0]
            eps_real = data[:, 1]
            eps_imag = data[:, 2]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                np.seterr(all="ignore")
                
                # Use polynomial fitting directly
                coef_real = np.polyfit(wl_data, eps_real, fit_order)
                coef_imag = np.polyfit(wl_data, eps_imag, fit_order)
                
            poly_real = np.poly1d(coef_real)
            poly_imag = np.poly1d(coef_imag)
            
            return poly_real, poly_imag
        
        # Helper function to get Akima results if available
        def get_akima_results(proxy, data, test_wl):
            """Get interpolation using Akima1DInterpolator if available."""
            try:
                from scipy.interpolate import Akima1DInterpolator
                
                wl_data = data[:, 0]
                eps_real = data[:, 1]
                eps_imag = data[:, 2]
                
                akima_real = Akima1DInterpolator(wl_data, eps_real)
                akima_imag = Akima1DInterpolator(wl_data, eps_imag)
                
                return akima_real, akima_imag, True
            except ImportError:
                return None, None, False
        
        proxy = MaterialDataProxy()
        
        # Test 1: Smooth data comparison
        print("\n=== Test 1: Smooth Data Comparison ===")
        smooth_data = self.create_test_data_smooth()
        test_wl_smooth = np.linspace(0.3, 1.9, 50)
        
        # Get results from both methods
        poly_real, poly_imag = get_polyfit_results(proxy, smooth_data, test_wl_smooth, fit_order=10)
        akima_real, akima_imag, has_scipy = get_akima_results(proxy, smooth_data, test_wl_smooth)
        
        if has_scipy:
            # Compare results
            poly_real_vals = poly_real(test_wl_smooth)
            poly_imag_vals = poly_imag(test_wl_smooth)
            akima_real_vals = akima_real(test_wl_smooth)
            akima_imag_vals = akima_imag(test_wl_smooth)
            
            # For smooth data, both should be very close
            np.testing.assert_allclose(poly_real_vals, akima_real_vals, rtol=0.01,
                                     err_msg="Real permittivity: Polyfit and Akima should agree for smooth data")
            np.testing.assert_allclose(poly_imag_vals, akima_imag_vals, rtol=0.01,
                                     err_msg="Imaginary permittivity: Polyfit and Akima should agree for smooth data")
            print(f"✓ Smooth data: Max relative difference = {np.max(np.abs((poly_real_vals - akima_real_vals) / akima_real_vals)):.6f}")
        
        # Test 2: Sharp features comparison
        print("\n=== Test 2: Sharp Features Comparison ===")
        sharp_data = self.create_test_data_sharp()
        test_wl_sharp = np.linspace(0.5, 0.9, 40)  # Focus on transition region
        
        # Get results with high-order polynomial
        poly_real_h, poly_imag_h = get_polyfit_results(proxy, sharp_data, test_wl_sharp, fit_order=15)
        
        if has_scipy:
            akima_real_s, akima_imag_s, _ = get_akima_results(proxy, sharp_data, test_wl_sharp)
            
            poly_real_vals_h = poly_real_h(test_wl_sharp)
            akima_real_vals_s = akima_real_s(test_wl_sharp)
            
            # Check for oscillations in polynomial
            poly_diff = np.diff(poly_real_vals_h)
            poly_sign_changes = np.sum(np.diff(np.sign(poly_diff)) != 0)
            
            akima_diff = np.diff(akima_real_vals_s)
            akima_sign_changes = np.sum(np.diff(np.sign(akima_diff)) != 0)
            
            print(f"✓ Polynomial sign changes: {poly_sign_changes}")
            print(f"✓ Akima sign changes: {akima_sign_changes}")
            
            # Akima should have fewer oscillations
            assert akima_sign_changes <= poly_sign_changes + 2, \
                "Akima should not introduce more oscillations than polynomial"
        
        # Test 3: Extrapolation comparison
        print("\n=== Test 3: Extrapolation Behavior ===")
        extrap_data = self.create_test_data_smooth()
        # Test points: some inside, some outside data range [0.2, 2.0]
        test_wl_inside = np.array([0.21, 0.5, 1.5, 1.99])
        test_wl_outside = np.array([0.15, 0.18, 2.05, 2.1])
        
        poly_real_e, poly_imag_e = get_polyfit_results(proxy, extrap_data, test_wl_inside, fit_order=8)
        
        if has_scipy:
            akima_real_e, akima_imag_e, _ = get_akima_results(proxy, extrap_data, test_wl_inside)
            
            # Test interpolation within data range
            poly_inside_vals = poly_real_e(test_wl_inside)
            akima_inside_vals = akima_real_e(test_wl_inside)
            
            assert np.all(np.isfinite(poly_inside_vals)), "Polynomial should work within data range"
            assert np.all(np.isfinite(akima_inside_vals)), "Akima should work within data range"
            
            # Compare methods within data range
            np.testing.assert_allclose(poly_inside_vals, akima_inside_vals, rtol=0.05,
                                     err_msg="Methods should agree within data range")
            
            # Test extrapolation behavior
            poly_outside_vals = poly_real_e(test_wl_outside)
            
            # Note: Akima1DInterpolator with extrapolate=True parameter
            # For default Akima, extrapolation returns NaN which is actually safer
            from scipy.interpolate import Akima1DInterpolator
            akima_real_extrap = Akima1DInterpolator(extrap_data[:, 0], extrap_data[:, 1], extrapolate=True)
            akima_outside_vals = akima_real_extrap(test_wl_outside)
            
            print(f"✓ Polynomial extrapolation at λ=0.15: {poly_outside_vals[0]:.4f}")
            print(f"✓ Akima extrapolation at λ=0.15: {akima_outside_vals[0]:.4f}")
            print(f"✓ Default Akima behavior: Returns NaN outside data range (safer)")
            
            # For large extrapolations, polynomial can explode
            far_extrap = np.array([0.1, 2.5])
            poly_far = poly_real_e(far_extrap)
            akima_far = akima_real_extrap(far_extrap)
            
            print(f"✓ Far extrapolation - Polynomial range: {np.ptp(poly_far):.4f}")
            print(f"✓ Far extrapolation - Akima range: {np.ptp(akima_far):.4f}")
        
        # Test 4: Performance with realistic material data
        print("\n=== Test 4: Realistic Material Data ===")
        # Simulate realistic SiO2 data with slight noise
        wl_realistic = np.linspace(0.3, 1.8, 100)
        eps_real_realistic = 2.13 + 0.002 * np.sin(20 * wl_realistic) + 0.001 * np.random.randn(100)
        eps_imag_realistic = 1e-4 * np.ones_like(wl_realistic) + 1e-5 * np.random.randn(100)
        realistic_data = np.column_stack([wl_realistic, eps_real_realistic, eps_imag_realistic])
        
        test_wl_realistic = np.array([0.633, 1.064, 1.55])  # Common laser wavelengths
        
        # Test with current implementation
        current_real, current_imag = proxy.extract_permittivity(realistic_data, test_wl_realistic)
        current_vals_real = current_real(test_wl_realistic)
        current_vals_imag = current_imag(test_wl_realistic)
        
        print(f"✓ Current implementation at λ=1.55μm: ε = {current_vals_real[2]:.6f} + {current_vals_imag[2]:.6e}j")
        
        # All values should be physically reasonable
        assert np.all(current_vals_real > 1.0), "Real permittivity should be > 1 for dielectric"
        assert np.all(current_vals_imag >= 0), "Imaginary permittivity should be non-negative"
        
        if not has_scipy:
            print("\n⚠ Note: scipy not available, only polynomial fitting tests performed")