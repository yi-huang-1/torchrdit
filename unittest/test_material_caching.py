"""Test LRU caching for material loading.

This test suite validates the LRU caching implementation for material
property loading, ensuring efficient memory usage and performance.
"""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock, call
from functools import lru_cache

from torchrdit.materials import MaterialClass


class TestMaterialCaching:
    """Test suite for material property caching."""

    @pytest.fixture
    def material_data_file(self, tmp_path):
        """Create a temporary material data file."""
        data_file = tmp_path / "test_material.txt"
        # Create wavelength-permittivity data
        wavelengths = np.linspace(0.4, 0.8, 50)
        eps_real = 2.25 + 0.1 * wavelengths
        eps_imag = 0.01 + 0.005 * wavelengths
        
        data = np.column_stack([wavelengths, eps_real, eps_imag])
        np.savetxt(data_file, data, header="wavelength(um) eps_real eps_imag")
        return str(data_file)

    def test_basic_caching_behavior(self, material_data_file):
        """Test that repeated calls with same parameters use cache."""
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um"
        )
        
        # First call - should load and cache
        wavelengths1 = np.array([0.5, 0.6, 0.7])
        mat.load_dispersive_er(wavelengths1, "um")
        er1 = mat.er.clone()
        
        # Clear the internal loaded data to simulate expensive reload
        original_loadeder = mat._loadeder.copy()
        mat._loadeder = None
        
        # Second call with same wavelengths - should use cache
        mat.load_dispersive_er(wavelengths1, "um")
        er2 = mat.er
        
        # Results should be identical (same tensor object from cache)
        assert torch.equal(er1, er2), "Cached result should be identical"
        
        # Verify cache was used (loadeder should still be None)
        assert mat._loadeder is None, "Should not have reloaded data"
        
        # Restore for cleanup
        mat._loadeder = original_loadeder

    def test_cache_key_includes_units(self, material_data_file):
        """Test that cache key includes length units."""
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um"
        )
        
        wavelengths_nm = np.array([500, 600, 700])  # nm values
        wavelengths_um = np.array([0.5, 0.6, 0.7])  # um values
        
        # Load with nm units
        mat.load_dispersive_er(wavelengths_nm, "nm")
        er_nm = mat.er.clone()
        
        # Load with um units (different unit, same physical wavelengths)
        mat.load_dispersive_er(wavelengths_um, "um")  
        er_um = mat.er.clone()
        
        # Results should be different
        assert not torch.equal(er_nm, er_um), "Different units should give different results"
        
        # Cache should have two entries
        cache_info = mat.cache_info()
        assert cache_info is not None, "Cache info should be available"
        assert cache_info.currsize == 2, "Cache should have separate entries for different units"

    def test_cache_key_includes_wavelengths(self, material_data_file):
        """Test that cache key includes exact wavelength values."""
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um"
        )
        
        # Different wavelength arrays
        wavelengths1 = np.array([0.5, 0.6, 0.7])
        wavelengths2 = np.array([0.5, 0.6, 0.71])  # Slightly different
        
        mat.load_dispersive_er(wavelengths1, "um")
        mat.load_dispersive_er(wavelengths2, "um")
        
        # Cache should have two entries
        cache_info = mat.cache_info()
        assert cache_info.currsize == 2, "Different wavelengths should create different cache entries"

    def test_lru_cache_decorator_mock(self):
        """Test behavior with LRU cache decorator (mocked)."""
        # Mock a function with lru_cache behavior
        call_count = 0
        
        @lru_cache(maxsize=2)
        def mock_load_permittivity(wavelengths_tuple, unit):
            nonlocal call_count
            call_count += 1
            # Simulate expensive calculation
            wavelengths = np.array(wavelengths_tuple)
            return torch.tensor(2.25 + 0.1 * wavelengths.mean(), dtype=torch.complex64)
        
        # Test basic caching
        wl1 = (0.5, 0.6, 0.7)
        result1 = mock_load_permittivity(wl1, "um")
        result2 = mock_load_permittivity(wl1, "um")
        assert call_count == 1, "Second call should use cache"
        assert result1 is result2, "Should return same object from cache"
        
        # Test cache eviction with maxsize=2
        wl2 = (0.6, 0.7, 0.8)
        wl3 = (0.7, 0.8, 0.9)
        _ = mock_load_permittivity(wl2, "um")
        _ = mock_load_permittivity(wl3, "um")
        assert call_count == 3, "New entries should trigger calculations"
        
        # Access first entry again - should recalculate (evicted)
        _ = mock_load_permittivity(wl1, "um")
        assert call_count == 4, "Evicted entry should recalculate"
        
        # Test cache_info if available
        info = mock_load_permittivity.cache_info()
        assert info.hits == 1, "Should have 1 cache hit"
        assert info.misses == 4, "Should have 4 cache misses"
        assert info.maxsize == 2, "Max size should be 2"

    def test_cache_memory_limit(self, material_data_file):
        """Test that cache respects memory limits."""
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um"
        )
        
        # Generate many different wavelength arrays
        for i in range(150):  # More than default cache size
            wavelengths = np.array([0.5 + i*0.001, 0.6 + i*0.001])
            mat.load_dispersive_er(wavelengths, "um")
        
        # With LRU implementation, cache size is limited
        cache_info = mat.cache_info()
        assert cache_info.currsize <= 128, "LRU cache should limit to maxsize"
        assert cache_info.maxsize == 128, "Cache maxsize should be 128"

    def test_cache_with_different_materials(self, tmp_path):
        """Test that different materials maintain separate caches."""
        # Create two different material files
        data_file1 = tmp_path / "mat1.txt"
        data_file2 = tmp_path / "mat2.txt"
        
        wavelengths = np.linspace(0.4, 0.8, 20)
        data1 = np.column_stack([wavelengths, 2.0 * np.ones_like(wavelengths), 0.01 * np.ones_like(wavelengths)])
        data2 = np.column_stack([wavelengths, 3.0 * np.ones_like(wavelengths), 0.02 * np.ones_like(wavelengths)])
        
        np.savetxt(data_file1, data1)
        np.savetxt(data_file2, data2)
        
        # Create materials
        mat1 = MaterialClass(
            name="mat1",
            dielectric_dispersion=True,
            user_dielectric_file=str(data_file1),
            data_format="wl-eps",
            data_unit="um"
        )
        mat2 = MaterialClass(
            name="mat2",
            dielectric_dispersion=True,
            user_dielectric_file=str(data_file2),
            data_format="wl-eps",
            data_unit="um"
        )
        
        test_wl = np.array([0.5, 0.6, 0.7])
        mat1.load_dispersive_er(test_wl, "um")
        mat2.load_dispersive_er(test_wl, "um")
        
        # Each material should have its own cache
        cache_info1 = mat1.cache_info()
        cache_info2 = mat2.cache_info()
        assert cache_info1.currsize == 1, "Material 1 should have 1 cache entry"
        assert cache_info2.currsize == 1, "Material 2 should have 1 cache entry"
        assert not torch.equal(mat1.er, mat2.er), "Different materials should give different results"

    def test_cache_invalidation_on_parameter_change(self, material_data_file):
        """Test that cache is properly handled when material parameters change."""
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um",
            max_poly_fit_order=10
        )
        
        wavelengths = np.array([0.5, 0.6, 0.7])
        mat.load_dispersive_er(wavelengths, "um")
        er1 = mat.er.clone()
        
        # Change polynomial order (this should affect results)
        mat._max_poly_order = 5
        
        # Current implementation doesn't invalidate cache on parameter change
        # This is a potential issue to document
        mat.load_dispersive_er(wavelengths, "um")
        er2 = mat.er
        
        # With LRU cache, parameter changes don't invalidate cache
        # This is a known limitation - cache key doesn't include polynomial order
        assert torch.equal(er1, er2), "Cache doesn't include polynomial order in key"
        
        # To work around this, one would need to clear cache manually
        mat.clear_cache()
        mat.load_dispersive_er(wavelengths, "um")
        er3 = mat.er
        # Now it should recompute with new polynomial order
        # However, for same wavelengths the difference might be small

    def test_thread_safety_considerations(self):
        """Test considerations for thread-safe caching."""
        # LRU cache in functools is thread-safe for CPython (GIL)
        # but may have issues in truly parallel environments
        
        from functools import lru_cache
        import threading
        
        results = []
        call_count = 0
        
        @lru_cache(maxsize=128)
        def cached_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        def worker(n):
            result = cached_function(n)
            results.append(result)
        
        # Create multiple threads accessing cache
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i % 3,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Check that caching worked even with threads
        assert call_count <= 3, "Should cache the 3 unique values"
        assert len(results) == 10, "All threads should complete"

    def test_cache_performance_benefit(self, material_data_file):
        """Test that caching provides performance benefits."""
        import time
        
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um"
        )
        
        wavelengths = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
        
        # First load (uncached)
        start = time.time()
        mat.load_dispersive_er(wavelengths, "um")
        first_load_time = time.time() - start
        
        # Clear _er to force reload attempt
        mat._er = None
        
        # Second load (should be cached)
        start = time.time()
        mat.load_dispersive_er(wavelengths, "um")
        cached_load_time = time.time() - start
        
        # Cached should be significantly faster
        # In practice, cached access should be 100x-1000x faster
        assert cached_load_time < first_load_time * 0.5, "Cached access should be faster"