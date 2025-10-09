"""Tests for LRU caching in material loading.

Focus on verifying observable caching behavior via the public API
(`load_dispersive_er`, `er`, and `cache_info`). Avoids testing
stdlib `lru_cache` itself or performance/timing characteristics.
"""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock, call

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
        """Repeated calls with the same inputs hit the cache."""
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um"
        )
        wavelengths = np.array([0.5, 0.6, 0.7])

        # First load populates cache (miss)
        mat.load_dispersive_er(wavelengths, "um")
        er_first = mat.er.clone()
        info_before = mat.cache_info()

        # Second load with identical inputs should be a cache hit
        mat.load_dispersive_er(wavelengths, "um")
        er_second = mat.er
        info_after = mat.cache_info()

        assert torch.equal(er_first, er_second), "Cached value should match"
        assert info_after.hits == info_before.hits + 1, "Expected one cache hit"

    def test_cache_key_includes_units(self, material_data_file):
        """Cache key differentiates based on length units."""
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um"
        )
        wavelengths_nm = np.array([500, 600, 700])  # nm values
        wavelengths_um = np.array([0.5, 0.6, 0.7])  # um values

        # Two loads with physically identical wavelengths but different units
        mat.load_dispersive_er(wavelengths_nm, "nm")
        mat.load_dispersive_er(wavelengths_um, "um")

        # Cache should hold separate entries for different units
        cache_info = mat.cache_info()
        assert cache_info is not None
        assert cache_info.currsize == 2, "Separate entries expected for different length units"

        # Loading again with one of the units should result in a cache hit
        hits_before = cache_info.hits
        mat.load_dispersive_er(wavelengths_nm, "nm")
        assert mat.cache_info().hits == hits_before + 1

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

    # Removed: mock tests of stdlib lru_cache; they don't exercise our code.

    def test_cache_memory_limit(self, material_data_file):
        """Cache respects the configured max size (LRU 128)."""
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um"
        )
        # Generate more unique keys than maxsize to fill the cache
        for i in range(130):  # Slightly over the default cache size
            wavelengths = np.array([0.5 + i * 1e-3, 0.6 + i * 1e-3])
            mat.load_dispersive_er(wavelengths, "um")

        cache_info = mat.cache_info()
        assert cache_info.maxsize == 128
        assert cache_info.currsize == 128, "LRU cache should cap at maxsize"

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

    # Removed: parameter-change behavior test; it asserts a known limitation
    # rather than validating correct behavior. Consider adding an xfail if needed.

    # Removed: generic thread-safety test of stdlib cache; not specific to materials.

    # Removed: performance/timing test; flaky across environments.

    def test_cache_invalidation_on_parameter_change(self, material_data_file):
        """Changing fit parameters should produce a distinct cache entry (desired).

        Desired behavior: when a fit-affecting parameter changes (e.g., polynomial
        order), the cache key should differ so the subsequent call is a miss that
        creates a new entry rather than a hit on the old entry.
        """
        mat = MaterialClass(
            name="test_mat",
            dielectric_dispersion=True,
            user_dielectric_file=material_data_file,
            data_format="wl-eps",
            data_unit="um",
            max_poly_fit_order=10,
        )

        wavelengths = np.array([0.5, 0.6, 0.7])

        mat.load_dispersive_er(wavelengths, "um")
        size_before = mat.cache_info().currsize

        # Change a parameter that affects fitting and therefore results
        mat._max_poly_order = 5

        # Expected: cache miss with a new distinct key (size increases)
        mat.load_dispersive_er(wavelengths, "um")
        size_after = mat.cache_info().currsize
        assert size_after == size_before + 1, "Parameter change should create a new cache entry"
