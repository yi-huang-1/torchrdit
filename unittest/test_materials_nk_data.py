"""Test suite for MaterialClass.from_nk_data method.

This module tests the correct handling of complex permittivity when creating
materials from refractive index (n) and extinction coefficient (k) data.
"""

import pytest
import torch
import numpy as np
from torchrdit.materials import MaterialClass


class TestFromNKData:
    """Test cases for MaterialClass.from_nk_data factory method."""
    
    def test_from_nk_data_preserves_complex_permittivity(self):
        """Test that from_nk_data correctly handles absorbing materials."""
        # Create material with n=1.5, k=0.1 (absorbing material)
        material = MaterialClass.from_nk_data("absorbing", n=1.5, k=0.1)
        
        # Calculate expected permittivity: ε = (n² - k²) + i(2nk)
        # Note: The constructor applies conj() for time convention, so imaginary part is negative
        expected_real = 1.5**2 - 0.1**2  # 2.24
        expected_imag = -(2 * 1.5 * 0.1)  # -0.3 (negative due to conj)
        
        # Test that complex permittivity is preserved
        assert torch.is_complex(material.er), "Permittivity should be complex"
        assert abs(material.er.real - expected_real) < 1e-6, f"Real part mismatch: {material.er.real} != {expected_real}"
        assert abs(material.er.imag - expected_imag) < 1e-6, f"Imaginary part mismatch: {material.er.imag} != {expected_imag}"
    
    def test_from_nk_data_lossless_material(self):
        """Test from_nk_data with k=0 (lossless material)."""
        # Create lossless material with n=2.0, k=0
        material = MaterialClass.from_nk_data("lossless", n=2.0, k=0.0)
        
        # For lossless: ε = n² + i0
        expected_real = 2.0**2  # 4.0
        expected_imag = 0.0
        
        assert torch.is_complex(material.er), "Permittivity should be complex even for lossless"
        assert abs(material.er.real - expected_real) < 1e-6
        assert abs(material.er.imag - expected_imag) < 1e-6
    
    def test_from_nk_data_metallic_material(self):
        """Test from_nk_data with typical metal values (large k)."""
        # Gold at 600nm: n≈0.2, k≈3.0
        material = MaterialClass.from_nk_data("gold", n=0.2, k=3.0)
        
        # ε = (n² - k²) + i(2nk)
        # Note: conj() applied for time convention
        expected_real = 0.2**2 - 3.0**2  # 0.04 - 9 = -8.96
        expected_imag = -(2 * 0.2 * 3.0)  # -1.2 (negative due to conj)
        
        assert torch.is_complex(material.er)
        assert abs(material.er.real - expected_real) < 1e-6
        assert abs(material.er.imag - expected_imag) < 1e-6
        # Verify negative real permittivity is allowed (important for metals)
        assert material.er.real < 0, "Metals should have negative real permittivity"
    
    def test_from_nk_data_perfect_conductor_limit(self):
        """Test from_nk_data with very large k (approaching perfect conductor)."""
        # Very large k simulating perfect conductor
        material = MaterialClass.from_nk_data("conductor", n=0.01, k=100.0)
        
        # ε = (n² - k²) + i(2nk)
        # Note: conj() applied for time convention
        expected_real = 0.01**2 - 100.0**2  # ≈ -10000
        expected_imag = -(2 * 0.01 * 100.0)  # -2.0 (negative due to conj)
        
        assert torch.is_complex(material.er)
        assert material.er.real < -9999, "Should have very large negative real part"
        assert abs(material.er.imag - expected_imag) < 1e-6
    
    def test_from_nk_data_default_permeability(self):
        """Test that default permeability is 1.0."""
        material = MaterialClass.from_nk_data("test", n=1.5, k=0.1)
        assert material.ur.item() == 1.0, "Default permeability should be 1.0"
    
    def test_from_nk_data_custom_permeability(self):
        """Test setting custom permeability."""
        material = MaterialClass.from_nk_data("test", n=1.5, k=0.1, permeability=2.5)
        assert material.ur.item() == 2.5, "Custom permeability should be preserved"
    
    def test_from_nk_data_name_preserved(self):
        """Test that material name is correctly set."""
        name = "my_special_material"
        material = MaterialClass.from_nk_data(name, n=1.5, k=0.1)
        assert material.name == name, f"Material name should be '{name}'"
    
    def test_from_nk_data_edge_case_n_equals_k(self):
        """Test edge case where n = k."""
        material = MaterialClass.from_nk_data("edge_case", n=1.0, k=1.0)
        
        # ε = (n² - k²) + i(2nk) = (1 - 1) + i(2) = 0 + 2i
        # Note: conj() applied for time convention
        expected_real = 0.0
        expected_imag = -2.0  # Negative due to conj
        
        assert abs(material.er.real - expected_real) < 1e-6
        assert abs(material.er.imag - expected_imag) < 1e-6
    
    def test_from_nk_data_very_small_values(self):
        """Test with very small n and k values."""
        material = MaterialClass.from_nk_data("vacuum_like", n=1.0001, k=1e-6)
        
        # Should handle small values without numerical issues
        assert torch.is_complex(material.er)
        assert material.er.real > 1.0, "Should be slightly above 1"
        assert material.er.imag < 0, "Should have small negative imaginary part (due to conj)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])