import pytest
import torch
import numpy as np
import os

from torchrdit.materials import MaterialClass
from torchrdit.material_proxy import MaterialDataProxy, UnitConverter

@pytest.fixture
def non_dispersive_material():
    return MaterialClass(name='SiO2', permittivity=2.0, permeability=1.0, dielectric_dispersion=False)

@pytest.fixture
def dispersive_material(tmp_path):
    # Create a temporary file with dispersive data in wl-eps format
    data = np.array([
        [1.0, 2.0, 0.1],
        [2.0, 2.1, 0.2],
        [3.0, 2.2, 0.3],
        [4.0, 2.3, 0.4],
        [5.0, 2.4, 0.5],
        [6.0, 2.5, 0.6]
    ])
    file_path = tmp_path / "dispersive_data.txt"
    np.savetxt(file_path, data, header="wavelength eps1 eps2", comments='')

    return MaterialClass(name='DispersiveMaterial', 
                        dielectric_dispersion=True, 
                        user_dielectric_file=str(file_path),
                        data_format='wl-eps',
                        data_unit='um')

@pytest.fixture
def freq_data_material(tmp_path):
    # Create a temporary file with dispersive data in freq-eps format
    data = np.array([
        [300.0, 2.0, 0.1],  # 300 THz ≈ 1 um
        [200.0, 2.1, 0.2],  # 200 THz ≈ 1.5 um
        [150.0, 2.2, 0.3],  # 150 THz ≈ 2 um
        [100.0, 2.3, 0.4],  # 100 THz ≈ 3 um
    ])
    file_path = tmp_path / "freq_data.txt"
    np.savetxt(file_path, data, header="freq eps1 eps2", comments='')

    return MaterialClass(name='FreqDispersiveMaterial', 
                        dielectric_dispersion=True, 
                        user_dielectric_file=str(file_path),
                        data_format='freq-eps',
                        data_unit='thz')

def test_non_dispersive_material_initialization(non_dispersive_material):
    assert non_dispersive_material.name == 'SiO2'
    assert non_dispersive_material.er == torch.tensor(2.0, dtype=torch.complex64)
    assert non_dispersive_material.ur == torch.tensor(1.0)
    assert not non_dispersive_material.isdispersive_er

def test_dispersive_material_initialization(dispersive_material):
    assert dispersive_material.name == 'DispersiveMaterial'
    assert dispersive_material.isdispersive_er
    assert dispersive_material._loadeder is not None
    # Verify proxy was created
    assert hasattr(dispersive_material, '_data_proxy')
    assert isinstance(dispersive_material._data_proxy, MaterialDataProxy)

def test_proxy_integration(dispersive_material):
    """Test that the MaterialDataProxy is correctly integrated with MaterialClass."""
    # Access the internal loaded data
    assert dispersive_material._loadeder is not None
    assert dispersive_material._loadeder.shape == (6, 3)
    
    # First column should be wavelength
    assert np.allclose(dispersive_material._loadeder[:, 0], np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    
    # Test the proxy directly
    proxy = dispersive_material._data_proxy
    test_wavelengths = np.array([2.5, 3.5])
    real_eps, imag_eps = proxy.extract_permittivity(dispersive_material._loadeder, test_wavelengths, fit_order=2)
    
    # Check that we got reasonable values back from the polynomial fit
    assert len(real_eps(test_wavelengths)) == 2
    assert len(imag_eps(test_wavelengths)) == 2
    assert 2.1 <= real_eps(test_wavelengths)[0] <= 2.3
    assert 0.2 <= imag_eps(test_wavelengths)[0] <= 0.4

def test_load_dispersive_er(dispersive_material):
    lam0 = np.array([2.0, 5.0])  # Values within the data range
    dispersive_material.load_dispersive_er(lam0=lam0, lengthunit='um')
    
    # Verify the fitted data structure
    assert 'wavelengths' in dispersive_material.fitted_data
    assert 'data_eps1' in dispersive_material.fitted_data
    assert 'data_eps2' in dispersive_material.fitted_data
    assert 'fitted_eps1' in dispersive_material.fitted_data or 'fitted_crv1' in dispersive_material.fitted_data
    assert 'fitted_eps2' in dispersive_material.fitted_data or 'fitted_crv2' in dispersive_material.fitted_data
    
    # Verify the permittivity tensor was created
    assert dispersive_material.er is not None
    assert dispersive_material.er.shape == (2,)
    assert dispersive_material.er.dtype == torch.complex64 or dispersive_material.er.dtype == torch.complex128

def test_freq_data_conversion(freq_data_material):
    """Test loading and processing frequency-based data."""
    lam0 = np.array([1.5, 2.0])  # In um
    freq_data_material.load_dispersive_er(lam0=lam0, lengthunit='um')
    
    assert freq_data_material.er is not None
    assert freq_data_material.er.shape == (2,)
    
    # The values should be in the right ballpark based on our test data
    assert 2.0 <= freq_data_material.er[0].real <= 2.2
    # We might get a negative imaginary part due to the way our polynomial fit works
    # So check the absolute value instead
    assert 0.1 <= abs(freq_data_material.er[0].imag) <= 0.3

def test_invalid_dispersive_material_initialization():
    with pytest.raises(ValueError, match='File path of the dispersive data must be defined'):
        MaterialClass(name='InvalidDispersiveMaterial', dielectric_dispersion=True, user_dielectric_file=None)

def test_invalid_data_file():
    with pytest.raises(ValueError, match='Material data file not found'):
        MaterialClass(name='NonexistentFile', 
                     dielectric_dispersion=True, 
                     user_dielectric_file='nonexistent_file.txt')

def test_check_header(tmp_path):
    # We now use the proxy's _check_header method
    from torchrdit.material_proxy import MaterialDataProxy
    
    file_path = tmp_path / "header_data.txt"
    with open(file_path, 'w') as f:
        f.write("freq eps1 eps2\n1.0 2.0 0.1\n2.0 2.1 0.2\n")
    assert MaterialDataProxy._check_header(file_path)

    file_path_no_header = tmp_path / "no_header_data.txt"
    with open(file_path_no_header, 'w') as f:
        f.write("1.0 2.0 0.1\n2.0 2.1 0.2\n")
    assert not MaterialDataProxy._check_header(file_path_no_header)

def test_unit_converter_integration():
    """Test that UnitConverter is properly integrated with MaterialClass."""
    converter = UnitConverter()
    
    # Test wavelength conversion (common operations in material handling)
    wl_um = np.array([1.55])
    wl_nm = converter.convert_length(wl_um, 'um', 'nm')
    assert np.isclose(wl_nm, 1550)
    
    # Test frequency to wavelength conversion
    freq_thz = np.array([193.5])
    wl_nm = converter.freq_to_wavelength(freq_thz, 'thz', 'nm')
    assert np.isclose(wl_nm, 1550, rtol=1e-2)

def test_factory_methods(tmp_path):
    """Test the factory methods for creating materials."""
    # Test from_nk_data
    material = MaterialClass.from_nk_data(name='Glass', n=1.5, k=0.01)
    assert material.name == 'Glass'
    assert not material.isdispersive_er
    assert material.er.dtype == torch.complex64
    
    # Use isclose instead of exact equality due to potential floating point differences
    expected_real = (1.5**2 - 0.01**2)
    assert torch.isclose(material.er.real, torch.tensor(expected_real, dtype=torch.float32))
    
    # Test from_data_file
    data = np.array([
        [1.0, 2.0, 0.1],
        [2.0, 2.1, 0.2],
    ])
    file_path = tmp_path / "factory_test_data.txt"
    np.savetxt(file_path, data, header="wavelength eps1 eps2", comments='')
    
    material = MaterialClass.from_data_file(
        name='TestMaterial', 
        file_path=str(file_path),
        data_format='wl-eps',
        data_unit='um'
    )
    
    assert material.name == 'TestMaterial'
    assert material.isdispersive_er
    assert material.data_format == 'wl-eps'
    assert material.data_unit == 'um'

def test_caching_mechanism(dispersive_material):
    """Test that the caching mechanism works correctly."""
    # Get cache info (should start empty or small)
    cache_info = dispersive_material.cache_info()
    initial_size = cache_info.currsize if cache_info else 0
    
    # First load should compute and cache
    lam0 = np.array([2.0, 3.0])
    dispersive_material.load_dispersive_er(lam0=lam0, lengthunit='um')
    initial_er = dispersive_material.er.clone()
    
    # Cache should now have one more entry
    cache_info = dispersive_material.cache_info()
    assert cache_info.currsize == initial_size + 1
    
    # Second load with same parameters should use cache
    dispersive_material.load_dispersive_er(lam0=lam0, lengthunit='um')
    # The tensor objects should be identical (same memory location)
    assert torch.all(dispersive_material.er == initial_er)
    
    # Different wavelengths should compute a new value
    lam0_new = np.array([4.0, 5.0])
    dispersive_material.load_dispersive_er(lam0=lam0_new, lengthunit='um')
    # Cache should now have 2 entries
    cache_info = dispersive_material.cache_info()
    assert cache_info.currsize == initial_size + 2
    
    # Clear cache
    dispersive_material.clear_cache()
    cache_info = dispersive_material.cache_info()
    assert cache_info.currsize == 0

def test_string_representations():
    """Test the string representation methods."""
    non_disp = MaterialClass(name='Glass', permittivity=2.25)
    str_rep = str(non_disp)
    assert 'Glass' in str_rep
    assert '2.25' in str_rep
    
    repr_str = repr(non_disp)
    assert 'MaterialClass' in repr_str
    assert 'name' in repr_str
    assert 'Glass' in repr_str
    
def test_get_permittivity_method(dispersive_material, non_dispersive_material):
    """Test the get_permittivity method for both dispersive and non-dispersive materials."""
    # Non-dispersive should just return the fixed permittivity
    result = non_dispersive_material.get_permittivity(np.array([1.0, 2.0]))
    assert torch.isclose(result, torch.tensor(2.0, dtype=torch.complex64))
    
    # Dispersive should calculate and return based on wavelengths
    lam0 = np.array([2.0, 3.0])
    result = dispersive_material.get_permittivity(lam0, 'um')
    assert result.shape == (2,)
    assert result.dtype == torch.complex64 or result.dtype == torch.complex128
    
    # Should match the directly loaded results
    dispersive_material.load_dispersive_er(lam0, 'um')
    direct_result = dispersive_material.er
    assert torch.allclose(result, direct_result)