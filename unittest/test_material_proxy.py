"""Unit tests for UnitConverter and MaterialDataProxy."""

import os
import numpy as np
import pytest
import tempfile

from torchrdit.material_proxy import UnitConverter, MaterialDataProxy


class TestUnitConverter:
    """Test unit conversion functionality (focused, non-duplicative)."""

    def test_basic_conversions_and_validation(self):
        converter = UnitConverter()

        # Length
        assert np.isclose(converter.convert_length(1, 'meter', 'mm'), 1000.0)
        assert np.allclose(converter.convert_length(np.array([1, 2, 3]), 'um', 'nm'), np.array([1000, 2000, 3000]))
        assert np.isclose(converter.convert_length(1, 'Meter', 'MM'), 1000.0)  # case-insensitive
        with pytest.raises(ValueError):
            converter.convert_length(1, 'invalid', 'meter')

        # Frequency
        assert np.isclose(converter.convert_frequency(1, 'ghz', 'mhz'), 1000.0)
        with pytest.raises(ValueError):
            converter.convert_frequency(1, 'hz', 'invalid')

        # Cross domain
        assert np.isclose(converter.freq_to_wavelength(300, 'thz', 'um'), 1.0, rtol=1e-2)
        assert np.isclose(converter.wavelength_to_freq(1550, 'nm', 'thz'), 193.5, rtol=1e-2)


class TestMaterialDataProxy:
    """Test MaterialDataProxy functionality."""
    
    def setup_method(self):
        """Set up test data files."""
        self.proxy = MaterialDataProxy()
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test data files
        self.freq_eps_file = os.path.join(self.temp_dir.name, 'freq_eps.txt')
        self.wl_eps_file = os.path.join(self.temp_dir.name, 'wl_eps.txt')
        self.freq_nk_file = os.path.join(self.temp_dir.name, 'freq_nk.txt')
        self.wl_nk_file = os.path.join(self.temp_dir.name, 'wl_nk.txt')
        
        # Write freq-eps data (193-195 THz range)
        with open(self.freq_eps_file, 'w') as f:
            f.write("# Frequency (THz), Real(ε), Imag(ε)\n")
            for freq in np.linspace(193, 195, 10):
                eps_real = 2.0 + 0.01 * (freq - 193)
                eps_imag = 0.001 * (freq - 193)
                f.write(f"{freq} {eps_real} {eps_imag}\n")
        
        # Write wl-eps data (1530-1570 nm range)
        with open(self.wl_eps_file, 'w') as f:
            f.write("# Wavelength (nm), Real(ε), Imag(ε)\n")
            for wl in np.linspace(1530, 1570, 10):
                eps_real = 2.0 + 0.01 * (wl - 1530)
                eps_imag = 0.001 * (wl - 1530)
                f.write(f"{wl} {eps_real} {eps_imag}\n")
        
        # Write freq-nk data (193-195 THz range)
        with open(self.freq_nk_file, 'w') as f:
            f.write("# Frequency (THz), n, k\n")
            for freq in np.linspace(193, 195, 10):
                n = 1.5 + 0.01 * (freq - 193)
                k = 0.001 * (freq - 193)
                f.write(f"{freq} {n} {k}\n")
        
        # Write wl-nk data (1530-1570 nm range)
        with open(self.wl_nk_file, 'w') as f:
            f.write("# Wavelength (nm), n, k\n")
            for wl in np.linspace(1530, 1570, 10):
                n = 1.5 + 0.01 * (wl - 1530)
                k = 0.001 * (wl - 1530)
                f.write(f"{wl} {n} {k}\n")
    
    def teardown_method(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_load_freq_eps_data(self):
        """Test loading frequency-epsilon data."""
        data = self.proxy.load_data(self.freq_eps_file, 'freq-eps', 'thz', 'um')
        
        # Check shape
        assert data.shape[0] == 10  # 10 data points
        assert data.shape[1] == 3   # wavelength, eps_real, eps_imag
        
        # Data is reversed to ascending wavelength; first row corresponds to highest freq (195 THz)
        c0 = 2.99792458e8  # m/s
        expected_um_195 = c0 / (195e12) / 1e-6
        assert np.isclose(data[0, 0], expected_um_195, rtol=1e-9)

        # Check permittivity values for 195 THz row
        eps_real_195 = 2.0 + 0.01 * (195.0 - 193.0)
        eps_imag_195 = 0.001 * (195.0 - 193.0)
        assert np.isclose(data[0, 1], eps_real_195, rtol=1e-9)
        assert np.isclose(data[0, 2], eps_imag_195, rtol=1e-9)
    
    def test_load_wl_eps_data(self):
        """Test loading wavelength-epsilon data."""
        data = self.proxy.load_data(self.wl_eps_file, 'wl-eps', 'nm', 'um')
        
        # Check conversion from nm to um
        assert np.isclose(data[0, 0], 1.53, rtol=1e-2)
        
        # Check permittivity values maintained
        assert np.isclose(data[0, 1], 2.0, rtol=1e-2)
    
    def test_load_freq_nk_data(self):
        """Test loading frequency-nk data."""
        data = self.proxy.load_data(self.freq_nk_file, 'freq-nk', 'thz', 'um')
        
        # Check conversion to wavelength
        c0 = 2.99792458e8  # m/s
        expected_um_195 = c0 / (195e12) / 1e-6
        assert np.isclose(data[0, 0], expected_um_195, rtol=1e-9)
        
        # Check conversion from n,k to permittivity
        # First row corresponds to 195 THz after reversal
        n = 1.5 + 0.01 * (195.0 - 193.0)  # 1.52
        k = 0.001 * (195.0 - 193.0)       # 0.002
        expected_eps_real = (n**2 - k**2)
        expected_eps_imag = 2 * n * k

        assert np.isclose(data[0, 1], expected_eps_real, rtol=1e-9)
        assert np.isclose(data[0, 2], expected_eps_imag, rtol=1e-9)
    
    def test_load_wl_nk_data(self):
        """Test loading wavelength-nk data."""
        data = self.proxy.load_data(self.wl_nk_file, 'wl-nk', 'nm', 'um')
        
        # Check conversion to um
        assert np.isclose(data[0, 0], 1.53, rtol=1e-2)
        
        # Check conversion from n,k to permittivity
        n, k = 1.5, 0.0  # First row values
        expected_eps_real = n**2 - k**2
        expected_eps_imag = 2 * n * k
        
        assert np.isclose(data[0, 1], expected_eps_real, rtol=1e-2)
        assert np.isclose(data[0, 2], expected_eps_imag, rtol=1e-2)
    
    def test_invalid_file(self):
        """Test handling invalid file."""
        with pytest.raises(ValueError):
            self.proxy.load_data('nonexistent_file.txt', 'freq-eps', 'thz', 'um')
    
    def test_invalid_format(self):
        """Test handling invalid format."""
        with pytest.raises(ValueError):
            self.proxy.load_data(self.freq_eps_file, 'invalid-format', 'thz', 'um')
    
    def test_invalid_unit(self):
        """Test handling invalid unit passed to loader."""
        with pytest.raises(ValueError):
            self.proxy.load_data(self.freq_eps_file, 'freq-eps', 'invalid', 'um')
