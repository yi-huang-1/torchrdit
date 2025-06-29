import unittest
import torch
import numpy as np
import os
from pathlib import Path

from torchrdit.material_proxy import UnitConverter, MaterialDataProxy
from torchrdit.constants import C_0


class TestMaterialProxyDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in material_proxy.py.
    
    This test suite ensures that the examples provided in the docstrings of
    material_proxy.py work as expected. It tests examples for:
    1. UnitConverter class and its methods
    2. MaterialDataProxy class and its methods
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a basic unit converter
        self.converter = UnitConverter()
        
        # Create a basic material data proxy
        self.proxy = MaterialDataProxy()
        
        # Test data files
        self.test_data_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.sio2_file = self.test_data_dir / "SiO2-e.txt"
        self.si_file = self.test_data_dir / "Si_C-e.txt"
    
    def test_module_examples(self):
        """Test the module-level examples shown in the docstring."""
        # Basic unit conversion example
        converter = UnitConverter()
        # Convert from nanometers to micrometers
        wavelength_um = converter.convert_length(1550, 'nm', 'um')
        self.assertAlmostEqual(wavelength_um, 1.55)
        
        # Convert from frequency to wavelength
        wavelength = converter.freq_to_wavelength(193.5, 'thz', 'um')
        self.assertAlmostEqual(wavelength, 1.55, places=2)
        
        # Loading material data example
        proxy = MaterialDataProxy()
        # We'll use a real data file for this test
        si_data = proxy.load_data(self.si_file, 'wl-eps', 'um')
        
        # Extract permittivity at specific wavelengths
        wavelengths = np.array([1.3, 1.55, 1.7])
        eps_real, eps_imag = proxy.extract_permittivity(si_data, wavelengths)
        
        # Verify we got reasonable results
        self.assertEqual(len(eps_real(wavelengths)), 3)
        self.assertEqual(len(eps_imag(wavelengths)), 3)
        # Silicon has high permittivity
        self.assertTrue(np.all(eps_real(wavelengths) > 8))
    
    def test_unit_converter_init(self):
        """Test UnitConverter initialization."""
        # Initialize UnitConverter
        converter = UnitConverter()
        
        # Verify attributes are set properly
        self.assertIsNotNone(converter._length_units)
        self.assertIsNotNone(converter._freq_units)
    
    def test_unit_converter_examples(self):
        """Test the UnitConverter examples."""
        # Example from UnitConverter class docstring
        converter = UnitConverter()
        
        # Length conversion
        result = converter.convert_length(1.0, 'mm', 'um')
        self.assertAlmostEqual(result, 1000.0)
        
        # Frequency conversion
        result = converter.convert_frequency(1.0, 'ghz', 'mhz')
        self.assertAlmostEqual(result, 1000.0)
        
        # Convert from frequency to wavelength
        result = converter.freq_to_wavelength(200, 'thz', 'nm')
        self.assertAlmostEqual(result, 1499.0, places=0)
        
        # Convert from wavelength to frequency
        result = converter.wavelength_to_freq(1.55, 'um', 'thz')
        # The actual value is closer to 193.41 THz rather than 193.5 THz
        self.assertAlmostEqual(result, 193.41, places=1)
    
    def test_validate_units(self):
        """Test validate_units method examples."""
        converter = UnitConverter()
        
        # Validate known units
        result = converter.validate_units('um', 'length')
        self.assertEqual(result, 'um')
        
        result = converter.validate_units('THz', 'frequency')
        self.assertEqual(result, 'thz')
        
        # Test validation error for invalid units
        with self.assertRaises(ValueError):
            converter.validate_units('invalid', 'length')
    
    def test_convert_length(self):
        """Test convert_length method examples."""
        converter = UnitConverter()
        
        # Test single value conversion
        result = converter.convert_length(1550, 'nm', 'um')
        self.assertAlmostEqual(result, 1.55)
        
        result = converter.convert_length(0.001, 'meter', 'mm')
        self.assertAlmostEqual(result, 1.0)
        
        # Test array conversion
        wavelengths_nm = np.array([1310, 1550, 1625])
        results = converter.convert_length(wavelengths_nm, 'nm', 'um')
        expected = np.array([1.31, 1.55, 1.625])
        np.testing.assert_allclose(results, expected, rtol=1e-5)
    
    def test_convert_frequency(self):
        """Test convert_frequency method examples."""
        converter = UnitConverter()
        
        # Test single value conversion
        result = converter.convert_frequency(1, 'thz', 'ghz')
        self.assertAlmostEqual(result, 1000.0)
        
        result = converter.convert_frequency(2.5, 'ghz', 'mhz')
        self.assertAlmostEqual(result, 2500.0)
        
        # Test array conversion
        frequencies_ghz = np.array([100, 200, 300])
        results = converter.convert_frequency(frequencies_ghz, 'ghz', 'thz')
        expected = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(results, expected, rtol=1e-5)
    
    def test_freq_to_wavelength(self):
        """Test freq_to_wavelength method examples."""
        converter = UnitConverter()
        
        # Test single value conversion
        result = converter.freq_to_wavelength(193.5, 'thz', 'um')
        self.assertAlmostEqual(result, 1.55, places=2)
        
        result = converter.freq_to_wavelength(100, 'ghz', 'mm')
        self.assertAlmostEqual(result, 3.0, places=1)
        
        # Test array conversion
        freqs_thz = np.array([193.5, 230.0, 350.0])
        results = converter.freq_to_wavelength(freqs_thz, 'thz', 'um')
        expected = np.array([1.55, 1.30, 0.86])
        np.testing.assert_allclose(results, expected, rtol=1e-2)
    
    def test_wavelength_to_freq(self):
        """Test wavelength_to_freq method examples."""
        converter = UnitConverter()
        
        # Test single value conversion
        result = converter.wavelength_to_freq(1.55, 'um', 'thz')
        # The actual value is closer to 193.41 THz rather than 193.5 THz
        self.assertAlmostEqual(result, 193.41, places=1)
        
        result = converter.wavelength_to_freq(3, 'mm', 'ghz')
        self.assertAlmostEqual(result, 100.0, places=0)
        
        # Test array conversion
        wavelengths_um = np.array([1.31, 1.55, 2.0])
        results = converter.wavelength_to_freq(wavelengths_um, 'um', 'thz')
        # Adjust expected values to match actual implementation
        expected = np.array([228.9, 193.4, 149.9])
        np.testing.assert_allclose(results, expected, rtol=1e-2)
    
    def test_material_data_proxy_init(self):
        """Test MaterialDataProxy initialization."""
        # Create with default unit converter
        proxy1 = MaterialDataProxy()
        self.assertIsInstance(proxy1._converter, UnitConverter)
        
        # Create with custom unit converter
        converter = UnitConverter()
        proxy2 = MaterialDataProxy(unit_converter=converter)
        self.assertEqual(proxy2._converter, converter)
    
    def test_load_data(self):
        """Test load_data method examples, using real data files."""
        proxy = MaterialDataProxy()
        
        # Load wavelength-permittivity data
        sio2_data = proxy.load_data(self.sio2_file, 'wl-eps', 'um')
        self.assertTrue(sio2_data.shape[0] > 0)
        self.assertEqual(sio2_data.shape[1], 3)
        
        # Check that data is sensible
        # First, print a data point near 1.55 um to see the actual value
        mask = np.abs(sio2_data[:, 0] - 1.55) < 0.1
        if np.any(mask):
            eps_1550 = sio2_data[mask, 1]
            # Just check that we got some data with reasonable values (positive permittivity)
            self.assertTrue(len(eps_1550) > 0)
            self.assertTrue(np.all(eps_1550 > 0))
        else:
            # If no data points near 1.55um, check general data validity
            self.assertTrue(np.all(sio2_data[:, 1] > 0))
    
    def test_extract_permittivity(self):
        """Test extract_permittivity method examples."""
        proxy = MaterialDataProxy()
        
        # Load material data from a real file
        data = proxy.load_data(self.sio2_file, 'wl-eps', 'um')
        
        # Extract permittivity at specific wavelengths
        operating_wl = np.array([1.31, 1.55, 1.7])
        eps_real, eps_imag = proxy.extract_permittivity(data, operating_wl)
        
        # Verify results
        self.assertEqual(len(eps_real(operating_wl)), 3)
        self.assertEqual(len(eps_imag(operating_wl)), 3)
        
        # For SiO2, check permittivity is positive and reasonable (silica varies with wavelength)
        self.assertTrue(np.all(eps_real(operating_wl) > 0))


if __name__ == '__main__':
    unittest.main() 