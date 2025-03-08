"""Module for handling material data with unit awareness."""

import numpy as np
from .constants import frequnit_dict, lengthunit_dict, C_0


class UnitConverter:
    """Handles all unit conversions for materials."""
    
    def __init__(self):
        self._length_units = lengthunit_dict
        self._freq_units = frequnit_dict
        
    def validate_units(self, unit, unit_type):
        """
        Validates that a unit string is recognized.
        
        Args:
            unit (str): Unit to validate
            unit_type (str): Type of unit ('length' or 'frequency')
            
        Returns:
            str: Lowercase validated unit string
            
        Raises:
            ValueError: If unit is not recognized
        """
        if unit_type == 'length' and unit.lower() not in self._length_units:
            raise ValueError(f"Unknown length unit: {unit}. Valid units: {list(self._length_units.keys())}")
        elif unit_type == 'frequency' and unit.lower() not in self._freq_units:
            raise ValueError(f"Unknown frequency unit: {unit}. Valid units: {list(self._freq_units.keys())}")
        return unit.lower()
    
    def convert_length(self, value, from_unit, to_unit):
        """
        Convert between length units.
        
        Args:
            value (float or array): Value to convert
            from_unit (str): Source unit
            to_unit (str): Target unit
            
        Returns:
            float or array: Converted value
        """
        from_unit = self.validate_units(from_unit, 'length')
        to_unit = self.validate_units(to_unit, 'length')
        return value * self._length_units[from_unit] / self._length_units[to_unit]
    
    def convert_frequency(self, value, from_unit, to_unit):
        """
        Convert between frequency units.
        
        Args:
            value (float or array): Value to convert
            from_unit (str): Source unit
            to_unit (str): Target unit
            
        Returns:
            float or array: Converted value
        """
        from_unit = self.validate_units(from_unit, 'frequency')
        to_unit = self.validate_units(to_unit, 'frequency')
        return value * self._freq_units[from_unit] / self._freq_units[to_unit]
    
    def freq_to_wavelength(self, freq, freq_unit, wl_unit):
        """
        Convert frequency to wavelength.
        
        Args:
            freq (float or array): Frequency value
            freq_unit (str): Frequency unit
            wl_unit (str): Target wavelength unit
            
        Returns:
            float or array: Wavelength in target unit
        """
        # Convert frequency to Hz
        freq_hz = self.convert_frequency(freq, freq_unit, 'hz')
        # Calculate wavelength in meters
        wl_m = C_0 / freq_hz
        # Convert to target unit
        return self.convert_length(wl_m, 'meter', wl_unit)
    
    def wavelength_to_freq(self, wavelength, wl_unit, freq_unit):
        """
        Convert wavelength to frequency.
        
        Args:
            wavelength (float or array): Wavelength value
            wl_unit (str): Wavelength unit
            freq_unit (str): Target frequency unit
            
        Returns:
            float or array: Frequency in target unit
        """
        # Convert wavelength to meters
        wl_m = self.convert_length(wavelength, wl_unit, 'meter')
        # Calculate frequency in Hz
        freq_hz = C_0 / wl_m
        # Convert to target unit
        return self.convert_frequency(freq_hz, 'hz', freq_unit)


class MaterialDataProxy:
    """Handles loading and processing material data with unit awareness."""
    
    def __init__(self, unit_converter=None):
        """
        Initialize the MaterialDataProxy.
        
        Args:
            unit_converter (UnitConverter, optional): Custom unit converter
        """
        self._converter = unit_converter or UnitConverter()
    
    def load_data(self, file_path, data_format, data_unit, target_unit='um'):
        """
        Load material data from file with unit conversion.
        
        Args:
            file_path (str): Path to data file
            data_format (str): Format of data ('freq-eps', 'wl-eps', 'freq-nk', 'wl-nk')
            data_unit (str): Unit of the first column (frequency or wavelength)
            target_unit (str): Length unit to convert data to
            
        Returns:
            numpy.ndarray: Array of [wavelength, eps_real, eps_imag] in target units
            
        Raises:
            ValueError: If file or format is invalid
        """
        # Check if file exists
        try:
            # Check file header
            skip_header = self._check_header(file_path)
            
            # Load raw data
            raw_data = np.loadtxt(file_path, skiprows=1 if skip_header else 0)
            
            # Process based on format
            converted_data = self._convert_data(raw_data, data_format, data_unit, target_unit)
            
            # Sort by wavelength (increasing)
            if converted_data[0, 0] > converted_data[-1, 0]:
                return converted_data[::-1]  # Reverse if descending
            return converted_data
            
        except FileNotFoundError:
            raise ValueError(f"Material data file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading material data: {str(e)}")
    
    def _convert_data(self, raw_data, data_format, data_unit, target_unit):
        """
        Apply appropriate unit conversions based on data format.
        
        Args:
            raw_data (numpy.ndarray): Raw data from file
            data_format (str): Format specification
            data_unit (str): Unit of input data
            target_unit (str): Desired output unit
            
        Returns:
            numpy.ndarray: Converted data as [wavelength, eps_real, eps_imag]
            
        Raises:
            ValueError: If format is not supported
        """
        if raw_data.shape[1] < 2:
            raise ValueError("Material data must have at least two columns")
            
        result = np.zeros((raw_data.shape[0], 3))
        
        # Extract format components
        format_parts = data_format.lower().split('-')
        if len(format_parts) != 2 or format_parts[0] not in ('freq', 'wl') or format_parts[1] not in ('eps', 'nk'):
            raise ValueError(f"Unsupported data format: {data_format}. " 
                            "Expected: freq-eps, wl-eps, freq-nk, or wl-nk")
                            
        x_type, y_type = format_parts[0], format_parts[1]
        
        # Handle x-axis conversion (always convert to wavelength)
        if x_type == 'freq':
            # Convert frequency to wavelength
            result[:, 0] = self._converter.freq_to_wavelength(
                raw_data[:, 0], data_unit, target_unit)
        else:  # wavelength
            # Just convert units
            result[:, 0] = self._converter.convert_length(
                raw_data[:, 0], data_unit, target_unit)
        
        # Handle y-axis conversion
        if y_type == 'eps':
            result[:, 1] = raw_data[:, 1]  # Real part
            # Imaginary part (if available)
            result[:, 2] = raw_data[:, 2] if raw_data.shape[1] > 2 else 0
        else:  # nk
            # Convert n,k to complex permittivity
            n = raw_data[:, 1]
            k = raw_data[:, 2] if raw_data.shape[1] > 2 else 0
            eps_complex = (n + 1j*k)**2
            result[:, 1] = np.real(eps_complex)
            result[:, 2] = np.imag(eps_complex)
        
        return result

    @staticmethod
    def _check_header(filename):
        """
        Check if the file has a header row.
        
        Args:
            filename (str): Path to the file
            
        Returns:
            bool: True if file has a header row
        """
        with open(filename, 'r') as file:
            first_line = file.readline().strip()
            return not all(char.isdigit() or char in '.- ' for char in first_line)
            
    def extract_permittivity(self, data, wavelengths, fit_order=10):
        """
        Extract permittivity values for specific wavelengths.
        
        Args:
            data (numpy.ndarray): Material data from load_data
            wavelengths (numpy.ndarray): Target wavelengths
            fit_order (int): Polynomial fit order
            
        Returns:
            tuple: (real_permittivity, imaginary_permittivity) at requested wavelengths
        """
        import warnings
        
        # Get the data columns
        wl_data = data[:, 0]
        eps_real = data[:, 1]
        eps_imag = data[:, 2]
        
        # Fit polynomials to the data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', Warning)
            np.seterr(all='ignore')
            
            coef_real = np.polyfit(wl_data, eps_real, fit_order)
            coef_imag = np.polyfit(wl_data, eps_imag, fit_order)
            
        poly_real = np.poly1d(coef_real)
        poly_imag = np.poly1d(coef_imag)
        
        # Evaluate at target wavelengths
        return poly_real(wavelengths), poly_imag(wavelengths) 