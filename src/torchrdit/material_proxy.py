"""Module for handling material data with unit awareness in TorchRDIT.

This module provides classes for loading, processing, and converting material data
with appropriate unit handling for electromagnetic simulations. It serves as a
foundation for the materials system in TorchRDIT, enabling support for different
data formats, units, and conversions between frequency and wavelength domains.

The module implements a proxy pattern for material data handling, separating the
concerns of data loading, unit conversion, and material representation. This allows
for efficient handling of both dispersive (wavelength/frequency dependent) and
non-dispersive materials.

Classes:
    UnitConverter: Handles all unit conversions for wavelength, frequency, and derived quantities.
    MaterialDataProxy: Loads and processes material data files with appropriate unit handling.
    
Examples:
```python
# Basic unit conversion:
from torchrdit.material_proxy import UnitConverter
converter = UnitConverter()
# Convert from nanometers to micrometers
wavelength_um = converter.convert_length(1550, 'nm', 'um')
print(f"1550 nm = {wavelength_um} μm")
# 1550 nm = 1.55 μm
# Convert from frequency to wavelength
wavelength = converter.freq_to_wavelength(193.5, 'thz', 'um')
print(f"193.5 THz = {wavelength:.2f} μm")
# 193.5 THz = 1.55 μm

# Loading material data:
from torchrdit.material_proxy import MaterialDataProxy
import numpy as np
# Load silicon data from file
proxy = MaterialDataProxy()
# File contains wavelength (um) and permittivity data
si_data = proxy.load_data('Si_data.txt', 'wl-eps', 'um')
# Extract permittivity at specific wavelengths
wavelengths = np.array([1.3, 1.55, 1.7])
eps_real, eps_imag = proxy.extract_permittivity(si_data, wavelengths)
```
    
Keywords:
    material data, unit conversion, wavelength, frequency, permittivity,
    material loading, dispersive materials, unit awareness, proxy pattern
"""

import numpy as np
from .constants import frequnit_dict, lengthunit_dict, C_0


class UnitConverter:
    """Handles unit conversions for wavelength, frequency, and derived quantities.
    
    This class provides a centralized system for handling unit conversions in
    TorchRDIT, particularly for wavelength and frequency units. It validates unit
    specifications, converts between different units of the same type, and provides
    utilities for converting between frequency and wavelength domains.
    
    The UnitConverter maintains dictionaries of supported units and their conversion
    factors, enabling consistent unit handling throughout the TorchRDIT package.
    
    Attributes:
        _length_units (dict): Dictionary of length unit conversion factors.
        _freq_units (dict): Dictionary of frequency unit conversion factors.
        
    Examples:
    ```python
    from torchrdit.material_proxy import UnitConverter
    converter = UnitConverter()
    # Length conversion
    print(converter.convert_length(1.0, 'mm', 'um'))
    # 1000.0
    # Frequency conversion
    print(converter.convert_frequency(1.0, 'ghz', 'mhz'))
    # 1000.0
    # Convert from frequency to wavelength
    print(converter.freq_to_wavelength(200, 'thz', 'nm'))
    # 1499.0
    # Convert from wavelength to frequency
    print(converter.wavelength_to_freq(1.55, 'um', 'thz'))
    # 193.5
    ```
    
    Keywords:
        unit conversion, wavelength, frequency, length units, frequency units,
        unit validation, unit transformation, electromagnetic units
    """
    
    def __init__(self):
        """Initialize the UnitConverter with supported units from constants.
        
        Initializes the converter with dictionaries of supported length and
        frequency units from the constants module.
        
        Keywords:
            initialization, unit converter creation
        """
        self._length_units = lengthunit_dict
        self._freq_units = frequnit_dict
        
    def validate_units(self, unit, unit_type):
        """Validate that a unit string is recognized and supported.
        
        This method checks if the provided unit string is valid for the specified
        unit type (length or frequency). It returns the lowercase version of the
        unit string if valid, making subsequent operations case-insensitive.
        
        Args:
            unit (str): Unit identifier to validate (e.g., 'nm', 'ghz').
            unit_type (str): Type of unit ('length' or 'frequency').
            
        Returns:
            str: Lowercase validated unit string.
            
        Raises:
            ValueError: If the unit is not recognized for the specified type.
            
        Examples:
        ```python
        from torchrdit.material_proxy import UnitConverter
        converter = UnitConverter()
        print(converter.validate_units('um', 'length'))
        # 'um'
        print(converter.validate_units('THz', 'frequency'))
        # 'thz'
        # This would raise an error
        try:
            print(converter.validate_units('invalid', 'length'))
        except ValueError as e:
            print(f"Error: {e}")
        # Error: Unknown length unit: invalid. Valid units: [...]
        ```
        
        Keywords:
            unit validation, error checking, case normalization
        """
        if unit_type == 'length' and unit.lower() not in self._length_units:
            raise ValueError(f"Unknown length unit: {unit}. Valid units: {list(self._length_units.keys())}")
        elif unit_type == 'frequency' and unit.lower() not in self._freq_units:
            raise ValueError(f"Unknown frequency unit: {unit}. Valid units: {list(self._freq_units.keys())}")
        return unit.lower()
    
    def convert_length(self, value, from_unit, to_unit):
        """Convert values between different length units.
        
        This method converts a length value from one unit to another using the
        conversion factors defined in the length units dictionary. Both units
        are validated before conversion.
        
        Args:
            value (float or array): Length value(s) to convert.
            from_unit (str): Source length unit (e.g., 'nm', 'um', 'mm').
            to_unit (str): Target length unit (e.g., 'nm', 'um', 'mm').
            
        Returns:
            float or array: Converted length value(s) in the target unit.
            
        Examples:
        ```python
        from torchrdit.material_proxy import UnitConverter
        converter = UnitConverter()
        print(converter.convert_length(1550, 'nm', 'um'))
        # 1.55
        print(converter.convert_length(0.001, 'meter', 'mm'))
        # 1.0
        # Converting an array of values
        import numpy as np
        wavelengths_nm = np.array([1310, 1550, 1625])
        print(converter.convert_length(wavelengths_nm, 'nm', 'um'))
        # array([1.31, 1.55, 1.625])
        ```
        
        Keywords:
            length conversion, unit transformation, dimension scaling
        """
        from_unit = self.validate_units(from_unit, 'length')
        to_unit = self.validate_units(to_unit, 'length')
        return value * self._length_units[from_unit] / self._length_units[to_unit]
    
    def convert_frequency(self, value, from_unit, to_unit):
        """Convert values between different frequency units.
        
        This method converts a frequency value from one unit to another using the
        conversion factors defined in the frequency units dictionary. Both units
        are validated before conversion.
        
        Args:
            value (float or array): Frequency value(s) to convert.
            from_unit (str): Source frequency unit (e.g., 'thz', 'ghz', 'mhz').
            to_unit (str): Target frequency unit (e.g., 'thz', 'ghz', 'mhz').
            
        Returns:
            float or array: Converted frequency value(s) in the target unit.
            
        Examples:
        ```python
        from torchrdit.material_proxy import UnitConverter
        converter = UnitConverter()
        print(converter.convert_frequency(1, 'thz', 'ghz'))
        # 1000.0
        print(converter.convert_frequency(2.5, 'ghz', 'mhz'))
        # 2500.0
        # Converting an array of values
        import numpy as np
        frequencies_ghz = np.array([100, 200, 300])
        print(converter.convert_frequency(frequencies_ghz, 'ghz', 'thz'))
        # array([0.1, 0.2, 0.3])
        ```
        
        Keywords:
            frequency conversion, unit transformation, scale conversion
        """
        from_unit = self.validate_units(from_unit, 'frequency')
        to_unit = self.validate_units(to_unit, 'frequency')
        return value * self._freq_units[from_unit] / self._freq_units[to_unit]
    
    def freq_to_wavelength(self, freq, freq_unit, wl_unit):
        """Convert from frequency to wavelength.
        
        This method converts frequency values to their corresponding wavelength
        values using the physical relationship λ = c/f, with appropriate unit
        conversions. Uses the speed of light constant from the constants module.
        
        Args:
            freq (float or array): Frequency value(s) to convert.
            freq_unit (str): Unit of the input frequency (e.g., 'thz', 'ghz').
            wl_unit (str): Desired output wavelength unit (e.g., 'um', 'nm').
            
        Returns:
            float or array: Wavelength value(s) in the target unit.
            
        Examples:
        ```python
        from torchrdit.material_proxy import UnitConverter
        converter = UnitConverter()
        # Convert 193.5 THz to wavelength in μm
        print(converter.freq_to_wavelength(193.5, 'thz', 'um'))
        # 1.55
        # Convert 100 GHz to wavelength in mm
        print(converter.freq_to_wavelength(100, 'ghz', 'mm'))
        # 3.0
        # Converting an array of frequencies
        import numpy as np
        freqs_thz = np.array([193.5, 230.0, 350.0])
        print(converter.freq_to_wavelength(freqs_thz, 'thz', 'um'))
        # array([1.55, 1.30, 0.86])
        ```
        
        Keywords:
            frequency to wavelength, unit conversion, electromagnetic spectrum,
            physical relationship, speed of light
        """
        # Convert frequency to Hz
        freq_hz = self.convert_frequency(freq, freq_unit, 'hz')
        # Calculate wavelength in meters
        wl_m = C_0 / freq_hz
        # Convert to target unit
        return self.convert_length(wl_m, 'meter', wl_unit)
    
    def wavelength_to_freq(self, wavelength, wl_unit, freq_unit):
        """Convert from wavelength to frequency.
        
        This method converts wavelength values to their corresponding frequency
        values using the physical relationship f = c/λ, with appropriate unit
        conversions. Uses the speed of light constant from the constants module.
        
        Args:
            wavelength (float or array): Wavelength value(s) to convert.
            wl_unit (str): Unit of the input wavelength (e.g., 'um', 'nm').
            freq_unit (str): Desired output frequency unit (e.g., 'thz', 'ghz').
            
        Returns:
            float or array: Frequency value(s) in the target unit.
            
        Examples:
        ```python
        from torchrdit.material_proxy import UnitConverter
        converter = UnitConverter()
        # Convert 1.55 μm to frequency in THz
        print(converter.wavelength_to_freq(1.55, 'um', 'thz'))
        # 193.5
        # Convert 3 mm to frequency in GHz
        print(converter.wavelength_to_freq(3, 'mm', 'ghz'))
        # 100.0
        # Converting an array of wavelengths
        import numpy as np
        wavelengths_um = np.array([1.31, 1.55, 2.0])
        print(converter.wavelength_to_freq(wavelengths_um, 'um', 'thz'))
        # array([229.0, 193.5, 150.0])
        ```
        
        Keywords:
            wavelength to frequency, unit conversion, electromagnetic spectrum,
            physical relationship, speed of light
        """
        # Convert wavelength to meters
        wl_m = self.convert_length(wavelength, wl_unit, 'meter')
        # Calculate frequency in Hz
        freq_hz = C_0 / wl_m
        # Convert to target unit
        return self.convert_frequency(freq_hz, 'hz', freq_unit)


class MaterialDataProxy:
    """Proxy for loading and processing material property data with unit awareness.
    
    This class implements the proxy pattern for handling material data loading and
    processing in TorchRDIT. It provides methods for loading material data from
    files with different formats and units, and extracting specific property values
    at desired wavelengths. The class handles unit conversions internally, providing
    a consistent interface regardless of the source data format.
    
    MaterialDataProxy supports multiple data formats including:
    - 'freq-eps': Frequency and complex permittivity data
    - 'wl-eps': Wavelength and complex permittivity data
    - 'freq-nk': Frequency and complex refractive index (n, k) data
    - 'wl-nk': Wavelength and complex refractive index (n, k) data
    
    This class is typically used internally by the MaterialClass to load and process
    dispersive material data, but can also be used directly for material data analysis
    and manipulation.
    
    Attributes:
        _converter (UnitConverter): Unit converter instance for handling unit transformations.
    
    Examples:
    ```python
    from torchrdit.material_proxy import MaterialDataProxy
    proxy = MaterialDataProxy()
    
    # Load permittivity data from a file with wavelength in μm
    data = proxy.load_data('materials/SiO2.txt', 'wl-eps', 'um')
    
    # Get the first few rows of the loaded and converted data
    print(f"Wavelength (μm) | ε_real | ε_imag")
    for i in range(min(3, data.shape[0])):
        print(f"{data[i, 0]:.2f} | {data[i, 1]:.2f} | {data[i, 2]:.4f}")
    
    # Extract permittivity at specific wavelengths
    import numpy as np
    wavelengths = np.array([1.31, 1.55, 1.85])
    eps_real, eps_imag = proxy.extract_permittivity(data, wavelengths)
    for wl, er, ei in zip(wavelengths, eps_real(wavelengths), eps_imag(wavelengths)):
        print(f"λ={wl}μm: ε={er:.4f}{'-' if ei < 0 else '+'}{abs(ei):.4f}j")
    ```
    
    Keywords:
        material data, data loading, proxy pattern, permittivity, refractive index,
        material properties, data processing, unit conversion, dispersive materials,
        optical constants, electromagnetic properties
    """
    
    def __init__(self, unit_converter=None):
        """Initialize the MaterialDataProxy with an optional custom unit converter.
        
        Args:
            unit_converter (UnitConverter, optional): Custom unit converter instance
                            to use for unit transformations. If None, a new
                            UnitConverter instance is created. Default is None.
        
        Examples:
        ```python
        from torchrdit.material_proxy import MaterialDataProxy
        # Create with default unit converter
        proxy = MaterialDataProxy()
        
        # Create with custom unit converter
        from torchrdit.material_proxy import UnitConverter
        converter = UnitConverter()
        proxy = MaterialDataProxy(unit_converter=converter)
        ```
        
        Keywords:
            initialization, proxy creation, unit converter
        """
        self._converter = unit_converter or UnitConverter()
    
    def load_data(self, file_path, data_format, data_unit, target_unit='um'):
        """Load and process material data from a file with unit conversion.
        
        This method loads material property data from a file, processes it according
        to the specified format, and converts it to a standardized representation
        with the target wavelength unit. It handles different data formats including
        permittivity (eps) and refractive index (n,k) data, in both frequency and
        wavelength domains.
        
        The returned data is always in the format [wavelength, eps_real, eps_imag],
        regardless of the input format, with wavelength values in the target_unit.
        
        Args:
            file_path (str): Path to the data file containing material properties.
            data_format (str): Format of the data file. Must be one of:
                       'freq-eps': Frequency and complex permittivity
                       'wl-eps': Wavelength and complex permittivity
                       'freq-nk': Frequency and complex refractive index
                       'wl-nk': Wavelength and complex refractive index
            data_unit (str): Unit of the first column (frequency or wavelength).
                     For wavelength: 'nm', 'um', etc.
                     For frequency: 'thz', 'ghz', etc.
            target_unit (str, optional): Length unit to convert wavelength data to.
                         Default is 'um' (micrometers).
            
        Returns:
            numpy.ndarray: Array with shape (N, 3) containing [wavelength, eps_real, eps_imag]
                        for each of N data points, with wavelength values in target_unit
                        and sorted in ascending order.
            
        Raises:
            ValueError: If the file doesn't exist, the format is invalid, or other
                      errors occur during loading or processing.
                      
        Examples:
        ```python
        from torchrdit.material_proxy import MaterialDataProxy
        proxy = MaterialDataProxy()
        
        # Load wavelength-permittivity data (wavelengths in nm)
        silica_data = proxy.load_data('SiO2.txt', 'wl-eps', 'nm', 'um')
        print(f"Loaded {silica_data.shape[0]} data points")
        
        # Load frequency-based n,k data (frequencies in THz)
        silicon_data = proxy.load_data('Si.txt', 'freq-nk', 'thz', 'um')
        print(f"Wavelength range: {silicon_data[:, 0].min():.2f} - "
              f"{silicon_data[:, 0].max():.2f} μm")
        ```
        
        Keywords:
            data loading, file reading, unit conversion, data processing,
            material properties, permittivity, refractive index
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
        """Apply appropriate unit conversions based on data format.
        
        This internal method processes raw data from a file according to the
        specified format, performing necessary unit conversions and transformations
        between different material property representations.
        
        Args:
            raw_data (numpy.ndarray): Raw data from file, typically with shape (N, 2) or (N, 3).
            data_format (str): Format specification ('freq-eps', 'wl-eps', 'freq-nk', 'wl-nk').
            data_unit (str): Unit of the first column in raw_data.
            target_unit (str): Desired wavelength unit for the output.
            
        Returns:
            numpy.ndarray: Processed data with shape (N, 3) containing
                        [wavelength, eps_real, eps_imag] for each data point.
            
        Raises:
            ValueError: If the data format is not supported or the data is invalid.
            
        Keywords:
            internal method, data conversion, format processing, permittivity calculation
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
        """Check if a file has a header row.
        
        This internal method examines the first line of a file to determine
        if it contains a header (non-numeric text) or data.
        
        Args:
            filename (str): Path to the file to check.
            
        Returns:
            bool: True if the file has a header row, False otherwise.
            
        Keywords:
            internal method, file header, data loading, preprocessing
        """
        with open(filename, 'r') as file:
            first_line = file.readline().strip()
            return not all(char.isdigit() or char in '.- ' for char in first_line)
            
    def extract_permittivity(self, data, wavelengths, fit_order=10):
        """Extract permittivity values at specific wavelengths using polynomial fitting.
        
        This method interpolates/extrapolates permittivity values for specified
        wavelengths from a set of material data. It uses polynomial fitting to
        generate smooth curves for both real and imaginary parts of the permittivity.
        
        Args:
            data (numpy.ndarray): Material data from load_data with shape (N, 3).
            wavelengths (numpy.ndarray): Target wavelengths to extract permittivity for.
            fit_order (int, optional): Polynomial fit order. Higher values provide
                       more accurate fits for complex curves but may lead to
                       overfitting. Default is 10.
            
        Returns:
            tuple: (real_permittivity, imaginary_permittivity) at requested wavelengths.
                 Each is a numpy.ndarray with the same shape as the input wavelengths.
        
        Examples:
        ```python
        import numpy as np
        from torchrdit.material_proxy import MaterialDataProxy
        proxy = MaterialDataProxy()
        
        # Load material data
        data = proxy.load_data('SiO2.txt', 'wl-eps', 'um')
        
        # Extract permittivity at specific wavelengths
        operating_wl = np.array([1.31, 1.55, 1.7])
        eps_real, eps_imag = proxy.extract_permittivity(data, operating_wl)
        
        # Display the results
        for wl, er, ei in zip(operating_wl, eps_real(operating_wl), eps_imag(operating_wl)):
            print(f"At {wl} μm: ε = {er:.4f} {ei:.4f}j")
        ```
        
        Keywords:
            permittivity extraction, interpolation, polynomial fitting, 
            material properties, dispersion, wavelength-dependent properties
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
        return poly_real, poly_imag