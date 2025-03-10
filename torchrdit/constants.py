"""Constants module for the TorchRDIT electromagnetic solver package.

This module defines physical constants, unit conversion factors, and enumerations
used throughout the TorchRDIT package. These include:

- Physical constants (vacuum permittivity, permeability, speed of light, etc.)
- Frequency unit conversion factors (Hz, kHz, MHz, etc.)
- Length unit conversion factors (meter, μm, nm, etc.)
- Algorithm enumerations (RCWA, R-DIT)
- Precision enumerations (single, double)

These constants provide a centralized reference for consistent values across the
package and allow for convenient unit conversions in calculations.
"""

import numpy as np
from enum import Enum, auto, unique

# Physical constants
EPS_0: float = 8.85418782e-12          # vacuum permittivity
MU_0: float = 1.25663706e-6            # vacuum permeability
C_0: float = 1 / np.sqrt(EPS_0 * MU_0)    # speed of light in vacuum
ETA_0: float = np.sqrt(MU_0 / EPS_0)      # vacuum impedance
Q_E: float = 1.602176634e-19           # fundamental charge

# Frequency unit conversion factors
frequnit_dict: dict[str, float] = {
    'hz': 1.0,
    'khz': 1.0e3,
    'mhz': 1.0e6,
    'ghz': 1.0e9,
    'thz': 1.0e12,
    'phz': 1.0e15,
}

# Length unit conversion factors
lengthunit_dict: dict[str, float] = {
    'meter': 1.0,
    'km': 1.0e3,
    'dm': 1.0e-1,
    'cm': 1.0e-2,
    'mm': 1.0e-3,
    'um': 1.0e-6,
    'nm': 1.0e-9,
    'pm': 1.0e-12,
    'angstrom': 1.0e-10,
}

@unique
class Algorithm(Enum):
    """Enumeration of supported electromagnetic solver algorithms.
    
    This enum defines the available algorithms that can be used for electromagnetic
    field calculations:
    
    - RCWA: Rigorous Coupled-Wave Analysis, the traditional approach using eigenmode
            decomposition.
    - RDIT: Rigorous Diffraction Interface Theory (R-DIT), the eigendecomposition-free
            approach that offers improved computational efficiency.
    """
    RCWA = auto()  # Rigorous Coupled-Wave Analysis
    RDIT = auto()  # Rigorous Diffraction Interface Theory (R-DIT)

@unique
class Precision(Enum):
    """Enumeration of numerical precision options for calculations.
    
    This enum defines the available precision options for calculations:
    
    - SINGLE: Single precision (32-bit floating point, torch.float32)
    - DOUBLE: Double precision (64-bit floating point, torch.float64)
    
    Higher precision provides more accurate results but requires more memory and
    computational resources.
    """
    SINGLE = auto()  # 32-bit floating point (torch.float32)
    DOUBLE = auto()  # 64-bit floating point (torch.float64)
