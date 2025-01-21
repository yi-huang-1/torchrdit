"""Module defines some constants used in the electromagnetic solver."""

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
    """Enumeration of supported algorithms."""
    RCWA = auto()
    RDIT = auto()

@unique
class Precision(Enum):
    SINGLE = auto()
    DOUBLE = auto()
