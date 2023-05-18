"""Module defines some constants."""
import numpy as np

# define constant values
EPS_0 = 8.85418782e-12          # vacuum permittivity
MU_0 = 1.25663706e-6            # vacuum permeability
C_0 = 1 / np.sqrt(EPS_0 * MU_0)    # speed of light in vacuum
ETA_0 = np.sqrt(MU_0 / EPS_0)      # vacuum impeadance
Q_E = 1.602176634e-19           # fundamental charge

frequnit_dict = {
    'hz': 1.0,
    'khz': 1.0e3,
    'mhz': 1.0e6,
    'ghz': 1.0e9,
    'thz': 1.0e12,
    'phz': 1.0e15,
}

lengthunit_dict = {
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
