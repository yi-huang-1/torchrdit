import unittest
import torch
import numpy as np

from torchrdit.solver import create_solver
from torchrdit.constants import (
    Algorithm, Precision, EPS_0, MU_0, C_0, ETA_0, Q_E,
    frequnit_dict, lengthunit_dict
)


class TestConstantsDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in constants.py.
    
    This test suite ensures that the examples provided in the docstrings of
    constants.py work as expected. It tests examples for:
    1. Using physical constants
    2. Unit conversions with provided dictionaries
    3. Algorithm enumeration usage
    4. Precision enumeration usage
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def test_physical_constants_example(self):
        """Test the example of using physical constants."""
        # Example from module docstring
        # Calculate the refractive index from relative permittivity
        epsilon_r = 2.25  # SiO2
        n = (epsilon_r)**0.5
        
        # Verify the calculation
        self.assertAlmostEqual(n, 1.5, places=6)
        
        # Additional verification of constants
        # Speed of light calculation from EPS_0 and MU_0
        c_calculated = 1 / np.sqrt(EPS_0 * MU_0)
        self.assertAlmostEqual(c_calculated, C_0, places=6)
        
        # Vacuum impedance calculation from MU_0 and EPS_0
        eta_calculated = np.sqrt(MU_0 / EPS_0)
        self.assertAlmostEqual(eta_calculated, ETA_0, places=6)
    
    def test_unit_conversion_example(self):
        """Test the example of converting between units."""
        # Example from module docstring
        # Convert 1550 nm to meters
        wavelength_nm = 1550
        wavelength_m = wavelength_nm * lengthunit_dict['nm']
        
        # Verify the conversion
        self.assertAlmostEqual(wavelength_m, 1.55e-6, places=12)
        
        # Additional tests for other unit conversions
        # Frequency conversion: THz to Hz
        freq_thz = 193.5  # Common optical frequency
        freq_hz = freq_thz * frequnit_dict['thz']
        self.assertEqual(freq_hz, 193.5e12)
        
        # Length conversion: μm to nm
        length_um = 2.0
        length_nm = length_um * (lengthunit_dict['um'] / lengthunit_dict['nm'])
        self.assertAlmostEqual(length_nm, 2000.0, places=10)
    
    def test_algorithm_enum_example(self):
        """Test the Algorithm enumeration examples."""
        # Example from Algorithm docstring
        # Create solver with RCWA algorithm
        rcwa_solver = create_solver(
            algorithm=Algorithm.RCWA,
            rdim=[256, 256],
            kdim=[5, 5]
        )
        
        # Create solver with RDIT algorithm
        rdit_solver = create_solver(
            algorithm=Algorithm.RDIT,
            rdim=[512, 512],
            kdim=[7, 7]
        )
        
        # Check which algorithm a solver uses
        self.assertEqual(rcwa_solver.algorithm.name, "RCWA")
        self.assertEqual(rdit_solver.algorithm.name, "R-DIT")
        
        # Verify solver dimensions were set correctly
        self.assertEqual(rcwa_solver.rdim, [256, 256])
        self.assertEqual(rcwa_solver.kdim, [5, 5])
        self.assertEqual(rdit_solver.rdim, [512, 512])
        self.assertEqual(rdit_solver.kdim, [7, 7])
    
    def test_precision_enum_example(self):
        """Test the Precision enumeration examples."""
        # Example from Precision docstring
        # Create solver with single precision
        single_prec_solver = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.SINGLE
        )
        
        # Create solver with double precision
        double_prec_solver = create_solver(
            algorithm=Algorithm.RCWA,
            precision=Precision.DOUBLE
        )
        
        # Verify precision settings affect the solver's data types
        self.assertEqual(single_prec_solver.tfloat, torch.float32)
        self.assertEqual(double_prec_solver.tfloat, torch.float64)
        
        # The precision affects internal tensor dtype - verify the statement
        self.assertEqual(torch.float32, torch.float32)  # Single precision
        self.assertEqual(torch.float64, torch.float64)  # Double precision


if __name__ == '__main__':
    unittest.main() 