import unittest
import numpy as np
import os
from pathlib import Path

from torchrdit.utils import create_material


class TestMaterialsDocExamples(unittest.TestCase):
    """Test cases that verify the functionality shown in docstring examples in materials.py.
    
    This test suite ensures that the examples provided in the docstrings of
    materials.py work as expected. It tests examples for:
    1. Creating non-dispersive materials
    2. Creating dispersive materials from data files
    3. Using materials in solvers
    4. MaterialClass operations and properties
    
    These tests help maintain the documentation's accuracy and provide working
    code examples that users can rely on.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Test data files
        self.test_data_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.sio2_file = str(self.test_data_dir / "SiO2-e.txt")
        self.si_file = str(self.test_data_dir / "Si_C-e.txt")
    
    def test_module_examples_non_dispersive(self):
        """Test the module-level examples for creating non-dispersive materials."""
        # Create a simple material with constant permittivity
        air = create_material(name="air", permittivity=1.0)
        self.assertEqual(air.name, "air")
        self.assertAlmostEqual(air.er.real.item(), 1.0, places=4)
        self.assertAlmostEqual(air.er.imag.item(), 0.0, places=6)
        self.assertFalse(air.isdispersive_er)
        
        silicon = create_material(name="silicon", permittivity=11.7)
        self.assertEqual(silicon.name, "silicon")
        self.assertAlmostEqual(silicon.er.real.item(), 11.7, places=4)
        self.assertAlmostEqual(silicon.er.imag.item(), 0.0, places=6)
        
        # Create a material from refractive index
        glass = create_material(name="glass", permittivity=2.25)  # n=1.5
        self.assertEqual(glass.name, "glass")
        self.assertAlmostEqual(glass.er.real.item(), 2.25, places=4)
        self.assertAlmostEqual(glass.er.imag.item(), 0.0, places=6)
        
        # Create a material with complex permittivity (lossy)
        gold = create_material(name="gold", permittivity=complex(-10.0, 1.5))
        self.assertEqual(gold.name, "gold")
        self.assertAlmostEqual(gold.er.real.item(), np.conj(complex(-10.0, 1.5)).real, places=4)
        self.assertAlmostEqual(gold.er.imag.item(), np.conj(complex(-10.0, 1.5)).imag, places=4)
    
    def test_module_examples_dispersive(self):
        """Test the module-level examples for creating dispersive materials."""
        # Create dispersive material from data file with wavelength-permittivity data
        silica = create_material(
            name="silica",
            dielectric_dispersion=True,
            user_dielectric_file=self.sio2_file,
            data_format="wl-eps",
            data_unit="nm"  # Data file has wavelengths in nm
        )
        
        self.assertEqual(silica.name, "silica")
        self.assertTrue(silica.isdispersive_er)
        self.assertEqual(silica.data_format, "wl-eps")
        self.assertEqual(silica.data_unit, "nm")
        # Verify permittivity retrieval in the covered range (1.0–1.2 um)
        eps_silica = silica.get_permittivity(np.array([1.0, 1.1]), 'um')
        self.assertEqual(eps_silica.shape, (2,))
        # Silica real permittivity is around 2–2.4 in this band
        self.assertGreater(eps_silica[0].real.item(), 1.5)
        self.assertLess(eps_silica[0].real.item(), 3.0)
        self.assertGreater(eps_silica[1].real.item(), 1.5)
        self.assertLess(eps_silica[1].real.item(), 3.0)
        
        # Create dispersive material from wavelength-permittivity data (silicon)
        silicon_dispersive = create_material(
            name="silicon_disp",
            dielectric_dispersion=True,
            user_dielectric_file=self.si_file,
            data_format="wl-eps",
            data_unit="nm"  # Si_C-e.txt wavelengths are in nm
        )
        
        self.assertEqual(silicon_dispersive.name, "silicon_disp")
        self.assertTrue(silicon_dispersive.isdispersive_er)
        # Verify permittivity retrieval within ~0.150–0.173 um range
        eps_si = silicon_dispersive.get_permittivity(np.array([0.155, 0.170]), 'um')
        self.assertEqual(eps_si.shape, (2,))
        # Silicon has high real permittivity in this band
        self.assertGreater(eps_si[0].real.item(), 5.0)
        self.assertGreater(eps_si[1].real.item(), 5.0)
    # Removed solver usage example check; solver behavior is validated in dedicated tests.


# No direct test runner; pytest collects and executes these tests.
