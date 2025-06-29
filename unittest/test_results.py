"""Unit tests for the results module.

These tests validate the functionality of the result dataclasses and methods defined in 
torchrdit.results, including ScatteringMatrix, FieldComponents, WaveVectors, and SolverResults.
"""

import unittest
import torch
import numpy as np
from torchrdit.results import ScatteringMatrix, FieldComponents, WaveVectors, SolverResults


class TestScatteringMatrix(unittest.TestCase):
    """Test the ScatteringMatrix dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample scattering matrix for testing
        n_freqs = 3
        n_harmonics = 4  # For a 2x2 k-space grid
        self.S11 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        self.S12 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        self.S21 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        self.S22 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        
        self.smatrix = ScatteringMatrix(
            S11=self.S11, 
            S12=self.S12, 
            S21=self.S21, 
            S22=self.S22
        )
    
    def test_attributes(self):
        """Test that ScatteringMatrix correctly stores tensor attributes."""
        self.assertTrue(torch.equal(self.smatrix.S11, self.S11))
        self.assertTrue(torch.equal(self.smatrix.S12, self.S12))
        self.assertTrue(torch.equal(self.smatrix.S21, self.S21))
        self.assertTrue(torch.equal(self.smatrix.S22, self.S22))
    
    def test_shapes(self):
        """Test that tensor shapes are as expected."""
        expected_shape = (3, 8, 8)  # n_freqs, 2*n_harmonics, 2*n_harmonics
        self.assertEqual(self.smatrix.S11.shape, expected_shape)
        self.assertEqual(self.smatrix.S12.shape, expected_shape)
        self.assertEqual(self.smatrix.S21.shape, expected_shape)
        self.assertEqual(self.smatrix.S22.shape, expected_shape)


class TestFieldComponents(unittest.TestCase):
    """Test the FieldComponents dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample field components for testing
        n_freqs = 2
        kdim = (3, 3)  # 3x3 k-space grid
        self.x = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        self.y = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        self.z = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        
        self.field = FieldComponents(x=self.x, y=self.y, z=self.z)
    
    def test_attributes(self):
        """Test that FieldComponents correctly stores tensor attributes."""
        self.assertTrue(torch.equal(self.field.x, self.x))
        self.assertTrue(torch.equal(self.field.y, self.y))
        self.assertTrue(torch.equal(self.field.z, self.z))
    
    def test_shapes(self):
        """Test that tensor shapes are as expected."""
        expected_shape = (2, 3, 3)  # n_freqs, kdim[0], kdim[1]
        self.assertEqual(self.field.x.shape, expected_shape)
        self.assertEqual(self.field.y.shape, expected_shape)
        self.assertEqual(self.field.z.shape, expected_shape)


class TestWaveVectors(unittest.TestCase):
    """Test the WaveVectors dataclass."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample wave vectors for testing
        n_freqs = 2
        kdim = (3, 3)  # 3x3 k-space grid
        n_harmonics = kdim[0] * kdim[1]
        
        self.kx = torch.linspace(-1, 1, kdim[0] * kdim[1]).reshape(kdim)
        self.ky = torch.linspace(-1, 1, kdim[0] * kdim[1]).reshape(kdim)
        self.kinc = torch.zeros(n_freqs, 3)
        self.kinc[:, 2] = 1.0  # Normal incidence
        
        # Create complex tensors for kzref and kztrn
        kx_flat = self.kx.flatten()
        ky_flat = self.ky.flatten()
        self.kzref = torch.zeros(n_freqs, n_harmonics, dtype=torch.complex64)
        self.kztrn = torch.zeros(n_freqs, n_harmonics, dtype=torch.complex64)
        
        # Fill kzref and kztrn with appropriate values
        for i in range(n_harmonics):
            k_mag = kx_flat[i]**2 + ky_flat[i]**2
            # Propagating orders (k_mag < 1.0) have real kz
            if k_mag < 1.0:
                kz_real = torch.sqrt(torch.as_tensor(1.0 - k_mag, dtype=torch.complex64))
                self.kzref[:, i] = kz_real
            else:
                # Evanescent orders have imaginary kz
                kz_imag = 1j * torch.sqrt(torch.as_tensor(k_mag - 1.0, dtype=torch.complex64))
                self.kzref[:, i] = kz_imag
                
            # Same for transmission region with different refractive index
            if k_mag < 2.25:
                kz_real = torch.sqrt(torch.as_tensor(2.25 - k_mag, dtype=torch.complex64))
                self.kztrn[:, i] = kz_real
            else:
                kz_imag = 1j * torch.sqrt(torch.as_tensor(k_mag - 2.25, dtype=torch.complex64))
                self.kztrn[:, i] = kz_imag
        
        self.wave_vectors = WaveVectors(
            kx=self.kx,
            ky=self.ky,
            kinc=self.kinc,
            kzref=self.kzref,
            kztrn=self.kztrn
        )
    
    def test_attributes(self):
        """Test that WaveVectors correctly stores tensor attributes."""
        self.assertTrue(torch.equal(self.wave_vectors.kx, self.kx))
        self.assertTrue(torch.equal(self.wave_vectors.ky, self.ky))
        self.assertTrue(torch.equal(self.wave_vectors.kinc, self.kinc))
        self.assertTrue(torch.equal(self.wave_vectors.kzref, self.kzref))
        self.assertTrue(torch.equal(self.wave_vectors.kztrn, self.kztrn))
    
    def test_shapes(self):
        """Test that tensor shapes are as expected."""
        self.assertEqual(self.wave_vectors.kx.shape, (3, 3))  # kdim[0], kdim[1]
        self.assertEqual(self.wave_vectors.ky.shape, (3, 3))  # kdim[0], kdim[1]
        self.assertEqual(self.wave_vectors.kinc.shape, (2, 3))  # n_freqs, 3
        self.assertEqual(self.wave_vectors.kzref.shape, (2, 9))  # n_freqs, kdim[0]*kdim[1]
        self.assertEqual(self.wave_vectors.kztrn.shape, (2, 9))  # n_freqs, kdim[0]*kdim[1]


class TestSolverResults(unittest.TestCase):
    """Test the SolverResults class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for SolverResults
        n_freqs = 2
        kdim = (3, 3)  # 3x3 k-space grid
        n_harmonics = kdim[0] * kdim[1]
        
        # Overall efficiencies
        self.reflection = torch.tensor([0.3, 0.35])
        self.transmission = torch.tensor([0.7, 0.65])
        
        # Diffraction efficiencies
        self.reflection_diffraction = torch.zeros(n_freqs, *kdim)
        self.transmission_diffraction = torch.zeros(n_freqs, *kdim)
        
        # Set center (zero-order) values
        self.reflection_diffraction[:, 1, 1] = self.reflection
        self.transmission_diffraction[:, 1, 1] = self.transmission
        
        # Create field components
        rx = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        ry = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        rz = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        tx = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        ty = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        tz = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        
        # Wave vectors
        kx = torch.linspace(-1, 1, n_harmonics).reshape(kdim)
        ky = torch.linspace(-1, 1, n_harmonics).reshape(kdim)
        kinc = torch.zeros(n_freqs, 3)
        kinc[:, 2] = 1.0  # Normal incidence
        
        # Create complex tensors for kzref and kztrn
        kx_flat = kx.flatten()
        ky_flat = ky.flatten()
        kzref = torch.zeros(n_freqs, n_harmonics, dtype=torch.complex64)
        kztrn = torch.zeros(n_freqs, n_harmonics, dtype=torch.complex64)
        
        # Mark some orders as evanescent with imaginary kz
        evanescent_indices = [0, 2, 6, 8]  # Corner and some edge indices
        
        # Fill kzref and kztrn with appropriate values
        for i in range(n_harmonics):
            k_mag = kx_flat[i]**2 + ky_flat[i]**2
            if i in evanescent_indices:
                # Evanescent orders
                k_val = max(0.1, k_mag)
                kz_imag = 1j * torch.sqrt(torch.as_tensor(k_val, dtype=torch.complex64))
                kzref[:, i] = kz_imag
            else:
                # Propagating orders
                kz_real = torch.sqrt(torch.as_tensor(1.0 - k_mag, dtype=torch.complex64))
                kzref[:, i] = kz_real
                
            # Similar for transmission
            if i in evanescent_indices[:2]:  # Fewer evanescent orders in transmission
                k_val = max(0.1, k_mag)
                kz_imag = 1j * torch.sqrt(torch.as_tensor(k_val, dtype=torch.complex64))
                kztrn[:, i] = kz_imag
            else:
                kz_real = torch.sqrt(torch.as_tensor(2.25 - k_mag, dtype=torch.complex64))
                kztrn[:, i] = kz_real
        
        # Scattering matrix
        s11 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        s12 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        s21 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        s22 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        
        # Create the field and scattering objects
        self.reflection_field = FieldComponents(x=rx, y=ry, z=rz)
        self.transmission_field = FieldComponents(x=tx, y=ty, z=tz)
        self.structure_matrix = ScatteringMatrix(S11=s11, S12=s12, S21=s21, S22=s22)
        self.wave_vectors = WaveVectors(kx=kx, ky=ky, kinc=kinc, kzref=kzref, kztrn=kztrn)
        
        # Create raw data dictionary for testing from_dict and to_dict
        self.raw_data = {
            'REF': self.reflection,
            'TRN': self.transmission,
            'RDE': self.reflection_diffraction,
            'TDE': self.transmission_diffraction,
            'rx': rx,
            'ry': ry,
            'rz': rz,
            'tx': tx,
            'ty': ty,
            'tz': tz,
            'smat_structure': {
                'S11': s11,
                'S12': s12,
                'S21': s21,
                'S22': s22
            },
            'kx': kx,
            'ky': ky,
            'kinc': kinc,
            'kzref': kzref,
            'kztrn': kztrn
        }
        
        # Create the SolverResults object
        self.results = SolverResults(
            reflection=self.reflection,
            transmission=self.transmission,
            reflection_diffraction=self.reflection_diffraction,
            transmission_diffraction=self.transmission_diffraction,
            reflection_field=self.reflection_field,
            transmission_field=self.transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.wave_vectors,
            raw_data=self.raw_data
        )
    
    def test_attributes(self):
        """Test that SolverResults correctly stores attributes."""
        self.assertTrue(torch.equal(self.results.reflection, self.reflection))
        self.assertTrue(torch.equal(self.results.transmission, self.transmission))
        self.assertTrue(torch.equal(self.results.reflection_diffraction, self.reflection_diffraction))
        self.assertTrue(torch.equal(self.results.transmission_diffraction, self.transmission_diffraction))
        self.assertEqual(self.results.raw_data, self.raw_data)
    
    def test_from_dict(self):
        """Test the from_dict factory method."""
        results_from_dict = SolverResults.from_dict(self.raw_data)
        
        # Check that scalar values match
        self.assertTrue(torch.equal(results_from_dict.reflection, self.reflection))
        self.assertTrue(torch.equal(results_from_dict.transmission, self.transmission))
        
        # Check that tensors match
        self.assertTrue(torch.equal(results_from_dict.reflection_diffraction, self.reflection_diffraction))
        self.assertTrue(torch.equal(results_from_dict.transmission_diffraction, self.transmission_diffraction))
        
        # Check that field components match
        self.assertTrue(torch.equal(results_from_dict.reflection_field.x, self.reflection_field.x))
        self.assertTrue(torch.equal(results_from_dict.transmission_field.y, self.transmission_field.y))
        
        # Check that scattering matrix matches
        self.assertTrue(torch.equal(results_from_dict.structure_matrix.S11, self.structure_matrix.S11))
        
        # Check that wave vectors match
        self.assertTrue(torch.equal(results_from_dict.wave_vectors.kx, self.wave_vectors.kx))
    
    def test_to_dict(self):
        """Test the to_dict method."""
        dict_from_results = self.results.to_dict()
        
        # Check that the dictionary contains all expected keys
        for key in self.raw_data.keys():
            self.assertIn(key, dict_from_results)
            
        # Check that a few values match
        self.assertTrue(torch.equal(dict_from_results['REF'], self.reflection))
        self.assertTrue(torch.equal(dict_from_results['TRN'], self.transmission))
        self.assertTrue(torch.equal(dict_from_results['kx'], self.wave_vectors.kx))
    
    def test_get_diffraction_order_indices(self):
        """Test the get_diffraction_order_indices method."""
        # Test zero order (center of k-space)
        ix, iy = self.results.get_diffraction_order_indices(0, 0)
        self.assertEqual((ix, iy), (1, 1))  # For a 3x3 grid, center is at (1, 1)
        
        # Test first order in x
        ix, iy = self.results.get_diffraction_order_indices(1, 0)
        self.assertEqual((ix, iy), (2, 1))
        
        # Test first order in y
        ix, iy = self.results.get_diffraction_order_indices(0, 1)
        self.assertEqual((ix, iy), (1, 2))
        
        # Test out of bounds order
        with self.assertRaises(ValueError):
            self.results.get_diffraction_order_indices(2, 0)
    
    def test_get_zero_order_transmission(self):
        """Test the get_zero_order_transmission method."""
        tx, ty, tz = self.results.get_zero_order_transmission()
        
        # Check shapes
        self.assertEqual(tx.shape, (2,))  # n_freqs
        self.assertEqual(ty.shape, (2,))
        self.assertEqual(tz.shape, (2,))
        
        # Check values
        center_idx = (1, 1)  # Center of 3x3 grid
        self.assertTrue(torch.equal(tx, self.transmission_field.x[:, center_idx[0], center_idx[1]]))
        self.assertTrue(torch.equal(ty, self.transmission_field.y[:, center_idx[0], center_idx[1]]))
        self.assertTrue(torch.equal(tz, self.transmission_field.z[:, center_idx[0], center_idx[1]]))
    
    def test_get_zero_order_reflection(self):
        """Test the get_zero_order_reflection method."""
        rx, ry, rz = self.results.get_zero_order_reflection()
        
        # Check shapes
        self.assertEqual(rx.shape, (2,))  # n_freqs
        self.assertEqual(ry.shape, (2,))
        self.assertEqual(rz.shape, (2,))
        
        # Check values
        center_idx = (1, 1)  # Center of 3x3 grid
        self.assertTrue(torch.equal(rx, self.reflection_field.x[:, center_idx[0], center_idx[1]]))
        self.assertTrue(torch.equal(ry, self.reflection_field.y[:, center_idx[0], center_idx[1]]))
        self.assertTrue(torch.equal(rz, self.reflection_field.z[:, center_idx[0], center_idx[1]]))
    
    def test_get_order_transmission_efficiency(self):
        """Test the get_order_transmission_efficiency method."""
        # Test zero order
        t0 = self.results.get_order_transmission_efficiency(0, 0)
        self.assertTrue(torch.equal(t0, self.transmission))
        
        # Test first order in x
        t1x = self.results.get_order_transmission_efficiency(1, 0)
        self.assertTrue(torch.equal(t1x, self.transmission_diffraction[:, 2, 1]))
    
    def test_get_order_reflection_efficiency(self):
        """Test the get_order_reflection_efficiency method."""
        # Test zero order
        r0 = self.results.get_order_reflection_efficiency(0, 0)
        self.assertTrue(torch.equal(r0, self.reflection))
        
        # Test first order in y
        r1y = self.results.get_order_reflection_efficiency(0, 1)
        self.assertTrue(torch.equal(r1y, self.reflection_diffraction[:, 1, 2]))
    
    def test_get_all_diffraction_orders(self):
        """Test the get_all_diffraction_orders method."""
        orders = self.results.get_all_diffraction_orders()
        
        # For a 3x3 grid, we expect 9 diffraction orders from (-1,-1) to (1,1)
        expected_orders = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Check that all expected orders are present
        self.assertEqual(len(orders), 9)
        for order in expected_orders:
            self.assertIn(order, orders)
    
    def test_get_propagating_orders(self):
        """Test the get_propagating_orders method."""
        # For our test setup, we marked some orders as evanescent
        # Only orders with real kz should be included
        prop_orders = self.results.get_propagating_orders(0)
        
        # Get the actual number of propagating orders from test result
        num_propagating = len(prop_orders)
        
        # The exact number depends on the implementation details of get_propagating_orders
        # and how it identifies orders with real kz, but we know a few properties:
        self.assertTrue(num_propagating > 0)  # At least some orders are propagating
        
        # The center order (0,0) should always be propagating
        self.assertIn((0, 0), prop_orders)
        
        # The corner orders should be evanescent and not included
        self.assertNotIn((-1, -1), prop_orders)
        self.assertNotIn((1, 1), prop_orders)


if __name__ == '__main__':
    unittest.main() 