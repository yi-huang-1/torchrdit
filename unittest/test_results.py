"""Unit tests for the results module.

These tests validate the functionality of the result dataclasses and methods defined in
torchrdit.results, including ScatteringMatrix, FieldComponents, WaveVectors, and SolverResults.
"""

import unittest
import torch
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
        harmonics = (3, 3)  # 3x3 k-space grid
        self.x = torch.randn(n_freqs, *harmonics, dtype=torch.complex64)
        self.y = torch.randn(n_freqs, *harmonics, dtype=torch.complex64)
        self.z = torch.randn(n_freqs, *harmonics, dtype=torch.complex64)

        self.field = FieldComponents(x=self.x, y=self.y, z=self.z)

    def test_shapes(self):
        """Test that tensor shapes are as expected."""
        expected_shape = (2, 3, 3)  # n_freqs, harmonics[0], harmonics[1]
        self.assertEqual(self.field.x.shape, expected_shape)
        self.assertEqual(self.field.y.shape, expected_shape)
        self.assertEqual(self.field.z.shape, expected_shape)


class TestWaveVectors(unittest.TestCase):
    """Test the WaveVectors dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample wave vectors for testing
        n_freqs = 2
        harmonics = (3, 3)  # 3x3 k-space grid
        n_harmonics = harmonics[0] * harmonics[1]

        self.kx = torch.linspace(-1, 1, harmonics[0] * harmonics[1]).reshape(harmonics)
        self.ky = torch.linspace(-1, 1, harmonics[0] * harmonics[1]).reshape(harmonics)
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

    def test_shapes(self):
        """Test that tensor shapes are as expected."""
        self.assertEqual(self.wave_vectors.kx.shape, (3, 3))  # harmonics[0], harmonics[1]
        self.assertEqual(self.wave_vectors.ky.shape, (3, 3))  # harmonics[0], harmonics[1]
        self.assertEqual(self.wave_vectors.kinc.shape, (2, 3))  # n_freqs, 3
        self.assertEqual(self.wave_vectors.kzref.shape, (2, 9))  # n_freqs, harmonics[0]*harmonics[1]
        self.assertEqual(self.wave_vectors.kztrn.shape, (2, 9))  # n_freqs, harmonics[0]*harmonics[1]


class TestSolverResults(unittest.TestCase):
    """Test the SolverResults class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for SolverResults
        n_freqs = 2
        harmonics = (3, 3)  # 3x3 k-space grid
        n_harmonics = harmonics[0] * harmonics[1]

        # Overall efficiencies
        self.reflection = torch.tensor([0.3, 0.35])
        self.transmission = torch.tensor([0.7, 0.65])

        # Diffraction efficiencies
        self.reflection_diffraction = torch.zeros(n_freqs, *harmonics)
        self.transmission_diffraction = torch.zeros(n_freqs, *harmonics)

        # Set center (zero-order) values
        self.reflection_diffraction[:, 1, 1] = self.reflection
        self.transmission_diffraction[:, 1, 1] = self.transmission

        # Create field components
        rx = torch.randn(n_freqs, *harmonics, dtype=torch.complex64)
        ry = torch.randn(n_freqs, *harmonics, dtype=torch.complex64)
        rz = torch.randn(n_freqs, *harmonics, dtype=torch.complex64)
        tx = torch.randn(n_freqs, *harmonics, dtype=torch.complex64)
        ty = torch.randn(n_freqs, *harmonics, dtype=torch.complex64)
        tz = torch.randn(n_freqs, *harmonics, dtype=torch.complex64)

        # Wave vectors
        kx = torch.linspace(-1, 1, n_harmonics).reshape(harmonics)
        ky = torch.linspace(-1, 1, n_harmonics).reshape(harmonics)
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

    def test_to_structured_dict(self):
        """Test the to_structured_dict method."""
        structured = self.results.to_structured_dict()

        self.assertIn("efficiency", structured)
        self.assertIn("diffraction_efficiency", structured)
        self.assertIn("field_fourier_coefficients", structured)
        self.assertIn("scattering_matrix", structured)
        self.assertIn("wavevectors", structured)

        self.assertTrue(torch.equal(structured["efficiency"]["reflection"], self.reflection))
        self.assertTrue(torch.equal(structured["efficiency"]["transmission"], self.transmission))
        self.assertTrue(torch.equal(structured["diffraction_efficiency"]["reflection"], self.reflection_diffraction))
        self.assertTrue(torch.equal(structured["diffraction_efficiency"]["transmission"], self.transmission_diffraction))

        self.assertTrue(torch.equal(structured["wavevectors"]["kx"], self.wave_vectors.kx))
        self.assertTrue(torch.equal(structured["wavevectors"]["ky"], self.wave_vectors.ky))
        self.assertTrue(torch.equal(structured["wavevectors"]["incident"], self.wave_vectors.kinc))
        self.assertTrue(torch.equal(structured["wavevectors"]["kz"]["reflection"], self.wave_vectors.kzref))
        self.assertTrue(torch.equal(structured["wavevectors"]["kz"]["transmission"], self.wave_vectors.kztrn))

        self.assertTrue(torch.equal(structured["scattering_matrix"]["structure"]["S11"], self.structure_matrix.S11))

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


class TestUnifiedSolverResults(unittest.TestCase):
    """Test the unified SolverResults interface for both single and batched sources.

    This test class validates the new unified interface that handles both single
    and batched source results transparently. It tests backward compatibility
    and the new batching functionality merged into SolverResults.
    """

    def setUp(self):
        """Set up test fixtures for unified results testing."""
        # Set up single source results data
        self.n_freqs = 2
        self.harmonics = [3, 3]  # 3x3 k-space grid
        self.n_harmonics = self.harmonics[0] * self.harmonics[1]  # 9 harmonics

        # Single source data (shape: (n_freqs, ...))
        self.single_reflection = torch.tensor([0.1, 0.2], dtype=torch.float32)
        self.single_transmission = torch.tensor([0.8, 0.7], dtype=torch.float32)
        self.single_reflection_diffraction = torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.float32)
        self.single_transmission_diffraction = torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.float32)

        # Field components for single source
        self.single_reflection_field = FieldComponents(
            x=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            y=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            z=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_x=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_y=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_z=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
        )

        self.single_transmission_field = FieldComponents(
            x=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            y=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            z=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_x=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_y=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_z=torch.randn(self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
        )

        # Batched source data (shape: (n_sources, n_freqs, ...))
        self.n_sources = 3
        self.batched_reflection = torch.tensor([
            [0.1, 0.2], [0.15, 0.25], [0.12, 0.22]
        ], dtype=torch.float32)
        self.batched_transmission = torch.tensor([
            [0.8, 0.7], [0.75, 0.65], [0.78, 0.68]
        ], dtype=torch.float32)
        self.batched_loss = 1.0 - self.batched_reflection - self.batched_transmission

        self.batched_reflection_diffraction = torch.randn(
            self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.float32
        )
        self.batched_transmission_diffraction = torch.randn(
            self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.float32
        )

        # Batched field components
        self.batched_reflection_field = FieldComponents(
            x=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            y=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            z=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_x=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_y=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_z=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
        )

        self.batched_transmission_field = FieldComponents(
            x=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            y=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            z=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_x=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_y=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            mag_z=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
        )

        # Shared data (structure matrix, wave vectors)
        self.structure_matrix = ScatteringMatrix(
            S11=torch.randn(self.n_freqs, 2*self.n_harmonics, 2*self.n_harmonics, dtype=torch.complex64),
            S12=torch.randn(self.n_freqs, 2*self.n_harmonics, 2*self.n_harmonics, dtype=torch.complex64),
            S21=torch.randn(self.n_freqs, 2*self.n_harmonics, 2*self.n_harmonics, dtype=torch.complex64),
            S22=torch.randn(self.n_freqs, 2*self.n_harmonics, 2*self.n_harmonics, dtype=torch.complex64)
        )
        self.batched_structure_matrix = ScatteringMatrix(
            S11=torch.randn(self.n_sources, self.n_freqs, 2*self.n_harmonics, 2*self.n_harmonics, dtype=torch.complex64),
            S12=torch.randn(self.n_sources, self.n_freqs, 2*self.n_harmonics, 2*self.n_harmonics, dtype=torch.complex64),
            S21=torch.randn(self.n_sources, self.n_freqs, 2*self.n_harmonics, 2*self.n_harmonics, dtype=torch.complex64),
            S22=torch.randn(self.n_sources, self.n_freqs, 2*self.n_harmonics, 2*self.n_harmonics, dtype=torch.complex64),
        )

        self.wave_vectors = WaveVectors(
            kx=torch.randn(self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            ky=torch.randn(self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            kinc=torch.randn(self.n_freqs, 3, dtype=torch.complex64),
            kzref=torch.randn(self.n_freqs, self.n_harmonics, dtype=torch.complex64),
            kztrn=torch.randn(self.n_freqs, self.n_harmonics, dtype=torch.complex64)
        )

        # Batched wave vectors (n_sources leading dim)
        self.batched_wave_vectors = WaveVectors(
            kx=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            ky=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            kinc=torch.randn(self.n_sources, self.n_freqs, 3, dtype=torch.complex64),
            kzref=torch.randn(self.n_sources, self.n_freqs, self.n_harmonics, dtype=torch.complex64),
            kztrn=torch.randn(self.n_sources, self.n_freqs, self.n_harmonics, dtype=torch.complex64),
        )

        # Lattice vectors for field reconstruction
        self.lattice_t1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
        self.lattice_t2 = torch.tensor([0.0, 1.0], dtype=torch.float32)
        self.default_grids = (64, 64)

        # Source parameters for batched case
        self.source_parameters = [
            {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 0.5, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"theta": 1.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        ]

    def _make_batched_results_with_structure_matrices(self):
        per_source_raw = []
        for i in range(self.n_sources):
            per_source_raw.append({
                "REF": self.batched_reflection[i],
                "TRN": self.batched_transmission[i],
                "smat_structure": ScatteringMatrix(
                    S11=self.batched_structure_matrix.S11[i],
                    S12=self.batched_structure_matrix.S12[i],
                    S21=self.batched_structure_matrix.S21[i],
                    S22=self.batched_structure_matrix.S22[i],
                ),
            })

        return SolverResults(
            reflection=self.batched_reflection,
            transmission=self.batched_transmission,
            reflection_diffraction=self.batched_reflection_diffraction,
            transmission_diffraction=self.batched_transmission_diffraction,
            reflection_field=self.batched_reflection_field,
            transmission_field=self.batched_transmission_field,
            structure_matrix=self.structure_matrix,
            _structure_matrix_batched=self.batched_structure_matrix,
            wave_vectors=self.batched_wave_vectors,
            raw_data={"_per_source": per_source_raw},
            n_sources=self.n_sources,
            source_parameters=self.source_parameters,
            loss=self.batched_loss,
        )

    def test_single_source_creation(self):
        """Test creating unified SolverResults for single source (backward compatibility)."""
        # Create single source results (existing behavior)
        results = SolverResults(
            reflection=self.single_reflection,
            transmission=self.single_transmission,
            reflection_diffraction=self.single_reflection_diffraction,
            transmission_diffraction=self.single_transmission_diffraction,
            reflection_field=self.single_reflection_field,
            transmission_field=self.single_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.wave_vectors,
            lattice_t1=self.lattice_t1,
            lattice_t2=self.lattice_t2,
            default_grids=self.default_grids,
        )

        # Test backward compatibility properties
        self.assertFalse(results.is_batched)
        self.assertEqual(results.n_sources, 1)
        self.assertEqual(len(results), 1)

        # Test existing field methods work
        tx, ty, tz = results.get_zero_order_transmission()
        self.assertEqual(tx.shape, (self.n_freqs,))

        # Test field API methods work
        fourier_coeffs = results.get_reflection_interface_fourier_coefficients()
        self.assertIn("S_x", fourier_coeffs)
        self.assertIn("U_x", fourier_coeffs)

    def test_batched_source_creation(self):
        """Test creating unified SolverResults for batched sources (new functionality)."""
        # Create batched source results (new behavior)
        results = SolverResults(
            reflection=self.batched_reflection,
            transmission=self.batched_transmission,
            reflection_diffraction=self.batched_reflection_diffraction,
            transmission_diffraction=self.batched_transmission_diffraction,
            reflection_field=self.batched_reflection_field,
            transmission_field=self.batched_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.batched_wave_vectors,
            lattice_t1=self.lattice_t1,
            lattice_t2=self.lattice_t2,
            default_grids=self.default_grids,
            n_sources=self.n_sources,
            source_parameters=self.source_parameters,
            loss=self.batched_loss,
        )

        # Test batched properties
        self.assertTrue(results.is_batched)
        self.assertEqual(results.n_sources, self.n_sources)
        self.assertEqual(len(results), self.n_sources)

        # Test indexing works
        single_result = results[0]
        self.assertIsInstance(single_result, SolverResults)
        self.assertFalse(single_result.is_batched)
        self.assertEqual(single_result.n_sources, 1)

        # Test slicing works
        subset_results = results[0:2]
        self.assertIsInstance(subset_results, SolverResults)
        self.assertTrue(subset_results.is_batched)
        self.assertEqual(subset_results.n_sources, 2)

        # Test iteration works
        results_list = list(results)
        self.assertEqual(len(results_list), self.n_sources)
        for result in results_list:
            self.assertIsInstance(result, SolverResults)
            self.assertFalse(result.is_batched)

    def test_batched_indexing_functionality(self):
        """Test indexing functionality for batched results."""
        results = SolverResults(
            reflection=self.batched_reflection,
            transmission=self.batched_transmission,
            reflection_diffraction=self.batched_reflection_diffraction,
            transmission_diffraction=self.batched_transmission_diffraction,
            reflection_field=self.batched_reflection_field,
            transmission_field=self.batched_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.batched_wave_vectors,
            n_sources=self.n_sources,
            source_parameters=self.source_parameters,
            loss=self.batched_loss,
        )

        # Test positive indexing
        result0 = results[0]
        self.assertTrue(torch.equal(result0.reflection, self.batched_reflection[0]))
        self.assertTrue(torch.equal(result0.transmission, self.batched_transmission[0]))

        # Test negative indexing
        result_last = results[-1]
        self.assertTrue(torch.equal(result_last.reflection, self.batched_reflection[-1]))

        # Test index out of range
        with self.assertRaises(IndexError):
            _ = results[self.n_sources]

        # Test invalid index type
        with self.assertRaises(TypeError):
            _ = results["invalid"]

    def test_batched_slicing_functionality(self):
        """Test slicing functionality for batched results."""
        results = SolverResults(
            reflection=self.batched_reflection,
            transmission=self.batched_transmission,
            reflection_diffraction=self.batched_reflection_diffraction,
            transmission_diffraction=self.batched_transmission_diffraction,
            reflection_field=self.batched_reflection_field,
            transmission_field=self.batched_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.batched_wave_vectors,
            n_sources=self.n_sources,
            source_parameters=self.source_parameters,
            loss=self.batched_loss,
        )

        # Test normal slice
        subset = results[0:2]
        self.assertEqual(subset.n_sources, 2)
        self.assertTrue(torch.equal(subset.reflection, self.batched_reflection[0:2]))

        # Test slice with step
        subset = results[::2]
        expected_sources = (self.n_sources + 1) // 2  # 3 sources, step 2 → 2 results
        self.assertEqual(subset.n_sources, expected_sources)

        # Test empty slice
        with self.assertRaises(ValueError):
            _ = results[3:3]  # Empty slice

    def test_batched_optimization_methods(self):
        """Test optimization methods for batched results."""
        results = SolverResults(
            reflection=self.batched_reflection,
            transmission=self.batched_transmission,
            reflection_diffraction=self.batched_reflection_diffraction,
            transmission_diffraction=self.batched_transmission_diffraction,
            reflection_field=self.batched_reflection_field,
            transmission_field=self.batched_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.batched_wave_vectors,
            n_sources=self.n_sources,
            source_parameters=self.source_parameters,
            loss=self.batched_loss,
        )

        # Test find_optimal_source with different metrics
        best_trans_idx = results.find_optimal_source('max_transmission')
        self.assertIsInstance(best_trans_idx, int)
        self.assertTrue(0 <= best_trans_idx < self.n_sources)

        best_refl_idx = results.find_optimal_source('min_reflection')
        self.assertIsInstance(best_refl_idx, int)
        self.assertTrue(0 <= best_refl_idx < self.n_sources)

        # Test with specific frequency
        best_freq_idx = results.find_optimal_source('max_transmission', frequency_idx=0)
        self.assertIsInstance(best_freq_idx, int)

        # Test invalid metric
        with self.assertRaises(ValueError):
            results.find_optimal_source('invalid_metric')

    def test_parameter_sweep_functionality(self):
        """Test parameter sweep data extraction for batched results."""
        results = SolverResults(
            reflection=self.batched_reflection,
            transmission=self.batched_transmission,
            reflection_diffraction=self.batched_reflection_diffraction,
            transmission_diffraction=self.batched_transmission_diffraction,
            reflection_field=self.batched_reflection_field,
            transmission_field=self.batched_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.batched_wave_vectors,
            n_sources=self.n_sources,
            source_parameters=self.source_parameters,
            loss=self.batched_loss,
        )

        # Test theta parameter sweep
        theta_values, trans_values = results.get_parameter_sweep_data('theta', 'transmission')
        self.assertEqual(theta_values.shape, (self.n_sources,))
        self.assertEqual(trans_values.shape, (self.n_sources,))

        # Test with different frequency
        _, refl_values = results.get_parameter_sweep_data('theta', 'reflection', frequency_idx=1)
        self.assertTrue(torch.equal(refl_values, self.batched_reflection[:, 1]))

        # Test invalid parameter
        with self.assertRaises(KeyError):
            results.get_parameter_sweep_data('invalid_param', 'transmission')

        # Test invalid metric
        with self.assertRaises(ValueError):
            results.get_parameter_sweep_data('theta', 'invalid_metric')

    def test_backward_compatibility_single_source(self):
        """Test that single source behavior is fully backward compatible."""
        # Create results as before
        results = SolverResults(
            reflection=self.single_reflection,
            transmission=self.single_transmission,
            reflection_diffraction=self.single_reflection_diffraction,
            transmission_diffraction=self.single_transmission_diffraction,
            reflection_field=self.single_reflection_field,
            transmission_field=self.single_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.wave_vectors,
        )

        # Test all existing methods still work
        tx, ty, tz = results.get_zero_order_transmission()
        rx, ry, rz = results.get_zero_order_reflection()
        t_eff = results.get_order_transmission_efficiency(0, 0)
        r_eff = results.get_order_reflection_efficiency(0, 0)
        all_orders = results.get_all_diffraction_orders()
        prop_orders = results.get_propagating_orders()

        # Test shapes and values are as expected
        self.assertEqual(tx.shape, (self.n_freqs,))
        self.assertEqual(len(all_orders), self.harmonics[0] * self.harmonics[1])

        # Test isinstance still works
        self.assertIsInstance(results, SolverResults)

        # Test that new batched attributes work for single source
        self.assertEqual(results.n_sources, 1)
        self.assertFalse(results.is_batched)

    def test_unbatched_slice_returns_self(self):
        """Slicing unbatched results with [0:1] should return self."""
        results = SolverResults(
            reflection=self.single_reflection,
            transmission=self.single_transmission,
            reflection_diffraction=self.single_reflection_diffraction,
            transmission_diffraction=self.single_transmission_diffraction,
            reflection_field=self.single_reflection_field,
            transmission_field=self.single_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.wave_vectors,
        )
        sliced = results[0:1]
        self.assertIs(sliced, results)

    def test_unbatched_slice_rejects_bad_slices(self):
        """Slicing unbatched results with empty or out-of-range slices must raise."""
        results = SolverResults(
            reflection=self.single_reflection,
            transmission=self.single_transmission,
            reflection_diffraction=self.single_reflection_diffraction,
            transmission_diffraction=self.single_transmission_diffraction,
            reflection_field=self.single_reflection_field,
            transmission_field=self.single_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.wave_vectors,
        )
        with self.assertRaises(ValueError):
            _ = results[:0]  # empty slice
        with self.assertRaises((ValueError, IndexError)):
            _ = results[1:2]  # out of range (clamps to empty)

    def test_batched_to_structured_dict_wavevectors_list(self):
        """to_structured_dict for batched results with source-major wave_vectors returns list."""
        results = SolverResults(
            reflection=self.batched_reflection,
            transmission=self.batched_transmission,
            reflection_diffraction=self.batched_reflection_diffraction,
            transmission_diffraction=self.batched_transmission_diffraction,
            reflection_field=self.batched_reflection_field,
            transmission_field=self.batched_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=self.batched_wave_vectors,
            n_sources=self.n_sources,
        )
        structured = results.to_structured_dict()
        wv_out = structured["wavevectors"]
        self.assertIsInstance(wv_out, list)
        self.assertEqual(len(wv_out), self.n_sources)
        # Each entry should have per-source (unbatched) tensors
        self.assertTrue(torch.equal(wv_out[0]["kx"], self.batched_wave_vectors.kx[0]))

    def test_single_element_list_batched_helpers(self):
        """solve([single_source]) must support find_optimal_source/get_parameter_sweep_data."""
        # Simulate solve([source]) — n_sources=1 but tensors are 2D (batched)
        results = SolverResults(
            reflection=self.batched_reflection[:1],  # (1, n_freqs)
            transmission=self.batched_transmission[:1],
            reflection_diffraction=self.batched_reflection_diffraction[:1],
            transmission_diffraction=self.batched_transmission_diffraction[:1],
            reflection_field=self.batched_reflection_field[:1],
            transmission_field=self.batched_transmission_field[:1],
            structure_matrix=self.structure_matrix,
            wave_vectors=self.batched_wave_vectors[:1],
            n_sources=1,
            source_parameters=[self.source_parameters[0]],
            loss=(1.0 - self.batched_reflection - self.batched_transmission)[:1],
        )
        self.assertTrue(results.is_batched)
        self.assertEqual(results.n_sources, 1)
        # These must not raise
        best_idx = results.find_optimal_source('max_transmission')
        self.assertEqual(best_idx, 0)
        theta_vals, trans_vals = results.get_parameter_sweep_data('theta', 'transmission')
        self.assertEqual(theta_vals.shape, (1,))
        self.assertEqual(trans_vals.shape, (1,))

    def test_batched_root_structure_matrix_keeps_legacy_shape(self):
        results = self._make_batched_results_with_structure_matrices()
        self.assertEqual(results.structure_matrix.S11.shape, self.structure_matrix.S11.shape)
        self.assertTrue(torch.equal(results.structure_matrix.S11, self.structure_matrix.S11))

    def test_batched_indexing_uses_per_source_structure_matrix(self):
        results = self._make_batched_results_with_structure_matrices()
        result1 = results[1]
        self.assertTrue(torch.equal(result1.structure_matrix.S11, self.batched_structure_matrix.S11[1]))
        self.assertTrue(torch.equal(result1.structure_matrix.S11, result1.raw_data["smat_structure"].S11))
        self.assertFalse(torch.equal(results[0].structure_matrix.S11, results[1].structure_matrix.S11))

    def test_batched_slice_preserves_structure_matrix_and_raw_data(self):
        results = self._make_batched_results_with_structure_matrices()
        subset = results[1:2]
        self.assertEqual(subset.structure_matrix.S11.shape, self.structure_matrix.S11.shape)
        self.assertTrue(torch.equal(subset.structure_matrix.S11, self.batched_structure_matrix.S11[1]))
        self.assertTrue(torch.equal(subset[0].structure_matrix.S11, results[1].structure_matrix.S11))
        self.assertTrue(torch.equal(subset[0].to_dict()["REF"], results[1].to_dict()["REF"]))


class TestBatchedDiffractionMethods(unittest.TestCase):
    def setUp(self):
        self.n_sources = 3
        self.n_freqs = 2
        self.harmonics = [3, 3]
        self.n_harmonics = self.harmonics[0] * self.harmonics[1]

        self.batched_reflection = torch.randn(self.n_sources, self.n_freqs, dtype=torch.float32)
        self.batched_transmission = torch.randn(self.n_sources, self.n_freqs, dtype=torch.float32)
        self.batched_reflection_diffraction = torch.randn(
            self.n_sources,
            self.n_freqs,
            self.harmonics[0],
            self.harmonics[1],
            dtype=torch.float32,
        )
        self.batched_transmission_diffraction = torch.randn(
            self.n_sources,
            self.n_freqs,
            self.harmonics[0],
            self.harmonics[1],
            dtype=torch.float32,
        )

        self.batched_reflection_field = FieldComponents(
            x=torch.randn(
                self.n_sources,
                self.n_freqs,
                self.harmonics[0],
                self.harmonics[1],
                dtype=torch.complex64,
            ),
            y=torch.randn(
                self.n_sources,
                self.n_freqs,
                self.harmonics[0],
                self.harmonics[1],
                dtype=torch.complex64,
            ),
            z=torch.randn(
                self.n_sources,
                self.n_freqs,
                self.harmonics[0],
                self.harmonics[1],
                dtype=torch.complex64,
            ),
        )
        self.batched_transmission_field = FieldComponents(
            x=torch.randn(
                self.n_sources,
                self.n_freqs,
                self.harmonics[0],
                self.harmonics[1],
                dtype=torch.complex64,
            ),
            y=torch.randn(
                self.n_sources,
                self.n_freqs,
                self.harmonics[0],
                self.harmonics[1],
                dtype=torch.complex64,
            ),
            z=torch.randn(
                self.n_sources,
                self.n_freqs,
                self.harmonics[0],
                self.harmonics[1],
                dtype=torch.complex64,
            ),
        )

        self.unbatched_reflection = self.batched_reflection[0]
        self.unbatched_transmission = self.batched_transmission[0]
        self.unbatched_reflection_diffraction = self.batched_reflection_diffraction[0]
        self.unbatched_transmission_diffraction = self.batched_transmission_diffraction[0]
        self.unbatched_reflection_field = FieldComponents(
            x=self.batched_reflection_field.x[0],
            y=self.batched_reflection_field.y[0],
            z=self.batched_reflection_field.z[0],
        )
        self.unbatched_transmission_field = FieldComponents(
            x=self.batched_transmission_field.x[0],
            y=self.batched_transmission_field.y[0],
            z=self.batched_transmission_field.z[0],
        )

        self.structure_matrix = ScatteringMatrix(
            S11=torch.randn(self.n_freqs, 2 * self.n_harmonics, 2 * self.n_harmonics, dtype=torch.complex64),
            S12=torch.randn(self.n_freqs, 2 * self.n_harmonics, 2 * self.n_harmonics, dtype=torch.complex64),
            S21=torch.randn(self.n_freqs, 2 * self.n_harmonics, 2 * self.n_harmonics, dtype=torch.complex64),
            S22=torch.randn(self.n_freqs, 2 * self.n_harmonics, 2 * self.n_harmonics, dtype=torch.complex64),
        )

        self.single_wave_vectors = WaveVectors(
            kx=torch.randn(self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            ky=torch.randn(self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            kinc=torch.randn(self.n_freqs, 3, dtype=torch.complex64),
            kzref=torch.randn(self.n_freqs, self.n_harmonics, dtype=torch.complex64),
            kztrn=torch.randn(self.n_freqs, self.n_harmonics, dtype=torch.complex64),
        )

    def _make_results(self, batched: bool, wave_vectors):
        if batched:
            return SolverResults(
                reflection=self.batched_reflection,
                transmission=self.batched_transmission,
                reflection_diffraction=self.batched_reflection_diffraction,
                transmission_diffraction=self.batched_transmission_diffraction,
                reflection_field=self.batched_reflection_field,
                transmission_field=self.batched_transmission_field,
                structure_matrix=self.structure_matrix,
                wave_vectors=wave_vectors,
                n_sources=self.n_sources,
            )
        return SolverResults(
            reflection=self.unbatched_reflection,
            transmission=self.unbatched_transmission,
            reflection_diffraction=self.unbatched_reflection_diffraction,
            transmission_diffraction=self.unbatched_transmission_diffraction,
            reflection_field=self.unbatched_reflection_field,
            transmission_field=self.unbatched_transmission_field,
            structure_matrix=self.structure_matrix,
            wave_vectors=wave_vectors,
        )

    def _make_batched_wave_vectors(self):
        """Create a single batched WaveVectors with per-source kzref values."""
        kzref = torch.full((self.n_sources, self.n_freqs, self.n_harmonics), 1j, dtype=torch.complex64)
        kztrn = torch.full((self.n_sources, self.n_freqs, self.n_harmonics), 1j, dtype=torch.complex64)
        # Source 0: two propagating orders
        kzref[0, 0, 4] = torch.tensor(1.0 + 0j, dtype=torch.complex64)
        kzref[0, 0, 7] = torch.tensor(0.5 + 0j, dtype=torch.complex64)
        # Source 1+: one propagating order
        for i in range(1, self.n_sources):
            kzref[i, 0, 4] = torch.tensor(1.0 + 0j, dtype=torch.complex64)
        return WaveVectors(
            kx=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            ky=torch.randn(self.n_sources, self.n_freqs, self.harmonics[0], self.harmonics[1], dtype=torch.complex64),
            kinc=torch.randn(self.n_sources, self.n_freqs, 3, dtype=torch.complex64),
            kzref=kzref,
            kztrn=kztrn,
        )

    def test_harmonics_property_unbatched(self):
        results = self._make_results(batched=False, wave_vectors=self.single_wave_vectors)
        self.assertEqual(results.harmonics, (3, 3))

    def test_harmonics_property_batched(self):
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        self.assertEqual(results.harmonics, (3, 3))

    def test_get_diffraction_order_indices_batched(self):
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        self.assertEqual(results.get_diffraction_order_indices(0, 0), (1, 1))
        self.assertEqual(results.get_diffraction_order_indices(1, 0), (2, 1))

    def test_get_zero_order_transmission_batched(self):
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        tx, ty, tz = results.get_zero_order_transmission()
        self.assertEqual(tx.shape, (self.n_sources, self.n_freqs))
        self.assertEqual(ty.shape, (self.n_sources, self.n_freqs))
        self.assertEqual(tz.shape, (self.n_sources, self.n_freqs))
        self.assertTrue(torch.equal(tx, self.batched_transmission_field.x[..., 1, 1]))
        self.assertTrue(torch.equal(ty, self.batched_transmission_field.y[..., 1, 1]))
        self.assertTrue(torch.equal(tz, self.batched_transmission_field.z[..., 1, 1]))

    def test_get_zero_order_reflection_batched(self):
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        rx, ry, rz = results.get_zero_order_reflection()
        self.assertEqual(rx.shape, (self.n_sources, self.n_freqs))
        self.assertEqual(ry.shape, (self.n_sources, self.n_freqs))
        self.assertEqual(rz.shape, (self.n_sources, self.n_freqs))
        self.assertTrue(torch.equal(rx, self.batched_reflection_field.x[..., 1, 1]))
        self.assertTrue(torch.equal(ry, self.batched_reflection_field.y[..., 1, 1]))
        self.assertTrue(torch.equal(rz, self.batched_reflection_field.z[..., 1, 1]))

    def test_get_order_transmission_efficiency_batched(self):
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        efficiency = results.get_order_transmission_efficiency(0, 0)
        self.assertEqual(efficiency.shape, (self.n_sources, self.n_freqs))
        self.assertTrue(torch.equal(efficiency, self.batched_transmission_diffraction[..., 1, 1]))

    def test_get_order_reflection_efficiency_batched(self):
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        efficiency = results.get_order_reflection_efficiency(0, 0)
        self.assertEqual(efficiency.shape, (self.n_sources, self.n_freqs))
        self.assertTrue(torch.equal(efficiency, self.batched_reflection_diffraction[..., 1, 1]))

    def test_get_all_diffraction_orders_batched(self):
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        orders = results.get_all_diffraction_orders()
        expected_orders = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        self.assertEqual(orders, expected_orders)

    def test_get_propagating_orders_batched_with_source_idx(self):
        wave_vectors = self._make_batched_wave_vectors()
        results = self._make_results(batched=True, wave_vectors=wave_vectors)
        propagating = results.get_propagating_orders(wavelength_idx=0, source_idx=0)
        self.assertIn((0, 0), propagating)
        self.assertEqual(set(propagating), {(0, 0), (1, 0)})

    def test_get_propagating_orders_batched_no_source_idx_raises(self):
        wave_vectors = self._make_batched_wave_vectors()
        results = self._make_results(batched=True, wave_vectors=wave_vectors)
        with self.assertRaisesRegex(ValueError, "source_idx"):
            results.get_propagating_orders(0)

    def test_resolve_wave_vectors_single(self):
        results = self._make_results(batched=False, wave_vectors=self.single_wave_vectors)
        self.assertIs(results._resolve_wave_vectors(), self.single_wave_vectors)
        # For unbatched results, source_idx is ignored — returns wave_vectors directly
        self.assertIs(results._resolve_wave_vectors(source_idx=0), self.single_wave_vectors)
        self.assertIs(results._resolve_wave_vectors(source_idx=1), self.single_wave_vectors)

    def test_resolve_wave_vectors_batched(self):
        wave_vectors = self._make_batched_wave_vectors()
        results = self._make_results(batched=True, wave_vectors=wave_vectors)
        resolved = results._resolve_wave_vectors(source_idx=0)
        self.assertTrue(torch.equal(resolved.kzref, wave_vectors.kzref[0]))
        self.assertTrue(torch.equal(resolved.kinc, wave_vectors.kinc[0]))
        with self.assertRaisesRegex(ValueError, "source_idx"):
            results._resolve_wave_vectors()

    def test_shared_wave_vectors_indexing(self):
        """Batched results with shared (unbatched) WaveVectors must not mis-slice."""
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        # Indexing should preserve shared wave_vectors, not index into them
        r0 = results[0]
        self.assertIs(r0.wave_vectors, self.single_wave_vectors)
        r1 = results[1]
        self.assertIs(r1.wave_vectors, self.single_wave_vectors)
        # Slicing should also preserve
        subset = results[0:2]
        self.assertIs(subset.wave_vectors, self.single_wave_vectors)

    def test_shared_wave_vectors_iteration(self):
        """Iterating batched results with shared WaveVectors must not crash."""
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        items = list(results)
        self.assertEqual(len(items), self.n_sources)
        for item in items:
            self.assertIs(item.wave_vectors, self.single_wave_vectors)

    def test_shared_wave_vectors_resolve_no_source_idx(self):
        """Shared wave_vectors should not require source_idx."""
        results = self._make_results(batched=True, wave_vectors=self.single_wave_vectors)
        # Shared wave_vectors have no source dim, so no source_idx needed
        resolved = results._resolve_wave_vectors()
        self.assertIs(resolved, self.single_wave_vectors)
        resolved_with_idx = results._resolve_wave_vectors(source_idx=0)
        self.assertIs(resolved_with_idx, self.single_wave_vectors)


if __name__ == '__main__':
    unittest.main()
