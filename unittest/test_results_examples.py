"""Examples of using the results module.

This module provides examples and tests for common usage patterns of the results
module in torchrdit, demonstrating how to work with simulation results, analyze
field components, and calculate diffraction efficiencies.
"""

import unittest
import torch
from torchrdit.results import SolverResults


class TestResultsExamples(unittest.TestCase):
    """Examples of working with the results module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create dummy results data for examples
        n_freqs = 3
        kdim = (5, 5)  # 5x5 k-space grid
        n_harmonics = kdim[0] * kdim[1]
        
        # Create reflection and transmission efficiencies
        self.reflection = torch.tensor([0.3, 0.25, 0.2])
        self.transmission = torch.tensor([0.7, 0.75, 0.8])
        
        # Create diffraction efficiencies
        self.rde = torch.zeros(n_freqs, *kdim)
        self.tde = torch.zeros(n_freqs, *kdim)
        
        # Set zero-order (specular) values
        center_x, center_y = 2, 2  # Center of 5x5 grid
        self.rde[:, center_x, center_y] = self.reflection
        self.tde[:, center_x, center_y] = self.transmission
        
        # Add some first-order diffraction
        self.rde[:, center_x+1, center_y] = 0.05  # (1,0) order
        self.rde[:, center_x, center_y+1] = 0.05  # (0,1) order
        self.tde[:, center_x+1, center_y] = 0.05  # (1,0) order
        self.tde[:, center_x, center_y+1] = 0.05  # (0,1) order
        
        # Create field components
        rx = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        ry = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        rz = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        tx = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        ty = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        tz = torch.randn(n_freqs, *kdim, dtype=torch.complex64)
        
        # Make the zero-order components stronger
        rx[:, center_x, center_y] = 2.0 + 1.0j
        ry[:, center_x, center_y] = 1.5 + 0.5j
        tx[:, center_x, center_y] = 3.0 + 0.0j
        ty[:, center_x, center_y] = 0.0 + 2.0j
        
        # Wave vectors
        kx = torch.linspace(-2, 2, kdim[0]*kdim[1]).reshape(kdim)
        ky = torch.linspace(-2, 2, kdim[0]*kdim[1]).reshape(kdim)
        kinc = torch.zeros(n_freqs, 3)
        kinc[:, 2] = 1.0  # Normal incidence
        
        # Create complex tensors for kzref and kztrn
        kx_flat = kx.flatten()
        ky_flat = ky.flatten()
        kzref = torch.zeros(n_freqs, n_harmonics, dtype=torch.complex64)
        kztrn = torch.zeros(n_freqs, n_harmonics, dtype=torch.complex64)
        
        # Fill kzref and kztrn with appropriate values
        for i in range(n_harmonics):
            k_mag = kx_flat[i]**2 + ky_flat[i]**2
            # Propagating orders
            if k_mag < 1.0:
                kz_real = torch.sqrt(torch.as_tensor(1.0 - k_mag, dtype=torch.complex64))
                kzref[:, i] = kz_real
            else:
                # Evanescent orders
                kz_imag = 1j * torch.sqrt(torch.as_tensor(k_mag - 1.0, dtype=torch.complex64))
                kzref[:, i] = kz_imag
                
            # Same for transmission region with different refractive index
            if k_mag < 2.25:
                kz_real = torch.sqrt(torch.as_tensor(2.25 - k_mag, dtype=torch.complex64))
                kztrn[:, i] = kz_real
            else:
                kz_imag = 1j * torch.sqrt(torch.as_tensor(k_mag - 2.25, dtype=torch.complex64))
                kztrn[:, i] = kz_imag
        
        # Scattering matrix
        s11 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        s12 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        s21 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        s22 = torch.randn(n_freqs, 2*n_harmonics, 2*n_harmonics, dtype=torch.complex64)
        
        # Create the raw data dictionary
        self.raw_data = {
            'REF': self.reflection,
            'TRN': self.transmission,
            'RDE': self.rde,
            'TDE': self.tde,
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
        self.results = SolverResults.from_dict(self.raw_data)
        
        # Store wavelengths for plotting examples
        self.wavelengths = torch.tensor([0.5, 0.6, 0.7])  # microns
        
    def test_basic_results_analysis(self):
        """Example of basic results analysis."""
        # Get overall transmission and reflection
        R = self.results.reflection
        T = self.results.transmission
        
        # Print the total reflection and transmission for the first wavelength
        print(f"Reflection: {R[0].item():.3f}")
        print(f"Transmission: {T[0].item():.3f}")
        
        # Check energy conservation
        total = R + T
        print(f"Total efficiency: {total[0].item():.3f} (should be close to 1.0)")
        
        # The above is a simple example and doesn't need assertions, 
        # but we'll add one for the unit test
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-5))
        
    def test_field_components_analysis(self):
        """Example of analyzing field components."""
        # Get the zero-order field components
        tx, ty, tz = self.results.get_zero_order_transmission()
        
        # Calculate field amplitudes
        amp_x = torch.abs(tx)
        amp_y = torch.abs(ty)
        amp_z = torch.abs(tz)
        
        # Calculate phases
        phase_x = torch.angle(tx)
        phase_y = torch.angle(ty)
        
        # Calculate phase difference between x and y components
        phase_diff = phase_x - phase_y
        
        # Example plotting code (commented out for unit testing)
        # plt.figure(figsize=(10, 6))
        # plt.plot(self.wavelengths, amp_x.detach().cpu().numpy(), 'r-', label='|Ex|')
        # plt.plot(self.wavelengths, amp_y.detach().cpu().numpy(), 'b-', label='|Ey|')
        # plt.xlabel('Wavelength (µm)')
        # plt.ylabel('Field amplitude')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        
        # Simple assertion for the test
        self.assertEqual(tx.shape, (3,))  # Should have 3 wavelengths
        self.assertTrue(torch.all(amp_x > 0))  # Amplitudes should be positive
        
    def test_diffraction_orders_analysis(self):
        """Example of analyzing diffraction orders."""
        # Get propagating diffraction orders for the first wavelength
        prop_orders = self.results.get_propagating_orders(0)
        print(f"Propagating orders: {prop_orders}")
        
        # Calculate total power in propagating orders
        total_eff = torch.tensor(0.0)
        for m, n in prop_orders:
            r_eff = self.results.get_order_reflection_efficiency(m, n)[0]
            t_eff = self.results.get_order_transmission_efficiency(m, n)[0]
            total_eff += r_eff + t_eff
            print(f"Order ({m}, {n}): R = {r_eff.item():.3f}, T = {t_eff.item():.3f}")
        
        print(f"Total efficiency in propagating orders: {total_eff.item():.3f}")
        
        # In our test setup, we set values for both the (0,0) order and (0,1)/(1,0) orders
        # This can make the total efficiency > 1.0, which is physically impossible in a real solver
        # but can happen in our test data
        
        # For a real example, you would expect:
        # self.assertTrue(torch.isclose(total_eff, torch.tensor(1.0), atol=1e-5))
        
        # For our test data, we expect 1.1 (0.3 + 0.7 from zero order, plus 0.05 + 0.05 from (0,1) order)
        expected_total = torch.tensor(1.1)
        self.assertTrue(torch.isclose(total_eff, expected_total, atol=1e-5))
        
    def test_create_from_solver(self):
        """Example showing how results would typically come from a solver."""
        # This is a mock example since we don't run a real solver in unit tests
        # In a real scenario, you would do:
        
        # solver = create_solver(algorithm=Algorithm.RCWA)
        # source = solver.add_source(theta=0, phi=0, pte=1, ptm=0)
        # results = solver.solve(source)
        
        # Instead, we'll use our existing results
        results = self.results
        
        # Analyze a specific diffraction order
        order_x, order_y = 0, 0  # Zero order
        zero_order_r = results.get_order_reflection_efficiency(order_x, order_y)
        zero_order_t = results.get_order_transmission_efficiency(order_x, order_y)
        
        # Check that the zero-order efficiencies match our expected values
        self.assertTrue(torch.allclose(zero_order_r, self.reflection))
        self.assertTrue(torch.allclose(zero_order_t, self.transmission))
        
    def test_dictionary_conversion(self):
        """Example of converting between SolverResults and dictionaries."""
        # Create a SolverResults from a dictionary
        results = SolverResults.from_dict(self.raw_data)
        
        # Convert back to a dictionary
        data_dict = results.to_dict()
        
        # Verify the conversion worked correctly
        self.assertTrue(torch.equal(data_dict['REF'], self.raw_data['REF']))
        self.assertTrue(torch.equal(data_dict['TRN'], self.raw_data['TRN']))
        
        # This pattern is useful when interfacing with legacy code
        # that expects dictionary outputs
        
        # Example of using a legacy function that expects a dictionary
        def legacy_function(data):
            return data['REF'][0] + data['TRN'][0]
        
        # Use our converted dictionary with the legacy function
        total = legacy_function(data_dict)
        self.assertAlmostEqual(total.item(), 1.0)


if __name__ == '__main__':
    unittest.main() 