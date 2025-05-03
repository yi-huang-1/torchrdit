"""Module for electromagnetic simulation results processing and analysis.

This module provides data structures for organizing and analyzing results from
electromagnetic simulations. It defines dataclasses that encapsulate reflection
and transmission coefficients, field components, scattering matrices, and wave
vectors in a structured, easy-to-use format.

Note:
    Throughout this module, n_harmonics = kdim[0] * kdim[1], where kdim is the
    k-space dimensions used in the simulation.

Classes:
    ScatteringMatrix: Container for S-parameter matrices (S11, S12, S21, S22).
    FieldComponents: Container for x, y, z components of electromagnetic fields.
    WaveVectors: Container for wave vector information in the simulation.
    SolverResults: Main results container with analysis methods for diffraction orders.
    
Examples:
Basic usage with a solver:

```python
import torch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm

# Create and configure a solver
solver = create_solver(algorithm=Algorithm.RCWA)

# Run the simulation
source = solver.add_source(theta=0, phi=0, pte=1, ptm=0)
results = solver.solve(source)  # Returns a SolverResults object

# Access results properties
print(f"Transmission: {results.transmission[0].item():.3f}")
print(f"Reflection: {results.reflection[0].item():.3f}")

# Get field components and analyze
tx, ty, tz = results.get_zero_order_transmission()
amplitude = torch.abs(tx[0])
phase = torch.angle(tx[0])

# Find propagating diffraction orders
prop_orders = results.get_propagating_orders()
```

Legacy dictionary conversion:

```python
# Convert legacy dictionary to structured results
old_data = legacy_solver.solve_dict()
results = SolverResults.from_dict(old_data)

# Convert back to dictionary if needed
data_dict = results.to_dict()
```
    
Keywords:
    electromagnetic simulation, results analysis, diffraction efficiency, scattering matrix,
    field components, wave vectors, RCWA, R-DIT, transmission, reflection, diffraction orders,
    Fourier optics, simulation output, efficiency calculation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

@dataclass
class ScatteringMatrix:
    """Scattering matrix components for electromagnetic simulation.
    
    Represents the four components of the scattering matrix (S-parameters) that
    characterize the reflection and transmission properties of an electromagnetic structure.
    
    Note:
        n_harmonics = kdim[0] * kdim[1], where kdim is the k-space dimensions
        used in the simulation.
    
    Attributes:
        S11 (torch.Tensor): Reflection coefficient matrix for waves incident from port 1.
            Shape: (n_freqs, 2*n_harmonics, 2*n_harmonics)
        S12 (torch.Tensor): Transmission coefficient matrix from port 2 to port 1.
            Shape: (n_freqs, 2*n_harmonics, 2*n_harmonics)
        S21 (torch.Tensor): Transmission coefficient matrix from port 1 to port 2.
            Shape: (n_freqs, 2*n_harmonics, 2*n_harmonics)
        S22 (torch.Tensor): Reflection coefficient matrix for waves incident from port 2.
            Shape: (n_freqs, 2*n_harmonics, 2*n_harmonics)
    
    Examples:
    ```python
    # Access reflection coefficient for first frequency
    s11_first_freq = smatrix.S11[0]
    
    # Calculate transmitted power through the structure
    transmitted_power = torch.abs(smatrix.S21)**2
    ```
    
    Keywords:
        scattering matrix, S-parameters, S11, S12, S21, S22, reflection, transmission,
        electromagnetic, Fourier harmonics
    """
    S11: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
    S12: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
    S21: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
    S22: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)

@dataclass
class FieldComponents:
    """Field components in x, y, z directions for electromagnetic fields.
    
    Contains the spatial distribution of electromagnetic field components along
    the x, y, and z directions in the Fourier domain.
    
    Attributes:
        x (torch.Tensor): X-component of the electromagnetic field.
            Shape: (n_freqs, kdim[0], kdim[1])
        y (torch.Tensor): Y-component of the electromagnetic field.
            Shape: (n_freqs, kdim[0], kdim[1])
        z (torch.Tensor): Z-component of the electromagnetic field.
            Shape: (n_freqs, kdim[0], kdim[1])
    
    Examples:
    ```python
    # Calculate field intensity (|E|²) at first frequency
    intensity = (abs(field.x[0])**2 + abs(field.y[0])**2 + abs(field.z[0])**2)
    
    # Extract phase of x-component
    phase_x = torch.angle(field.x)
    ```
    
    Keywords:
        field components, electromagnetic field, electric field, magnetic field,
        Fourier domain, x-component, y-component, z-component, vectorial
    """
    x: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
    y: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
    z: torch.Tensor  # (n_freqs, kdim[0], kdim[1])

@dataclass
class WaveVectors:
    """Wave vector components for the electromagnetic simulation.
    
    Contains the wave vector information for the simulation, including incident,
    reflected, and transmitted wave vectors in the Fourier domain.
    
    Attributes:
        kx (torch.Tensor): X-component of the wave vector in reciprocal space.
            Shape: (kdim[0], kdim[1])
        ky (torch.Tensor): Y-component of the wave vector in reciprocal space.
            Shape: (kdim[0], kdim[1])
        kinc (torch.Tensor): Incident wave vector.
            Shape: (n_freqs, 3)
        kzref (torch.Tensor): Z-component of the wave vector in reflection region.
            Shape: (n_freqs, kdim[0]*kdim[1])
        kztrn (torch.Tensor): Z-component of the wave vector in transmission region.
            Shape: (n_freqs, kdim[0]*kdim[1])
    
    Examples:
    ```python
    # Get propagation constant in z-direction for reflected waves
    kz = wave_vectors.kzref
    
    # Calculate wave vector magnitude
    k_magnitude = torch.sqrt(wave_vectors.kx**2 + wave_vectors.ky**2 + kz**2)
    ```
    
    Keywords:
        wave vector, k-vector, propagation constant, reciprocal space, Fourier harmonics,
        incident wave, reflected wave, transmitted wave, dispersion
    """
    kx: torch.Tensor  # (kdim[0], kdim[1])
    ky: torch.Tensor  # (kdim[0], kdim[1])
    kinc: torch.Tensor  # (n_freqs, 3)
    kzref: torch.Tensor  # (n_freqs, kdim[0]*kdim[1])
    kztrn: torch.Tensor  # (n_freqs, kdim[0]*kdim[1])

@dataclass
class SolverResults:
    """Complete results from electromagnetic solver.
    
    Comprehensive container for all results from an electromagnetic simulation,
    including reflection and transmission coefficients, diffraction efficiencies,
    field components, scattering matrices, and wave vectors. This class also provides
    methods for analyzing diffraction orders and extracting specific field components.
    
    Note:
        n_harmonics = kdim[0] * kdim[1], where kdim is the k-space dimensions
        used in the simulation. This relationship appears in the ScatteringMatrix
        component's tensor shapes.
    
    Attributes:
        reflection (torch.Tensor): Total reflection efficiency for each wavelength.
            Shape: (n_freqs)
        transmission (torch.Tensor): Total transmission efficiency for each wavelength.
            Shape: (n_freqs)
        reflection_diffraction (torch.Tensor): Reflection efficiencies for each diffraction order.
            Shape: (n_freqs, kdim[0], kdim[1])
        transmission_diffraction (torch.Tensor): Transmission efficiencies for each diffraction order.
            Shape: (n_freqs, kdim[0], kdim[1])
        reflection_field (FieldComponents): Field components in the reflection region.
        transmission_field (FieldComponents): Field components in the transmission region.
        structure_matrix (ScatteringMatrix): Scattering matrix for the entire structure.
        wave_vectors (WaveVectors): Wave vector components for the simulation.
        raw_data (Dict): Raw dictionary data for backward compatibility.
    
    Examples:
    ```python
    # Get overall reflection and transmission
    total_reflection = results.reflection[0]  # First wavelength
    total_transmission = results.transmission[0]  # First wavelength
    
    # Access field components
    tx, ty, tz = results.get_zero_order_transmission()
    
    # Calculate field amplitude and phase
    amplitude = torch.abs(tx[0])  # Amplitude of x-component
    phase = torch.angle(tx[0])    # Phase in radians
    
    # Analyze diffraction orders
    orders = results.get_propagating_orders()
    efficiency = results.get_order_transmission_efficiency(1, 0)  # First order in x
    ```
    
    Keywords:
        electromagnetic simulation, results, reflection, transmission, diffraction,
        scattering matrix, field components, wave vectors, RCWA, R-DIT, diffraction order,
        efficiency, Fourier optics
    """
    # Overall efficiencies
    reflection: torch.Tensor  # (n_freqs)
    transmission: torch.Tensor  # (n_freqs)
    
    # Diffraction efficiencies
    reflection_diffraction: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
    transmission_diffraction: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
    
    # Field components
    reflection_field: FieldComponents
    transmission_field: FieldComponents
    
    # Scattering matrices
    structure_matrix: ScatteringMatrix
    
    # Wave vectors
    wave_vectors: WaveVectors
    
    # Raw dictionary for backward compatibility
    raw_data: Dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SolverResults':
        """Create a SolverResults instance from a dictionary.
        
        Factory method to construct a SolverResults object from a raw dictionary
        of simulation results. This is useful for converting legacy format data
        into the structured SolverResults format.
        
        Args:
            data (Dict): Raw dictionary containing simulation results with keys like
                'REF', 'TRN', 'RDE', 'TDE', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz', 
                'smat_structure', 'kx', 'ky', 'kinc', 'kzref', 'kztrn'.
        
        Returns:
            SolverResults: A structured results object with organized data.
        
        Examples:
        ```python
        # Convert legacy dictionary format to SolverResults
        legacy_data = solver.solve_legacy()
        results = SolverResults.from_dict(legacy_data)
        
        # Now use the structured API
        print(f"Transmission: {results.transmission[0].item():.3f}")
        ```
        
        Keywords:
            factory method, conversion, dictionary, legacy format, backward compatibility
        """
        return cls(
            reflection=data['REF'],
            transmission=data['TRN'],
            reflection_diffraction=data['RDE'],
            transmission_diffraction=data['TDE'],
            reflection_field=FieldComponents(
                x=data['rx'],
                y=data['ry'],
                z=data['rz']
            ),
            transmission_field=FieldComponents(
                x=data['tx'],
                y=data['ty'],
                z=data['tz']
            ),
            structure_matrix=ScatteringMatrix(
                S11=data['smat_structure']['S11'],
                S12=data['smat_structure']['S12'],
                S21=data['smat_structure']['S21'],
                S22=data['smat_structure']['S22']
            ),
            wave_vectors=WaveVectors(
                kx=data['kx'],
                ky=data['ky'],
                kinc=data['kinc'],
                kzref=data['kzref'],
                kztrn=data['kztrn']
            ),
            raw_data=data
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for backward compatibility.
        
        Exports the SolverResults object back to a raw dictionary format for
        backward compatibility with code expecting the legacy format.
        
        Returns:
            Dict: Raw dictionary containing all simulation results.
        
        Examples:
        ```python
        # Get dictionary format for legacy code
        results = device.solve(source)
        data_dict = results.to_dict()
        
        # Use with legacy code
        legacy_function(data_dict)
        ```
        
        Keywords:
            conversion, dictionary, legacy format, backward compatibility
        """
        return self.raw_data
    
    def get_diffraction_order_indices(self, order_x: int = 0, order_y: int = 0) -> Tuple[int, int]:
        """Get the indices for a specific diffraction order.
        
        Converts diffraction order coordinates (m,n) to array indices (i,j) in the
        diffraction efficiency and field tensors. The zero order (0,0) corresponds
        to the center of the k-space grid.
        
        Args:
            order_x (int, optional): The x-component of the diffraction order.
                Defaults to 0 (specular).
            order_y (int, optional): The y-component of the diffraction order.
                Defaults to 0 (specular).
            
        Returns:
            Tuple[int, int]: The indices (ix, iy) corresponding to the requested order
                in the diffraction efficiency and field tensors.
        
        Raises:
            ValueError: If the requested diffraction order is outside the simulation bounds.
        
        Examples:
        ```python
        # Get indices for the (1,1) diffraction order
        ix, iy = results.get_diffraction_order_indices(1, 1)
        
        # Access corresponding transmission efficiency
        efficiency = results.transmission_diffraction[0, ix, iy]
        ```
        
        Keywords:
            diffraction order, indices, k-space, Fourier harmonics, specular, array index
        """
        kdim_x = self.reflection_field.x.shape[1]
        kdim_y = self.reflection_field.x.shape[2]
        
        # The center indices correspond to the zero order
        center_x = kdim_x // 2
        center_y = kdim_y // 2
        
        # Calculate indices relative to center
        ix = center_x + order_x
        iy = center_y + order_y
        
        # Check if the requested order is within bounds
        if ix < 0 or ix >= kdim_x or iy < 0 or iy >= kdim_y:
            raise ValueError(f"Diffraction order ({order_x}, {order_y}) is out of bounds")
        
        return (ix, iy)
    
    def get_zero_order_transmission(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the zero-order transmission field components.
        
        Returns the electric field components (Ex, Ey, Ez) for the zero-order
        (specular) transmission. This is useful for analyzing phase, polarization,
        and amplitude of the directly transmitted light.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The (x, y, z) field components
                for the zero diffraction order. Each tensor has shape (n_freqs).
        
        Examples:
        ```python
        # Get field components for the directly transmitted light
        tx, ty, tz = results.get_zero_order_transmission()
        
        # Calculate amplitude of the x-component
        amplitude_x = torch.abs(tx[0])  # For first wavelength
        
        # Calculate phase of the x-component
        phase_x = torch.angle(tx[0])  # For first wavelength
        
        # Calculate polarization ellipse parameters
        phase_diff = torch.angle(tx[0]) - torch.angle(ty[0])
        ```
        
        Keywords:
            zero order, specular, transmission, field components, electric field,
            amplitude, phase, polarization
        """
        ix, iy = self.get_diffraction_order_indices(0, 0)
        return (
            self.transmission_field.x[:, ix, iy],
            self.transmission_field.y[:, ix, iy],
            self.transmission_field.z[:, ix, iy]
        )
    
    def get_zero_order_reflection(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the zero-order reflection field components.
        
        Returns the electric field components (Ex, Ey, Ez) for the zero-order
        (specular) reflection. This is useful for analyzing phase, polarization,
        and amplitude of the directly reflected light.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The (x, y, z) field components
                for the zero diffraction order. Each tensor has shape (n_freqs).
        
        Examples:
        ```python
        # Get field components for the directly reflected light
        rx, ry, rz = results.get_zero_order_reflection()
        
        # Calculate field intensity
        intensity = torch.abs(rx[0])**2 + torch.abs(ry[0])**2 + torch.abs(rz[0])**2
        
        # Analyze polarization state
        major_axis = torch.maximum(torch.abs(rx[0]), torch.abs(ry[0]))
        minor_axis = torch.minimum(torch.abs(rx[0]), torch.abs(ry[0]))
        ```
        
        Keywords:
            zero order, specular, reflection, field components, electric field,
            amplitude, phase, polarization
        """
        ix, iy = self.get_diffraction_order_indices(0, 0)
        return (
            self.reflection_field.x[:, ix, iy],
            self.reflection_field.y[:, ix, iy],
            self.reflection_field.z[:, ix, iy]
        )
    
    def get_order_transmission_efficiency(self, order_x: int = 0, order_y: int = 0) -> torch.Tensor:
        """Get the transmission diffraction efficiency for a specific order.
        
        Returns the diffraction efficiency (ratio of transmitted power to incident power)
        for the specified diffraction order. For the zero order (0,0), this gives the
        direct transmission efficiency.
        
        Args:
            order_x (int, optional): The x-component of the diffraction order.
                Defaults to 0 (specular).
            order_y (int, optional): The y-component of the diffraction order.
                Defaults to 0 (specular).
            
        Returns:
            torch.Tensor: The transmission diffraction efficiency for the specified order.
                Shape: (n_freqs)
        
        Examples:
        ```python
        # Get zero-order (direct) transmission efficiency
        t0 = results.get_order_transmission_efficiency()
        
        # Get first-order diffraction efficiency
        t1 = results.get_order_transmission_efficiency(1, 0)
        
        # Compare efficiencies across wavelengths
        plt.plot(wavelengths, t0.detach().cpu().numpy(), label='Zero order')
        plt.plot(wavelengths, t1.detach().cpu().numpy(), label='First order')
        ```
        
        Keywords:
            transmission, diffraction efficiency, diffraction order, specular transmission,
            power ratio, wavelength dependence
        """
        ix, iy = self.get_diffraction_order_indices(order_x, order_y)
        return self.transmission_diffraction[:, ix, iy]
    
    def get_order_reflection_efficiency(self, order_x: int = 0, order_y: int = 0) -> torch.Tensor:
        """Get the reflection diffraction efficiency for a specific order.
        
        Returns the diffraction efficiency (ratio of reflected power to incident power)
        for the specified diffraction order. For the zero order (0,0), this gives the
        direct reflection efficiency.
        
        Args:
            order_x (int, optional): The x-component of the diffraction order.
                Defaults to 0 (specular).
            order_y (int, optional): The y-component of the diffraction order.
                Defaults to 0 (specular).
            
        Returns:
            torch.Tensor: The reflection diffraction efficiency for the specified order.
                Shape: (n_freqs)
        
        Examples:
        ```python
        # Get zero-order (specular) reflection efficiency
        r0 = results.get_order_reflection_efficiency()
        
        # Compare multiple diffraction orders
        r1 = results.get_order_reflection_efficiency(1, 0)
        r_1 = results.get_order_reflection_efficiency(-1, 0)
        
        # Energy conservation check
        total = results.reflection[0] + results.transmission[0]
        print(f"Total efficiency: {total.item():.4f} (should be close to 1.0)")
        ```
        
        Keywords:
            reflection, diffraction efficiency, diffraction order, specular reflection,
            power ratio, wavelength dependence, energy conservation
        """
        ix, iy = self.get_diffraction_order_indices(order_x, order_y)
        return self.reflection_diffraction[:, ix, iy]
    
    def get_all_diffraction_orders(self) -> List[Tuple[int, int]]:
        """Get a list of all available diffraction orders as (m, n) tuples.
        
        Returns all diffraction orders included in the simulation, whether propagating
        or evanescent. Each order is represented as a tuple (m, n) where m is the
        x-component and n is the y-component of the diffraction order.
        
        Returns:
            List[Tuple[int, int]]: List of all diffraction orders as (m, n) tuples.
        
        Examples:
        ```python
        # Get all available diffraction orders
        all_orders = results.get_all_diffraction_orders()
        
        # Count total number of diffraction orders
        num_orders = len(all_orders)
        print(f"Simulation includes {num_orders} diffraction orders")
        
        # Filter for specific orders
        first_orders = [(m, n) for m, n in all_orders if abs(m) + abs(n) == 1]
        ```
        
        Keywords:
            diffraction orders, Fourier harmonics, grating orders, reciprocal lattice,
            k-space, simulation grid
        """
        kdim_x = self.reflection_field.x.shape[1]
        kdim_y = self.reflection_field.x.shape[2]
        center_x = kdim_x // 2
        center_y = kdim_y // 2
        
        orders = []
        for ix in range(kdim_x):
            for iy in range(kdim_y):
                order_x = ix - center_x
                order_y = iy - center_y
                orders.append((order_x, order_y))
        
        return orders
    
    def get_propagating_orders(self, wavelength_idx: int = 0) -> List[Tuple[int, int]]:
        """Get a list of propagating diffraction orders for a specific wavelength.
        
        Identifies which diffraction orders are propagating (rather than evanescent)
        for the specified wavelength. Propagating orders have real-valued z-component
        of the wave vector and contribute to far-field diffraction patterns.
        
        Args:
            wavelength_idx (int, optional): Index of the wavelength in the simulation.
                Defaults to 0 (first wavelength).
            
        Returns:
            List[Tuple[int, int]]: List of propagating diffraction orders as (m, n) tuples.
        
        Examples:
        ```python
        # Get propagating orders for the first wavelength
        prop_orders = results.get_propagating_orders()
        
        # Calculate total efficiency in propagating orders
        total_eff = 0
        for m, n in prop_orders:
            total_eff += results.get_order_transmission_efficiency(m, n)[0]
            total_eff += results.get_order_reflection_efficiency(m, n)[0]
        
        # Compare number of propagating orders at different wavelengths
        for i, wl in enumerate(wavelengths):
            orders = results.get_propagating_orders(i)
            print(f"Wavelength {wl:.3f} µm: {len(orders)} propagating orders")
        ```
        
        Keywords:
            propagating orders, evanescent orders, diffraction, wave vector, far-field,
            wavelength dependence, grating equation, k-space
        """
        kdim_x = self.reflection_field.x.shape[1]
        kdim_y = self.reflection_field.x.shape[2]
        
        # Reshape kzref to match the shape of the field tensors
        kzref_reshaped = self.wave_vectors.kzref[wavelength_idx].reshape(kdim_x, kdim_y)
        
        propagating_orders = []
        center_x = kdim_x // 2
        center_y = kdim_y // 2
        
        for ix in range(kdim_x):
            for iy in range(kdim_y):
                # Check if kz is real (imaginary part is very small)
                if torch.abs(torch.imag(kzref_reshaped[ix, iy])) < 1e-6:
                    order_x = ix - center_x
                    order_y = iy - center_y
                    propagating_orders.append((order_x, order_y))
        
        return propagating_orders