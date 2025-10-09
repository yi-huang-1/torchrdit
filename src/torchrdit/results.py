"""Module for electromagnetic simulation results processing and analysis.

This module provides data structures for organizing and analyzing results from
electromagnetic simulations. It defines dataclasses that encapsulate reflection
and transmission coefficients, field components, scattering matrices, and wave
vectors in a structured, easy-to-use format.

Note:
    Throughout this module, kdim_0_tims_1 = kdim[0] * kdim[1], where kdim is the
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
from typing import Dict, List, Tuple, Optional, Union
import torch


@dataclass
class ScatteringMatrix:
    """Scattering matrix components for electromagnetic simulation.

    Represents the four components of the scattering matrix (S-parameters) that
    characterize the reflection and transmission properties of an electromagnetic structure.

    Note:
        kdim_0_tims_1 = kdim[0] * kdim[1], where kdim is the k-space dimensions
        used in the simulation.

    Attributes:
        S11 (torch.Tensor): Reflection coefficient matrix for waves incident from port 1.
            Shape: (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
        S12 (torch.Tensor): Transmission coefficient matrix from port 2 to port 1.
            Shape: (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
        S21 (torch.Tensor): Transmission coefficient matrix from port 1 to port 2.
            Shape: (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
        S22 (torch.Tensor): Reflection coefficient matrix for waves incident from port 2.
            Shape: (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)

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

    S11: torch.Tensor  # (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
    S12: torch.Tensor  # (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
    S21: torch.Tensor  # (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
    S22: torch.Tensor  # (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)


@dataclass
class _FourierCoefficients:
    """Internal class for k-space Fourier coefficients (not real-space fields).

    Contains Fourier coefficients in k-space representing spectral amplitudes.
    These are the raw coefficients from the electromagnetic solver

    Note: This is an internal class - use public field API methods instead.
    """

    s_x: torch.Tensor  # Electric field Fourier coefficient x-component
    s_y: torch.Tensor  # Electric field Fourier coefficient y-component
    s_z: torch.Tensor  # Electric field Fourier coefficient z-component
    u_x: Optional[torch.Tensor] = None  # Magnetic field Fourier coefficient x-component
    u_y: Optional[torch.Tensor] = None  # Magnetic field Fourier coefficient y-component
    u_z: Optional[torch.Tensor] = None  # Magnetic field Fourier coefficient z-component

@dataclass
class FieldComponents:
    """Electromagnetic field Fourier coefficients in k-space.

    Contains Fourier coefficients (not real-space fields) of electromagnetic field
    components in k-space. These represent the spectral amplitudes s_x, s_y, s_z for
    electric fields and u_x, u_y, u_z for magnetic fields, indexed by harmonic orders.

    Important: These are Fourier coefficients in k-space, not real-space field values.

    Attributes:
        x (torch.Tensor): X-component Fourier coefficient of electric field (s_x).
            Shape: (n_freqs, n_harmonics_x, n_harmonics_y)
        y (torch.Tensor): Y-component Fourier coefficient of electric field (s_y).
            Shape: (n_freqs, n_harmonics_x, n_harmonics_y)
        z (torch.Tensor): Z-component Fourier coefficient of electric field (s_z).
            Shape: (n_freqs, n_harmonics_x, n_harmonics_y)
        mag_x (Optional[torch.Tensor]): X-component Fourier coefficient of magnetic field (u_x).
            Shape: (n_freqs, n_harmonics_x, n_harmonics_y). Default: None
            Note: These are Fourier coefficients of normalized magnetic field h_hat(x,y,z) where h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z)
        mag_y (Optional[torch.Tensor]): Y-component Fourier coefficient of magnetic field (u_y).
            Shape: (n_freqs, n_harmonics_x, n_harmonics_y). Default: None
            Note: These are Fourier coefficients of normalized magnetic field h_hat(x,y,z) where h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z)
        mag_z (Optional[torch.Tensor]): Z-component Fourier coefficient of magnetic field (u_z).
            Shape: (n_freqs, n_harmonics_x, n_harmonics_y). Default: None
            Note: These are Fourier coefficients of normalized magnetic field h_hat(x,y,z) where h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z)

    Examples:
    ```python
    # Calculate electric field intensity from Fourier coefficients at first frequency
    e_intensity = (abs(field.x[0])**2 + abs(field.y[0])**2 + abs(field.z[0])**2)

    # Access Fourier coefficients of normalized magnetic field h_hat(x,y,z)
    if field.mag_x is not None:
        u_x = field.mag_x  # Fourier coefficients of h_hat_x(x,y,z)
        u_y = field.mag_y  # Fourier coefficients of h_hat_y(x,y,z)

    ```

    Keywords:
        field components, electromagnetic field, electric field, magnetic field,
        Fourier domain, x-component, y-component, z-component, vectorial, Poynting vector
    """

    x: torch.Tensor  # Electric field x-component (n_freqs, kdim[0], kdim[1])
    y: torch.Tensor  # Electric field y-component (n_freqs, kdim[0], kdim[1])
    z: torch.Tensor  # Electric field z-component (n_freqs, kdim[0], kdim[1])
    mag_x: Optional[torch.Tensor] = None  # Magnetic field x-component
    mag_y: Optional[torch.Tensor] = None  # Magnetic field y-component
    mag_z: Optional[torch.Tensor] = None  # Magnetic field z-component


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
    """Unified results container for electromagnetic solver with single/batched source support.

    Comprehensive container for all results from electromagnetic simulations, supporting
    both single and batched source solving. Includes reflection and transmission coefficients,
    diffraction efficiencies, field components, scattering matrices, wave vectors.

    Note:
        kdim_0_tims_1 = kdim[0] * kdim[1], where kdim is the k-space dimensions
        used in the simulation. This relationship appears in the ScatteringMatrix
        component's tensor shapes.

    Attributes:
        reflection (torch.Tensor): Total reflection efficiency.
            Shape: (n_freqs) for single source, (n_sources, n_freqs) for batched
        transmission (torch.Tensor): Total transmission efficiency.
            Shape: (n_freqs) for single source, (n_sources, n_freqs) for batched
        reflection_diffraction (torch.Tensor): Reflection efficiencies for each diffraction order.
            Shape: (n_freqs, kdim[0], kdim[1]) or (n_sources, n_freqs, kdim[0], kdim[1])
        transmission_diffraction (torch.Tensor): Transmission efficiencies for each diffraction order.
            Shape: (n_freqs, kdim[0], kdim[1]) or (n_sources, n_freqs, kdim[0], kdim[1])
        reflection_field (FieldComponents): Field Fourier coefficients in the reflection region.
        transmission_field (FieldComponents): Field Fourier coefficients in the transmission region.
        structure_matrix (ScatteringMatrix): Scattering matrix for the entire structure.
        wave_vectors (WaveVectors): Wave vector components for the simulation.
        n_sources (int): Number of sources (1 for single, >1 for batched).
        lattice_t1, lattice_t2 (Optional[torch.Tensor]): Lattice vectors.
        default_rdim (Optional[Tuple[int, int]]): Default spatial resolution from solver.

    Examples:
    ```python
    # Single source results
    results = solver.solve(source)
    total_reflection = results.reflection[0]  # First wavelength
    total_transmission = results.transmission[0]

    # Batched source results
    sources = [solver.add_source(theta=angle) for angle in angles]
    results = solver.solve(sources)  # Returns SolverResults with batching

    # Access individual source results
    for i, result in enumerate(results):
        print(f"Source {i}: T={result.transmission[0]:.3f}")

    # Find optimal source (batched only)
    if results.is_batched:
        best_idx = results.find_optimal_source('max_transmission')

    # Interface Fourier coefficients
    coeffs = results.get_reflection_interface_fourier_coefficients()
    S_x, S_y, S_z = coeffs['S_x'], coeffs['S_y'], coeffs['S_z']  # E-field coefficients
    U_x, U_y, U_z = coeffs['U_x'], coeffs['U_y'], coeffs['U_z']  # H-field coefficients
    ```

    Keywords:
        electromagnetic simulation, results, reflection, transmission, diffraction,
        scattering matrix, field components, wave vectors, RCWA, R-DIT, diffraction order,
        efficiency, Fourier optics, batched sources
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

    # Field data
    mat_v_ref: Optional[torch.Tensor] = None  # V matrix for reflection region (magnetic field mode matrix)
    mat_v_trn: Optional[torch.Tensor] = None  # V matrix for transmission region (magnetic field mode matrix)
    polarization_data: Optional[Dict] = None  # Polarization data containing esrc and other vectors
    solver_config: Optional[Dict] = None  # Solver configuration (kdim, n_freqs, device, etc.)
    smat_layers: Optional[Dict] = None  # Individual layer S-matrices for proper mode coefficient calculation

    # Lattice vectors
    lattice_t1: Optional[torch.Tensor] = None  # First lattice vector [x, y] from solver
    lattice_t2: Optional[torch.Tensor] = None  # Second lattice vector [x, y] from solver
    default_rdim: Optional[Tuple[int, int]] = None  # Default spatial resolution [height, width] from solver

    # Unified batching support (new fields for merged BatchedSolverResults functionality)
    n_sources: int = field(default=1)  # Number of sources (1 for single, >1 for batched)
    source_parameters: Optional[List[Dict]] = None  # Original source dictionaries for batched case
    loss: Optional[torch.Tensor] = None  # Total loss for each source/wavelength (batched only)
    _is_batched: Optional[bool] = field(default=None)  # Explicit batched flag to override n_sources logic

    @property
    def is_batched(self) -> bool:
        """Return True if this represents batched source results.

        This is True when:
        1. Explicitly set via _is_batched (for single-element list inputs)
        2. When n_sources > 1 (multiple sources)
        """
        if self._is_batched is not None:
            return self._is_batched
        return self.n_sources > 1

    def __len__(self) -> int:
        """Return number of sources in the results."""
        return self.n_sources

    def __getitem__(self, idx: Union[int, slice]) -> Union["SolverResults", "SolverResults"]:
        """Get results for specific source(s).

        Args:
            idx: Integer index or slice for source selection.

        Returns:
            SolverResults for single index, SolverResults for slice (both single and batched compatible).
        """
        if isinstance(idx, int):
            # Handle negative indexing
            if idx < 0:
                idx = self.n_sources + idx
            if idx < 0 or idx >= self.n_sources:
                raise IndexError(f"Source index {idx} out of range for {self.n_sources} sources")

            if not self.is_batched:
                # For single source, only index 0 is valid
                if idx != 0:
                    raise IndexError(f"Single source result only supports index 0, got {idx}")
                return self

            # Extract single source from batched results
            # Handle both single and batched wave_vectors
            single_wave_vectors = self.wave_vectors
            if isinstance(self.wave_vectors, list):
                single_wave_vectors = self.wave_vectors[idx]

            return SolverResults(
                reflection=self.reflection[idx],
                transmission=self.transmission[idx],
                reflection_diffraction=self.reflection_diffraction[idx],
                transmission_diffraction=self.transmission_diffraction[idx],
                reflection_field=FieldComponents(
                    x=self.reflection_field.x[idx],
                    y=self.reflection_field.y[idx],
                    z=self.reflection_field.z[idx],
                    mag_x=self.reflection_field.mag_x[idx] if self.reflection_field.mag_x is not None else None,
                    mag_y=self.reflection_field.mag_y[idx] if self.reflection_field.mag_y is not None else None,
                    mag_z=self.reflection_field.mag_z[idx] if self.reflection_field.mag_z is not None else None,
                ),
                transmission_field=FieldComponents(
                    x=self.transmission_field.x[idx],
                    y=self.transmission_field.y[idx],
                    z=self.transmission_field.z[idx],
                    mag_x=self.transmission_field.mag_x[idx] if self.transmission_field.mag_x is not None else None,
                    mag_y=self.transmission_field.mag_y[idx] if self.transmission_field.mag_y is not None else None,
                    mag_z=self.transmission_field.mag_z[idx] if self.transmission_field.mag_z is not None else None,
                ),
                structure_matrix=self.structure_matrix,
                wave_vectors=single_wave_vectors,
                raw_data={},  # Could populate with source-specific data
                lattice_t1=self.lattice_t1,
                lattice_t2=self.lattice_t2,
                default_rdim=self.default_rdim,
                n_sources=1,  # Single source result
                source_parameters=[self.source_parameters[idx]] if self.source_parameters else None,
                loss=self.loss[idx] if self.loss is not None else None,
                _is_batched=None,  # Single source extracted from batch is not batched
            )

        elif isinstance(idx, slice):
            # Handle slicing
            indices = range(*idx.indices(self.n_sources))
            if len(indices) == 0:
                raise ValueError("Empty slice")

            # Extract subset using slicing
            subset_wave_vectors = self.wave_vectors
            if isinstance(self.wave_vectors, list):
                subset_wave_vectors = [self.wave_vectors[i] for i in indices]

            return SolverResults(
                reflection=self.reflection[idx],
                transmission=self.transmission[idx],
                reflection_diffraction=self.reflection_diffraction[idx],
                transmission_diffraction=self.transmission_diffraction[idx],
                reflection_field=FieldComponents(
                    x=self.reflection_field.x[idx],
                    y=self.reflection_field.y[idx],
                    z=self.reflection_field.z[idx],
                    mag_x=self.reflection_field.mag_x[idx] if self.reflection_field.mag_x is not None else None,
                    mag_y=self.reflection_field.mag_y[idx] if self.reflection_field.mag_y is not None else None,
                    mag_z=self.reflection_field.mag_z[idx] if self.reflection_field.mag_z is not None else None,
                ),
                transmission_field=FieldComponents(
                    x=self.transmission_field.x[idx],
                    y=self.transmission_field.y[idx],
                    z=self.transmission_field.z[idx],
                    mag_x=self.transmission_field.mag_x[idx] if self.transmission_field.mag_x is not None else None,
                    mag_y=self.transmission_field.mag_y[idx] if self.transmission_field.mag_y is not None else None,
                    mag_z=self.transmission_field.mag_z[idx] if self.transmission_field.mag_z is not None else None,
                ),
                structure_matrix=self.structure_matrix,
                wave_vectors=subset_wave_vectors,
                lattice_t1=self.lattice_t1,
                lattice_t2=self.lattice_t2,
                default_rdim=self.default_rdim,
                n_sources=len(indices),
                source_parameters=[self.source_parameters[i] for i in indices] if self.source_parameters else None,
                loss=self.loss[idx] if self.loss is not None else None,
                _is_batched=True if len(indices) >= 1 else None,  # Sliced results maintain batched status
            )
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __iter__(self):
        """Iterate over individual source results."""
        for i in range(self.n_sources):
            yield self[i]

    @property
    def as_list(self) -> List["SolverResults"]:
        """Get all results as a list of SolverResults objects."""
        return list(self)

    def get_source_result(self, idx: int) -> "SolverResults":
        """Get results for a specific source.

        Args:
            idx: Source index.

        Returns:
            SolverResults for the specified source.
        """
        return self[idx]

    @classmethod
    def from_dict(cls, data: Dict) -> "SolverResults":
        """Create a SolverResults instance from a dictionary.

        Factory method to construct a SolverResults object from a raw dictionary
        of simulation results. This is useful for converting legacy format data
        into the structured SolverResults format.

        Args:
            data (Dict): Raw dictionary containing simulation results with keys like
                'REF', 'TRN', 'RDE', 'TDE', 'ref_s_x', 'ref_s_y', 'ref_s_z', 'trn_s_x', 'trn_s_y', 'trn_s_z',
                'ref_u_x', 'ref_u_y', 'ref_u_z', 'trn_u_x', 'trn_u_y', 'trn_u_z',
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
            reflection=data["REF"],
            transmission=data["TRN"],
            reflection_diffraction=data["RDE"],
            transmission_diffraction=data["TDE"],
            reflection_field=FieldComponents(
                # Use new Fourier coefficient naming with backward compatibility
                x=data.get("ref_s_x", data.get("rx")),  # Reflection E-field Fourier coeff, x
                y=data.get("ref_s_y", data.get("ry")),  # Reflection E-field Fourier coeff, y
                z=data.get("ref_s_z", data.get("rz")),  # Reflection E-field Fourier coeff, z
                mag_x=data.get("ref_u_x", data.get("rmag_x")),  # Reflection H-field Fourier coeff, x
                mag_y=data.get("ref_u_y", data.get("rmag_y")),  # Reflection H-field Fourier coeff, y
                mag_z=data.get("ref_u_z", data.get("rmag_z")),  # Reflection H-field Fourier coeff, z
            ),
            transmission_field=FieldComponents(
                # Use new Fourier coefficient naming with backward compatibility
                x=data.get("trn_s_x", data.get("tx")),  # Transmission E-field Fourier coeff, x
                y=data.get("trn_s_y", data.get("ty")),  # Transmission E-field Fourier coeff, y
                z=data.get("trn_s_z", data.get("tz")),  # Transmission E-field Fourier coeff, z
                mag_x=data.get("trn_u_x", data.get("tmag_x")),  # Transmission H-field Fourier coeff, x
                mag_y=data.get("trn_u_y", data.get("tmag_y")),  # Transmission H-field Fourier coeff, y
                mag_z=data.get("trn_u_z", data.get("tmag_z")),  # Transmission H-field Fourier coeff, z
            ),
            structure_matrix=ScatteringMatrix(
                S11=data["smat_structure"]["S11"],
                S12=data["smat_structure"]["S12"],
                S21=data["smat_structure"]["S21"],
                S22=data["smat_structure"]["S22"],
            ),
            wave_vectors=WaveVectors(
                kx=data["kx"], ky=data["ky"], kinc=data["kinc"], kzref=data["kzref"], kztrn=data["kztrn"]
            ),
            raw_data=data,
            # New fields with backward compatibility (use .get() for safe access)
            mat_v_ref=data.get("mat_v_ref"),
            mat_v_trn=data.get("mat_v_trn"),
            polarization_data=data.get("polarization_data"),
            solver_config=data.get("solver_config"),
            smat_layers=data.get("smat_layers"),
            # Lattice vectors
            lattice_t1=data.get("lattice_t1"),
            lattice_t2=data.get("lattice_t2"),
            default_rdim=data.get("default_rdim"),
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
            self.transmission_field.z[:, ix, iy],
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
            self.reflection_field.z[:, ix, iy],
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

    def get_reflection_interface_fourier_coefficients(self) -> Dict[str, torch.Tensor]:
        """Get E and H Fourier coefficients at the reflection interface.

        Returns the electromagnetic field Fourier coefficients (s and u components)
        at the reflection interface between the incident region and the first layer.
        This provides complete field information needed for energy flow analysis
        and field visualization.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing field coefficients:
                - 'S_x': Electric field Fourier coefficient x-component (s_x) at reflection interface
                - 'S_y': Electric field Fourier coefficient y-component (s_y) at reflection interface
                - 'S_z': Electric field Fourier coefficient z-component (s_z) at reflection interface
                - 'U_x': Magnetic field Fourier coefficient x-component (u_x) at reflection interface
                - 'U_y': Magnetic field Fourier coefficient y-component (u_y) at reflection interface
                - 'U_z': Magnetic field Fourier coefficient z-component (u_z) at reflection interface

            Each tensor has shape (n_freqs, kdim[0], kdim[1]).
            Returns None for magnetic components if not available.

            Note:
                U components are Fourier coefficients of normalized magnetic fields h_hat(x,y,z).
                The normalization is applied in real space: h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z).

        Examples:
        ```python
        # Get complete Fourier coefficient information at reflection interface
        ref_fields = results.get_reflection_interface_fourier_coefficients()

        # Access electric field Fourier coefficients
        S_x = ref_fields['S_x']  # x-component electric field coefficient
        S_y = ref_fields['S_y']  # y-component electric field coefficient

        # Access normalized magnetic field Fourier coefficients
        # Note: U components are normalized and require denormalization for physical calculations
        U_x = ref_fields['U_x']  # x-component magnetic field coefficient (normalized)
        U_y = ref_fields['U_y']  # y-component magnetic field coefficient (normalized)

        # Analyze electric field intensity from Fourier coefficients
        electric_field_intensity = (
            torch.abs(ref_fields['S_x'])**2 +
            torch.abs(ref_fields['S_y'])**2 +
            torch.abs(ref_fields['S_z'])**2
        )
        ```

        Keywords:
            interface fields, reflection interface, Fourier coefficients, Poynting vector,
            field enhancement, electromagnetic fields, energy flow
        """
        return {
            "S_x": self.reflection_field.x,
            "S_y": self.reflection_field.y,
            "S_z": self.reflection_field.z,
            "U_x": self.reflection_field.mag_x,
            "U_y": self.reflection_field.mag_y,
            "U_z": self.reflection_field.mag_z,
        }

    def get_transmission_interface_fourier_coefficients(self) -> Dict[str, torch.Tensor]:
        """Get E and H Fourier coefficients at the transmission interface.

        Returns the electromagnetic field Fourier coefficients (s and u components)
        at the transmission interface between the last layer and the transmission region.
        This provides complete field information needed for energy flow analysis
        and field visualization.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing field coefficients:
                - 'S_x': Electric field Fourier coefficient x-component (s_x) at transmission interface
                - 'S_y': Electric field Fourier coefficient y-component (s_y) at transmission interface
                - 'S_z': Electric field Fourier coefficient z-component (s_z) at transmission interface
                - 'U_x': Magnetic field Fourier coefficient x-component (u_x) at transmission interface
                - 'U_y': Magnetic field Fourier coefficient y-component (u_y) at transmission interface
                - 'U_z': Magnetic field Fourier coefficient z-component (u_z) at transmission interface

            Each tensor has shape (n_freqs, kdim[0], kdim[1]).
            Returns None for magnetic components if not available.

            Note:
                U components are Fourier coefficients of normalized magnetic fields h_hat(x,y,z).
                The normalization is applied in real space: h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z).

        Examples:
        ```python
        # Get complete Fourier coefficient information at transmission interface
        trn_fields = results.get_transmission_interface_fourier_coefficients()

        # Access electric field Fourier coefficients
        S_x = trn_fields['S_x']  # x-component electric field coefficient
        S_y = trn_fields['S_y']  # y-component electric field coefficient

        # Access normalized magnetic field Fourier coefficients
        # Note: U components are normalized and require denormalization for physical calculations
        U_x = trn_fields['U_x']  # x-component magnetic field coefficient (normalized)
        U_y = trn_fields['U_y']  # y-component magnetic field coefficient (normalized)

        # Analyze electric field intensity from Fourier coefficients
        electric_field_intensity = (
            torch.abs(trn_fields['S_x'])**2 +
            torch.abs(trn_fields['S_y'])**2 +
            torch.abs(trn_fields['S_z'])**2
        )

        # TODO: Energy conservation analysis and Poynting vector calculations require
        # denormalization of U coefficients. This will be implemented in future versions.
        ```

        Keywords:
            interface fields, transmission interface, Fourier coefficients, Poynting vector,
            energy conservation, electromagnetic fields, energy flow
        """
        return {
            "S_x": self.transmission_field.x,
            "S_y": self.transmission_field.y,
            "S_z": self.transmission_field.z,
            "U_x": self.transmission_field.mag_x,
            "U_y": self.transmission_field.mag_y,
            "U_z": self.transmission_field.mag_z,
        }

    def calculate_interface_fourier_coefficients(self, interface: str = "both") -> Dict[str, Dict[str, torch.Tensor]]:
        """Calculate E and H Fourier coefficients at specified interface(s).

        Main API method for calculating electromagnetic Fourier coefficients at
        the reflection and/or transmission interfaces. This method provides
        the foundation for field monitoring, energy flow analysis, and
        electromagnetic field visualization.

        Args:
            interface (str): Which interface(s) to calculate. Options:
                - 'reflection': Only reflection interface coefficients
                - 'transmission': Only transmission interface coefficients
                - 'both': Both interfaces (default)

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Nested dictionary containing:
                - 'reflection': Reflection interface coefficients (if requested)
                - 'transmission': Transmission interface coefficients (if requested)

            Each interface dict contains:
                - 'S_x', 'S_y', 'S_z': Electric field Fourier coefficients
                - 'U_x', 'U_y', 'U_z': Magnetic field Fourier coefficients (normalized)

            Note:
                U components are Fourier coefficients of normalized magnetic fields h_hat(x,y,z).
                The normalization is applied in real space: h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z).

        Examples:
        ```python
        # Get coefficients at both interfaces
        all_fields = results.calculate_interface_fourier_coefficients('both')

        # Access electric field Fourier coefficients
        reflection_S_x = all_fields['reflection']['S_x']
        transmission_S_x = all_fields['transmission']['S_x']

        # Access normalized magnetic field Fourier coefficients
        # Note: U components are normalized and require denormalization for physical calculations
        reflection_U_x = all_fields['reflection']['U_x']  # normalized magnetic coefficient
        transmission_U_x = all_fields['transmission']['U_x']  # normalized magnetic coefficient

        # Analyze field properties at interfaces
        for iface in ['reflection', 'transmission']:
            fields = all_fields[iface]
            # Electric field intensity analysis
            E_intensity = (
                torch.abs(fields['S_x'])**2 +
                torch.abs(fields['S_y'])**2 +
                torch.abs(fields['S_z'])**2
            )
            print(f"Electric field intensity at {iface} interface: {torch.sum(E_intensity)}")

        # Get coefficients at just transmission interface
        trn_only = results.calculate_interface_fourier_coefficients('transmission')
        ```

        Raises:
            ValueError: If interface parameter is not valid.

        Keywords:
            interface fields, electromagnetic fields, Fourier coefficients, field monitoring,
            energy flow analysis, reflection interface, transmission interface
        """
        if interface not in ["reflection", "transmission", "both"]:
            raise ValueError(f"Interface must be 'reflection', 'transmission', or 'both', got '{interface}'")

        result = {}

        if interface in ["reflection", "both"]:
            result["reflection"] = self.get_reflection_interface_fourier_coefficients()

        if interface in ["transmission", "both"]:
            result["transmission"] = self.get_transmission_interface_fourier_coefficients()

        return result

    def find_optimal_source(self, metric: str = "max_transmission", frequency_idx: Optional[int] = None) -> int:
        """Find the source index that optimizes the specified metric.

        This method is only available for batched results (n_sources > 1).

        Args:
            metric: Optimization criterion. Options:
                - 'max_transmission': Maximum transmission
                - 'min_reflection': Minimum reflection
                - 'max_efficiency': Maximum total efficiency (T+R)
            frequency_idx: Specific frequency index to optimize for.
                If None, uses average over all frequencies.

        Returns:
            Index of the optimal source.

        Raises:
            ValueError: If called on single source results or invalid metric.

        Examples:
        ```python
        # Find source with highest transmission (batched results only)
        best_idx = results.find_optimal_source('max_transmission')
        best_result = results[best_idx]

        # Find source with lowest reflection at specific frequency
        best_idx = results.find_optimal_source('min_reflection', frequency_idx=0)
        ```
        """
        if not self.is_batched:
            raise ValueError("find_optimal_source() is only available for batched results (n_sources > 1)")

        if frequency_idx is None:
            # Use average over frequencies
            if metric == "max_transmission":
                values = self.transmission.mean(dim=1)
                return torch.argmax(values).item()
            elif metric == "min_reflection":
                values = self.reflection.mean(dim=1)
                return torch.argmin(values).item()
            elif metric == "max_efficiency":
                values = (self.transmission + self.reflection).mean(dim=1)
                return torch.argmax(values).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            # Use specific frequency
            if metric == "max_transmission":
                return torch.argmax(self.transmission[:, frequency_idx]).item()
            elif metric == "min_reflection":
                return torch.argmin(self.reflection[:, frequency_idx]).item()
            elif metric == "max_efficiency":
                values = self.transmission[:, frequency_idx] + self.reflection[:, frequency_idx]
                return torch.argmax(values).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")

    def get_parameter_sweep_data(
        self, parameter: str, metric: str, frequency_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract parameter sweep data for plotting.

        This method is only available for batched results (n_sources > 1).

        Args:
            parameter: Source parameter name ('theta', 'phi', 'pte', 'ptm').
            metric: Result metric ('transmission', 'reflection', 'loss').
            frequency_idx: Frequency index to extract data for.

        Returns:
            Tuple of (parameter_values, metric_values) tensors.

        Raises:
            ValueError: If called on single source results.
            KeyError: If parameter not found in source_parameters.

        Examples:
        ```python
        # Get data for angle sweep (batched results only)
        angles, trans = results.get_parameter_sweep_data('theta', 'transmission')

        # Plot the sweep
        plt.plot(angles * 180/np.pi, trans)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Transmission')
        ```
        """
        if not self.is_batched:
            raise ValueError("get_parameter_sweep_data() is only available for batched results (n_sources > 1)")

        if self.source_parameters is None:
            raise ValueError("Source parameters not available. Ensure solver stores source_parameters in results.")

        # Extract parameter values
        param_values = torch.tensor([src[parameter] for src in self.source_parameters])

        # Extract metric values
        if metric == "transmission":
            metric_values = self.transmission[:, frequency_idx]
        elif metric == "reflection":
            metric_values = self.reflection[:, frequency_idx]
        elif metric == "loss":
            if self.loss is None:
                raise ValueError("Loss data not available for this results object")
            metric_values = self.loss[:, frequency_idx]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return param_values, metric_values
