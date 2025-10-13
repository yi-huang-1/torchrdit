# Results Module

## Overview

The `torchrdit.results` module provides structured data containers for electromagnetic simulation results. It defines dataclasses that organize field components, scattering matrices, wave vectors, and diffraction efficiencies, making simulation results more accessible and easier to work with.

## Key Components

The module consists of several dataclasses that organize different aspects of simulation results:

### ScatteringMatrix

Contains the four components of the scattering matrix:

```python
@dataclass
class ScatteringMatrix:
    """Scattering matrix components for electromagnetic simulation"""
    S11: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
    S12: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
    S21: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
    S22: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
```

### FieldComponents

Organizes the x, y, and z components of electromagnetic fields:

```python
@dataclass
class FieldComponents:
    """Field components in x, y, z directions"""
    x: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
    y: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
    z: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
```

### WaveVectors

Stores wave vector components for the simulation:

```python
@dataclass
class WaveVectors:
    """Wave vector components for the simulation"""
    kx: torch.Tensor  # (kdim[0], kdim[1])
    ky: torch.Tensor  # (kdim[0], kdim[1])
    kinc: torch.Tensor  # (n_freqs, 3)
    kzref: torch.Tensor  # (n_freqs, kdim[0]*kdim[1])
    kztrn: torch.Tensor  # (n_freqs, kdim[0]*kdim[1])
```

### SolverResults

The main results container that includes reflection and transmission data, field components, and methods for analyzing diffraction orders.

## Usage Examples

### Basic Usage

```python
import torch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm

# Create a solver
solver = create_solver(algorithm=Algorithm.RCWA)

# Set up the simulation
# ...

# Run the simulation
source = solver.add_source(theta=0, phi=0, pte=1, ptm=0)
results = solver.solve(source)  # Returns a SolverResults object

# Access overall efficiencies
print(f"Reflection: {results.reflection[0].item():.3f}")
print(f"Transmission: {results.transmission[0].item():.3f}")
```

### Accessing Field Components

```python
# Get field components for the zero-order diffraction
tx, ty, tz = results.get_zero_order_transmission()

# Calculate field amplitude and phase
amplitude = torch.abs(tx[0])  # Amplitude of x-component (first wavelength)
phase = torch.angle(tx[0])    # Phase in radians
```

### Analyzing Diffraction Orders

```python
# Get efficiency of specific diffraction order
order_1_1_efficiency = results.get_order_transmission_efficiency(order_x=1, order_y=1)

# Get all available diffraction orders
all_orders = results.get_all_diffraction_orders()

# Find all propagating orders
propagating = results.get_propagating_orders(wavelength_idx=0)
print(f"Propagating orders: {propagating}")
```

### Backward Compatibility

```python
# For code that expects the old dictionary format
data_dict = results.to_dict()

# Access data using the old dictionary keys
trn = data_dict['TRN']
ref = data_dict['REF']
```

## API Reference

Below is the complete API reference for the results module, automatically generated from the source code.

# Table of Contents

* [torchrdit.results](#torchrdit.results)
  * [ScatteringMatrix](#torchrdit.results.ScatteringMatrix)
    * [S11](#torchrdit.results.ScatteringMatrix.S11)
    * [S12](#torchrdit.results.ScatteringMatrix.S12)
    * [S21](#torchrdit.results.ScatteringMatrix.S21)
    * [S22](#torchrdit.results.ScatteringMatrix.S22)
  * [\_FourierCoefficients](#torchrdit.results._FourierCoefficients)
    * [s\_x](#torchrdit.results._FourierCoefficients.s_x)
    * [s\_y](#torchrdit.results._FourierCoefficients.s_y)
    * [s\_z](#torchrdit.results._FourierCoefficients.s_z)
    * [u\_x](#torchrdit.results._FourierCoefficients.u_x)
    * [u\_y](#torchrdit.results._FourierCoefficients.u_y)
    * [u\_z](#torchrdit.results._FourierCoefficients.u_z)
  * [FieldComponents](#torchrdit.results.FieldComponents)
    * [x](#torchrdit.results.FieldComponents.x)
    * [y](#torchrdit.results.FieldComponents.y)
    * [z](#torchrdit.results.FieldComponents.z)
    * [mag\_x](#torchrdit.results.FieldComponents.mag_x)
    * [mag\_y](#torchrdit.results.FieldComponents.mag_y)
    * [mag\_z](#torchrdit.results.FieldComponents.mag_z)
  * [WaveVectors](#torchrdit.results.WaveVectors)
    * [kx](#torchrdit.results.WaveVectors.kx)
    * [ky](#torchrdit.results.WaveVectors.ky)
    * [kinc](#torchrdit.results.WaveVectors.kinc)
    * [kzref](#torchrdit.results.WaveVectors.kzref)
    * [kztrn](#torchrdit.results.WaveVectors.kztrn)
  * [SolverResults](#torchrdit.results.SolverResults)
    * [reflection](#torchrdit.results.SolverResults.reflection)
    * [transmission](#torchrdit.results.SolverResults.transmission)
    * [reflection\_diffraction](#torchrdit.results.SolverResults.reflection_diffraction)
    * [transmission\_diffraction](#torchrdit.results.SolverResults.transmission_diffraction)
    * [mat\_v\_ref](#torchrdit.results.SolverResults.mat_v_ref)
    * [mat\_v\_trn](#torchrdit.results.SolverResults.mat_v_trn)
    * [polarization\_data](#torchrdit.results.SolverResults.polarization_data)
    * [solver\_config](#torchrdit.results.SolverResults.solver_config)
    * [smat\_layers](#torchrdit.results.SolverResults.smat_layers)
    * [lattice\_t1](#torchrdit.results.SolverResults.lattice_t1)
    * [lattice\_t2](#torchrdit.results.SolverResults.lattice_t2)
    * [default\_rdim](#torchrdit.results.SolverResults.default_rdim)
    * [n\_sources](#torchrdit.results.SolverResults.n_sources)
    * [source\_parameters](#torchrdit.results.SolverResults.source_parameters)
    * [loss](#torchrdit.results.SolverResults.loss)
    * [is\_batched](#torchrdit.results.SolverResults.is_batched)
    * [\_\_len\_\_](#torchrdit.results.SolverResults.__len__)
    * [\_\_getitem\_\_](#torchrdit.results.SolverResults.__getitem__)
    * [\_\_iter\_\_](#torchrdit.results.SolverResults.__iter__)
    * [as\_list](#torchrdit.results.SolverResults.as_list)
    * [get\_source\_result](#torchrdit.results.SolverResults.get_source_result)
    * [from\_dict](#torchrdit.results.SolverResults.from_dict)
    * [to\_dict](#torchrdit.results.SolverResults.to_dict)
    * [get\_diffraction\_order\_indices](#torchrdit.results.SolverResults.get_diffraction_order_indices)
    * [get\_zero\_order\_transmission](#torchrdit.results.SolverResults.get_zero_order_transmission)
    * [get\_zero\_order\_reflection](#torchrdit.results.SolverResults.get_zero_order_reflection)
    * [get\_order\_transmission\_efficiency](#torchrdit.results.SolverResults.get_order_transmission_efficiency)
    * [get\_order\_reflection\_efficiency](#torchrdit.results.SolverResults.get_order_reflection_efficiency)
    * [get\_all\_diffraction\_orders](#torchrdit.results.SolverResults.get_all_diffraction_orders)
    * [get\_propagating\_orders](#torchrdit.results.SolverResults.get_propagating_orders)
    * [get\_reflection\_interface\_fourier\_coefficients](#torchrdit.results.SolverResults.get_reflection_interface_fourier_coefficients)
    * [get\_transmission\_interface\_fourier\_coefficients](#torchrdit.results.SolverResults.get_transmission_interface_fourier_coefficients)
    * [calculate\_interface\_fourier\_coefficients](#torchrdit.results.SolverResults.calculate_interface_fourier_coefficients)
    * [find\_optimal\_source](#torchrdit.results.SolverResults.find_optimal_source)
    * [get\_parameter\_sweep\_data](#torchrdit.results.SolverResults.get_parameter_sweep_data)

<a id="torchrdit.results"></a>

# torchrdit.results

Module for electromagnetic simulation results processing and analysis.

This module provides data structures for organizing and analyzing results from
electromagnetic simulations. It defines dataclasses that encapsulate reflection
and transmission coefficients, field components, scattering matrices, and wave
vectors in a structured, easy-to-use format.

**Notes**:

  Throughout this module, kdim_0_tims_1 = kdim[0] * kdim[1], where kdim is the
  k-space dimensions used in the simulation.
  
  Classes:
- `ScatteringMatrix` - Container for S-parameter matrices (S11, S12, S21, S22).
- `FieldComponents` - Container for x, y, z components of electromagnetic fields.
- `WaveVectors` - Container for wave vector information in the simulation.
- `SolverResults` - Main results container with analysis methods for diffraction orders.
  

**Examples**:

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

<a id="torchrdit.results.ScatteringMatrix"></a>

## ScatteringMatrix Objects

```python
@dataclass
class ScatteringMatrix()
```

Scattering matrix components for electromagnetic simulation.

Represents the four components of the scattering matrix (S-parameters) that
characterize the reflection and transmission properties of an electromagnetic structure.

**Notes**:

  kdim_0_tims_1 = kdim[0] * kdim[1], where kdim is the k-space dimensions
  used in the simulation.
  

**Attributes**:

- `S11` _torch.Tensor_ - Reflection coefficient matrix for waves incident from port 1.
- `Shape` - (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
- `S12` _torch.Tensor_ - Transmission coefficient matrix from port 2 to port 1.
- `Shape` - (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
- `S21` _torch.Tensor_ - Transmission coefficient matrix from port 1 to port 2.
- `Shape` - (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
- `S22` _torch.Tensor_ - Reflection coefficient matrix for waves incident from port 2.
- `Shape` - (n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)
  

**Examples**:

```python
# Access reflection coefficient for first frequency
s11_first_freq = smatrix.S11[0]

# Calculate transmitted power through the structure
transmitted_power = torch.abs(smatrix.S21)**2
```
  
  Keywords:
  scattering matrix, S-parameters, S11, S12, S21, S22, reflection, transmission,
  electromagnetic, Fourier harmonics

<a id="torchrdit.results.ScatteringMatrix.S11"></a>

#### S11

(n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)

<a id="torchrdit.results.ScatteringMatrix.S12"></a>

#### S12

(n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)

<a id="torchrdit.results.ScatteringMatrix.S21"></a>

#### S21

(n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)

<a id="torchrdit.results.ScatteringMatrix.S22"></a>

#### S22

(n_freqs, 2*kdim_0_tims_1, 2*kdim_0_tims_1)

<a id="torchrdit.results._FourierCoefficients"></a>

## \_FourierCoefficients Objects

```python
@dataclass
class _FourierCoefficients()
```

Internal class for k-space Fourier coefficients (not real-space fields).

Contains Fourier coefficients in k-space representing spectral amplitudes.
These are the raw coefficients from the electromagnetic solver

Note: This is an internal class - use public field API methods instead.

<a id="torchrdit.results._FourierCoefficients.s_x"></a>

#### s\_x

Electric field Fourier coefficient x-component

<a id="torchrdit.results._FourierCoefficients.s_y"></a>

#### s\_y

Electric field Fourier coefficient y-component

<a id="torchrdit.results._FourierCoefficients.s_z"></a>

#### s\_z

Electric field Fourier coefficient z-component

<a id="torchrdit.results._FourierCoefficients.u_x"></a>

#### u\_x

Magnetic field Fourier coefficient x-component

<a id="torchrdit.results._FourierCoefficients.u_y"></a>

#### u\_y

Magnetic field Fourier coefficient y-component

<a id="torchrdit.results._FourierCoefficients.u_z"></a>

#### u\_z

Magnetic field Fourier coefficient z-component

<a id="torchrdit.results.FieldComponents"></a>

## FieldComponents Objects

```python
@dataclass
class FieldComponents()
```

Electromagnetic field Fourier coefficients in k-space.

Contains Fourier coefficients (not real-space fields) of electromagnetic field
components in k-space. These represent the spectral amplitudes s_x, s_y, s_z for
electric fields and u_x, u_y, u_z for magnetic fields, indexed by harmonic orders.

Important: These are Fourier coefficients in k-space, not real-space field values.

**Attributes**:

- `x` _torch.Tensor_ - X-component Fourier coefficient of electric field (s_x).
- `Shape` - (n_freqs, n_harmonics_x, n_harmonics_y)
- `y` _torch.Tensor_ - Y-component Fourier coefficient of electric field (s_y).
- `Shape` - (n_freqs, n_harmonics_x, n_harmonics_y)
- `z` _torch.Tensor_ - Z-component Fourier coefficient of electric field (s_z).
- `Shape` - (n_freqs, n_harmonics_x, n_harmonics_y)
- `mag_x` _Optional[torch.Tensor]_ - X-component Fourier coefficient of magnetic field (u_x).
- `Shape` - (n_freqs, n_harmonics_x, n_harmonics_y). Default: None
- `Note` - These are Fourier coefficients of normalized magnetic field h_hat(x,y,z) where h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z)
- `mag_y` _Optional[torch.Tensor]_ - Y-component Fourier coefficient of magnetic field (u_y).
- `Shape` - (n_freqs, n_harmonics_x, n_harmonics_y). Default: None
- `Note` - These are Fourier coefficients of normalized magnetic field h_hat(x,y,z) where h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z)
- `mag_z` _Optional[torch.Tensor]_ - Z-component Fourier coefficient of magnetic field (u_z).
- `Shape` - (n_freqs, n_harmonics_x, n_harmonics_y). Default: None
- `Note` - These are Fourier coefficients of normalized magnetic field h_hat(x,y,z) where h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z)
  

**Examples**:

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

<a id="torchrdit.results.FieldComponents.x"></a>

#### x

Electric field x-component (n_freqs, kdim[0], kdim[1])

<a id="torchrdit.results.FieldComponents.y"></a>

#### y

Electric field y-component (n_freqs, kdim[0], kdim[1])

<a id="torchrdit.results.FieldComponents.z"></a>

#### z

Electric field z-component (n_freqs, kdim[0], kdim[1])

<a id="torchrdit.results.FieldComponents.mag_x"></a>

#### mag\_x

Magnetic field x-component

<a id="torchrdit.results.FieldComponents.mag_y"></a>

#### mag\_y

Magnetic field y-component

<a id="torchrdit.results.FieldComponents.mag_z"></a>

#### mag\_z

Magnetic field z-component

<a id="torchrdit.results.WaveVectors"></a>

## WaveVectors Objects

```python
@dataclass
class WaveVectors()
```

Wave vector components for the electromagnetic simulation.

Contains the wave vector information for the simulation, including incident,
reflected, and transmitted wave vectors in the Fourier domain.

**Attributes**:

- `kx` _torch.Tensor_ - X-component of the wave vector in reciprocal space.
- `Shape` - (kdim[0], kdim[1])
- `ky` _torch.Tensor_ - Y-component of the wave vector in reciprocal space.
- `Shape` - (kdim[0], kdim[1])
- `kinc` _torch.Tensor_ - Incident wave vector.
- `Shape` - (n_freqs, 3)
- `kzref` _torch.Tensor_ - Z-component of the wave vector in reflection region.
- `Shape` - (n_freqs, kdim[0]*kdim[1])
- `kztrn` _torch.Tensor_ - Z-component of the wave vector in transmission region.
- `Shape` - (n_freqs, kdim[0]*kdim[1])
  

**Examples**:

```python
# Get propagation constant in z-direction for reflected waves
kz = wave_vectors.kzref

# Calculate wave vector magnitude
k_magnitude = torch.sqrt(wave_vectors.kx**2 + wave_vectors.ky**2 + kz**2)
```
  
  Keywords:
  wave vector, k-vector, propagation constant, reciprocal space, Fourier harmonics,
  incident wave, reflected wave, transmitted wave, dispersion

<a id="torchrdit.results.WaveVectors.kx"></a>

#### kx

(kdim[0], kdim[1])

<a id="torchrdit.results.WaveVectors.ky"></a>

#### ky

(kdim[0], kdim[1])

<a id="torchrdit.results.WaveVectors.kinc"></a>

#### kinc

(n_freqs, 3)

<a id="torchrdit.results.WaveVectors.kzref"></a>

#### kzref

(n_freqs, kdim[0]*kdim[1])

<a id="torchrdit.results.WaveVectors.kztrn"></a>

#### kztrn

(n_freqs, kdim[0]*kdim[1])

<a id="torchrdit.results.SolverResults"></a>

## SolverResults Objects

```python
@dataclass
class SolverResults()
```

Unified results container for electromagnetic solver with single/batched source support.

Comprehensive container for all results from electromagnetic simulations, supporting
both single and batched source solving. Includes reflection and transmission coefficients,
diffraction efficiencies, field components, scattering matrices, wave vectors.

**Notes**:

  kdim_0_tims_1 = kdim[0] * kdim[1], where kdim is the k-space dimensions
  used in the simulation. This relationship appears in the ScatteringMatrix
  component's tensor shapes.
  

**Attributes**:

- `reflection` _torch.Tensor_ - Total reflection efficiency.
- `Shape` - (n_freqs) for single source, (n_sources, n_freqs) for batched
- `transmission` _torch.Tensor_ - Total transmission efficiency.
- `Shape` - (n_freqs) for single source, (n_sources, n_freqs) for batched
- `reflection_diffraction` _torch.Tensor_ - Reflection efficiencies for each diffraction order.
- `Shape` - (n_freqs, kdim[0], kdim[1]) or (n_sources, n_freqs, kdim[0], kdim[1])
- `transmission_diffraction` _torch.Tensor_ - Transmission efficiencies for each diffraction order.
- `Shape` - (n_freqs, kdim[0], kdim[1]) or (n_sources, n_freqs, kdim[0], kdim[1])
- `reflection_field` _FieldComponents_ - Field Fourier coefficients in the reflection region.
- `transmission_field` _FieldComponents_ - Field Fourier coefficients in the transmission region.
- `structure_matrix` _ScatteringMatrix_ - Scattering matrix for the entire structure.
- `wave_vectors` _WaveVectors_ - Wave vector components for the simulation.
- `n_sources` _int_ - Number of sources (1 for single, >1 for batched).
  lattice_t1, lattice_t2 (Optional[torch.Tensor]): Lattice vectors.
- `default_rdim` _Optional[Tuple[int, int]]_ - Default spatial resolution from solver.
  

**Examples**:

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

<a id="torchrdit.results.SolverResults.reflection"></a>

#### reflection

(n_freqs)

<a id="torchrdit.results.SolverResults.transmission"></a>

#### transmission

(n_freqs)

<a id="torchrdit.results.SolverResults.reflection_diffraction"></a>

#### reflection\_diffraction

(n_freqs, kdim[0], kdim[1])

<a id="torchrdit.results.SolverResults.transmission_diffraction"></a>

#### transmission\_diffraction

(n_freqs, kdim[0], kdim[1])

<a id="torchrdit.results.SolverResults.mat_v_ref"></a>

#### mat\_v\_ref

V matrix for reflection region (magnetic field mode matrix)

<a id="torchrdit.results.SolverResults.mat_v_trn"></a>

#### mat\_v\_trn

V matrix for transmission region (magnetic field mode matrix)

<a id="torchrdit.results.SolverResults.polarization_data"></a>

#### polarization\_data

Polarization data containing esrc and other vectors

<a id="torchrdit.results.SolverResults.solver_config"></a>

#### solver\_config

Solver configuration (kdim, n_freqs, device, etc.)

<a id="torchrdit.results.SolverResults.smat_layers"></a>

#### smat\_layers

Individual layer S-matrices for proper mode coefficient calculation

<a id="torchrdit.results.SolverResults.lattice_t1"></a>

#### lattice\_t1

First lattice vector [x, y] from solver

<a id="torchrdit.results.SolverResults.lattice_t2"></a>

#### lattice\_t2

Second lattice vector [x, y] from solver

<a id="torchrdit.results.SolverResults.default_rdim"></a>

#### default\_rdim

Default spatial resolution [height, width] from solver

<a id="torchrdit.results.SolverResults.n_sources"></a>

#### n\_sources

Number of sources (1 for single, >1 for batched)

<a id="torchrdit.results.SolverResults.source_parameters"></a>

#### source\_parameters

Original source dictionaries for batched case

<a id="torchrdit.results.SolverResults.loss"></a>

#### loss

Total loss for each source/wavelength (batched only)

<a id="torchrdit.results.SolverResults.is_batched"></a>

#### is\_batched

```python
@property
def is_batched() -> bool
```

Return True if this represents batched source results.

This is True when:
1. Explicitly set via _is_batched (for single-element list inputs)
2. When n_sources > 1 (multiple sources)

<a id="torchrdit.results.SolverResults.__len__"></a>

#### \_\_len\_\_

```python
def __len__() -> int
```

Return number of sources in the results.

<a id="torchrdit.results.SolverResults.__getitem__"></a>

#### \_\_getitem\_\_

```python
def __getitem__(
        idx: Union[int, slice]) -> Union["SolverResults", "SolverResults"]
```

Get results for specific source(s).

**Arguments**:

- `idx` - Integer index or slice for source selection.
  

**Returns**:

  SolverResults for single index, SolverResults for slice (both single and batched compatible).

<a id="torchrdit.results.SolverResults.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__()
```

Iterate over individual source results.

<a id="torchrdit.results.SolverResults.as_list"></a>

#### as\_list

```python
@property
def as_list() -> List["SolverResults"]
```

Get all results as a list of SolverResults objects.

<a id="torchrdit.results.SolverResults.get_source_result"></a>

#### get\_source\_result

```python
def get_source_result(idx: int) -> "SolverResults"
```

Get results for a specific source.

**Arguments**:

- `idx` - Source index.
  

**Returns**:

  SolverResults for the specified source.

<a id="torchrdit.results.SolverResults.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: Dict) -> "SolverResults"
```

Create a SolverResults instance from a dictionary.

Factory method to construct a SolverResults object from a raw dictionary
of simulation results. This is useful for converting legacy format data
into the structured SolverResults format.

**Arguments**:

- `data` _Dict_ - Raw dictionary containing simulation results with keys like
  'REF', 'TRN', 'RDE', 'TDE', 'ref_s_x', 'ref_s_y', 'ref_s_z', 'trn_s_x', 'trn_s_y', 'trn_s_z',
  'ref_u_x', 'ref_u_y', 'ref_u_z', 'trn_u_x', 'trn_u_y', 'trn_u_z',
  'smat_structure', 'kx', 'ky', 'kinc', 'kzref', 'kztrn'.
  

**Returns**:

- `SolverResults` - A structured results object with organized data.
  

**Examples**:

```python
# Convert legacy dictionary format to SolverResults
legacy_data = solver.solve_legacy()
results = SolverResults.from_dict(legacy_data)

# Now use the structured API
print(f"Transmission: {results.transmission[0].item():.3f}")
```
  
  Keywords:
  factory method, conversion, dictionary, legacy format, backward compatibility

<a id="torchrdit.results.SolverResults.to_dict"></a>

#### to\_dict

```python
def to_dict() -> Dict
```

Convert to dictionary for backward compatibility.

Exports the SolverResults object back to a raw dictionary format for
backward compatibility with code expecting the legacy format.

**Returns**:

- `Dict` - Raw dictionary containing all simulation results.
  

**Examples**:

```python
# Get dictionary format for legacy code
results = device.solve(source)
data_dict = results.to_dict()

# Use with legacy code
legacy_function(data_dict)
```
  
  Keywords:
  conversion, dictionary, legacy format, backward compatibility

<a id="torchrdit.results.SolverResults.get_diffraction_order_indices"></a>

#### get\_diffraction\_order\_indices

```python
def get_diffraction_order_indices(order_x: int = 0,
                                  order_y: int = 0) -> Tuple[int, int]
```

Get the indices for a specific diffraction order.

Converts diffraction order coordinates (m,n) to array indices (i,j) in the
diffraction efficiency and field tensors. The zero order (0,0) corresponds
to the center of the k-space grid.

**Arguments**:

- `order_x` _int, optional_ - The x-component of the diffraction order.
  Defaults to 0 (specular).
- `order_y` _int, optional_ - The y-component of the diffraction order.
  Defaults to 0 (specular).
  

**Returns**:

  Tuple[int, int]: The indices (ix, iy) corresponding to the requested order
  in the diffraction efficiency and field tensors.
  

**Raises**:

- `ValueError` - If the requested diffraction order is outside the simulation bounds.
  

**Examples**:

```python
# Get indices for the (1,1) diffraction order
ix, iy = results.get_diffraction_order_indices(1, 1)

# Access corresponding transmission efficiency
efficiency = results.transmission_diffraction[0, ix, iy]
```
  
  Keywords:
  diffraction order, indices, k-space, Fourier harmonics, specular, array index

<a id="torchrdit.results.SolverResults.get_zero_order_transmission"></a>

#### get\_zero\_order\_transmission

```python
def get_zero_order_transmission(
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Get the zero-order transmission field components.

Returns the electric field components (Ex, Ey, Ez) for the zero-order
(specular) transmission. This is useful for analyzing phase, polarization,
and amplitude of the directly transmitted light.

**Returns**:

  Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The (x, y, z) field components
  for the zero diffraction order. Each tensor has shape (n_freqs).
  

**Examples**:

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

<a id="torchrdit.results.SolverResults.get_zero_order_reflection"></a>

#### get\_zero\_order\_reflection

```python
def get_zero_order_reflection(
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Get the zero-order reflection field components.

Returns the electric field components (Ex, Ey, Ez) for the zero-order
(specular) reflection. This is useful for analyzing phase, polarization,
and amplitude of the directly reflected light.

**Returns**:

  Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The (x, y, z) field components
  for the zero diffraction order. Each tensor has shape (n_freqs).
  

**Examples**:

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

<a id="torchrdit.results.SolverResults.get_order_transmission_efficiency"></a>

#### get\_order\_transmission\_efficiency

```python
def get_order_transmission_efficiency(order_x: int = 0,
                                      order_y: int = 0) -> torch.Tensor
```

Get the transmission diffraction efficiency for a specific order.

Returns the diffraction efficiency (ratio of transmitted power to incident power)
for the specified diffraction order. For the zero order (0,0), this gives the
direct transmission efficiency.

**Arguments**:

- `order_x` _int, optional_ - The x-component of the diffraction order.
  Defaults to 0 (specular).
- `order_y` _int, optional_ - The y-component of the diffraction order.
  Defaults to 0 (specular).
  

**Returns**:

- `torch.Tensor` - The transmission diffraction efficiency for the specified order.
- `Shape` - (n_freqs)
  

**Examples**:

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

<a id="torchrdit.results.SolverResults.get_order_reflection_efficiency"></a>

#### get\_order\_reflection\_efficiency

```python
def get_order_reflection_efficiency(order_x: int = 0,
                                    order_y: int = 0) -> torch.Tensor
```

Get the reflection diffraction efficiency for a specific order.

Returns the diffraction efficiency (ratio of reflected power to incident power)
for the specified diffraction order. For the zero order (0,0), this gives the
direct reflection efficiency.

**Arguments**:

- `order_x` _int, optional_ - The x-component of the diffraction order.
  Defaults to 0 (specular).
- `order_y` _int, optional_ - The y-component of the diffraction order.
  Defaults to 0 (specular).
  

**Returns**:

- `torch.Tensor` - The reflection diffraction efficiency for the specified order.
- `Shape` - (n_freqs)
  

**Examples**:

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

<a id="torchrdit.results.SolverResults.get_all_diffraction_orders"></a>

#### get\_all\_diffraction\_orders

```python
def get_all_diffraction_orders() -> List[Tuple[int, int]]
```

Get a list of all available diffraction orders as (m, n) tuples.

Returns all diffraction orders included in the simulation, whether propagating
or evanescent. Each order is represented as a tuple (m, n) where m is the
x-component and n is the y-component of the diffraction order.

**Returns**:

  List[Tuple[int, int]]: List of all diffraction orders as (m, n) tuples.
  

**Examples**:

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

<a id="torchrdit.results.SolverResults.get_propagating_orders"></a>

#### get\_propagating\_orders

```python
def get_propagating_orders(wavelength_idx: int = 0) -> List[Tuple[int, int]]
```

Get a list of propagating diffraction orders for a specific wavelength.

Identifies which diffraction orders are propagating (rather than evanescent)
for the specified wavelength. Propagating orders have real-valued z-component
of the wave vector and contribute to far-field diffraction patterns.

**Arguments**:

- `wavelength_idx` _int, optional_ - Index of the wavelength in the simulation.
  Defaults to 0 (first wavelength).
  

**Returns**:

  List[Tuple[int, int]]: List of propagating diffraction orders as (m, n) tuples.
  

**Examples**:

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

<a id="torchrdit.results.SolverResults.get_reflection_interface_fourier_coefficients"></a>

#### get\_reflection\_interface\_fourier\_coefficients

```python
def get_reflection_interface_fourier_coefficients() -> Dict[str, torch.Tensor]
```

Get E and H Fourier coefficients at the reflection interface.

Returns the electromagnetic field Fourier coefficients (s and u components)
at the reflection interface between the incident region and the first layer.
This provides complete field information needed for energy flow analysis
and field visualization.

**Returns**:

  Dict[str, torch.Tensor]: Dictionary containing field coefficients:
  - 'S_x': Electric field Fourier coefficient x-component (s_x) at reflection interface
  - 'S_y': Electric field Fourier coefficient y-component (s_y) at reflection interface
  - 'S_z': Electric field Fourier coefficient z-component (s_z) at reflection interface
  - 'U_x': Magnetic field Fourier coefficient x-component (u_x) at reflection interface
  - 'U_y': Magnetic field Fourier coefficient y-component (u_y) at reflection interface
  - 'U_z': Magnetic field Fourier coefficient z-component (u_z) at reflection interface
  
  Each tensor has shape (n_freqs, kdim[0], kdim[1]).
  Returns None for magnetic components if not available.
  

**Notes**:

  U components are Fourier coefficients of normalized magnetic fields h_hat(x,y,z).
  The normalization is applied in real space: h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z).
  

**Examples**:

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

<a id="torchrdit.results.SolverResults.get_transmission_interface_fourier_coefficients"></a>

#### get\_transmission\_interface\_fourier\_coefficients

```python
def get_transmission_interface_fourier_coefficients(
) -> Dict[str, torch.Tensor]
```

Get E and H Fourier coefficients at the transmission interface.

Returns the electromagnetic field Fourier coefficients (s and u components)
at the transmission interface between the last layer and the transmission region.
This provides complete field information needed for energy flow analysis
and field visualization.

**Returns**:

  Dict[str, torch.Tensor]: Dictionary containing field coefficients:
  - 'S_x': Electric field Fourier coefficient x-component (s_x) at transmission interface
  - 'S_y': Electric field Fourier coefficient y-component (s_y) at transmission interface
  - 'S_z': Electric field Fourier coefficient z-component (s_z) at transmission interface
  - 'U_x': Magnetic field Fourier coefficient x-component (u_x) at transmission interface
  - 'U_y': Magnetic field Fourier coefficient y-component (u_y) at transmission interface
  - 'U_z': Magnetic field Fourier coefficient z-component (u_z) at transmission interface
  
  Each tensor has shape (n_freqs, kdim[0], kdim[1]).
  Returns None for magnetic components if not available.
  

**Notes**:

  U components are Fourier coefficients of normalized magnetic fields h_hat(x,y,z).
  The normalization is applied in real space: h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z).
  

**Examples**:

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

<a id="torchrdit.results.SolverResults.calculate_interface_fourier_coefficients"></a>

#### calculate\_interface\_fourier\_coefficients

```python
def calculate_interface_fourier_coefficients(
        interface: str = "both") -> Dict[str, Dict[str, torch.Tensor]]
```

Calculate E and H Fourier coefficients at specified interface(s).

Main API method for calculating electromagnetic Fourier coefficients at
the reflection and/or transmission interfaces. This method provides
the foundation for field monitoring, energy flow analysis, and
electromagnetic field visualization.

**Arguments**:

- `interface` _str_ - Which interface(s) to calculate. Options:
  - 'reflection': Only reflection interface coefficients
  - 'transmission': Only transmission interface coefficients
  - 'both': Both interfaces (default)
  

**Returns**:

  Dict[str, Dict[str, torch.Tensor]]: Nested dictionary containing:
  - 'reflection': Reflection interface coefficients (if requested)
  - 'transmission': Transmission interface coefficients (if requested)
  
  Each interface dict contains:
  - 'S_x', 'S_y', 'S_z': Electric field Fourier coefficients
  - 'U_x', 'U_y', 'U_z': Magnetic field Fourier coefficients (normalized)
  

**Notes**:

  U components are Fourier coefficients of normalized magnetic fields h_hat(x,y,z).
  The normalization is applied in real space: h_hat(x,y,z) = -j * sqrt(mu0/epsilon0) * h(x,y,z).
  

**Examples**:

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
  

**Raises**:

- `ValueError` - If interface parameter is not valid.
  
  Keywords:
  interface fields, electromagnetic fields, Fourier coefficients, field monitoring,
  energy flow analysis, reflection interface, transmission interface

<a id="torchrdit.results.SolverResults.find_optimal_source"></a>

#### find\_optimal\_source

```python
def find_optimal_source(metric: str = "max_transmission",
                        frequency_idx: Optional[int] = None) -> int
```

Find the source index that optimizes the specified metric.

This method is only available for batched results (n_sources > 1).

**Arguments**:

- `metric` - Optimization criterion. Options:
  - 'max_transmission': Maximum transmission
  - 'min_reflection': Minimum reflection
  - 'max_efficiency': Maximum total efficiency (T+R)
- `frequency_idx` - Specific frequency index to optimize for.
  If None, uses average over all frequencies.
  

**Returns**:

  Index of the optimal source.
  

**Raises**:

- `ValueError` - If called on single source results or invalid metric.
  

**Examples**:

```python
# Find source with highest transmission (batched results only)
best_idx = results.find_optimal_source('max_transmission')
best_result = results[best_idx]

# Find source with lowest reflection at specific frequency
best_idx = results.find_optimal_source('min_reflection', frequency_idx=0)
```

<a id="torchrdit.results.SolverResults.get_parameter_sweep_data"></a>

#### get\_parameter\_sweep\_data

```python
def get_parameter_sweep_data(
        parameter: str,
        metric: str,
        frequency_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]
```

Extract parameter sweep data for plotting.

This method is only available for batched results (n_sources > 1).

**Arguments**:

- `parameter` - Source parameter name ('theta', 'phi', 'pte', 'ptm').
- `metric` - Result metric ('transmission', 'reflection', 'loss').
- `frequency_idx` - Frequency index to extract data for.
  

**Returns**:

  Tuple of (parameter_values, metric_values) tensors.
  

**Raises**:

- `ValueError` - If called on single source results.
- `KeyError` - If parameter not found in source_parameters.
  

**Examples**:

```python
# Get data for angle sweep (batched results only)
angles, trans = results.get_parameter_sweep_data('theta', 'transmission')

# Plot the sweep
plt.plot(angles * 180/np.pi, trans)
plt.xlabel('Angle (degrees)')
plt.ylabel('Transmission')
```

