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
  * [FieldComponents](#torchrdit.results.FieldComponents)
    * [x](#torchrdit.results.FieldComponents.x)
    * [y](#torchrdit.results.FieldComponents.y)
    * [z](#torchrdit.results.FieldComponents.z)
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
    * [from\_dict](#torchrdit.results.SolverResults.from_dict)
    * [to\_dict](#torchrdit.results.SolverResults.to_dict)
    * [get\_diffraction\_order\_indices](#torchrdit.results.SolverResults.get_diffraction_order_indices)
    * [get\_zero\_order\_transmission](#torchrdit.results.SolverResults.get_zero_order_transmission)
    * [get\_zero\_order\_reflection](#torchrdit.results.SolverResults.get_zero_order_reflection)
    * [get\_order\_transmission\_efficiency](#torchrdit.results.SolverResults.get_order_transmission_efficiency)
    * [get\_order\_reflection\_efficiency](#torchrdit.results.SolverResults.get_order_reflection_efficiency)
    * [get\_all\_diffraction\_orders](#torchrdit.results.SolverResults.get_all_diffraction_orders)
    * [get\_propagating\_orders](#torchrdit.results.SolverResults.get_propagating_orders)

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

<a id="torchrdit.results.FieldComponents"></a>

## FieldComponents Objects

```python
@dataclass
class FieldComponents()
```

Field components in x, y, z directions for electromagnetic fields.

Contains the spatial distribution of electromagnetic field components along
the x, y, and z directions in the Fourier domain.

**Attributes**:

- `x` _torch.Tensor_ - X-component of the electromagnetic field.
- `Shape` - (n_freqs, kdim[0], kdim[1])
- `y` _torch.Tensor_ - Y-component of the electromagnetic field.
- `Shape` - (n_freqs, kdim[0], kdim[1])
- `z` _torch.Tensor_ - Z-component of the electromagnetic field.
- `Shape` - (n_freqs, kdim[0], kdim[1])
  

**Examples**:

```python
# Calculate field intensity (|E|²) at first frequency
intensity = (abs(field.x[0])**2 + abs(field.y[0])**2 + abs(field.z[0])**2)

# Extract phase of x-component
phase_x = torch.angle(field.x)
```
  
  Keywords:
  field components, electromagnetic field, electric field, magnetic field,
  Fourier domain, x-component, y-component, z-component, vectorial

<a id="torchrdit.results.FieldComponents.x"></a>

#### x

(n_freqs, kdim[0], kdim[1])

<a id="torchrdit.results.FieldComponents.y"></a>

#### y

(n_freqs, kdim[0], kdim[1])

<a id="torchrdit.results.FieldComponents.z"></a>

#### z

(n_freqs, kdim[0], kdim[1])

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

Complete results from electromagnetic solver.

Comprehensive container for all results from an electromagnetic simulation,
including reflection and transmission coefficients, diffraction efficiencies,
field components, scattering matrices, and wave vectors. This class also provides
methods for analyzing diffraction orders and extracting specific field components.

**Notes**:

  kdim_0_tims_1 = kdim[0] * kdim[1], where kdim is the k-space dimensions
  used in the simulation. This relationship appears in the ScatteringMatrix
  component's tensor shapes.
  

**Attributes**:

- `reflection` _torch.Tensor_ - Total reflection efficiency for each wavelength.
- `Shape` - (n_freqs)
- `transmission` _torch.Tensor_ - Total transmission efficiency for each wavelength.
- `Shape` - (n_freqs)
- `reflection_diffraction` _torch.Tensor_ - Reflection efficiencies for each diffraction order.
- `Shape` - (n_freqs, kdim[0], kdim[1])
- `transmission_diffraction` _torch.Tensor_ - Transmission efficiencies for each diffraction order.
- `Shape` - (n_freqs, kdim[0], kdim[1])
- `reflection_field` _FieldComponents_ - Field components in the reflection region.
- `transmission_field` _FieldComponents_ - Field components in the transmission region.
- `structure_matrix` _ScatteringMatrix_ - Scattering matrix for the entire structure.
- `wave_vectors` _WaveVectors_ - Wave vector components for the simulation.
- `raw_data` _Dict_ - Raw dictionary data for backward compatibility.
  

**Examples**:

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
  'REF', 'TRN', 'RDE', 'TDE', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz',
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

