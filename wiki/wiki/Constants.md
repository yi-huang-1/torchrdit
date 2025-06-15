# Constants Module

## Overview
The `torchrdit.constants` module defines physical constants, unit conversion factors, and enumerations used throughout the TorchRDIT package for electromagnetic simulations.

## Key Components

### Physical Constants
- `EPS_0`: Vacuum permittivity (8.85418782e-12 F/m)
- `MU_0`: Vacuum permeability (1.25663706e-6 H/m)
- `C_0`: Speed of light in vacuum (2.99792458e8 m/s)
- `ETA_0`: Vacuum impedance (376.730313668 Ω)
- `Q_E`: Elementary charge (1.602176634e-19 C)

### Unit Conversion Dictionaries
- `frequnit_dict`: Frequency unit conversion factors (Hz to PHz)
- `lengthunit_dict`: Length unit conversion factors (meter to angstrom)

### Enumerations
- `Algorithm`: Supported electromagnetic solver algorithms (RCWA, RDIT)
- `Precision`: Numerical precision options (SINGLE, DOUBLE)

## Usage Examples

```python
# Using physical constants
from torchrdit.constants import EPS_0, MU_0, C_0
# Calculate the refractive index from relative permittivity
epsilon_r = 2.25  # SiO2
n = (epsilon_r)**0.5
print(f"Refractive index: {n}")  # Refractive index: 1.5

# Converting between units
from torchrdit.constants import lengthunit_dict
# Convert 1550 nm to meters
wavelength_nm = 1550
wavelength_m = wavelength_nm * lengthunit_dict['nm']
print(f"Wavelength in meters: {wavelength_m}")  # Wavelength in meters: 1.55e-06

# Using algorithm enumeration
from torchrdit.constants import Algorithm
# Create a solver with specific algorithm
from torchrdit.solver import create_solver
solver = create_solver(algorithm=Algorithm.RCWA)
```

## API Reference

Below is the complete API reference for the constants module, automatically generated from the source code.

# Table of Contents

* [torchrdit.constants](#torchrdit.constants)
  * [EPS\_0](#torchrdit.constants.EPS_0)
  * [MU\_0](#torchrdit.constants.MU_0)
  * [C\_0](#torchrdit.constants.C_0)
  * [ETA\_0](#torchrdit.constants.ETA_0)
  * [Q\_E](#torchrdit.constants.Q_E)
  * [Algorithm](#torchrdit.constants.Algorithm)
    * [RCWA](#torchrdit.constants.Algorithm.RCWA)
    * [RDIT](#torchrdit.constants.Algorithm.RDIT)
  * [Precision](#torchrdit.constants.Precision)
    * [SINGLE](#torchrdit.constants.Precision.SINGLE)
    * [DOUBLE](#torchrdit.constants.Precision.DOUBLE)

<a id="torchrdit.constants"></a>

# torchrdit.constants

Constants module for the TorchRDIT electromagnetic solver package.

This module defines physical constants, unit conversion factors, and enumerations
used throughout the TorchRDIT package for electromagnetic simulations. These
constants provide a centralized reference for consistent values across the package
and facilitate convenient unit conversions in calculations.

Classes:
Algorithm: Enumeration of supported electromagnetic solver algorithms (RCWA, RDIT).
Precision: Enumeration of numerical precision options (SINGLE, DOUBLE).

Constants:
EPS_0: Vacuum permittivity (8.85418782e-12 F/m)
MU_0: Vacuum permeability (1.25663706e-6 H/m)
C_0: Speed of light in vacuum (2.99792458e8 m/s)
ETA_0: Vacuum impedance (376.730313668 Ω)
Q_E: Elementary charge (1.602176634e-19 C)

Dictionaries:
frequnit_dict: Frequency unit conversion factors (Hz to PHz)
lengthunit_dict: Length unit conversion factors (meter to angstrom)

**Examples**:

```python
# Using physical constants
from torchrdit.constants import EPS_0, MU_0, C_0
# Calculate the refractive index from relative permittivity
epsilon_r = 2.25  # SiO2
n = (epsilon_r)**0.5
print(f"Refractive index: {n}")
# Refractive index: 1.5

# Converting between units
from torchrdit.constants import lengthunit_dict
# Convert 1550 nm to meters
wavelength_nm = 1550
wavelength_m = wavelength_nm * lengthunit_dict['nm']
print(f"Wavelength in meters: {wavelength_m}")
# Wavelength in meters: 1.55e-06

# Using algorithm enumeration
from torchrdit.constants import Algorithm
# Create a solver with specific algorithm
from torchrdit.solver import create_solver
solver = create_solver(algorithm=Algorithm.RCWA)
```
  
  Keywords:
  physical constants, electromagnetic constants, unit conversion, frequency,
  length, permittivity, permeability, speed of light, impedance, RCWA, RDIT,
  precision, algorithms, enums

<a id="torchrdit.constants.EPS_0"></a>

#### EPS\_0

vacuum permittivity (F/m)

<a id="torchrdit.constants.MU_0"></a>

#### MU\_0

vacuum permeability (H/m)

<a id="torchrdit.constants.C_0"></a>

#### C\_0

speed of light in vacuum (m/s)

<a id="torchrdit.constants.ETA_0"></a>

#### ETA\_0

vacuum impedance (Ω)

<a id="torchrdit.constants.Q_E"></a>

#### Q\_E

elementary charge (C)

<a id="torchrdit.constants.Algorithm"></a>

## Algorithm Objects

```python
@unique
class Algorithm(Enum)
```

Enumeration of supported electromagnetic solver algorithms.

This enum defines the available algorithms that can be used for electromagnetic
field calculations in TorchRDIT. Each algorithm has different computational
approaches and performance characteristics.

**Attributes**:

- `RCWA` - Rigorous Coupled-Wave Analysis, the traditional approach using
  eigenmode decomposition. This method provides high accuracy for
  periodic structures but may be computationally intensive for
  large problems.
  
- `RDIT` - Rigorous Diffraction Interface Theory (R-DIT), the
  eigendecomposition-free approach that offers improved computational
  efficiency. This method avoids matrix eigendecomposition and is
  typically faster than RCWA for many problems.
  

**Examples**:

```python
# Create solver with RCWA algorithm
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
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
print(f"Solver uses: {rcwa_solver.algorithm.name}")
# Solver uses: RCWA
```
  

**Notes**:

  The choice of algorithm affects both accuracy and performance. RCWA is
  the traditional gold standard for periodic structure simulations, while
  RDIT offers performance improvements for many cases. The optimal choice
  depends on your specific simulation requirements.
  
  Keywords:
  algorithms, RCWA, RDIT, electromagnetic solvers, simulation methods,
  computational electromagnetics, diffraction, periodic structures,
  eigenmode decomposition, performance optimization

<a id="torchrdit.constants.Algorithm.RCWA"></a>

#### RCWA

Rigorous Coupled-Wave Analysis

<a id="torchrdit.constants.Algorithm.RDIT"></a>

#### RDIT

Rigorous Diffraction Interface Theory (R-DIT)

<a id="torchrdit.constants.Precision"></a>

## Precision Objects

```python
@unique
class Precision(Enum)
```

Enumeration of numerical precision options for calculations.

This enum defines the available precision options for calculations in
TorchRDIT. The choice of precision affects both the accuracy of results
and the computational resources required.

**Attributes**:

- `SINGLE` - Single precision (32-bit floating point, torch.float32).
  Provides good accuracy with reduced memory usage and
  faster computation, especially beneficial for large
  simulations or GPU-accelerated calculations.
  
- `DOUBLE` - Double precision (64-bit floating point, torch.float64).
  Provides higher numerical accuracy at the cost of increased
  memory usage and potentially slower computation. Recommended
  for simulations requiring high precision or when dealing
  with ill-conditioned matrices.
  

**Examples**:

```python
# Create solver with single precision
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm, Precision
single_prec_solver = create_solver(
    algorithm=Algorithm.RCWA,
    precision=Precision.SINGLE
)

# Create solver with double precision
double_prec_solver = create_solver(
    algorithm=Algorithm.RCWA,
    precision=Precision.DOUBLE
)

# The precision affects internal tensor dtype
import torch
print(f"Single precision uses dtype: {torch.float32}")
# Single precision uses dtype: torch.float32
print(f"Double precision uses dtype: {torch.float64}")
# Double precision uses dtype: torch.float64
```
  

**Notes**:

  For most simulations, SINGLE precision provides sufficient accuracy
  while offering better performance. DOUBLE precision should be used
  when higher numerical accuracy is critical or when dealing with
  numerically sensitive calculations.
  
  Keywords:
  numerical precision, floating point, computation accuracy,
  single precision, double precision, float32, float64,
  numerical stability, computational performance, memory usage

<a id="torchrdit.constants.Precision.SINGLE"></a>

#### SINGLE

32-bit floating point (torch.float32)

<a id="torchrdit.constants.Precision.DOUBLE"></a>

#### DOUBLE

64-bit floating point (torch.float64)

