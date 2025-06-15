# Table of Contents

* [torchrdit.materials](#torchrdit.materials)
  * [MaterialClass](#torchrdit.materials.MaterialClass)
    * [\_\_init\_\_](#torchrdit.materials.MaterialClass.__init__)
    * [isdispersive\_er](#torchrdit.materials.MaterialClass.isdispersive_er)
    * [er](#torchrdit.materials.MaterialClass.er)
    * [ur](#torchrdit.materials.MaterialClass.ur)
    * [name](#torchrdit.materials.MaterialClass.name)
    * [fitted\_data](#torchrdit.materials.MaterialClass.fitted_data)
    * [data\_format](#torchrdit.materials.MaterialClass.data_format)
    * [data\_unit](#torchrdit.materials.MaterialClass.data_unit)
    * [supported\_formats](#torchrdit.materials.MaterialClass.supported_formats)
    * [get\_permittivity](#torchrdit.materials.MaterialClass.get_permittivity)
    * [load\_dispersive\_er](#torchrdit.materials.MaterialClass.load_dispersive_er)
    * [from\_nk\_data](#torchrdit.materials.MaterialClass.from_nk_data)
    * [from\_data\_file](#torchrdit.materials.MaterialClass.from_data_file)
    * [clear\_cache](#torchrdit.materials.MaterialClass.clear_cache)
    * [\_\_str\_\_](#torchrdit.materials.MaterialClass.__str__)
    * [\_\_repr\_\_](#torchrdit.materials.MaterialClass.__repr__)

<a id="torchrdit.materials"></a>

# torchrdit.materials

Module for defining and managing material properties in TorchRDIT electromagnetic simulations.

This module provides classes and utilities for creating, managing, and manipulating
material properties used in electromagnetic simulations with TorchRDIT. It handles
both dispersive (wavelength/frequency dependent) and non-dispersive materials with
a unified interface.

The material system is designed with a proxy pattern, separating material property
definitions from their data loading and processing. This allows for efficient handling
of different material data formats and sources, including:
- Direct permittivity/permeability specification
- Refractive index and extinction coefficient (n, k) specification
- Loading from data files with different formats (freq-eps, wl-eps, freq-nk, wl-nk)

Classes:
MaterialClass: Main class for representing materials with their electromagnetic properties.

Functions:
(No module-level functions, but MaterialClass provides factory methods)

**Examples**:

```python
from torchrdit.utils import create_material
# Create a simple material with constant permittivity
air = create_material(name="air", permittivity=1.0)
silicon = create_material(name="silicon", permittivity=11.7)
# Create a material from refractive index
glass = create_material(name="glass", permittivity=2.25)  # n=1.5
# Create a material with complex permittivity (lossy)
gold = create_material(name="gold", permittivity=complex(-10.0, 1.5))

# Create dispersive material from data file with wavelength-permittivity data
silica = create_material(
    name="silica",
    dielectric_dispersion=True,
    user_dielectric_file="materials/SiO2.txt",
    data_format="wl-eps",
    data_unit="um"
)

# Create dispersive material from data file with frequency-nk data
silicon_dispersive = create_material(
    name="silicon_disp",
    dielectric_dispersion=True,
    user_dielectric_file="materials/Si.txt",
    data_format="freq-nk",
    data_unit="thz"
)

# Using materials in solvers
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
import torch
# Create a solver and add materials
solver = create_solver(algorithm=Algorithm.RCWA)
solver.add_materials([air, silicon, glass])
# Add layers using these materials
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
solver.add_layer(material_name="glass", thickness=torch.tensor(0.1))
# Set input/output materials
solver.update_ref_material("air")
solver.update_trn_material("air")
```
  
  Keywords:
  materials, permittivity, permeability, optical properties, dispersive materials,
  refractive index, extinction coefficient, material data, complex permittivity,
  electromagnetic properties, dielectric function, optical constants, wavelength-dependent

<a id="torchrdit.materials.MaterialClass"></a>

## MaterialClass Objects

```python
class MaterialClass()
```

Class for representing materials and their electromagnetic properties in TorchRDIT.

This class implements a comprehensive representation of materials used in electromagnetic
simulations, supporting both dispersive (wavelength/frequency dependent) and
non-dispersive (constant) material properties. It provides a unified interface
for managing permittivity and permeability data regardless of the source format.

MaterialClass uses a proxy pattern for handling material data loading and processing,
allowing it to support multiple data formats and unit systems. For dispersive
materials, it can load data from files and perform polynomial fitting to interpolate
property values at specific wavelengths needed for simulations.

**Attributes**:

- `name` _str_ - Name identifier for the material.
- `er` _torch.Tensor_ - Complex permittivity tensor.
- `ur` _torch.Tensor_ - Permeability tensor.
- `isdispersive_er` _bool_ - Whether the material has wavelength/frequency-dependent permittivity.
- `data_format` _str_ - Format of the loaded data file for dispersive materials.
- `data_unit` _str_ - Unit used in the data file for dispersive materials.
- `fitted_data` _Dict[str, Any]_ - Fitted profile data for dispersive materials.
  

**Notes**:

  Users typically create MaterialClass instances through the `create_material`
  function rather than directly instantiating this class.
  

**Examples**:

```python
from torchrdit.utils import create_material
# Material with constant permittivity
silicon = create_material(name="silicon", permittivity=11.7)
print(f"Material: {silicon.name}, ε = {silicon.er.real:.1f}")
# Material: silicon, ε = 11.7
# Material with complex permittivity (lossy)
gold = create_material(name="gold", permittivity=complex(-10.0, 1.5))
print(f"Material: {gold.name}, ε = {gold.er.real:.1f}{gold.er.imag:+.1f}j")
# Material: gold, ε = -10.0+1.5j

# Load permittivity data from wavelength-permittivity data file
silica = create_material(
    name="silica",
    dielectric_dispersion=True,
    user_dielectric_file="materials/SiO2.txt",
    data_format="wl-eps",
    data_unit="um"
)
# Use in a simulation with specific wavelengths
import numpy as np
wavelengths = np.array([1.31, 1.55])
permittivity = silica.get_permittivity(wavelengths, 'um')
print(f"Silica permittivity at λ=1.55 μm: {permittivity[1].real:.4f}")
# Silica permittivity at λ=1.55 μm: 2.1521
```
  
  Keywords:
  material properties, permittivity, permeability, optical constants,
  dispersive materials, refractive index, wavelength-dependent properties,
  material data, complex permittivity, dielectric function

<a id="torchrdit.materials.MaterialClass.__init__"></a>

#### \_\_init\_\_

```python
def __init__(name: str = "material1",
             permittivity: float = 1.0,
             permeability: float = 1.0,
             dielectric_dispersion: bool = False,
             user_dielectric_file: str = "",
             data_format: str = "freq-eps",
             data_unit: str = "thz",
             max_poly_fit_order: int = 10,
             data_proxy: Optional[MaterialDataProxy] = None) -> None
```

Initialize a MaterialClass instance with electromagnetic properties.

Creates a new material with specified properties. For non-dispersive materials,
only name, permittivity, and permeability need to be specified. For dispersive
materials (with wavelength/frequency-dependent properties), additional parameters
are required to specify the data source and format.

**Arguments**:

- `name` - Unique identifier for the material. Used when referencing the material
  in solvers and layer definitions.
- `permittivity` - Relative permittivity (εr) for non-dispersive materials.
  Can be a real or complex value to represent lossless or lossy materials.
  This value is ignored for dispersive materials.
- `permeability` - Relative permeability (μr) of the material. Default is 1.0
  (non-magnetic material).
- `dielectric_dispersion` - Whether the material has wavelength/frequency-dependent
  permittivity. If True, a data file must be provided.
- `user_dielectric_file` - Path to the data file containing the dispersive properties.
  Required if dielectric_dispersion is True.
- `data_format` - Format of the data in the file. Must be one of:
- `'freq-eps'` - Frequency and permittivity (real, imaginary)
- `'wl-eps'` - Wavelength and permittivity (real, imaginary)
- `'freq-nk'` - Frequency and refractive index (n, k)
- `'wl-nk'` - Wavelength and refractive index (n, k)
- `data_unit` - Unit of the frequency or wavelength in the data file (e.g., 'thz', 'um').
- `max_poly_fit_order` - Maximum polynomial order for fitting dispersive data.
  Higher values provide more accurate fits for complex
  dispersion curves but may lead to overfitting.
- `data_proxy` - Custom data proxy instance for handling material data loading.
  Uses the shared class-level proxy if None.
  

**Raises**:

- `ValueError` - If dispersive material is missing a data file,
  if the data file doesn't exist, or if the data format is invalid.
  

**Examples**:

```python
from torchrdit.materials import MaterialClass
# Create a simple air material
air = MaterialClass(name="air", permittivity=1.0)

# Create a lossy metal with complex permittivity
gold = MaterialClass(
    name="gold",
    permittivity=complex(-10.0, 1.5)
)

# Create a dispersive material from data file
silica = MaterialClass(
    name="silica",
    dielectric_dispersion=True,
    user_dielectric_file="materials/SiO2.txt",
    data_format="wl-eps",
    data_unit="um"
)
```
  
  Keywords:
  material creation, permittivity, permeability, dispersive material,
  optical constants, electromagnetic properties, material initialization

<a id="torchrdit.materials.MaterialClass.isdispersive_er"></a>

#### isdispersive\_er

```python
@property
def isdispersive_er() -> bool
```

Whether the material has dispersive permittivity.

<a id="torchrdit.materials.MaterialClass.er"></a>

#### er

```python
@property
def er() -> torch.Tensor
```

Permittivity tensor of the material.

<a id="torchrdit.materials.MaterialClass.ur"></a>

#### ur

```python
@property
def ur() -> torch.Tensor
```

Permeability tensor of the material.

<a id="torchrdit.materials.MaterialClass.name"></a>

#### name

```python
@property
def name() -> str
```

Name of the material.

<a id="torchrdit.materials.MaterialClass.fitted_data"></a>

#### fitted\_data

```python
@property
def fitted_data() -> Dict[str, Any]
```

Fitted profile data for dispersive materials.

<a id="torchrdit.materials.MaterialClass.data_format"></a>

#### data\_format

```python
@property
def data_format() -> str
```

Format of the material data.

<a id="torchrdit.materials.MaterialClass.data_unit"></a>

#### data\_unit

```python
@property
def data_unit() -> str
```

Unit of the material data.

<a id="torchrdit.materials.MaterialClass.supported_formats"></a>

#### supported\_formats

```python
@property
def supported_formats() -> List[str]
```

List of supported data formats.

<a id="torchrdit.materials.MaterialClass.get_permittivity"></a>

#### get\_permittivity

```python
def get_permittivity(wavelengths: np.ndarray,
                     wavelength_unit: str = "um") -> torch.Tensor
```

Get the material's permittivity at specified wavelengths.

This is a standardized interface that handles both dispersive and
non-dispersive materials.

**Arguments**:

- `wavelengths` - Array of wavelengths to calculate permittivity at
- `wavelength_unit` - Unit of the provided wavelengths
  

**Returns**:

  Permittivity tensor at the specified wavelengths
  

**Raises**:

- `ValueError` - If the wavelengths are out of the available data range

<a id="torchrdit.materials.MaterialClass.load_dispersive_er"></a>

#### load\_dispersive\_er

```python
def load_dispersive_er(lam0: np.ndarray, lengthunit: str = "um") -> None
```

Load the dispersive profile and fit with the wavelengths to be simulated.

**Arguments**:

- `lam0` - Wavelengths to simulate
- `lengthunit` - Length unit used in the solver
  

**Raises**:

- `ValueError` - If wavelengths are out of the available data range

<a id="torchrdit.materials.MaterialClass.from_nk_data"></a>

#### from\_nk\_data

```python
@classmethod
def from_nk_data(cls,
                 name: str,
                 n: float,
                 k: float = 0.0,
                 permeability: float = 1.0) -> "MaterialClass"
```

Create a material from refractive index and extinction coefficient.

**Arguments**:

- `name` - Name of the material
- `n` - Refractive index
- `k` - Extinction coefficient
- `permeability` - Relative permeability
  

**Returns**:

  Instantiated MaterialClass with permittivity derived from n and k

<a id="torchrdit.materials.MaterialClass.from_data_file"></a>

#### from\_data\_file

```python
@classmethod
def from_data_file(cls,
                   name: str,
                   file_path: str,
                   data_format: str = "wl-eps",
                   data_unit: str = "um",
                   permeability: float = 1.0,
                   max_poly_fit_order: int = 10) -> "MaterialClass"
```

Create a dispersive material from a data file.

**Arguments**:

- `name` - Name of the material
- `file_path` - Path to the data file
- `data_format` - Format of the data file ('freq-eps', 'wl-eps', etc.)
- `data_unit` - Unit of the data in the file ('thz', 'um', etc.)
- `permeability` - Relative permeability
- `max_poly_fit_order` - Maximum polynomial order for fitting
  

**Returns**:

  Instantiated dispersive MaterialClass
  

**Raises**:

- `ValueError` - If the file doesn't exist or has an invalid format

<a id="torchrdit.materials.MaterialClass.clear_cache"></a>

#### clear\_cache

```python
def clear_cache() -> None
```

Clear the permittivity cache to free memory.

<a id="torchrdit.materials.MaterialClass.__str__"></a>

#### \_\_str\_\_

```python
def __str__() -> str
```

String representation of the material.

<a id="torchrdit.materials.MaterialClass.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__() -> str
```

Detailed representation of the material.

