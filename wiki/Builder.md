# Table of Contents

* [torchrdit.builder](#torchrdit.builder)
  * [flip\_config](#torchrdit.builder.flip_config)
  * [SolverBuilder](#torchrdit.builder.SolverBuilder)
    * [\_\_init\_\_](#torchrdit.builder.SolverBuilder.__init__)
    * [with\_algorithm](#torchrdit.builder.SolverBuilder.with_algorithm)
    * [with\_algorithm\_instance](#torchrdit.builder.SolverBuilder.with_algorithm_instance)
    * [with\_precision](#torchrdit.builder.SolverBuilder.with_precision)
    * [with\_wavelengths](#torchrdit.builder.SolverBuilder.with_wavelengths)
    * [with\_length\_unit](#torchrdit.builder.SolverBuilder.with_length_unit)
    * [with\_real\_dimensions](#torchrdit.builder.SolverBuilder.with_real_dimensions)
    * [with\_k\_dimensions](#torchrdit.builder.SolverBuilder.with_k_dimensions)
    * [with\_materials](#torchrdit.builder.SolverBuilder.with_materials)
    * [add\_material](#torchrdit.builder.SolverBuilder.add_material)
    * [with\_trn\_material](#torchrdit.builder.SolverBuilder.with_trn_material)
    * [with\_ref\_material](#torchrdit.builder.SolverBuilder.with_ref_material)
    * [with\_lattice\_vectors](#torchrdit.builder.SolverBuilder.with_lattice_vectors)
    * [with\_fff](#torchrdit.builder.SolverBuilder.with_fff)
    * [with\_device](#torchrdit.builder.SolverBuilder.with_device)
    * [with\_rdit\_order](#torchrdit.builder.SolverBuilder.with_rdit_order)
    * [add\_layer](#torchrdit.builder.SolverBuilder.add_layer)
    * [from\_config](#torchrdit.builder.SolverBuilder.from_config)
    * [build](#torchrdit.builder.SolverBuilder.build)

<a id="torchrdit.builder"></a>

# torchrdit.builder

<a id="torchrdit.builder.flip_config"></a>

#### flip\_config

```python
def flip_config(config: Dict[str, Any]) -> Dict[str, Any]
```

Create a flipped version of the configuration with reversed layer stack.

This function creates a new configuration with the layer stack reversed and
the transmission and reflection materials swapped. This is useful for simulating
the same structure from the opposite direction.

**Arguments**:

- `config` _Dict[str, Any]_ - Original configuration dictionary.
  

**Returns**:

  Dict[str, Any]: Flipped configuration with reversed layer stack and
  swapped transmission/reflection materials.
  

**Examples**:

```python
original_config = {
    "layers": [{"material": "Air"}, {"material": "Si"}, {"material": "SiO2"}],
    "trn_material": "Air",
    "ref_material": "SiO2"
}
flipped = flip_config(original_config)
flipped["layers"]
# [{"material": "SiO2"}, {"material": "Si"}, {"material": "Air"}]
flipped["trn_material"]
# "SiO2"
flipped["ref_material"]
# "Air"
```
  
  Keywords:
  configuration, flip, reverse, layer stack, transmission, reflection

<a id="torchrdit.builder.SolverBuilder"></a>

## SolverBuilder Objects

```python
class SolverBuilder()
```

Builder for creating and configuring electromagnetic solver instances.

This class implements the Builder pattern for constructing and configuring
RCWA and R-DIT solver instances. It provides a fluent interface with method
chaining to set various solver parameters.

The SolverBuilder is the recommended way to create and configure solvers
in the TorchRDIT package, allowing for a clean and expressive API.

**Attributes**:

  Various internal attributes storing the configuration state.
  

**Examples**:

```python
# Create a basic RCWA solver with default settings
from torchrdit.builder import SolverBuilder
from torchrdit.constants import Algorithm
solver = SolverBuilder().build()

# Create a more customized RCWA solver
solver = SolverBuilder() \
    .with_algorithm(Algorithm.RCWA) \
    .with_wavelengths(1.55) \
    .with_k_dimensions([5, 5]) \
    .with_device("cuda") \
    .build()

# Create an R-DIT solver with custom order
solver = SolverBuilder() \
    .with_algorithm(Algorithm.RDIT) \
    .with_rdit_order(8) \
    .build()

# Load configuration from a JSON file
solver = SolverBuilder().from_config("config.json").build()
```
  
  Keywords:
  builder pattern, solver configuration, RCWA, R-DIT, electromagnetic solver

<a id="torchrdit.builder.SolverBuilder.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

Initialize the builder with default values for all solver parameters.

<a id="torchrdit.builder.SolverBuilder.with_algorithm"></a>

#### with\_algorithm

```python
def with_algorithm(algorithm_type: Algorithm) -> "SolverBuilder"
```

Set the algorithm type for the solver.

**Arguments**:

- `algorithm_type` _Algorithm_ - The algorithm to use, either Algorithm.RCWA
  or Algorithm.RDIT.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.constants import Algorithm
builder = SolverBuilder().with_algorithm(Algorithm.RCWA)
builder = SolverBuilder().with_algorithm(Algorithm.RDIT)
```
  
  Keywords:
  algorithm, RCWA, R-DIT, solver configuration

<a id="torchrdit.builder.SolverBuilder.with_algorithm_instance"></a>

#### with\_algorithm\_instance

```python
def with_algorithm_instance(algorithm: SolverAlgorithm) -> "SolverBuilder"
```

Set a specific algorithm instance directly.

This is an advanced method that allows directly setting a custom algorithm
instance rather than using the built-in implementations.

**Arguments**:

- `algorithm` _SolverAlgorithm_ - A custom algorithm instance that implements
  the SolverAlgorithm interface.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
# Advanced usage with custom algorithm
from torchrdit.constants import Algorithm
from torchrdit.builder import SolverBuilder
custom_algorithm = RCWAAlgorithm(None)  # Configure as needed
builder = SolverBuilder().with_algorithm_instance(custom_algorithm)
```
  
  Keywords:
  custom algorithm, algorithm instance, advanced configuration

<a id="torchrdit.builder.SolverBuilder.with_precision"></a>

#### with\_precision

```python
def with_precision(precision: Precision) -> "SolverBuilder"
```

Set the numerical precision for computations.

**Arguments**:

- `precision` _Precision_ - The precision to use, either Precision.SINGLE
  for float32 or Precision.DOUBLE for float64.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
from torchrdit.constants import Precision
# Use single precision (float32)
builder = SolverBuilder().with_precision(Precision.SINGLE)
# Use double precision (float64) for higher accuracy
builder = SolverBuilder().with_precision(Precision.DOUBLE)
```
  
  Keywords:
  precision, float32, float64, numerical accuracy

<a id="torchrdit.builder.SolverBuilder.with_wavelengths"></a>

#### with\_wavelengths

```python
def with_wavelengths(lam0: Union[float, np.ndarray]) -> "SolverBuilder"
```

Set the wavelengths for the solver to compute solutions at.

**Arguments**:

- `lam0` _Union[float, np.ndarray]_ - The wavelength(s) to solve for. Can be
  a single value or an array of values for spectral calculations.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
import numpy as np
# Set a single wavelength (1.55 μm)
builder = SolverBuilder().with_wavelengths(1.55)
# Set multiple wavelengths for spectral calculations
wavelengths = np.linspace(1.2, 1.8, 100)  # 100 points from 1.2 to 1.8 μm
builder = SolverBuilder().with_wavelengths(wavelengths)
```
  
  Keywords:
  wavelength, spectral, optical frequency, electromagnetic spectrum

<a id="torchrdit.builder.SolverBuilder.with_length_unit"></a>

#### with\_length\_unit

```python
def with_length_unit(unit: str) -> "SolverBuilder"
```

Set the length unit for all dimensional parameters.

**Arguments**:

- `unit` _str_ - The length unit to use (e.g., 'um', 'nm', 'mm').
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
# Set length unit to micrometers
builder = SolverBuilder().with_length_unit('um')
# Set length unit to nanometers
builder = SolverBuilder().with_length_unit('nm')
```
  
  Keywords:
  length unit, dimensions, scale, micrometers, nanometers

<a id="torchrdit.builder.SolverBuilder.with_real_dimensions"></a>

#### with\_real\_dimensions

```python
def with_real_dimensions(rdim: List[int]) -> "SolverBuilder"
```

Set the dimensions in real space for discretization.

**Arguments**:

- `rdim` _List[int]_ - The dimensions in real space as [height, width].
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
# Set real space dimensions to 512x512
builder = SolverBuilder().with_real_dimensions([512, 512])
# Set real space dimensions to 1024x1024 for higher resolution
builder = SolverBuilder().with_real_dimensions([1024, 1024])
```
  
  Keywords:
  real dimensions, discretization, spatial resolution, grid size

<a id="torchrdit.builder.SolverBuilder.with_k_dimensions"></a>

#### with\_k\_dimensions

```python
def with_k_dimensions(kdim: List[int]) -> "SolverBuilder"
```

Set the dimensions in k-space (Fourier space) for harmonics.

**Arguments**:

- `kdim` _List[int]_ - The dimensions in k-space as [height, width].
  Higher values include more harmonics for better accuracy at
  the cost of computational complexity.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
# Set k-space dimensions to 3x3 (9 harmonics)
builder = SolverBuilder().with_k_dimensions([3, 3])
# Set k-space dimensions to 5x5 (25 harmonics) for higher accuracy
builder = SolverBuilder().with_k_dimensions([5, 5])
```
  
  Keywords:
  k-space, Fourier harmonics, diffraction orders, computational accuracy

<a id="torchrdit.builder.SolverBuilder.with_materials"></a>

#### with\_materials

```python
def with_materials(materials: List[Any]) -> "SolverBuilder"
```

Set the list of materials for the simulation.

**Arguments**:

- `materials` _List[Any]_ - List of material objects to use in the simulation.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
# Set materials for the simulation
from torchrdit.utils import create_material
from torchrdit.builder import SolverBuilder
air = create_material(name="Air", permittivity=1.0)
silicon = create_material(name="Si", permittivity=12.0)
builder = SolverBuilder().with_materials([air, silicon])
```
  
  Keywords:
  materials, permittivity, dielectric properties, optical materials

<a id="torchrdit.builder.SolverBuilder.add_material"></a>

#### add\_material

```python
def add_material(material: Any) -> "SolverBuilder"
```

Add a single material to the materials list.

**Arguments**:

- `material` _Any_ - The material object to add to the simulation.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
# Add materials one by one
from torchrdit.utils import create_material
from torchrdit.builder import SolverBuilder
builder = SolverBuilder()
builder.add_material(create_material(name="Air", permittivity=1.0))
builder.add_material(create_material(name="Si", permittivity=12.0))
```
  
  Keywords:
  material, add material, permittivity, optical properties

<a id="torchrdit.builder.SolverBuilder.with_trn_material"></a>

#### with\_trn\_material

```python
def with_trn_material(material: Union[str, Any]) -> "SolverBuilder"
```

Set the transmission material for the simulation.

The transmission material defines the material properties of the medium
on the transmission side of the structure.

**Arguments**:

- `material` _Union[str, Any]_ - Either a material name (string) that exists
  in the material list, or a material object that will be added to
  the material list.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
# Set transmission material by name (must already be in materials list)
builder = SolverBuilder().with_trn_material("Air")
# Set transmission material by object
from torchrdit.utils import create_material
air = create_material(name="Air", permittivity=1.0)
builder = SolverBuilder().with_trn_material(air)
```
  
  Keywords:
  transmission material, output medium, optical interface

<a id="torchrdit.builder.SolverBuilder.with_ref_material"></a>

#### with\_ref\_material

```python
def with_ref_material(material: Union[str, Any]) -> "SolverBuilder"
```

Set the reflection material (or incident material) for the simulation.

The reflection material (or incident material) defines the material properties of the medium
on the reflection side (or incident side) of the structure.

**Arguments**:

- `material` _Union[str, Any]_ - Either a material name (string) that exists
  in the material list, or a material object that will be added to
  the material list.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
# Set reflection material by name (must already be in materials list)
builder = SolverBuilder().with_ref_material("Si")
# Set reflection material by object
from torchrdit.utils import create_material
silicon = create_material(name="Si", permittivity=12.0)
builder = SolverBuilder().with_ref_material(silicon)
```
  
  Keywords:
  reflection material, incident material, optical interface

<a id="torchrdit.builder.SolverBuilder.with_lattice_vectors"></a>

#### with\_lattice\_vectors

```python
def with_lattice_vectors(t1: torch.Tensor,
                         t2: torch.Tensor) -> "SolverBuilder"
```

Set the lattice vectors defining the unit cell geometry.

**Arguments**:

- `t1` _torch.Tensor_ - First lattice vector.
- `t2` _torch.Tensor_ - Second lattice vector.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
# Set square lattice with 1 μm period
import torch
from torchrdit.builder import SolverBuilder
t1 = torch.tensor([[1.0, 0.0]])
t2 = torch.tensor([[0.0, 1.0]])
builder = SolverBuilder().with_lattice_vectors(t1, t2)
# Set rectangular lattice
t1 = torch.tensor([[1.5, 0.0]])
t2 = torch.tensor([[0.0, 1.0]])
builder = SolverBuilder().with_lattice_vectors(t1, t2)
```
  
  Keywords:
  lattice vectors, unit cell, periodicity, photonic crystal

<a id="torchrdit.builder.SolverBuilder.with_fff"></a>

#### with\_fff

```python
def with_fff(use_fff: bool) -> "SolverBuilder"
```

Set whether to use Fast Fourier Factorization for improved convergence.

Fast Fourier Factorization (FFF) improves the convergence of the RCWA
algorithm, especially for metallic structures and TM polarization.

**Arguments**:

- `use_fff` _bool_ - Whether to use Fast Fourier Factorization.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
# Enable Fast Fourier Factorization (default)
builder = SolverBuilder().with_fff(True)
# Disable Fast Fourier Factorization
builder = SolverBuilder().with_fff(False)
```
  
  Keywords:
  Fast Fourier Factorization, FFF, convergence, numerical stability

<a id="torchrdit.builder.SolverBuilder.with_device"></a>

#### with\_device

```python
def with_device(device: Union[str, torch.device]) -> "SolverBuilder"
```

Set the computation device (CPU or GPU).

**Arguments**:

- `device` _Union[str, torch.device]_ - The device to use for computations,
  either as a string ('cpu', 'cuda', 'cuda:0', etc.) or as a torch.device.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
# Use CPU for computations
builder = SolverBuilder().with_device('cpu')
# Use GPU for computations
builder = SolverBuilder().with_device('cuda')
# Use specific GPU
builder = SolverBuilder().with_device('cuda:1')
```
  
  Keywords:
  device, CPU, GPU, CUDA, computation acceleration

<a id="torchrdit.builder.SolverBuilder.with_rdit_order"></a>

#### with\_rdit\_order

```python
def with_rdit_order(order: int) -> "SolverBuilder"
```

Set the R-DIT order for the R-DIT algorithm.

The R-DIT order determines the approximation used in the diffraction
interface theory. Higher orders provide better accuracy at the cost of
computational efficiency.

**Notes**:

  This parameter is only used when the R-DIT algorithm is selected.
  

**Arguments**:

- `order` _int_ - The order of the R-DIT algorithm (typically 1-10).
  Higher values provide more accurate results.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
# Use R-DIT algorithm with order 8
from torchrdit.constants import Algorithm
from torchrdit.builder import SolverBuilder
builder = SolverBuilder() \
    .with_algorithm(Algorithm.RDIT) \
    .with_rdit_order(8)
```
  
  Keywords:
  R-DIT, order, approximation, accuracy, performance

<a id="torchrdit.builder.SolverBuilder.add_layer"></a>

#### add\_layer

```python
def add_layer(layer_config: Dict[str, Any]) -> "SolverBuilder"
```

Add a layer configuration to be added after solver creation.

**Arguments**:

- `layer_config` _Dict[str, Any]_ - Configuration dictionary for a layer,
  containing at minimum "material" and "thickness" keys.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
# Add a homogeneous silicon layer
builder = SolverBuilder()
builder.add_layer({
    "material": "Si",
    "thickness": 0.5,
    "is_homogeneous": True
})
# Add a non-homogeneous (patterned) layer
builder.add_layer({
    "material": "SiO2",
    "thickness": 0.2,
    "is_homogeneous": False
})
```
  
  Keywords:
  layer, material, thickness, homogeneous, patterned

<a id="torchrdit.builder.SolverBuilder.from_config"></a>

#### from\_config

```python
def from_config(config: Union[str, Dict[str, Any]],
                flip: bool = False) -> "SolverBuilder"
```

Configure the builder from a JSON configuration file or dictionary.

This method allows loading all solver configuration from a JSON file or
dictionary, which is useful for storing and sharing complex configurations.

**Arguments**:

- `config` _Union[str, Dict[str, Any]]_ - Either a path to a JSON configuration
  file or a configuration dictionary.
- `flip` _bool, optional_ - Whether to flip the configuration (reverse layer
  stack and swap transmission/reflection materials). Defaults to False.
  

**Returns**:

- `SolverBuilder` - The builder instance for method chaining.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
# Load configuration from a JSON file
builder = SolverBuilder().from_config("config.json")
# Load and flip configuration (reverse layer stack)
builder = SolverBuilder().from_config("config.json", flip=True)
# Load from a dictionary
config_dict = {
    "algorithm": "RDIT",
    "wavelengths": [1.55],
    "kdim": [5, 5],
    "materials": {...},
    "layers": [...]
}
builder = SolverBuilder().from_config(config_dict)
```
  

**Raises**:

- `ValueError` - If unknown configuration keys are detected.
  
  Keywords:
  configuration, JSON, file loading, serialization, flip

<a id="torchrdit.builder.SolverBuilder.build"></a>

#### build

```python
def build()
```

Build and return the configured solver instance.

This method creates a solver instance based on all the configuration
parameters that have been set on the builder.

**Returns**:

  Union[RCWASolver, RDITSolver]: The configured solver instance.
  

**Examples**:

```python
from torchrdit.builder import SolverBuilder
from torchrdit.constants import Algorithm
# Build a basic RCWA solver
solver = SolverBuilder().with_algorithm(Algorithm.RCWA).build()
# Build a fully configured solver
solver = SolverBuilder() \
    .with_algorithm(Algorithm.RDIT) \
    .with_rdit_order(8) \
    .with_wavelengths([1.3, 1.4, 1.5, 1.6]) \
    .with_k_dimensions([5, 5]) \
    .with_device("cuda") \
    .build()
```
  
  Keywords:
  build, create solver, instantiate, solver instance

