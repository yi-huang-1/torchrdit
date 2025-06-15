# Table of Contents

* [torchrdit.cell](#torchrdit.cell)
  * [CellType](#torchrdit.cell.CellType)
  * [Cell3D](#torchrdit.cell.Cell3D)
    * [\_\_init\_\_](#torchrdit.cell.Cell3D.__init__)
    * [get\_shape\_generator\_params](#torchrdit.cell.Cell3D.get_shape_generator_params)
    * [add\_materials](#torchrdit.cell.Cell3D.add_materials)
    * [add\_layer](#torchrdit.cell.Cell3D.add_layer)
    * [layers](#torchrdit.cell.Cell3D.layers)
    * [update\_trn\_material](#torchrdit.cell.Cell3D.update_trn_material)
    * [update\_ref\_material](#torchrdit.cell.Cell3D.update_ref_material)
    * [get\_layer\_structure](#torchrdit.cell.Cell3D.get_layer_structure)
    * [get\_layout](#torchrdit.cell.Cell3D.get_layout)
    * [er1](#torchrdit.cell.Cell3D.er1)
    * [er2](#torchrdit.cell.Cell3D.er2)
    * [ur1](#torchrdit.cell.Cell3D.ur1)
    * [ur2](#torchrdit.cell.Cell3D.ur2)
    * [lengthunit](#torchrdit.cell.Cell3D.lengthunit)
    * [get\_cell\_type](#torchrdit.cell.Cell3D.get_cell_type)

<a id="torchrdit.cell"></a>

# torchrdit.cell

Module for defining base class (Cell3D) of solver classes in electromagnetic simulations.

This module provides classes for creating and manipulating 3D unit cells for use in electromagnetic simulations with TorchRDIT. The key
components include:

- CellType: Enumeration of supported cell geometry types
- Cell3D: Base class for defining 3D unit cells with material properties

These classes form the foundation for defining the geometric and material properties
of structures to be simulated with RCWA or R-DIT algorithms.

**Examples**:

```python
# Basic usage: The Cell3D class is not intended to be used directly. Instead, it serves as a
# parent class for solver implementations in solver.py (RCWASolver and RDITSolver).
```
  
  Keywords:
  unit cell, geometry, shape generation, mask, material, layer, photonics

<a id="torchrdit.cell.CellType"></a>

## CellType Objects

```python
class CellType()
```

Enumeration of supported cell geometry types in TorchRDIT.

This class defines the types of coordinate systems that can be used for
the unit cell in electromagnetic simulations. The cell type affects how
coordinates are interpreted and how Fourier transforms are computed.

**Attributes**:

- `Cartesian` - Standard Cartesian coordinate system with rectangular grid.
  Used for rectangular lattices aligned with coordinate axes.
- `Other` - Alternative coordinate systems used for non-rectangular lattices
  or rotated structures.
  

**Examples**:

```python
# Check the cell type
import torch
from torchrdit.cell import Cell3D
cell = Cell3D()
print(cell.cell_type)
# Cartesian
# Cell type is automatically determined from lattice vectors
cell = Cell3D(t1=torch.tensor([[1.0, 0.5]]),
              t2=torch.tensor([[0.0, 1.0]]))
print(cell.cell_type)
# Other
```
  
  Keywords:
  coordinate system, Cartesian, unit cell, lattice type, grid

<a id="torchrdit.cell.Cell3D"></a>

## Cell3D Objects

```python
class Cell3D()
```

Base parent class for defining and structuring cells in electromagnetic solvers.

The Cell3D class provides the foundational structure for defining and managing
the geometric and material properties of a unit cell in TorchRDIT. It serves
as the core building block for creating photonic structures to be simulated
with electromagnetic solvers.

**Notes**:

  Cell3D is not intended to be used directly. Instead, it serves as a
  parent class for solver implementations in solver.py (RCWASolver and RDITSolver).
  Most users should create solver instances using the create_solver() function
  rather than instantiating Cell3D directly.
  
  Key capabilities:
  - Setting up computational grids in real and Fourier space
  - Managing material properties and multilayer stacks
  - Creating and manipulating binary shape masks for complex geometries
  - Controlling reference and transmission materials
  - Providing coordinate transformation utilities
  
  This class serves as a base for solver classes (FourierBaseSolver, RCWASolver, RDITSolver)
  and provides all the necessary functionality to define the structure being simulated,
  including material layers, shape patterns, and boundary conditions.
  

**Attributes**:

- `cell_type` _CellType_ - Type of coordinate system (Cartesian or Other)
- `tcomplex` _torch.dtype_ - Complex data type used for calculations
- `tfloat` _torch.dtype_ - Float data type used for calculations
- `tint` _torch.dtype_ - Integer data type used for calculations
- `nfloat` _np.dtype_ - NumPy float data type for compatibility
- `rdim` _List[int]_ - Dimensions in real space [height, width]
- `kdim` _List[int]_ - Dimensions in k-space [kheight, kwidth]
- `layer_manager` _LayerManager_ - Manager for handling material layers
  

**Examples**:

```python
# Create a solver (recommended way) instead of using Cell3D directly
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm

# Create an RCWA solver
solver = create_solver(
    algorithm=Algorithm.RCWA,
    lam0=np.array([1.55]),
    rdim=[256, 256],
    kdim=[5, 5]
)

# Add materials
silicon = create_material(name="silicon", permittivity=11.7)
sio2 = create_material(name="sio2", permittivity=2.25)
solver.add_materials([silicon, sio2])

# Add layers
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
solver.add_layer(material_name="sio2", thickness=torch.tensor(0.1))

# Create a patterned layer
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.3),
               is_homogeneous=False)
circle_mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
```
  
  Keywords:
  base class, parent class, solver foundation, electromagnetic simulation,
  photonics, layers, materials, computational grid, Fourier transform,
  shape generation, inheritance

<a id="torchrdit.cell.Cell3D.__init__"></a>

#### \_\_init\_\_

```python
def __init__(lengthunit: str = "um",
             rdim: List[int] = [512, 512],
             kdim: List[int] = [3, 3],
             materiallist: List[MaterialClass] = [],
             t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),
             t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),
             device: Union[str, torch.device] = "cpu") -> None
```

Initialize a Cell3D object with geometric and material properties.

This constructor sets up the computational grid in both real and Fourier space,
initializes materials, and creates a coordinate system based on the
provided lattice vectors. It also initializes the layer manager for handling
material layers in the structure.

**Notes**:

  Users should generally not instantiate Cell3D directly, but rather
  create a solver instance using the create_solver() function.
  

**Arguments**:

- `lengthunit` _str_ - Unit of length for all dimensions in the simulation.
  Common values: 'um' (micrometers), 'nm' (nanometers).
  Default is 'um'.
- `rdim` _List[int]_ - Dimensions of the real-space grid as [height, width].
  This determines the spatial resolution of the simulation.
  Default is [512, 512].
- `kdim` _List[int]_ - Dimensions in Fourier space as [kheight, kwidth].
  This determines the number of Fourier harmonics used in the simulation.
  Default is [3, 3].
- `materiallist` _List[MaterialClass]_ - List of material objects to be used in the simulation.
  Each material should be an instance of MaterialClass.
  Default is an empty list.
- `t1` _torch.Tensor_ - First lattice vector defining the unit cell.
  Default is [[1.0, 0.0]] (unit vector in x-direction).
- `t2` _torch.Tensor_ - Second lattice vector defining the unit cell.
  Default is [[0.0, 1.0]] (unit vector in y-direction).
- `device` _Union[str, torch.device]_ - The device to run computations on ('cpu' or 'cuda').
  Default is 'cpu'.
  

**Raises**:

- `ValueError` - If any of the input parameters have invalid values or formats.
  

**Examples**:

```python
# Instead of creating Cell3D directly, create a solver:
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
import torch
# Create an RDIT solver
rdit_solver = create_solver(
    algorithm=Algorithm.RDIT,
    rdim=[1024, 1024],
    kdim=[7, 7]
)

# Create an RCWA solver with non-rectangular lattice
rcwa_solver = create_solver(
    algorithm=Algorithm.RCWA,
    t1=torch.tensor([[1.0, 0.0]]),
    t2=torch.tensor([[0.5, 0.866]]),  # 30-degree lattice
    rdim=[512, 512],
    kdim=[5, 5]
)

# Create a solver with GPU acceleration
gpu_solver = create_solver(
    algorithm=Algorithm.RCWA,
    device="cuda"
)
```
  
  Keywords:
  initialization, base class, parent class, computational grid, lattice vectors,
  spatial resolution, Fourier harmonics

<a id="torchrdit.cell.Cell3D.get_shape_generator_params"></a>

#### get\_shape\_generator\_params

```python
def get_shape_generator_params()
```

Get the parameters for the shape generator.

**Returns**:

- `dict` - Dictionary containing the parameters for the shape generator.
  

**Examples**:

```python
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
import torch
from torchrdit.shapes import ShapeGenerator
solver = create_solver(
    algorithm=Algorithm.RDIT,
    rdim=[1024, 1024],
    kdim=[7, 7]
)
params = solver.get_shape_generator_params()
shape_gen = ShapeGenerator(**params)
```

<a id="torchrdit.cell.Cell3D.add_materials"></a>

#### add\_materials

```python
def add_materials(material_list: list = [])
```

Add materials to the solver's material library.

This method adds one or more materials to the cell's internal material
library, making them available for use in layers. Materials can be either
non-dispersive (constant properties) or dispersive (wavelength-dependent).

**Arguments**:

- `material_list` _list_ - List of MaterialClass instances to add to the material library.
  Each material should have a unique name that will be used to
  reference it when creating layers.
  

**Raises**:

- `ValueError` - If any element in the list is not a MaterialClass instance or
  if the input is not a list.
  

**Examples**:

```python
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
solver = create_solver(
    algorithm=Algorithm.RDIT,
    rdim=[1024, 1024],
    kdim=[7, 7]
)
# Create and add materials
from torchrdit.utils import create_material
silicon = create_material(name='silicon', permittivity=11.7)
sio2 = create_material(name='sio2', permittivity=2.25)
# Add multiple materials at once
solver.add_materials([silicon, sio2])
# Add a single material
gold = create_material(name='gold', permittivity=complex(-10.0, 1.5))
solver.add_materials([gold])
```
  

**Notes**:

  The 'air' material (permittivity=1.0) is automatically added to all
  Cell3D instances by default and does not need to be explicitly added.
  
  Keywords:
  materials, permittivity, dielectric properties, material library,
  optical materials, dispersive materials

<a id="torchrdit.cell.Cell3D.add_layer"></a>

#### add\_layer

```python
def add_layer(material_name: Any,
              thickness: torch.Tensor,
              is_homogeneous: bool = True,
              is_optimize: bool = False)
```

Add a new material layer to the structure.

This method adds a layer with specified material and thickness to the
simulation structure. Layers are stacked in the order they are added,
with the first layer added being at the bottom of the stack (closest to
the reference region) and subsequent layers building upward.

**Arguments**:

- `material_name` _Union[str, MaterialClass]_ - Name of the material for this layer,
  or a MaterialClass instance. Must be a material that exists in
  the material library or a new material to be added.
- `thickness` _torch.Tensor_ - Thickness of the layer as a torch.Tensor.
  The units are determined by the lengthunit parameter of the solver.
- `is_homogeneous` _bool_ - Whether the layer has uniform material properties (True) or
  is patterned with a spatial distribution (False).
  Default is True.
- `is_optimize` _bool_ - Whether this layer's parameters (e.g., thickness) should be
  included in optimization. Set to True if you plan to optimize
  this layer's properties. Default is False.
  

**Raises**:

- `RuntimeError` - If the specified material does not exist in the material library.
  

**Examples**:

```python
# Create a cell and add materials
from torchrdit.cell import Cell3D
from torchrdit.utils import create_material
import torch
cell = Cell3D()
silicon = create_material(name='silicon', permittivity=11.7)
sio2 = create_material(name='sio2', permittivity=2.25)
cell.add_materials([silicon, sio2])

# Add a homogeneous silicon layer with thickness 0.2 μm
cell.add_layer(material_name='silicon', thickness=torch.tensor(0.2))

# Add a layer using a material object directly
air = create_material(name='air2', permittivity=1.0)
cell.add_layer(material_name=air, thickness=torch.tensor(0.1))

# Add a patterned (non-homogeneous) layer
cell.add_layer(
    material_name='silicon',
    thickness=torch.tensor(0.3),
    is_homogeneous=False
)

# Add a layer that will be optimized
cell.add_layer(
    material_name='sio2',
    thickness=torch.tensor(0.15),
    is_optimize=True
)
```
  
  Keywords:
  layer, material, thickness, homogeneous, patterned, photonic structure,
  multilayer, stack, optimization

<a id="torchrdit.cell.Cell3D.layers"></a>

#### layers

```python
@property
def layers()
```

Get all created layers in the layer structure.

**Returns**:

- `List` - All created layers in the layer_manager instance.
  

**Examples**:

```python
from torchrdit.cell import Cell3D
import torch
# Access all layers
cell = Cell3D()
cell.add_layer(material_name='air', thickness=torch.tensor(0.2))
cell.add_layer(material_name='air', thickness=torch.tensor(0.3))
print(f"Number of layers: {len(cell.layers)}")
# Access properties of specific layers
for i, layer in enumerate(cell.layers):
    print(f"Layer {i}: {layer.material_name}, thickness={layer.thickness}")
```
  
  Keywords:
  layers, layer access, layer properties, layer stack

<a id="torchrdit.cell.Cell3D.update_trn_material"></a>

#### update\_trn\_material

```python
def update_trn_material(trn_material: Any) -> None
```

Update the transmission layer material.

This method sets or changes the material used for the transmission layer,
which is the semi-infinite region above the layered structure. This material
affects the boundary conditions and the calculation of transmission coefficients.

**Arguments**:

- `trn_material` _Union[str, MaterialClass]_ - Name of the material to use for
  the transmission layer, or a MaterialClass instance.
  Must be a material that exists in the material library
  or a new material to be added.
  

**Raises**:

- `RuntimeError` - If the specified material does not exist in the material library
  and is not a MaterialClass instance.
  

**Examples**:

```python
from torchrdit.cell import Cell3D
from torchrdit.utils import create_material
# Set transmission material by name
cell = Cell3D()
silicon = create_material(name='silicon', permittivity=11.7)
cell.add_materials([silicon])
cell.update_trn_material(trn_material='silicon')

# Set transmission material by providing a material object
water = create_material(name='water', permittivity=1.77)
cell.update_trn_material(trn_material=water)
```
  
  Keywords:
  transmission layer, output medium, boundary condition, semi-infinite region

<a id="torchrdit.cell.Cell3D.update_ref_material"></a>

#### update\_ref\_material

```python
def update_ref_material(ref_material: Any) -> None
```

Update the reflection layer material.

This method sets or changes the material used for the reflection layer,
which is the semi-infinite region below the layered structure. This material
affects the boundary conditions and the calculation of reflection coefficients.

**Arguments**:

- `ref_material` _Union[str, MaterialClass]_ - Name of the material to use for
  the reflection layer, or a MaterialClass instance.
  Must be a material that exists in the material library
  or a new material to be added.
  

**Raises**:

- `RuntimeError` - If the specified material does not exist in the material library
  and is not a MaterialClass instance.
  

**Examples**:

```python
from torchrdit.cell import Cell3D
from torchrdit.utils import create_material
# Set reflection material by name
cell = Cell3D()
silicon = create_material(name='silicon', permittivity=11.7)
cell.add_materials([silicon])
cell.update_ref_material(ref_material='silicon')

# Set reflection material by providing a material object
metal = create_material(name='silver', permittivity=complex(-15.0, 1.0))
cell.update_ref_material(ref_material=metal)
```
  
  Keywords:
  reflection layer, input medium, boundary condition, semi-infinite region

<a id="torchrdit.cell.Cell3D.get_layer_structure"></a>

#### get\_layer\_structure

```python
def get_layer_structure()
```

Print information about all layers in the structure.

This method displays detailed information about the current layer structure,
including the reflection layer, all intermediate layers, and the transmission
layer. For each layer, it shows:
- Material name
- Thickness (for intermediate layers)
- Permittivity and permeability
- Whether the layer is dispersive, homogeneous, or to be optimized

**Examples**:

```python
from torchrdit.cell import Cell3D
from torchrdit.utils import create_material
import torch
# Create a cell with multiple layers and display information
cell = Cell3D()
silicon = create_material(name='silicon', permittivity=11.7)
sio2 = create_material(name='sio2', permittivity=2.25)
cell.add_materials([silicon, sio2])
cell.add_layer(material_name='silicon', thickness=torch.tensor(0.2))
cell.add_layer(material_name='sio2', thickness=torch.tensor(0.1))
cell.get_layer_structure()
```
  
  Keywords:
  layer structure, information display, debugging, layer properties

<a id="torchrdit.cell.Cell3D.get_layout"></a>

#### get\_layout

```python
def get_layout() -> Tuple[torch.Tensor, torch.Tensor]
```

Get the coordinate grid tensors of the cell.

This method returns the x and y coordinate tensors that define the real-space
grid of the unit cell. These tensors can be used for visualization or for
creating custom shape masks.

**Returns**:

  Tuple[torch.Tensor, torch.Tensor]: A tuple containing (X, Y) coordinate tensors,
  each with shape (rdim[0], rdim[1]).
  

**Examples**:

```python
# Get coordinate grids and use them for visualization
import matplotlib.pyplot as plt
import torch
from torchrdit.cell import Cell3D
cell = Cell3D()
X, Y = cell.get_layout()
plt.figure(figsize=(6, 6))
plt.pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), torch.ones_like(X).cpu().numpy())
plt.axis('equal')
plt.title('Unit Cell Coordinate Grid')
plt.colorbar(label='Grid Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```
  
  Keywords:
  coordinate grid, layout, visualization, real-space coordinates

<a id="torchrdit.cell.Cell3D.er1"></a>

#### er1

```python
@property
def er1()
```

Get the permittivity of the reflection layer.

**Returns**:

- `torch.Tensor` - Complex tensor containing the permittivity (εᵣ) of the
  reflection layer material.
  
  Keywords:
  permittivity, reflection layer, material property, dielectric constant

<a id="torchrdit.cell.Cell3D.er2"></a>

#### er2

```python
@property
def er2()
```

Get the permittivity of the transmission layer.

**Returns**:

- `torch.Tensor` - Complex tensor containing the permittivity (εᵣ) of the
  transmission layer material.
  
  Keywords:
  permittivity, transmission layer, material property, dielectric constant

<a id="torchrdit.cell.Cell3D.ur1"></a>

#### ur1

```python
@property
def ur1()
```

Get the permeability of the reflection layer.

**Returns**:

- `torch.Tensor` - Complex tensor containing the permeability (μᵣ) of the
  reflection layer material.
  
  Keywords:
  permeability, reflection layer, material property, magnetic property

<a id="torchrdit.cell.Cell3D.ur2"></a>

#### ur2

```python
@property
def ur2()
```

Get the permeability of the transmission layer.

**Returns**:

- `torch.Tensor` - Complex tensor containing the permeability (μᵣ) of the
  transmission layer material.
  
  Keywords:
  permeability, transmission layer, material property, magnetic property

<a id="torchrdit.cell.Cell3D.lengthunit"></a>

#### lengthunit

```python
@property
def lengthunit()
```

Get the length unit used in the simulation.

**Returns**:

- `str` - The unit of length used in the simulation (e.g., 'um', 'nm').
  
  Keywords:
  length unit, dimension, measurement unit

<a id="torchrdit.cell.Cell3D.get_cell_type"></a>

#### get\_cell\_type

```python
def get_cell_type()
```

Determine the type of cell based on lattice vectors.

This method analyzes the lattice vectors and determines whether the cell
is a standard Cartesian cell (with vectors aligned to the coordinate axes)
or a more general cell type.

**Returns**:

- `CellType` - The type of cell (CellType.Cartesian or CellType.Other).
  

**Examples**:

```python
# Create cells with different lattice vectors and check their types
import torch
from torchrdit.cell import Cell3D
cell1 = Cell3D(t1=torch.tensor([[1.0, 0.0]]), t2=torch.tensor([[0.0, 1.0]]))
print(cell1.get_cell_type())  # Cartesian

cell2 = Cell3D(t1=torch.tensor([[1.0, 0.2]]), t2=torch.tensor([[0.0, 1.0]]))
print(cell2.get_cell_type())  # Other
```
  
  Keywords:
  cell type, lattice vectors, coordinate system, Cartesian

