# Table of Contents

* [torchrdit.layers](#torchrdit.layers)
  * [Layer](#torchrdit.layers.Layer)
    * [\_\_init\_\_](#torchrdit.layers.Layer.__init__)
    * [\_\_str\_\_](#torchrdit.layers.Layer.__str__)
    * [thickness](#torchrdit.layers.Layer.thickness)
    * [thickness](#torchrdit.layers.Layer.thickness)
    * [material\_name](#torchrdit.layers.Layer.material_name)
    * [material\_name](#torchrdit.layers.Layer.material_name)
    * [is\_homogeneous](#torchrdit.layers.Layer.is_homogeneous)
    * [is\_dispersive](#torchrdit.layers.Layer.is_dispersive)
    * [is\_dispersive](#torchrdit.layers.Layer.is_dispersive)
    * [is\_optimize](#torchrdit.layers.Layer.is_optimize)
    * [is\_optimize](#torchrdit.layers.Layer.is_optimize)
    * [is\_solved](#torchrdit.layers.Layer.is_solved)
    * [is\_solved](#torchrdit.layers.Layer.is_solved)
  * [LayerBuilder](#torchrdit.layers.LayerBuilder)
    * [\_\_init\_\_](#torchrdit.layers.LayerBuilder.__init__)
    * [create\_layer](#torchrdit.layers.LayerBuilder.create_layer)
    * [update\_thickness](#torchrdit.layers.LayerBuilder.update_thickness)
    * [update\_material\_name](#torchrdit.layers.LayerBuilder.update_material_name)
    * [set\_optimize](#torchrdit.layers.LayerBuilder.set_optimize)
    * [get\_layer\_instance](#torchrdit.layers.LayerBuilder.get_layer_instance)
    * [set\_dispersive](#torchrdit.layers.LayerBuilder.set_dispersive)
  * [HomogeneousLayer](#torchrdit.layers.HomogeneousLayer)
    * [\_\_init\_\_](#torchrdit.layers.HomogeneousLayer.__init__)
    * [\_\_str\_\_](#torchrdit.layers.HomogeneousLayer.__str__)
  * [GratingLayer](#torchrdit.layers.GratingLayer)
    * [\_\_init\_\_](#torchrdit.layers.GratingLayer.__init__)
    * [\_\_str\_\_](#torchrdit.layers.GratingLayer.__str__)
  * [HomogeneousLayerBuilder](#torchrdit.layers.HomogeneousLayerBuilder)
    * [\_\_init\_\_](#torchrdit.layers.HomogeneousLayerBuilder.__init__)
    * [create\_layer](#torchrdit.layers.HomogeneousLayerBuilder.create_layer)
  * [GratingLayerBuilder](#torchrdit.layers.GratingLayerBuilder)
    * [\_\_init\_\_](#torchrdit.layers.GratingLayerBuilder.__init__)
    * [create\_layer](#torchrdit.layers.GratingLayerBuilder.create_layer)
  * [LayerDirector](#torchrdit.layers.LayerDirector)
    * [\_\_init\_\_](#torchrdit.layers.LayerDirector.__init__)
    * [build\_layer](#torchrdit.layers.LayerDirector.build_layer)
  * [LayerManager](#torchrdit.layers.LayerManager)
    * [\_\_init\_\_](#torchrdit.layers.LayerManager.__init__)
    * [gen\_toeplitz\_matrix](#torchrdit.layers.LayerManager.gen_toeplitz_matrix)
    * [add\_layer](#torchrdit.layers.LayerManager.add_layer)
    * [replace\_layer\_to\_homogeneous](#torchrdit.layers.LayerManager.replace_layer_to_homogeneous)
    * [replace\_layer\_to\_grating](#torchrdit.layers.LayerManager.replace_layer_to_grating)
    * [update\_layer\_thickness](#torchrdit.layers.LayerManager.update_layer_thickness)
    * [update\_trn\_layer](#torchrdit.layers.LayerManager.update_trn_layer)
    * [update\_ref\_layer](#torchrdit.layers.LayerManager.update_ref_layer)
    * [ref\_material\_name](#torchrdit.layers.LayerManager.ref_material_name)
    * [trn\_material\_name](#torchrdit.layers.LayerManager.trn_material_name)
    * [is\_ref\_dispersive](#torchrdit.layers.LayerManager.is_ref_dispersive)
    * [is\_trn\_dispersive](#torchrdit.layers.LayerManager.is_trn_dispersive)
    * [nlayer](#torchrdit.layers.LayerManager.nlayer)

<a id="torchrdit.layers"></a>

# torchrdit.layers

Module for defining and managing material layers in TorchRDIT electromagnetic simulations.

This module provides classes to define, create, and manage the material layers that
make up the structure being simulated in TorchRDIT. It implements a builder pattern
to create different types of layers (homogeneous and patterned/grating) and a manager
class to handle collections of layers.

The layer system is a core component of the TorchRDIT electromagnetic solver,
representing the physical structure of the device being simulated. Each layer
has properties such as thickness, material composition, and can be either
homogeneous (uniform material) or patterned (spatially varying material distribution).

Classes:
Layer: Abstract base class for all layer types.
HomogeneousLayer: Layer with uniform material properties.
GratingLayer: Layer with spatially varying material properties.
LayerBuilder: Abstract base class for layer builders.
HomogeneousLayerBuilder: Builder for homogeneous layers.
GratingLayerBuilder: Builder for grating/patterned layers.
LayerDirector: Coordinates the layer building process.
LayerManager: Manages collections of layers and layer operations.

**Notes**:

  Users typically don't interact with these classes directly, but rather through
  the solver classes defined in solver.py. The solver provides more user-friendly
  methods for adding and manipulating layers.
  

**Examples**:

```python
# Users typically add layers through solver interfaces:
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
import torch
# Create a solver
solver = create_solver(algorithm=Algorithm.RCWA)
# Add a material to the solver
silicon = create_material(name="silicon", permittivity=11.7)
solver.add_materials([silicon])
# Add a homogeneous layer
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
# Add a patterned layer
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.3),
               is_homogeneous=False)
# Pattern the layer with a circle
circle_mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
solver.update_er_with_mask(circle_mask, layer_index=1)
```
  
  Keywords:
  layers, material layers, layer management, homogeneous layers, grating layers,
  patterned layers, layer stack, photonic structure, simulation structure,
  builder pattern, layer properties, electromagnetic simulation

<a id="torchrdit.layers.Layer"></a>

## Layer Objects

```python
class Layer(metaclass=ABCMeta)
```

Abstract base class for all layer types in TorchRDIT electromagnetic simulations.

This class defines the interface for all layer types in the simulation. It is an
abstract base class that cannot be instantiated directly. Instead, concrete
layer classes (HomogeneousLayer, GratingLayer) should be used, typically through
the solver's add_layer method rather than direct instantiation.

Each Layer represents a physical layer in the simulated structure with properties
such as thickness, material composition, and homogeneity. Layers are used to model
the electromagnetic response of different regions in the simulated structure.

**Notes**:

  This class is not intended for direct use by end users. Users should create
  and manipulate layers through the solver interface (create_solver, add_layer, etc.)
  rather than instantiating Layer subclasses directly.
  

**Attributes**:

- `thickness` _float_ - Thickness of the layer in the simulation's length units.
- `material_name` _str_ - Name of the material used in this layer.
- `is_homogeneous` _bool_ - Whether the layer has uniform material properties.
- `is_dispersive` _bool_ - Whether the layer's material has frequency-dependent properties.
- `is_optimize` _bool_ - Whether the layer's parameters are subject to optimization.
- `is_solved` _bool_ - Whether the electromagnetic response of this layer has been solved.
- `ermat` _torch.Tensor, optional_ - Permittivity distribution matrix.
- `urmat` _torch.Tensor, optional_ - Permeability distribution matrix.
- `kermat` _torch.Tensor, optional_ - Fourier-transformed permittivity matrix.
- `kurmat` _torch.Tensor, optional_ - Fourier-transformed permeability matrix.
  
  Keywords:
  abstract class, layer interface, layer properties, material layer,
  electromagnetic simulation, photonic structure

<a id="torchrdit.layers.Layer.__init__"></a>

#### \_\_init\_\_

```python
@abstractmethod
def __init__(thickness: float = 0.0,
             material_name: str = "",
             is_optimize: bool = False,
             **kwargs) -> None
```

Initialize a Layer instance with given properties.

This abstract method initializes basic properties common to all layer types.
It must be implemented by all concrete Layer subclasses.

**Arguments**:

- `thickness` _float_ - Thickness of the layer in the simulation's length units.
  Default is 0.0.
- `material_name` _str_ - Name of the material used for this layer.
  The material must exist in the solver's material library.
  Default is an empty string.
- `is_optimize` _bool_ - Flag indicating if the layer's parameters
  (e.g., thickness) should be optimized during parameter sweeps.
  Default is False.
- `**kwargs` - Additional keyword arguments specific to subclasses.
  

**Notes**:

  This method is not intended to be called directly by users. Layers
  should be created through the solver interface.
  
  Keywords:
  initialization, layer creation, layer properties, material assignment

<a id="torchrdit.layers.Layer.__str__"></a>

#### \_\_str\_\_

```python
@abstractmethod
def __str__() -> str
```

Return a string representation of the layer.

<a id="torchrdit.layers.Layer.thickness"></a>

#### thickness

```python
@property
def thickness() -> float
```

Get the thickness of the layer.

**Returns**:

- `float` - The thickness of the layer in the simulation's length units.
  
  Keywords:
  thickness, layer property, geometric parameter

<a id="torchrdit.layers.Layer.thickness"></a>

#### thickness

```python
@thickness.setter
def thickness(thickness: float)
```

Set the thickness of the layer.

**Arguments**:

- `thickness` _float_ - The new thickness value for the layer
  in the simulation's length units.
  
  Keywords:
  thickness, layer property, update parameter

<a id="torchrdit.layers.Layer.material_name"></a>

#### material\_name

```python
@property
def material_name() -> str
```

Get the name of the material assigned to this layer.

**Returns**:

- `str` - The name of the material used in this layer.
  
  Keywords:
  material, layer property, material assignment

<a id="torchrdit.layers.Layer.material_name"></a>

#### material\_name

```python
@material_name.setter
def material_name(material_name: str)
```

Set the material name for this layer.

**Arguments**:

- `material_name` _str_ - The name of the material to assign to this layer.
  The material must exist in the solver's material library.
  
  Keywords:
  material, layer property, update material

<a id="torchrdit.layers.Layer.is_homogeneous"></a>

#### is\_homogeneous

```python
@property
def is_homogeneous() -> bool
```

Check if the layer has uniform material properties.

**Returns**:

- `bool` - True if the layer has uniform material properties,
  False if it has spatially varying properties.
  
  Keywords:
  homogeneous, uniform, layer property, material distribution

<a id="torchrdit.layers.Layer.is_dispersive"></a>

#### is\_dispersive

```python
@property
def is_dispersive() -> bool
```

Check if the layer's material has frequency-dependent properties.

**Returns**:

- `bool` - True if the material properties depend on frequency/wavelength,
  False otherwise.
  
  Keywords:
  dispersive, frequency-dependent, wavelength-dependent, material property

<a id="torchrdit.layers.Layer.is_dispersive"></a>

#### is\_dispersive

```python
@is_dispersive.setter
def is_dispersive(is_dispersive: bool)
```

Set the dispersive flag for the layer's material.

**Arguments**:

- `is_dispersive` _bool_ - True if the material properties should be
  treated as frequency/wavelength-dependent, False otherwise.
  
  Keywords:
  dispersive, frequency-dependent, wavelength-dependent, material property

<a id="torchrdit.layers.Layer.is_optimize"></a>

#### is\_optimize

```python
@property
def is_optimize() -> bool
```

Check if the layer's parameters are subject to optimization.

**Returns**:

- `bool` - True if the layer's parameters should be included in optimization
  procedures, False otherwise.
  
  Keywords:
  optimization, parameter sweep, inverse design

<a id="torchrdit.layers.Layer.is_optimize"></a>

#### is\_optimize

```python
@is_optimize.setter
def is_optimize(is_optimize: bool)
```

Set the optimization flag for the layer.

**Arguments**:

- `is_optimize` _bool_ - True if the layer's parameters should be included
  in optimization procedures, False otherwise.
  
  Keywords:
  optimization, parameter sweep, inverse design

<a id="torchrdit.layers.Layer.is_solved"></a>

#### is\_solved

```python
@property
def is_solved() -> bool
```

Check if the electromagnetic response of this layer has been solved.

**Returns**:

- `bool` - True if the layer's electromagnetic response has been calculated,
  False otherwise.
  
  Keywords:
  solved, calculated, electromagnetic response

<a id="torchrdit.layers.Layer.is_solved"></a>

#### is\_solved

```python
@is_solved.setter
def is_solved(is_solved: bool)
```

Set the solved flag for the layer.

**Arguments**:

- `is_solved` _bool_ - True if the layer's electromagnetic response
  has been calculated, False otherwise.
  
  Keywords:
  solved, calculated, electromagnetic response

<a id="torchrdit.layers.LayerBuilder"></a>

## LayerBuilder Objects

```python
class LayerBuilder(metaclass=ABCMeta)
```

Abstract base class for builders that create layer instances.

This class implements the builder pattern for creating different types of layers
in TorchRDIT. It provides the interface for concrete builders that create
specific types of layers (homogeneous or grating/patterned).

The builder pattern separates the construction of complex objects from their
representation, allowing the same construction process to create different
representations.

**Notes**:

  This class is primarily used internally by TorchRDIT and is not intended
  for direct use by end users. Users should create layers through the
  solver interface.
  

**Attributes**:

- `layer` _Layer, optional_ - The layer instance being constructed.
  
  Keywords:
  builder pattern, layer creation, design pattern, factory, abstract class

<a id="torchrdit.layers.LayerBuilder.__init__"></a>

#### \_\_init\_\_

```python
@abstractmethod
def __init__() -> None
```

Initialize the LayerBuilder instance.

Sets the layer attribute to None, to be populated by create_layer.

<a id="torchrdit.layers.LayerBuilder.create_layer"></a>

#### create\_layer

```python
@abstractmethod
def create_layer()
```

Create a new layer instance of the appropriate type.

This abstract method must be implemented by subclasses to instantiate
the specific type of layer they build.

Keywords:
    layer creation, instantiation, builder pattern

<a id="torchrdit.layers.LayerBuilder.update_thickness"></a>

#### update\_thickness

```python
def update_thickness(thickness)
```

Update the thickness of the layer being built.

**Arguments**:

- `thickness` _float_ - The thickness value to set for the layer.
  
  Keywords:
  thickness, layer property, parameter update

<a id="torchrdit.layers.LayerBuilder.update_material_name"></a>

#### update\_material\_name

```python
def update_material_name(material_name)
```

Set the material for the layer being built.

**Arguments**:

- `material_name` _str_ - The name of the material to use for the layer.
  Must be a material that exists in the solver's material library.
  
  Keywords:
  material, layer property, material assignment

<a id="torchrdit.layers.LayerBuilder.set_optimize"></a>

#### set\_optimize

```python
def set_optimize(is_optimize)
```

Set whether the layer's parameters should be optimized.

**Arguments**:

- `is_optimize` _bool_ - True if the layer should be included in
  optimization procedures, False otherwise.
  
  Keywords:
  optimization, parameter sweep, design optimization

<a id="torchrdit.layers.LayerBuilder.get_layer_instance"></a>

#### get\_layer\_instance

```python
def get_layer_instance()
```

Get the constructed layer instance.

**Returns**:

- `Layer` - The fully configured layer instance.
  
  Keywords:
  layer instance, builder result, object creation

<a id="torchrdit.layers.LayerBuilder.set_dispersive"></a>

#### set\_dispersive

```python
def set_dispersive(is_dispersive)
```

Set whether the layer's material has frequency-dependent properties.

**Arguments**:

- `is_dispersive` _bool_ - True if the material should be treated as
  having frequency/wavelength-dependent properties, False otherwise.
  
  Keywords:
  dispersive, frequency-dependent, wavelength-dependent, material property

<a id="torchrdit.layers.HomogeneousLayer"></a>

## HomogeneousLayer Objects

```python
class HomogeneousLayer(Layer)
```

Layer with uniform material properties throughout its volume.

This class represents a layer with homogeneous (spatially uniform) material
properties. The permittivity and permeability are constant throughout the
entire layer, making it suitable for representing bulk materials, thin films,
or other uniform regions.

Homogeneous layers are computationally efficient as they don't require
spatial discretization of material properties.

**Notes**:

  Users typically do not instantiate this class directly, but rather
  create homogeneous layers through the solver's add_layer method with
  is_homogeneous=True (the default).
  

**Attributes**:

  Inherits all attributes from the Layer base class.
  

**Examples**:

```python
# Create a homogeneous layer through the solver interface
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
import torch
solver = create_solver()
silicon = create_material(name="silicon", permittivity=11.7)
solver.add_materials([silicon])
# Add a homogeneous layer with thickness 0.2 μm
solver.add_layer(
    material_name="silicon",
    thickness=torch.tensor(0.2)
)
```
  
  Keywords:
  homogeneous layer, uniform material, bulk material, thin film, constant properties

<a id="torchrdit.layers.HomogeneousLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(thickness: float = 0.0,
             material_name: str = "",
             is_optimize: bool = False,
             **kwargs) -> None
```

Initialize a HomogeneousLayer instance.

**Arguments**:

- `thickness` _float_ - Thickness of the layer in the simulation's length units.
  Default is 0.0.
- `material_name` _str_ - Name of the material to use for this layer.
  Must be a material that exists in the solver's material library.
  Default is an empty string.
- `is_optimize` _bool_ - Whether the layer's parameters should be included in
  optimization procedures. Default is False.
- `**kwargs` - Additional keyword arguments passed to the parent class.
  
  Keywords:
  initialize homogeneous layer, create uniform layer

<a id="torchrdit.layers.HomogeneousLayer.__str__"></a>

#### \_\_str\_\_

```python
def __str__() -> str
```

Get a string representation of the homogeneous layer.

**Returns**:

- `str` - A string describing the homogeneous layer instance.

<a id="torchrdit.layers.GratingLayer"></a>

## GratingLayer Objects

```python
class GratingLayer(Layer)
```

Layer with spatially varying material properties.

This class represents a layer with inhomogeneous (spatially varying) material
properties, such as photonic crystals, metamaterials, diffraction gratings,
or other patterned structures. The permittivity and permeability can vary
throughout the layer based on binary masks or other patterns.

Grating layers require spatial discretization of material properties and are
generally more computationally intensive than homogeneous layers.

**Notes**:

  Users typically do not instantiate this class directly, but rather
  create grating layers through the solver's add_layer method with
  is_homogeneous=False.
  

**Attributes**:

  Inherits all attributes from the Layer base class.
  The attribute is_homogeneous is always False for grating layers.
  

**Examples**:

```python
# Create a patterned (grating) layer through the solver interface
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
import torch
solver = create_solver()
silicon = create_material(name="silicon", permittivity=11.7)
air = create_material(name="air", permittivity=1.0)
solver.add_materials([silicon, air])

# Add a patterned layer with thickness 0.5 μm
solver.add_layer(
    material_name="silicon",
    thickness=torch.tensor(0.5),
    is_homogeneous=False
)

# Create a circular pattern in the layer
circle_mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
solver.update_er_with_mask(
    mask=circle_mask,
    layer_index=0,
    bg_material="air"
)
```
  
  Keywords:
  grating layer, patterned layer, inhomogeneous, photonic crystal, metamaterial,
  spatial variation, material distribution, photonic device

<a id="torchrdit.layers.GratingLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(thickness: float = 0.0,
             material_name: str = "",
             is_optimize: bool = False,
             **kwargs) -> None
```

Initialize a GratingLayer instance.

**Arguments**:

- `thickness` _float_ - Thickness of the layer in the simulation's length units.
  Default is 0.0.
- `material_name` _str_ - Name of the material to use as the foreground material.
  Must be a material that exists in the solver's material library.
  Default is an empty string.
- `is_optimize` _bool_ - Whether the layer's parameters should be included in
  optimization procedures. Default is False.
- `**kwargs` - Additional keyword arguments passed to the parent class.
  
  Keywords:
  initialize grating layer, create patterned layer

<a id="torchrdit.layers.GratingLayer.__str__"></a>

#### \_\_str\_\_

```python
def __str__() -> str
```

Get a string representation of the grating layer.

**Returns**:

- `str` - A string describing the grating layer instance.

<a id="torchrdit.layers.HomogeneousLayerBuilder"></a>

## HomogeneousLayerBuilder Objects

```python
class HomogeneousLayerBuilder(LayerBuilder)
```

Builder for creating HomogeneousLayer instances.

This class is a concrete implementation of the LayerBuilder abstract class
that creates and configures HomogeneousLayer instances. It follows the
builder pattern to separate the construction details from the representation.

HomogeneousLayerBuilder creates layers with uniform material properties
throughout their volume, suitable for bulk materials, thin films, or other
uniform regions.

**Notes**:

  This class is used internally by the LayerDirector and is not typically
  accessed directly by users. Users should create layers through the solver
  interface.
  
  Keywords:
  builder pattern, homogeneous layer, uniform material, layer creation

<a id="torchrdit.layers.HomogeneousLayerBuilder.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

Initialize a HomogeneousLayerBuilder instance.

Initializes the builder with no layer instance created yet.

<a id="torchrdit.layers.HomogeneousLayerBuilder.create_layer"></a>

#### create\_layer

```python
def create_layer()
```

Create a new empty HomogeneousLayer instance.

This method instantiates a new HomogeneousLayer with default values.
The properties of the layer are set through subsequent method calls
to update_thickness, update_material_name, etc.

Keywords:
    create layer, instantiate layer, homogeneous layer

<a id="torchrdit.layers.GratingLayerBuilder"></a>

## GratingLayerBuilder Objects

```python
class GratingLayerBuilder(HomogeneousLayerBuilder)
```

Builder for creating GratingLayer instances.

This class is a concrete implementation of the LayerBuilder abstract class
that creates and configures GratingLayer instances. It follows the
builder pattern to separate the construction details from the representation.

GratingLayerBuilder creates layers with spatially varying material properties,
suitable for photonic crystals, metamaterials, diffraction gratings, or other
patterned structures.

**Notes**:

  This class is used internally by the LayerDirector and is not typically
  accessed directly by users. Users should create layers through the solver
  interface with is_homogeneous=False.
  
  Keywords:
  builder pattern, grating layer, patterned layer, inhomogeneous material,
  layer creation, photonic crystal, metamaterial

<a id="torchrdit.layers.GratingLayerBuilder.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

Initialize a GratingLayerBuilder instance.

Initializes the builder with no layer instance created yet.

<a id="torchrdit.layers.GratingLayerBuilder.create_layer"></a>

#### create\_layer

```python
def create_layer()
```

Create a new empty GratingLayer instance.

This method instantiates a new GratingLayer with default values.
The properties of the layer are set through subsequent method calls
to update_thickness, update_material_name, etc.

Keywords:
    create layer, instantiate layer, grating layer, patterned layer

<a id="torchrdit.layers.LayerDirector"></a>

## LayerDirector Objects

```python
class LayerDirector()
```

Director class that coordinates the layer building process.

This class implements the director component of the builder pattern,
coordinating the sequence of steps needed to create fully configured
layer instances using the appropriate builder based on the layer type.

The director abstracts the layer creation process, allowing the client
code (typically LayerManager) to create layers without knowing the
details of how they are constructed.

**Notes**:

  This class is used internally by TorchRDIT and is not intended for
  direct use by end users.
  
  Keywords:
  builder pattern, layer creation, director, instantiation coordinator

<a id="torchrdit.layers.LayerDirector.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

Initialize the LayerDirector instance.

Creates a new LayerDirector with no configuration required.

<a id="torchrdit.layers.LayerDirector.build_layer"></a>

#### build\_layer

```python
def build_layer(layer_type,
                thickness,
                material_name,
                is_optimize=False,
                is_dispersive=False) -> Layer
```

Build a fully configured layer of the specified type.

This method coordinates the layer building process, selecting the appropriate
builder based on the layer_type, creating a new layer instance, and configuring
it with the provided parameters.

**Arguments**:

- `layer_type` _str_ - Type of layer to create. Must be one of:
  'homogeneous' for a layer with uniform material properties
  'grating' for a layer with spatially varying material properties
- `thickness` _float_ - Thickness of the layer in the simulation's length units.
- `material_name` _str_ - Name of the material for the layer. Must be a material
  that exists in the solver's material library.
- `is_optimize` _bool_ - Whether the layer's parameters should be included in
  optimization procedures. Default is False.
- `is_dispersive` _bool_ - Whether the material has frequency-dependent properties.
  Default is False.
  

**Returns**:

- `Layer` - A fully configured layer instance of the specified type.
  
  Keywords:
  build layer, create layer, layer construction, layer configuration

<a id="torchrdit.layers.LayerManager"></a>

## LayerManager Objects

```python
class LayerManager()
```

Manager for organizing and manipulating layers in an electromagnetic simulation.

The LayerManager class is responsible for maintaining a collection of layers
that form the structure being simulated in TorchRDIT. It provides methods for
adding, updating, and configuring layers, as well as handling the material
properties of the semi-infinite regions above (transmission) and below
(reflection) the layered structure.

This class also provides utilities for converting between real-space material
distributions and their Fourier-space representations using Toeplitz matrices,
which are crucial for the RCWA and RDIT algorithms.

**Notes**:

  This class is not typically accessed directly by users but is rather used
  internally by solver classes. Users interact with layers through the solver
  interface.
  

**Attributes**:

- `layers` _List[Layer]_ - List of Layer instances that make up the structure.
- `layer_director` _LayerDirector_ - Director that coordinates layer creation.
- `_ref_material_name` _str_ - Name of the material for the reflection region.
- `_trn_material_name` _str_ - Name of the material for the transmission region.
- `_is_ref_dispers` _bool_ - Whether the reflection material is dispersive.
- `_is_trn_dispers` _bool_ - Whether the transmission material is dispersive.
- `lattice_t1` _torch.Tensor_ - First lattice vector defining the unit cell.
- `lattice_t2` _torch.Tensor_ - Second lattice vector defining the unit cell.
- `vec_p` _torch.Tensor_ - Vector of p-coordinates in the unit cell.
- `vec_q` _torch.Tensor_ - Vector of q-coordinates in the unit cell.
  

**Examples**:

```python
# Users typically interact with layers through the solver interface:
from torchrdit.solver import create_solver
from torchrdit.utils import create_material
import torch
# Create solver with RCWA algorithm
solver = create_solver()
# Add materials
silicon = create_material(name="silicon", permittivity=11.7)
sio2 = create_material(name="sio2", permittivity=2.25)
solver.add_materials([silicon, sio2])
# Add layers
solver.add_layer(material_name="sio2", thickness=torch.tensor(0.1))
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
# Set bottom (reflection) region to silicon
solver.update_ref_material("silicon")
# Set top (transmission) region to sio2
solver.update_trn_material("sio2")
```
  
  Keywords:
  layer management, layer stack, structure definition, layer organization,
  layer manipulation, reflection region, transmission region, toeplitz matrix

<a id="torchrdit.layers.LayerManager.__init__"></a>

#### \_\_init\_\_

```python
def __init__(lattice_t1, lattice_t2, vec_p, vec_q) -> None
```

Initialize a LayerManager instance.

**Arguments**:

- `lattice_t1` _torch.Tensor_ - First lattice vector defining the unit cell.
- `lattice_t2` _torch.Tensor_ - Second lattice vector defining the unit cell.
- `vec_p` _torch.Tensor_ - Vector of p-coordinates in the unit cell grid.
- `vec_q` _torch.Tensor_ - Vector of q-coordinates in the unit cell grid.
  
  Keywords:
  initialization, layer manager creation

<a id="torchrdit.layers.LayerManager.gen_toeplitz_matrix"></a>

#### gen\_toeplitz\_matrix

```python
def gen_toeplitz_matrix(layer_index: int,
                        n_harmonic1: int,
                        n_harmonic2: int,
                        param: str = "er",
                        method: str = "FFT")
```

Generate Toeplitz matrix for the specified layer and parameter.

This method converts the real-space material distribution of a layer
to its Fourier-space representation using a Toeplitz matrix. This is a
critical operation for the RCWA and RDIT algorithms that operate in
Fourier space.

**Arguments**:

- `layer_index` _int_ - Index of the layer in the layers list.
- `n_harmonic1` _int_ - Number of harmonics for the first dimension.
- `n_harmonic2` _int_ - Number of harmonics for the second dimension.
- `param` _str_ - Parameter to convert, either 'er' for permittivity or
  'ur' for permeability. Default is 'er'.
- `method` _str_ - Method for computing the Toeplitz matrix.
- `'FFT'` - Uses Fast Fourier Transform, works for all cell types.
- `'Analytical'` - Uses analytical formulation, only for Cartesian cells.
  Default is 'FFT'.
  

**Raises**:

- `ValueError` - If the specified layer does not have the required material
  property distribution set.
  
  Keywords:
  toeplitz matrix, fourier transform, RCWA, spatial harmonics,
  material distribution, permittivity, permeability

<a id="torchrdit.layers.LayerManager.add_layer"></a>

#### add\_layer

```python
@tensor_params_check(check_start_index=2, check_stop_index=2)
def add_layer(layer_type,
              thickness,
              material_name,
              is_optimize=False,
              is_dispersive=False)
```

Add a new layer to the layer structure.

This method creates a new layer of the specified type with the given properties
and adds it to the layer structure. Layers are stacked in the order they are added,
with the first layer added being at the bottom of the stack (closest to the
reflection region) and subsequent layers building upward.

**Arguments**:

- `layer_type` _str_ - Type of layer to create. Must be one of:
  'homogeneous' for a layer with uniform material properties
  'grating' for a layer with spatially varying material properties
- `thickness` _torch.Tensor_ - Thickness of the layer in the simulation's length units.
- `material_name` _str_ - Name of the material for the layer. Must be a material
  that exists in the solver's material library.
- `is_optimize` _bool_ - Whether the layer's parameters should be included in
  optimization procedures. Default is False.
- `is_dispersive` _bool_ - Whether the material has frequency-dependent properties.
  Default is False.
  

**Notes**:

  Users typically do not call this method directly but rather use the
  solver's add_layer method, which provides a more user-friendly interface.
  
  Keywords:
  add layer, create layer, layer stack, material layer, layer configuration

<a id="torchrdit.layers.LayerManager.replace_layer_to_homogeneous"></a>

#### replace\_layer\_to\_homogeneous

```python
def replace_layer_to_homogeneous(layer_index)
```

Convert a layer to a homogeneous layer.

This method replaces the layer at the specified index with a new homogeneous
layer that has the same thickness, material name, and other properties.

**Arguments**:

- `layer_index` _int_ - Index of the layer to convert in the layers list.
  
  Keywords:
  convert layer, homogeneous layer, layer modification

<a id="torchrdit.layers.LayerManager.replace_layer_to_grating"></a>

#### replace\_layer\_to\_grating

```python
def replace_layer_to_grating(layer_index)
```

Convert a layer to a grating (patterned) layer.

This method replaces the layer at the specified index with a new grating
layer that has the same thickness, material name, and other properties.

**Arguments**:

- `layer_index` _int_ - Index of the layer to convert in the layers list.
  
  Keywords:
  convert layer, grating layer, patterned layer, layer modification

<a id="torchrdit.layers.LayerManager.update_layer_thickness"></a>

#### update\_layer\_thickness

```python
@tensor_params_check(check_start_index=2, check_stop_index=2)
def update_layer_thickness(layer_index, thickness)
```

Update the thickness of a layer.

This method changes the thickness of the layer at the specified index.

**Arguments**:

- `layer_index` _int_ - Index of the layer to update in the layers list.
- `thickness` _torch.Tensor_ - New thickness value for the layer in the
  simulation's length units.
  
  Keywords:
  update thickness, layer modification, geometric parameter

<a id="torchrdit.layers.LayerManager.update_trn_layer"></a>

#### update\_trn\_layer

```python
def update_trn_layer(material_name: str, is_dispersive: bool)
```

Update the transmission region material.

This method sets the material for the semi-infinite region above the
layer stack (transmission region).

**Arguments**:

- `material_name` _str_ - Name of the material to use for the transmission region.
  Must be a material that exists in the solver's material library.
- `is_dispersive` _bool_ - Whether the material has frequency-dependent properties.
  
  Keywords:
  transmission region, output medium, top medium, semi-infinite region

<a id="torchrdit.layers.LayerManager.update_ref_layer"></a>

#### update\_ref\_layer

```python
def update_ref_layer(material_name: str, is_dispersive: bool)
```

Update the reflection region material.

This method sets the material for the semi-infinite region below the
layer stack (reflection region).

**Arguments**:

- `material_name` _str_ - Name of the material to use for the reflection region.
  Must be a material that exists in the solver's material library.
- `is_dispersive` _bool_ - Whether the material has frequency-dependent properties.
  
  Keywords:
  reflection region, input medium, bottom medium, semi-infinite region

<a id="torchrdit.layers.LayerManager.ref_material_name"></a>

#### ref\_material\_name

```python
@property
def ref_material_name() -> str
```

Get the name of the material used in the reflection region.

**Returns**:

- `str` - Name of the material for the semi-infinite region below the layer stack.
  
  Keywords:
  reflection material, bottom material, input medium

<a id="torchrdit.layers.LayerManager.trn_material_name"></a>

#### trn\_material\_name

```python
@property
def trn_material_name() -> str
```

Get the name of the material used in the transmission region.

**Returns**:

- `str` - Name of the material for the semi-infinite region above the layer stack.
  
  Keywords:
  transmission material, top material, output medium

<a id="torchrdit.layers.LayerManager.is_ref_dispersive"></a>

#### is\_ref\_dispersive

```python
@property
def is_ref_dispersive() -> bool
```

Check if the reflection region material has frequency-dependent properties.

**Returns**:

- `bool` - True if the reflection material is dispersive, False otherwise.
  
  Keywords:
  dispersive material, frequency-dependent, wavelength-dependent

<a id="torchrdit.layers.LayerManager.is_trn_dispersive"></a>

#### is\_trn\_dispersive

```python
@property
def is_trn_dispersive() -> bool
```

Check if the transmission region material has frequency-dependent properties.

**Returns**:

- `bool` - True if the transmission material is dispersive, False otherwise.
  
  Keywords:
  dispersive material, frequency-dependent, wavelength-dependent

<a id="torchrdit.layers.LayerManager.nlayer"></a>

#### nlayer

```python
@property
def nlayer() -> int
```

Get the number of layers in the structure.

**Returns**:

- `int` - Number of material layers in the layer stack.
  
  Keywords:
  layer count, stack size, structure depth

