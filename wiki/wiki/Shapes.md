# Shape Generation Module

## Overview
The `torchrdit.shapes` module provides tools for generating binary masks representing various photonic structures. These masks can be used to define the geometry of patterned layers in electromagnetic simulations.

## Key Features
- Create common shapes (circles, rectangles, polygons)
- Support for both hard and soft edges
- Combine shapes using boolean operations (union, intersection, difference)
- Non-Cartesian coordinate system support through lattice vectors
- Full PyTorch integration for GPU acceleration and differentiability

## Usage
```python
import torch
from torchrdit.shapes import ShapeGenerator
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm

# Create a solver
solver = create_solver(algorithm=Algorithm.RDIT, rdim=[512, 512], kdim=[7, 7])

# Create a shape generator
shape_gen = ShapeGenerator.from_solver(solver)

# Generate shapes
circle = shape_gen.generate_circle_mask(center=(0.0, 0.0), radius=0.3)
rect = shape_gen.generate_rectangle_mask(width=0.4, height=0.2, angle=30)

# Combine shapes
combined = shape_gen.combine_masks(circle, rect, operation='union')

# Use the mask in a simulation
solver.update_er_with_mask(combined, layer_index=0)
```

## API Reference

Below is the complete API reference for the shapes module, automatically generated from the source code.

# Table of Contents

* [torchrdit.shapes](#torchrdit.shapes)
  * [ShapeGenerator](#torchrdit.shapes.ShapeGenerator)
    * [\_\_init\_\_](#torchrdit.shapes.ShapeGenerator.__init__)
    * [from\_solver](#torchrdit.shapes.ShapeGenerator.from_solver)
    * [generate\_circle\_mask](#torchrdit.shapes.ShapeGenerator.generate_circle_mask)
    * [generate\_rectangle\_mask](#torchrdit.shapes.ShapeGenerator.generate_rectangle_mask)
    * [generate\_polygon\_mask](#torchrdit.shapes.ShapeGenerator.generate_polygon_mask)
    * [combine\_masks](#torchrdit.shapes.ShapeGenerator.combine_masks)

<a id="torchrdit.shapes"></a>

# torchrdit.shapes

<a id="torchrdit.shapes.ShapeGenerator"></a>

## ShapeGenerator Objects

```python
class ShapeGenerator()
```

Class to generate binary shape masks for photonic structures with lattice vector support.

This class provides methods to create various shapes (circles, rectangles, polygons)
as binary masks for use in photonic simulations. It supports both Cartesian and
non-Cartesian coordinate systems through lattice vectors.

**Attributes**:

- `XO` _torch.Tensor_ - Tensor containing the x-coordinates of each point in the grid.
- `YO` _torch.Tensor_ - Tensor containing the y-coordinates of each point in the grid.
- `rdim` _Tuple[int, int]_ - Dimensions of the real-space grid as (height, width).
- `lattice_t1` _torch.Tensor_ - First lattice vector, defaults to [1,0] if not provided.
- `lattice_t2` _torch.Tensor_ - Second lattice vector, defaults to [0,1] if not provided.
- `tcomplex` _torch.dtype_ - Complex tensor data type.
- `tfloat` _torch.dtype_ - Float tensor data type.
- `tint` _torch.dtype_ - Integer tensor data type.
- `nfloat` _torch.dtype_ - Number tensor data type for calculations.
- `is_cartesian` _bool_ - Flag indicating if the coordinate system is Cartesian.
  

**Examples**:

```python
import torch
from torchrdit.shapes import ShapeGenerator
# Create coordinate grids
x = torch.linspace(-1, 1, 128)
y = torch.linspace(-1, 1, 128)
X, Y = torch.meshgrid(x, y, indexing='ij')
# Initialize shape generator
sg = ShapeGenerator(X, Y, (128, 128))
# Generate a circle mask
circle = sg.generate_circle_mask(center=(0.0, 0.0), radius=0.5)
```
  
  Keywords:
  shape, mask, photonics, circle, rectangle, polygon, lattice, binary mask

<a id="torchrdit.shapes.ShapeGenerator.__init__"></a>

#### \_\_init\_\_

```python
def __init__(XO: torch.Tensor,
             YO: torch.Tensor,
             rdim: Tuple[int, int],
             lattice_t1=None,
             lattice_t2=None,
             tcomplex=torch.complex128,
             tfloat=torch.float64,
             tint=torch.int64,
             nfloat=torch.float64)
```

Initialize a shape generator with coordinate grids and lattice vectors.

Creates a new ShapeGenerator instance with the provided coordinate grids and
optional lattice vectors for non-Cartesian coordinate systems.

**Arguments**:

- `XO` _torch.Tensor_ - Tensor containing the x-coordinates of each point in the grid.
- `YO` _torch.Tensor_ - Tensor containing the y-coordinates of each point in the grid.
- `rdim` _Tuple[int, int]_ - Dimensions of the real-space grid as (height, width).
- `lattice_t1` _torch.Tensor, optional_ - First lattice vector. Defaults to [1,0].
- `lattice_t2` _torch.Tensor, optional_ - Second lattice vector. Defaults to [0,1].
- `tcomplex` _torch.dtype, optional_ - Complex tensor data type. Defaults to torch.complex128.
- `tfloat` _torch.dtype, optional_ - Float tensor data type. Defaults to torch.float64.
- `tint` _torch.dtype, optional_ - Integer tensor data type. Defaults to torch.int64.
- `nfloat` _torch.dtype, optional_ - Number type for calculations. Defaults to torch.float64.
  

**Raises**:

- `AssertionError` - If XO and YO are not torch.Tensor objects.
  

**Examples**:

```python
import torch
from torchrdit.shapes import ShapeGenerator
# Create coordinate grids
x = torch.linspace(-1, 1, 128)
y = torch.linspace(-1, 1, 128)
X, Y = torch.meshgrid(x, y, indexing='ij')
# Initialize shape generator with default Cartesian coordinates
sg = ShapeGenerator(X, Y, (128, 128))
```
  
  Keywords:
  initialization, lattice, coordinates, grid, tensor types

<a id="torchrdit.shapes.ShapeGenerator.from_solver"></a>

#### from\_solver

```python
@classmethod
def from_solver(cls, solver)
```

Create a ShapeGenerator from a solver.

This factory method creates a ShapeGenerator instance using the coordinate grids
and lattice vectors from an existing solver object, ensuring consistency between
the solver and shape generator.

**Arguments**:

- `solver` - A solver object containing coordinate grids and lattice vectors.
  

**Returns**:

- `ShapeGenerator` - A new ShapeGenerator initialized with the solver's parameters.
  

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
shape_gen = ShapeGenerator.from_solver(solver)
# Now shape_gen uses the same coordinate system as solver
circle_mask = shape_gen.generate_circle_mask(radius=0.3)
```
  
  Keywords:
  factory, solver, initialization, coordinate system

<a id="torchrdit.shapes.ShapeGenerator.generate_circle_mask"></a>

#### generate\_circle\_mask

```python
def generate_circle_mask(center=None, radius=0.1, soft_edge=0.001)
```

Generate a mask for a circle in Cartesian coordinates.

Creates a binary mask representing a circle with the specified center and radius.
The mask can have hard or soft edges based on the soft_edge parameter.

**Arguments**:

- `center` _Tuple[float, float], optional_ - Center coordinates (x, y) of the circle.
  Defaults to (0.0, 0.0).
- `radius` _float, optional_ - Radius of the circle. Defaults to 0.1.
- `soft_edge` _float, optional_ - Width of the soft transition at the edge.
  Use 0 for a binary hard edge. Defaults to 0.001.
  

**Returns**:

- `torch.Tensor` - A tensor of shape rdim containing the circle mask.
  Values are 1.0 inside the circle and 0.0 outside,
  with a smooth transition at the edge if soft_edge > 0.
  

**Examples**:

```python
import torch
from torchrdit.shapes import ShapeGenerator
# Create coordinate grids
x = torch.linspace(-1, 1, 128)
y = torch.linspace(-1, 1, 128)
X, Y = torch.meshgrid(x, y, indexing='ij')
sg = ShapeGenerator(X, Y, (128, 128))
# Generate a circle mask with hard edges
hard_circle = sg.generate_circle_mask(center=(0.2, -0.3), radius=0.4, soft_edge=0)
# Generate a circle mask with soft edges
soft_circle = sg.generate_circle_mask(center=(0.2, -0.3), radius=0.4, soft_edge=0.02)
```
  
  Keywords:
  circle, mask, binary mask, shape generation, photonics

<a id="torchrdit.shapes.ShapeGenerator.generate_rectangle_mask"></a>

#### generate\_rectangle\_mask

```python
def generate_rectangle_mask(
        center=(0.0, 0.0), width=0.2, height=0.2, angle=0.0, soft_edge=0.001)
```

Generate a mask for a rectangle in Cartesian coordinates.

Creates a binary mask representing a rectangle with the specified center,
dimensions, and orientation. The mask can have hard or soft edges.

**Arguments**:

- `center` _Tuple[float, float], optional_ - Center coordinates (x, y) of the rectangle.
  Defaults to (0.0, 0.0).
- `width` _float, optional_ - Width of the rectangle. Defaults to 0.2.
- `height` _float, optional_ - Height of the rectangle. Defaults to 0.2.
- `angle` _float, optional_ - Rotation angle in degrees. Defaults to 0.0.
- `soft_edge` _float, optional_ - Width of the soft transition at the edge.
  Use 0 for a binary hard edge. Defaults to 0.001.
  

**Returns**:

- `torch.Tensor` - A tensor of shape rdim containing the rectangle mask.
  Values are 1.0 inside the rectangle and 0.0 outside,
  with a smooth transition at the edge if soft_edge > 0.
  

**Examples**:

```python
import torch
from torchrdit.shapes import ShapeGenerator
# Create coordinate grids
x = torch.linspace(-1, 1, 128)
y = torch.linspace(-1, 1, 128)
X, Y = torch.meshgrid(x, y, indexing='ij')
sg = ShapeGenerator(X, Y, (128, 128))
# Generate a rectangle mask
rect = sg.generate_rectangle_mask(width=0.5, height=0.3, angle=45)
# Generate a square mask
square = sg.generate_rectangle_mask(width=0.4, height=0.4, angle=0)
```
  
  Keywords:
  rectangle, square, mask, binary mask, shape generation, photonics, rotation

<a id="torchrdit.shapes.ShapeGenerator.generate_polygon_mask"></a>

#### generate\_polygon\_mask

```python
def generate_polygon_mask(polygon_points,
                          center=None,
                          angle=None,
                          invert=False,
                          soft_edge=0.001)
```

Generate a mask for a polygon in Cartesian coordinates.

Creates a binary mask representing an arbitrary polygon defined by its vertices.
The polygon can be rotated, translated, and have soft or hard edges.

**Arguments**:

- `polygon_points` _List[Tuple] or torch.Tensor_ - List of (x, y) coordinates
  defining the polygon vertices, or a tensor of shape (n, 2) where n is
  the number of vertices.
- `center` _Tuple[float, float], optional_ - Center coordinates (x, y) for the polygon.
  If provided, the polygon will be translated to this position. Defaults to None.
- `angle` _float, optional_ - Rotation angle in degrees. If provided, the polygon
  will be rotated around its center. Defaults to None.
- `invert` _bool, optional_ - If True, inverts the mask (0s inside, 1s outside).
  Defaults to False.
- `soft_edge` _float, optional_ - Width of the soft transition at the edge.
  Use 0 for a binary hard edge. Defaults to 0.001.
  

**Returns**:

- `torch.Tensor` - A tensor of shape rdim containing the polygon mask.
  Values are 1.0 inside the polygon and 0.0 outside (unless inverted),
  with a smooth transition at the edge if soft_edge > 0.
  

**Examples**:

```python
import torch
from torchrdit.shapes import ShapeGenerator
# Create coordinate grids
x = torch.linspace(-1, 1, 128)
y = torch.linspace(-1, 1, 128)
X, Y = torch.meshgrid(x, y, indexing='ij')
sg = ShapeGenerator(X, Y, (128, 128))
# Generate a triangle mask
triangle_points = [(-0.2, -0.2), (0.2, -0.2), (0.0, 0.2)]
triangle = sg.generate_polygon_mask(triangle_points)
# Generate a hexagon mask
import numpy as np
n = 6  # hexagon
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
radius = 0.3
hexagon_points = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]
hexagon = sg.generate_polygon_mask(hexagon_points, center=(0.1, 0.1), angle=30)
```
  
  Keywords:
  polygon, mask, binary mask, shape generation, photonics, arbitrary shape

<a id="torchrdit.shapes.ShapeGenerator.combine_masks"></a>

#### combine\_masks

```python
def combine_masks(mask1, mask2, operation="union")
```

Combine two masks using a specified operation.

Performs boolean operations on two binary masks to create complex shapes.
Supported operations include union, intersection, difference, and subtraction.

**Arguments**:

- `mask1` _torch.Tensor_ - First binary mask tensor.
- `mask2` _torch.Tensor_ - Second binary mask tensor.
- `operation` _str, optional_ - The operation to perform. Options are:
  - "union": Logical OR (max) of the masks
  - "intersection": Logical AND (min) of the masks
  - "difference": Absolute difference between masks
  - "subtract": Remove mask2 from mask1
  Defaults to "union".
  

**Returns**:

- `torch.Tensor` - The combined mask resulting from the specified operation.
  

**Raises**:

- `ValueError` - If an invalid operation is specified.
  

**Examples**:

```python
import torch
from torchrdit.shapes import ShapeGenerator
# Create coordinate grids
x = torch.linspace(-1, 1, 128)
y = torch.linspace(-1, 1, 128)
X, Y = torch.meshgrid(x, y, indexing='ij')
sg = ShapeGenerator(X, Y, (128, 128))
# Create two circular masks
circle1 = sg.generate_circle_mask(center=(-0.1, 0), radius=0.3)
circle2 = sg.generate_circle_mask(center=(0.1, 0), radius=0.3)
# Combine masks using different operations
union = sg.combine_masks(circle1, circle2, operation="union")
intersection = sg.combine_masks(circle1, circle2, operation="intersection")
difference = sg.combine_masks(circle1, circle2, operation="difference")
circle1_minus_circle2 = sg.combine_masks(circle1, circle2, operation="subtract")
```
  
  Keywords:
  mask combination, boolean operations, union, intersection, difference, compound shape

