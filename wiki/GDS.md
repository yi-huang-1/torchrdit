# GDS Module

## Overview
The `torchrdit.gds` module provides functionality to export photonic structure masks to GDS (GDSII) files and import masks from GDS JSON vertex data. It enables interoperability with standard photonics design tools and fabrication workflows.

## Key Features
- Export binary masks to GDS format with JSON vertex data
- Import masks from GDS JSON files  
- Support for complex topologies including holes and disconnected regions
- Batch processing for multiple masks
- Both Cartesian and non-Cartesian lattice systems
- Smart coordinate extrapolation for boundary handling
- High fidelity reconstruction (IoU > 0.9)

## Main Functions

### mask_to_gds
```python
mask_to_gds(mask, layout, cell_name, file_path, smooth=0.01, padding=20, 
            connectivity=1, morph_separate=False)
```

Exports binary mask(s) to GDS file(s) with JSON vertices. Handles complex shapes including holes and disconnected regions.

**Parameters:**
- `mask`: Binary mask tensor/array or list for batch export
- `layout`: Tuple of (X, Y) coordinate meshgrid matrices
- `cell_name`: Name for the GDS cell
- `file_path`: Output file path (should end with .gds)
- `smooth`: Smoothing parameter for boundary splines (default: 0.01)
- `padding`: Padding pixels for boundary shapes (default: 20)
- `connectivity`: 1 for 4-connectivity, 2 for 8-connectivity (default: 1)
- `morph_separate`: Apply morphological opening to separate components

### gds_to_mask
```python
gds_to_mask(json_path, shape_generator, soft_edge=0.0)
```

Import mask from GDS JSON vertices. Reconstructs a binary mask from polygon vertices.

**Parameters:**
- `json_path`: Path to JSON file containing vertex data
- `shape_generator`: ShapeGenerator instance for coordinate system
- `soft_edge`: Soft edge parameter for polygon generation (default: 0.0)

### load_gds_vertices
```python
load_gds_vertices(json_path)
```

Load polygon vertices from JSON file created during GDS export.

## Usage Examples

### Basic Export
```python
from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds

# Create shape generator and mask
shape_gen = ShapeGenerator(X, Y, rdim)
mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.5)

# Export to GDS
mask_to_gds(mask, shape_gen.get_layout(), "CIRCLE", "output.gds")
```

### Complex Shapes with Holes
```python
# For shapes with holes, use higher smoothing
mask_to_gds(ring_mask, shape_gen.get_layout(), "RING", "ring.gds", smooth=0.1)
```

### Batch Processing
```python
# Export multiple masks
masks = [mask1, mask2, mask3]
paths = mask_to_gds(masks, shape_gen.get_layout(), "BATCH", "batch.gds")
# Creates: batch_0.gds, batch_1.gds, batch_2.gds
```

### Import and Reconstruction
```python
from torchrdit.gds import gds_to_mask

# Import mask from GDS JSON
reconstructed = gds_to_mask("output.json", shape_gen)

# Use soft edge for smoother boundaries
smooth_mask = gds_to_mask("output.json", shape_gen, soft_edge=0.001)
```

## Parameter Guidelines

### Smoothing Parameter
- Simple shapes (circles, rectangles): `smooth=0.001-0.005`
- Complex shapes with holes: `smooth=0.1` or higher
- Sharp corners needed: `smooth=0` (no smoothing)

### Padding
- Shapes fully within bounds: `padding=0`
- Shapes near/crossing boundaries: `padding=20-40`

### Connectivity
- `connectivity=1`: 4-connectivity (better separation)
- `connectivity=2`: 8-connectivity (preserves thin connections)

## Implementation Details

The GDS export process:
1. Converts mask to binary (threshold at 0.5)
2. Applies padding if specified
3. Identifies connected components and holes
4. Extracts contours using skimage
5. Smooths boundaries using spline interpolation
6. Exports to GDS with gdspy
7. Saves vertices to JSON for reconstruction

The coordinate extrapolation handles both:
- **Cartesian grids**: Linear extrapolation based on grid spacing
- **Non-Cartesian grids**: Parametric transformation using lattice vectors

## API Reference

Below is the complete API reference for the gds module, automatically generated from the source code.

# Table of Contents

* [torchrdit.gds](#torchrdit.gds)
  * [smoothen\_boundary](#torchrdit.gds.smoothen_boundary)
  * [get\_coord](#torchrdit.gds.get_coord)
  * [gds\_export](#torchrdit.gds.gds_export)
  * [mask\_to\_gds](#torchrdit.gds.mask_to_gds)
  * [load\_gds\_vertices](#torchrdit.gds.load_gds_vertices)
  * [gds\_to\_mask](#torchrdit.gds.gds_to_mask)

<a id="torchrdit.gds"></a>

# torchrdit.gds

GDS (GDSII) file format support for TorchRDIT.

This module provides functionality to export photonic structure masks to GDS files
and import masks from GDS JSON vertex data. It enables interoperability with
standard photonics design tools and fabrication workflows.

The module supports:
- Export of binary masks to GDS files with JSON vertex data
- Import of masks from GDS JSON vertices
- Complex topologies including holes and disconnected regions
- Both Cartesian and non-Cartesian lattice systems
- High fidelity reconstruction (IoU > 0.9)

**Examples**:

```python
from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds, gds_to_mask

# Create shape generator
shape_gen = ShapeGenerator(X, Y, rdim)

# Generate a mask
mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.5)

# Export to GDS
mask_to_gds(mask, shape_gen.get_layout(), "MY_DEVICE", "output.gds")

# Import from GDS
reconstructed = gds_to_mask("output.json", shape_gen)
```
  
  Keywords:
  GDS, GDSII, export, import, mask, photonics, fabrication, vertex, polygon

<a id="torchrdit.gds.smoothen_boundary"></a>

#### smoothen\_boundary

```python
def smoothen_boundary(boundary: List[Tuple[float, float]],
                      s: float = 0.02) -> List[Tuple[float, float]]
```

Smooth boundary points using cubic spline interpolation.

**Arguments**:

- `boundary` - List of (x, y) boundary points.
- `s` - Smoothing parameter for spline (default: 0.02).
  

**Returns**:

  List of smoothed boundary points.

<a id="torchrdit.gds.get_coord"></a>

#### get\_coord

```python
def get_coord(row, col, X0, Y0)
```

Get coordinate for given indices, extrapolating if out of bounds.

Dispatcher that determines coordinate system and delegates to
appropriate handler (_get_cartesian_coord or _get_non_cartesian_coord).

**Arguments**:

- `row` - Row index (can be float, will be converted to int)
- `col` - Column index (can be float, will be converted to int)
- `X0` - X coordinate grid
- `Y0` - Y coordinate grid
  

**Returns**:

  Tuple of (x, y) coordinates

<a id="torchrdit.gds.gds_export"></a>

#### gds\_export

```python
def gds_export(boundary_list: list,
               layout: tuple,
               cell_name: str,
               file_path: str,
               smooth: float = 0.1)
```

Export boundary list to GDSII file with support for holes.

**Arguments**:

- `boundary_list` - Nested list of boundary points. Each element is a pattern group
  containing [pattern_boundary, hole_boundaries...].
- `Format` - [[[outer_boundary], [hole1], [hole2], ...], ...]
- `layout` - Layout, or meshgrid matrices X and Y of the mask
- `cell_name` - Cell name in GDSII file
- `file_path` - File path of the generated gds file
- `smooth` - Parameter in the UnivariateSpline function controls the smoothness

<a id="torchrdit.gds.mask_to_gds"></a>

#### mask\_to\_gds

```python
def mask_to_gds(mask: Union[torch.Tensor, List[torch.Tensor], np.ndarray,
                            List[np.ndarray]],
                layout: tuple,
                cell_name: str,
                file_path: str,
                smooth: float = 0.01,
                padding: int = 20,
                connectivity: int = 1,
                morph_separate: bool = False) -> Union[str, List[str]]
```

Export mask(s) to GDS file(s) with JSON vertices.

Exports binary mask(s) to GDS format, creating both .gds file(s) and .json
file(s) containing the polygon vertices. Handles complex shapes including holes
and disconnected regions. This function properly handles shapes that extend
beyond image boundaries by using padding and smart coordinate extrapolation.

Supports both single masks and batch processing of multiple masks.

**Arguments**:

- `mask` - Binary mask tensor/array to export, or list of masks for batch export.
  For batch export: list of 2D tensors/arrays, or 3D tensor (batch, height, width).
- `layout` - Tuple of (X, Y) coordinate meshgrid matrices.
- `cell_name` - Name for the GDS cell. For batches, will be appended with index.
- `file_path` - Output file path (should end with .gds). For batches, index will be
  inserted before extension (e.g., "output.gds" -> "output_0.gds").
- `smooth` - Smoothing parameter for boundary splines (default: 0.01).
  Lower values preserve shape fidelity better.
- `padding` - Padding pixels to handle shapes at boundaries (default: 20).
  Set to 0 to disable padding. Recommended values: 20-40 for shapes
  that may extend beyond image boundaries.
- `connectivity` - Connectivity for labeling (1 for 4-connectivity, 2 for 8-connectivity).
  Default is 1 (4-connectivity) which better separates touching components.
- `morph_separate` - If True, apply morphological opening to separate touching components.
  Useful for masks where components are connected by thin bridges.
  

**Returns**:

  For single mask: string path to the created GDS file
  For batch: list of string paths to the created GDS files
  

**Notes**:

  - Creates two files per mask: .gds (GDS format) and .json (vertex data)
  - Uses 4-connectivity for component separation by default
  - Handles shapes extending beyond image boundaries
  - For complex shapes with holes, use higher smooth values (>= 0.1)
  - Supports both Cartesian and non-Cartesian lattice systems
  

**Examples**:

```python
from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds

# Create shape generator and mask
shape_gen = ShapeGenerator(X, Y, rdim)
mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.5)

# Export to GDS
mask_to_gds(mask, shape_gen.get_layout(), "CIRCLE", "output.gds")

# For shapes with holes, use higher smoothing
mask_to_gds(ring_mask, shape_gen.get_layout(), "RING", "ring.gds", smooth=0.1)

# Direct usage with coordinate grids
X, Y = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256))
mask_to_gds(mask, (X, Y), "DEVICE", "device.gds")
```
  
  Keywords:
  export, GDS, mask, GDSII, fabrication, polygon, vertices

<a id="torchrdit.gds.load_gds_vertices"></a>

#### load\_gds\_vertices

```python
def load_gds_vertices(json_path: str) -> List[List[Tuple[float, float]]]
```

Load polygon vertices from JSON file.

Reads the JSON file created during GDS export that contains the polygon
vertices for all boundaries in the mask.

**Arguments**:

- `json_path` - Path to JSON file containing vertex data.
  

**Returns**:

  List of polygon vertex lists. The structure depends on the complexity:
  - Simple shapes: List of boundaries
  - Complex shapes: Nested list with pattern groups
  

**Raises**:

- `FileNotFoundError` - If the JSON file does not exist.
- `json.JSONDecodeError` - If the JSON file is invalid.
  

**Examples**:

```python
# Load vertices from JSON file
vertices = load_gds_vertices("device.json")

# For simple shapes
print(f"Number of boundaries: {len(vertices)}")

# For complex shapes with holes
for group in vertices:
    print(f"Pattern group with {len(group)} boundaries")
```
  
  Keywords:
  load, JSON, vertices, polygon, GDS, import

<a id="torchrdit.gds.gds_to_mask"></a>

#### gds\_to\_mask

```python
def gds_to_mask(json_path: str,
                shape_generator,
                soft_edge: float = 0.0) -> torch.Tensor
```

Import mask from GDS JSON vertices.

Reconstructs a binary mask from the polygon vertices stored in the JSON file
created during GDS export. Supports complex topologies with holes and
disconnected regions.

**Arguments**:

- `json_path` - Path to JSON file containing vertex data.
- `shape_generator` - ShapeGenerator instance for coordinate system.
- `soft_edge` - Soft edge parameter for polygon generation (default: 0.0).
  

**Returns**:

  Reconstructed mask as torch.Tensor with same dimensions as the
  original mask.
  

**Raises**:

- `FileNotFoundError` - If the JSON file does not exist.
- `json.JSONDecodeError` - If the JSON file is invalid.
  

**Notes**:

  The reconstruction process handles nested JSON structures:
  - Pattern groups are processed sequentially
  - First boundary in each group is the outer boundary (added)
  - Subsequent boundaries are holes (subtracted)
  

**Examples**:

```python
from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import gds_to_mask

# Create shape generator
shape_gen = ShapeGenerator(X, Y, rdim)

# Import mask from GDS JSON
mask = gds_to_mask("device.json", shape_gen)

# Use soft edge for smoother boundaries
smooth_mask = gds_to_mask("device.json", shape_gen, soft_edge=0.001)
```
  
  Keywords:
  import, GDS, mask, reconstruction, JSON, vertices, polygon

