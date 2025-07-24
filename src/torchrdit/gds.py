"""GDS (GDSII) file format support for TorchRDIT.

This module provides functionality to export photonic structure masks to GDS files
and import masks from GDS JSON vertex data. It enables interoperability with
standard photonics design tools and fabrication workflows.

The module supports:
- Export of binary masks to GDS files with JSON vertex data
- Import of masks from GDS JSON vertices
- Complex topologies including holes and disconnected regions
- Both Cartesian and non-Cartesian lattice systems
- High fidelity reconstruction (IoU > 0.9)

Examples:
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
"""

import json
import warnings
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from scipy.interpolate import UnivariateSpline
from .gds_utils import convert_mask_to_numpy


def smoothen_boundary(boundary: List[Tuple[float, float]], s: float = 0.02) -> List[Tuple[float, float]]:
    """Smooth boundary points using cubic spline interpolation.
    
    Args:
        boundary: List of (x, y) boundary points.
        s: Smoothing parameter for spline (default: 0.02).
        
    Returns:
        List of smoothed boundary points.
    """
    # Remove consecutive duplicate points
    unique_boundary = [boundary[0]]
    for i in range(1, len(boundary)):
        if boundary[i] != boundary[i-1]:
            unique_boundary.append(boundary[i])
    
    # If we removed too many points, return original boundary
    if len(unique_boundary) < 3:
        return boundary
    
    boundary = unique_boundary
    
    # Compute the arclength of the boundary
    dx = np.diff([p[0] for p in boundary])
    dy = np.diff([p[1] for p in boundary])
    dt = np.sqrt(dx**2 + dy**2)
    u = np.zeros(len(boundary))
    u[1:] = np.cumsum(dt)
    
    # Check for duplicate u values and add small epsilon if needed
    if len(np.unique(u)) < len(u):
        # Add small increments to ensure u is strictly increasing
        epsilon = 1e-10
        for i in range(1, len(u)):
            if u[i] <= u[i-1]:
                u[i] = u[i-1] + epsilon

    try:
        # Interpolate using cubic spline
        # Suppress the specific warning about maxiter for small smoothing values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", 
                                  message="The maximal number of iterations maxit.*allowed for finding a smoothing spline",
                                  category=UserWarning)
            fx = UnivariateSpline(u, [p[0] for p in boundary], s=s)
            fy = UnivariateSpline(u, [p[1] for p in boundary], s=s)
            
            # Note: For small smoothing values (s < 0.01), the spline may not fully converge
            # within the iteration limit, but still provides a usable approximation
            if s < 0.01:
                # This is expected behavior - the approximation is still valid
                pass

        # Generate dense boundary points
        dense_u = np.linspace(0, u[-1], len(boundary) * 10)  # 10x denser
        dense_x = fx(dense_u)
        dense_y = fy(dense_u)
        
        # Check for NaN values
        if np.any(np.isnan(dense_x)) or np.any(np.isnan(dense_y)):
            print(f"Warning: NaN values detected in spline interpolation with smoothing parameter s={s}")
            print("  This may indicate the boundary is too complex for the current smoothing value.")
            # Try with a larger smoothing parameter
            if s < 0.1:
                print("  Retrying with s=0.1 for better stability...")
                return smoothen_boundary(boundary, s=0.1)
            else:
                # Return original boundary if we can't smooth it
                print("  Returning original boundary without smoothing.")
                return boundary

        return list(zip(dense_x, dense_y))
    
    except Exception as e:
        print(f"Error in boundary smoothing: {e}")
        print(f"  Boundary has {len(boundary)} points, smoothing parameter s={s}")
        print(f"  Arc length parameterization resulted in {len(np.unique(u))} unique values")
        print("  Consider using a larger smoothing parameter or checking boundary geometry")
        # Return original boundary if smoothing fails
        return boundary


def _get_cartesian_coord(row, col, X0, Y0):
    """Handle Cartesian coordinate extraction with bounds checking.
    
    For Cartesian grids, extrapolation is simple linear extension
    based on the regular grid spacing.
    
    Args:
        row: Row index (will be converted to int)
        col: Column index (will be converted to int) 
        X0: X coordinate grid
        Y0: Y coordinate grid
        
    Returns:
        Tuple of (x, y) coordinates
    """
    # Ensure indices are integers
    row = int(row)
    col = int(col)
    
    max_row, max_col = X0.shape[0] - 1, X0.shape[1] - 1
    
    if 0 <= row <= max_row and 0 <= col <= max_col:
        # In bounds - use direct lookup
        return X0[row, col].item(), Y0[row, col].item()
    
    # Out of bounds - extrapolate based on grid spacing
    if X0.shape[0] > 1 and X0.shape[1] > 1:
        dx_col = (X0[0, 1] - X0[0, 0]).item()  # X spacing along columns
        dy_row = (Y0[1, 0] - Y0[0, 0]).item()  # Y spacing along rows
        
        x_coord = X0[0, 0].item() + col * dx_col
        y_coord = Y0[0, 0].item() + row * dy_row
    else:
        # Edge case: very small grid
        x_coord = X0[0, 0].item()
        y_coord = Y0[0, 0].item()
        
    return x_coord, y_coord


def _get_non_cartesian_coord(row, col, X0, Y0):
    """Handle non-Cartesian (parametric) coordinate extraction.
    
    For non-Cartesian grids (e.g., hexagonal), uses parametric
    transformation based on lattice vectors.
    
    Args:
        row: Row index (will be converted to int)
        col: Column index (will be converted to int)
        X0: X coordinate grid 
        Y0: Y coordinate grid
        
    Returns:
        Tuple of (x, y) coordinates
    """
    # Ensure indices are integers
    row = int(row)
    col = int(col)
    
    max_row, max_col = X0.shape[0] - 1, X0.shape[1] - 1
    
    if 0 <= row <= max_row and 0 <= col <= max_col:
        # In bounds - use direct lookup
        return X0[row, col].item(), Y0[row, col].item()
    
    # Out of bounds - use parametric transformation
    n_rows, n_cols = X0.shape
    
    if n_rows > 1 and n_cols > 1:
        # Parametric coordinates typically range from -0.5 to 0.5
        p = -0.5 + 1.0 * row / (n_rows - 1)
        q = -0.5 + 1.0 * col / (n_cols - 1)
        
        # Reconstruct lattice vectors from the grid
        # t1 direction: from (0,0) to (0,1) in index space
        t1_x = X0[0, min(1, max_col)].item() - X0[0, 0].item()
        t1_y = Y0[0, min(1, max_col)].item() - Y0[0, 0].item()
        
        # t2 direction: from (0,0) to (1,0) in index space  
        t2_x = X0[min(1, max_row), 0].item() - X0[0, 0].item()
        t2_y = Y0[min(1, max_row), 0].item() - Y0[0, 0].item()
        
        # Scale by number of steps
        t1_x *= (n_cols - 1)
        t1_y *= (n_cols - 1)
        t2_x *= (n_rows - 1)
        t2_y *= (n_rows - 1)
        
        # Calculate coordinates using lattice transformation
        x_coord = p * t2_x + q * t1_x
        y_coord = p * t2_y + q * t1_y
    else:
        # Edge case: very small grid
        x_coord = X0[0, 0].item()
        y_coord = Y0[0, 0].item()
        
    return x_coord, y_coord


def get_coord(row, col, X0, Y0):
    """Get coordinate for given indices, extrapolating if out of bounds.
    
    Dispatcher that determines coordinate system and delegates to
    appropriate handler (_get_cartesian_coord or _get_non_cartesian_coord).
    
    Args:
        row: Row index (can be float, will be converted to int)
        col: Column index (can be float, will be converted to int)
        X0: X coordinate grid
        Y0: Y coordinate grid
        
    Returns:
        Tuple of (x, y) coordinates
    """
    # Check if this is a Cartesian grid
    if X0.shape[0] > 1 and X0.shape[1] > 1:
        # Check grid regularity by comparing spacings
        dx_row = abs((X0[1, 0] - X0[0, 0]).item())  # X spacing along rows
        dy_col = abs((Y0[0, 1] - Y0[0, 0]).item())  # Y spacing along columns
        
        # For Cartesian grid: dx_row ≈ 0 and dy_col ≈ 0
        is_cartesian = dx_row < 1e-6 and dy_col < 1e-6
        
        if is_cartesian:
            return _get_cartesian_coord(row, col, X0, Y0)
        else:
            return _get_non_cartesian_coord(row, col, X0, Y0)
    else:
        # For very small grids, use Cartesian approach
        return _get_cartesian_coord(row, col, X0, Y0)


def gds_export(boundary_list: list, layout: tuple, cell_name: str, file_path: Union[str, Path], smooth: float = 0.1):
    """Export boundary list to GDSII file with support for holes.

    Args:
        boundary_list: Nested list of boundary points. Each element is a pattern group
                      containing [pattern_boundary, hole_boundaries...]. 
                      Format: [[[outer_boundary], [hole1], [hole2], ...], ...]
        layout: Layout, or meshgrid matrices X and Y of the mask
        cell_name: Cell name in GDSII file
        file_path: File path of the generated gds file (str or pathlib.Path)
        smooth: Parameter in the UnivariateSpline function controls the smoothness
    """
    import gdstk

    # Convert to Path object
    file_path = Path(file_path)

    # Initialize GDSII Library
    gds_lib = gdstk.Library()

    X0, Y0 = layout

    # Create a new cell
    cell = gds_lib.new_cell(cell_name)
    
    coord_list = []

    # Process each pattern group
    for sub_boundary_list in boundary_list:
        # Each sub_boundary_list contains [pattern_boundary, hole_boundaries...]
        # First boundary is the pattern, rest are holes
        
        sub_coord_list = []
        
        for boundary_idx, bond in enumerate(sub_boundary_list):
            sub_coords = []
            
            # Smooth the boundary if requested
            if smooth > 0:
                bond = smoothen_boundary(bond, s=smooth)
            
            # Convert boundary points to coordinates
            for item in bond:
                # Get coordinate, handling out-of-bounds indices
                x_coord, y_coord = get_coord(item[0], item[1], X0, Y0)
                sub_coords.append((x_coord, y_coord))
            
            sub_coord_list.append(sub_coords)
        
        coord_list.append(sub_coord_list)
        
        # Create GDS polygon using Boolean operations for holes
        if len(sub_coord_list) > 0 and len(sub_coord_list[0]) > 2:
            # Create the main pattern polygon
            polygon = gdstk.Polygon(sub_coord_list[0])
            
            # Subtract holes using Boolean operations
            if len(sub_coord_list) > 1:
                holes = []
                for hole_coords in sub_coord_list[1:]:
                    if len(hole_coords) > 2:
                        hole = gdstk.Polygon(hole_coords)
                        holes.append(hole)
                
                if holes:
                    # Subtract all holes from the pattern
                    result = gdstk.boolean(polygon, holes, "not")
                    if result:
                        # gdstk returns a list of polygons
                        if isinstance(result, list):
                            for poly in result:
                                cell.add(poly)
                        else:
                            cell.add(result)
                else:
                    cell.add(polygon)
            else:
                cell.add(polygon)

    # Save the GDS file (gdstk accepts string path)
    gds_lib.write_gds(str(file_path))

    # Save JSON coordinates
    json_file_path = file_path.with_suffix(".json")
    with json_file_path.open("w") as jsonfile:
        json.dump(coord_list, jsonfile)


def mask_to_gds(
    mask: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]], 
    layout: tuple, cell_name: str, file_path: Union[str, Path], 
    smooth: float = 0.01, padding: int = 20, connectivity: int = 1, 
    morph_separate: bool = False
) -> Union[str, List[str]]:
    """Export mask(s) to GDS file(s) with JSON vertices.
    
    Exports binary mask(s) to GDS format, creating both .gds file(s) and .json
    file(s) containing the polygon vertices. Handles complex shapes including holes
    and disconnected regions. This function properly handles shapes that extend 
    beyond image boundaries by using padding and smart coordinate extrapolation.
    
    Supports both single masks and batch processing of multiple masks.
    
    Args:
        mask: Binary mask tensor/array to export, or list of masks for batch export.
              For batch export: list of 2D tensors/arrays, or 3D tensor (batch, height, width).
        layout: Tuple of (X, Y) coordinate meshgrid matrices.
        cell_name: Name for the GDS cell. For batches, will be appended with index.
        file_path: Output file path (str or pathlib.Path, should end with .gds). 
                  For batches, index will be inserted before extension 
                  (e.g., "output.gds" -> "output_0.gds").
        smooth: Smoothing parameter for boundary splines (default: 0.01).
                Lower values preserve shape fidelity better.
        padding: Padding pixels to handle shapes at boundaries (default: 20).
                Set to 0 to disable padding. Recommended values: 20-40 for shapes
                that may extend beyond image boundaries.
        connectivity: Connectivity for labeling (1 for 4-connectivity, 2 for 8-connectivity).
                     Default is 1 (4-connectivity) which better separates touching components.
        morph_separate: If True, apply morphological opening to separate touching components.
                       Useful for masks where components are connected by thin bridges.
                       
    Returns:
        For single mask: string path to the created GDS file
        For batch: list of string paths to the created GDS files
                 
    Notes:
        - Creates two files per mask: .gds (GDS format) and .json (vertex data)
        - Uses 4-connectivity for component separation by default
        - Handles shapes extending beyond image boundaries
        - For complex shapes with holes, use higher smooth values (>= 0.1)
        - Supports both Cartesian and non-Cartesian lattice systems
        
    Examples:
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
    """
    from skimage import measure
    from scipy import ndimage
    
    # Handle batch processing
    if isinstance(mask, list):
        # List of masks - process each one
        results = []
        for i, single_mask in enumerate(mask):
            # Modify file path to include index
            file_path_obj = Path(file_path)
            base = file_path_obj.stem
            ext = file_path_obj.suffix or '.gds'
            indexed_path = file_path_obj.parent / f"{base}_{i}{ext}"
            indexed_cell = f"{cell_name}_{i}"
            
            # Recursive call for single mask
            result = mask_to_gds(
                single_mask, layout, indexed_cell, indexed_path,
                smooth=smooth, padding=padding, connectivity=connectivity,
                morph_separate=morph_separate
            )
            results.append(result)
        return results
    
    # Handle 3D tensor (batch dimension)
    if isinstance(mask, (torch.Tensor, np.ndarray)) and mask.ndim == 3:
        # Convert to list of 2D masks
        mask_list = []
        for i in range(mask.shape[0]):
            mask_list.append(mask[i])
        
        # Process as batch
        return mask_to_gds(
            mask_list, layout, cell_name, file_path,
            smooth=smooth, padding=padding, connectivity=connectivity,
            morph_separate=morph_separate
        )
    
    # Single mask processing continues below
    # Convert to Path object
    file_path = Path(file_path).expanduser()
    
    # If relative path without directory, make it relative to current directory
    if not file_path.is_absolute() and len(file_path.parts) == 1:
        file_path = Path.cwd() / file_path
    
    # Handle directory input
    if file_path.is_dir():
        dir_path = file_path
        file_path = dir_path / "top.gds"
    else:
        dir_path = file_path.parent
    
    # Ensure .gds extension
    if file_path.suffix != ".gds":
        file_path = file_path.with_suffix(".gds")
    
    # Create directory if it doesn't exist
    dir_path.mkdir(parents=True, exist_ok=True)

    boundary_list = []
    # Convert mask to numpy array without unnecessary copying
    binary_matrix = convert_mask_to_numpy(mask)

    # Binarize the matrix
    binary_matrix = (binary_matrix >= 0.5).astype(np.uint8)
    
    # Add padding to handle boundary clipping
    if padding > 0:
        binary_matrix = np.pad(binary_matrix, padding, mode='constant', constant_values=0)

    # Apply morphological separation if requested
    if morph_separate:
        # Use morphological opening to break thin connections
        binary_matrix = ndimage.binary_opening(binary_matrix, iterations=1).astype(np.uint8)

    # Fill all holes to identify them later
    filled_matrix = ndimage.binary_fill_holes(binary_matrix).astype(np.uint8)
    
    # Find holes by subtracting original from filled
    holes_matrix = filled_matrix - binary_matrix
    
    # Label connected components in the original mask with specified connectivity
    labeled_patterns = measure.label(binary_matrix, connectivity=connectivity)
    pattern_regions = measure.regionprops(labeled_patterns)
    
    # Label holes with same connectivity
    labeled_holes = measure.label(holes_matrix, connectivity=connectivity)
    hole_regions = measure.regionprops(labeled_holes)
    
    # Process each pattern region
    for pattern_region in pattern_regions:
        sub_boundary_list = []
        
        # Create a mask for this specific pattern
        pattern_mask = (labeled_patterns == pattern_region.label).astype(np.float64)
        
        # Find the outer contour of the pattern
        pattern_contours = measure.find_contours(pattern_mask, 0.5)
        
        if len(pattern_contours) == 0:
            continue
            
        # Take the longest contour as the outer boundary
        outer_contour = max(pattern_contours, key=len)
        # Convert to integer indices, ensuring they're within bounds
        max_row, max_col = binary_matrix.shape[0] - 1, binary_matrix.shape[1] - 1
        
        # Adjust for padding if applied
        if padding > 0:
            # Subtract padding offset to get back to original coordinate system
            pattern_bond = []
            for p in outer_contour:
                # Adjust coordinates by subtracting padding
                row = int(round(p[0])) - padding
                col = int(round(p[1])) - padding
                pattern_bond.append((row, col))
        else:
            pattern_bond = [(min(max(int(round(p[0])), 0), max_row),   # row
                            min(max(int(round(p[1])), 0), max_col))    # col
                           for p in outer_contour]
        
        if len(pattern_bond) > 2:  # Need at least 3 points for a valid polygon
            sub_boundary_list.append(pattern_bond)
        
        # Find holes that belong to this pattern
        pattern_bbox = pattern_region.bbox
        for hole_region in hole_regions:
            hole_bbox = hole_region.bbox
            # Check if hole is within pattern's bounding box
            if (hole_bbox[0] >= pattern_bbox[0] and hole_bbox[1] >= pattern_bbox[1] and
                hole_bbox[2] <= pattern_bbox[2] and hole_bbox[3] <= pattern_bbox[3]):
                
                # Create mask for this hole
                hole_mask = (labeled_holes == hole_region.label).astype(np.float64)
                hole_contours = measure.find_contours(hole_mask, 0.5)
                
                if len(hole_contours) > 0:
                    # Take the longest contour as the hole boundary
                    hole_contour = max(hole_contours, key=len)
                    
                    # Adjust for padding if applied
                    if padding > 0:
                        # Subtract padding offset to get back to original coordinate system
                        hole_bond = []
                        for p in hole_contour:
                            # Adjust coordinates by subtracting padding
                            row = int(round(p[0])) - padding
                            col = int(round(p[1])) - padding
                            hole_bond.append((row, col))
                    else:
                        hole_bond = [(min(max(int(round(p[0])), 0), max_row),   # row
                                     min(max(int(round(p[1])), 0), max_col))    # col
                                    for p in hole_contour]
                    
                    if len(hole_bond) > 2:  # Need at least 3 points for a valid polygon
                        sub_boundary_list.append(hole_bond)
        
        if len(sub_boundary_list) > 0:
            boundary_list.append(sub_boundary_list)

    gds_export(
        boundary_list=boundary_list,
        layout=layout,
        cell_name=cell_name,
        file_path=file_path,
        smooth=smooth,
    )
    
    # Return the file path as string for single mask
    return str(file_path)


def load_gds_vertices(json_path: Union[str, Path]) -> List[List[Tuple[float, float]]]:
    """Load polygon vertices from JSON file.
    
    Reads the JSON file created during GDS export that contains the polygon
    vertices for all boundaries in the mask.
    
    Args:
        json_path: Path to JSON file containing vertex data (str or pathlib.Path).
        
    Returns:
        List of polygon vertex lists. The structure depends on the complexity:
        - Simple shapes: List of boundaries
        - Complex shapes: Nested list with pattern groups
        
    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
        
    Examples:
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
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
        
    with json_path.open('r') as f:
        data = json.load(f)
        
    return data


def gds_to_mask(json_path: Union[str, Path], shape_generator, soft_edge: float = 0.0) -> torch.Tensor:
    """Import mask from GDS JSON vertices.
    
    Reconstructs a binary mask from the polygon vertices stored in the JSON file
    created during GDS export. Supports complex topologies with holes and
    disconnected regions.
    
    Args:
        json_path: Path to JSON file containing vertex data (str or pathlib.Path).
        shape_generator: ShapeGenerator instance for coordinate system.
        soft_edge: Soft edge parameter for polygon generation (default: 0.0).
        
    Returns:
        Reconstructed mask as torch.Tensor with same dimensions as the
        original mask.
        
    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
        
    Notes:
        The reconstruction process handles nested JSON structures:
        - Pattern groups are processed sequentially
        - First boundary in each group is the outer boundary (added)
        - Subsequent boundaries are holes (subtracted)
        
    Examples:
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
    """
    # Load vertices from JSON file
    vertices_list = load_gds_vertices(json_path)
    
    # Start with empty mask
    reconstructed = torch.zeros(
        shape_generator.rdim, 
        dtype=shape_generator.tfloat, 
        device=shape_generator.XO.device
    )
    
    # Check if this is nested format (from gds_export) or flat format
    if len(vertices_list) > 0 and isinstance(vertices_list[0], list) and len(vertices_list[0]) > 0:
        # Check if it's a list of coordinates (flat) or list of boundaries (nested)
        first_element = vertices_list[0][0]
        if isinstance(first_element, (tuple, list)) and len(first_element) == 2:
            # This is a flat list of boundaries - wrap each in a list for compatibility
            vertices_list = [[boundary] for boundary in vertices_list]
    
    # Process each pattern group
    for pattern_group in vertices_list:
        if isinstance(pattern_group, list) and len(pattern_group) > 0:
            # For each pattern group, first boundary is the outer boundary,
            # subsequent boundaries are holes
            for boundary_idx, boundary in enumerate(pattern_group):
                if len(boundary) > 2:  # Need at least 3 vertices for a polygon
                    poly_mask = shape_generator.generate_polygon_mask(
                        polygon_points=boundary,
                        soft_edge=soft_edge
                    )
                    
                    if boundary_idx == 0:
                        # First boundary in group - add to mask using maximum
                        reconstructed = torch.maximum(reconstructed, poly_mask)
                    else:
                        # Subsequent boundaries are holes - subtract from mask
                        reconstructed = reconstructed * (1 - poly_mask)
    
    return reconstructed


