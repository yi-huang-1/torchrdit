"""
# Example - GDS Export/Import for Photonic Structures

This example demonstrates the GDS (GDSII) export and import functionality in TorchRDIT.
The GDS format is the industry standard for photonics and semiconductor fabrication,
enabling seamless integration with fabrication workflows.

This example covers:
1. Basic GDS export for simple shapes
2. Complex topology with holes and islands
3. Batch processing of multiple masks
4. Import and reconstruction from GDS files
5. Parameter optimization for different geometries

The GDS functionality supports:
- Complex geometries with holes and disconnected regions
- Both Cartesian and non-Cartesian lattice systems
- High-fidelity reconstruction (IoU > 0.9)
- Shapes extending beyond image boundaries
- Batch processing for design variations

Keywords: GDS, GDSII, export, import, fabrication, mask, photonics, complex topology, holes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds, gds_to_mask

# Create output directory for GDS files
output_dir = Path(__file__).parent / "gds_examples"
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Part 1: Basic Setup and Simple Shapes
# =============================================================================

print("GDS Export/Import Example")
print("=" * 60)

# Define units (all dimensions in micrometers)
um = 1.0
nm = 1e-3 * um

# Create coordinate grid
rdim = [256, 256]  # Resolution
size = 2.0 * um    # Physical size

# Cartesian coordinate system
X, Y = torch.meshgrid(
    torch.linspace(-size/2, size/2, rdim[0]),
    torch.linspace(-size/2, size/2, rdim[1]),
    indexing='xy'
)

# Initialize shape generator
shape_gen = ShapeGenerator(X, Y, rdim)

print("\n1. Simple Shapes Export")
print("-" * 40)

# Example 1.1: Circle
# A simple circular pattern - the most basic photonic structure
circle_mask = shape_gen.generate_circle_mask(
    center=(0.0, 0.0),      # Center at origin
    radius=0.5 * um,        # 500 nm radius
    soft_edge=0.001         # Small soft edge for smooth boundaries
)

# Export to GDS
# The mask_to_gds function creates both .gds and .json files
# - .gds file: Standard GDSII format readable by CAD tools
# - .json file: Polygon vertices for reconstruction
circle_gds_path = output_dir / "circle.gds"
mask_to_gds(
    mask=circle_mask,
    layout=shape_gen.get_layout(),  # Coordinate system information
    cell_name="CIRCLE_500NM",       # Cell name in GDS file
    file_path=circle_gds_path,
    smooth=0.001,                   # Low smoothing preserves circle shape
    padding=20                      # Handle shapes near boundaries
)
print(f"✓ Exported circle to {circle_gds_path}")

# Example 1.2: Rectangle with rotation
# Rotated rectangles are common in photonic waveguides and couplers
rect_mask = shape_gen.generate_rectangle_mask(
    center=(0.2 * um, 0.1 * um),    # Offset from center
    width=0.8 * um,
    height=0.4 * um,
    angle=30.0,                     # 30 degrees rotation
    soft_edge=0.001
)

rect_gds_path = output_dir / "rectangle.gds"
mask_to_gds(
    mask=rect_mask,
    layout=shape_gen.get_layout(),
    cell_name="RECT_ROTATED_30DEG",
    file_path=rect_gds_path,
    smooth=0.001                    # Low smoothing for sharp corners
)
print(f"✓ Exported rectangle to {rect_gds_path}")

# =============================================================================
# Part 2: Complex Topology - Rectangle with Holes and Island
# =============================================================================

print("\n2. Complex Topology Export")
print("-" * 40)

# This example creates a complex structure that tests all GDS capabilities:
# - A main rectangle (the device body)
# - Two circular holes (could be air holes in a photonic crystal)
# - One hole contains a triangular island (nested structure)

# Initialize mask
complex_mask = torch.zeros(rdim)

# Step 1: Create main rectangle
# This could represent a photonic device or waveguide section
rect_top, rect_bottom = 28, 228
rect_left, rect_right = 53, 203
complex_mask[rect_top:rect_bottom, rect_left:rect_right] = 1

# Step 2: Create Hole 1 (simple circular hole)
# Such holes are common in photonic crystals for bandgap engineering
hole1_center_row, hole1_center_col = 128, 80
hole1_radius = 20

for i in range(rdim[0]):
    for j in range(rdim[1]):
        if rect_top <= i < rect_bottom and rect_left <= j < rect_right:
            dist = np.sqrt((i - hole1_center_row)**2 + (j - hole1_center_col)**2)
            if dist < hole1_radius:
                complex_mask[i, j] = 0

# Step 3: Create Hole 2 (larger hole that will contain an island)
# This demonstrates support for nested structures
hole2_center_row, hole2_center_col = 128, 150
hole2_radius = 35

for i in range(rdim[0]):
    for j in range(rdim[1]):
        if rect_top <= i < rect_bottom and rect_left <= j < rect_right:
            dist = np.sqrt((i - hole2_center_row)**2 + (j - hole2_center_col)**2)
            if dist < hole2_radius:
                complex_mask[i, j] = 0

# Step 4: Add triangular island inside Hole 2
# This could represent a resonator or coupling element
triangle_side = 25
triangle_height = triangle_side * np.sqrt(3) / 2

# Define triangle vertices (equilateral triangle)
triangle_vertices = [
    (hole2_center_row - triangle_height * 2/3, hole2_center_col),
    (hole2_center_row + triangle_height/3, hole2_center_col - triangle_side/2),
    (hole2_center_row + triangle_height/3, hole2_center_col + triangle_side/2)
]

# Helper function to check if point is inside triangle
def point_in_triangle(px, py, v0, v1, v2):
    """Check if point (px, py) is inside triangle defined by vertices v0, v1, v2."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign((px, py), v0, v1)
    d2 = sign((px, py), v1, v2)
    d3 = sign((px, py), v2, v0)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

# Add triangle inside Hole 2
for i in range(rdim[0]):
    for j in range(rdim[1]):
        dist_to_hole2 = np.sqrt((i - hole2_center_row)**2 + (j - hole2_center_col)**2)
        if dist_to_hole2 < hole2_radius:
            if point_in_triangle(i, j, triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]):
                complex_mask[i, j] = 1

# Export complex topology
# For shapes with holes, higher smoothing values prevent numerical issues
complex_gds_path = output_dir / "complex_topology.gds"
mask_to_gds(
    mask=complex_mask,
    layout=shape_gen.get_layout(),
    cell_name="COMPLEX_WITH_HOLES",
    file_path=complex_gds_path,
    smooth=0.1,                     # Higher smoothing for complex shapes
    padding=20,                     # Handle boundary shapes
    connectivity=1,                 # 4-connectivity for better separation
    morph_separate=False            # Don't apply morphological separation
)
print(f"✓ Exported complex topology to {complex_gds_path}")

# =============================================================================
# Part 3: Import and Reconstruction
# =============================================================================

print("\n3. Import and Reconstruction")
print("-" * 40)

# Import the complex topology back from GDS
# The gds_to_mask function reads the JSON file created during export
json_path = complex_gds_path.with_suffix('.json')
reconstructed_complex = gds_to_mask(json_path, shape_gen, soft_edge=0.0)

# Calculate reconstruction quality using IoU (Intersection over Union)
def calculate_iou(mask1, mask2, threshold=0.5):
    """Calculate IoU between two masks."""
    binary1 = (mask1 > threshold).float()
    binary2 = (mask2 > threshold).float()
    
    intersection = (binary1 * binary2).sum()
    union = binary1.sum() + binary2.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()

iou = calculate_iou(complex_mask, reconstructed_complex)
print(f"✓ Reconstruction IoU: {iou:.4f} (>0.9 is excellent)")

# Visualize the comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original mask
im1 = axes[0].imshow(complex_mask.numpy(), cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Original Mask')
axes[0].axis('off')

# Reconstructed mask
im2 = axes[1].imshow(reconstructed_complex.numpy(), cmap='gray', vmin=0, vmax=1)
axes[1].set_title(f'Reconstructed (IoU={iou:.3f})')
axes[1].axis('off')

# Difference
diff = torch.abs(complex_mask - reconstructed_complex)
im3 = axes[2].imshow(diff.numpy(), cmap='hot', vmin=0, vmax=1)
axes[2].set_title('Absolute Difference')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "complex_topology_comparison.png", dpi=150)
plt.close()

# =============================================================================
# Part 4: Batch Processing
# =============================================================================

print("\n4. Batch Processing")
print("-" * 40)

# Create multiple design variations
# This is useful for parameter sweeps or design optimization
batch_masks = []
radii = [0.3, 0.4, 0.5, 0.6]  # Different circle radii in um

for i, radius in enumerate(radii):
    # Create circles with different radii
    mask = shape_gen.generate_circle_mask(
        center=(0.0, 0.0),
        radius=radius * um,
        soft_edge=0.001
    )
    batch_masks.append(mask)

# Batch export - creates multiple GDS files with index suffixes
batch_gds_path = output_dir / "batch_circles.gds"
gds_paths = mask_to_gds(
    mask=batch_masks,                   # List of masks
    layout=shape_gen.get_layout(),
    cell_name="CIRCLE_SWEEP",          # Will be appended with _0, _1, etc.
    file_path=batch_gds_path,          # Will create batch_circles_0.gds, etc.
    smooth=0.001
)

print(f"✓ Exported {len(gds_paths)} masks in batch mode")
for i, path in enumerate(gds_paths):
    print(f"  - Radius {radii[i]} um: {path}")

# =============================================================================
# Part 5: Non-Cartesian Lattice Example
# =============================================================================

print("\n5. Non-Cartesian Lattice")
print("-" * 40)

# Create a hexagonal lattice system
# Common in photonic crystals and metamaterials
a = 1.0 * um  # Lattice constant
lattice_t1 = torch.tensor([a/2, -a*np.sqrt(3)/2])
lattice_t2 = torch.tensor([a/2, a*np.sqrt(3)/2])

# Create parametric coordinates
vec_p = torch.linspace(-0.5, 0.5, rdim[0])
vec_q = torch.linspace(-0.5, 0.5, rdim[1])
mesh_q, mesh_p = torch.meshgrid(vec_q, vec_p, indexing="xy")

# Transform to physical coordinates using lattice vectors
X_hex = mesh_p * lattice_t1[0] + mesh_q * lattice_t2[0]
Y_hex = mesh_p * lattice_t1[1] + mesh_q * lattice_t2[1]

# Create shape generator for hexagonal lattice
shape_gen_hex = ShapeGenerator(X_hex, Y_hex, rdim, lattice_t1=lattice_t1, lattice_t2=lattice_t2)

# Create a pattern on hexagonal lattice
hex_mask = shape_gen_hex.generate_circle_mask(
    center=(0.0, 0.0),
    radius=0.3 * um,
    soft_edge=0.001
)

# Export hexagonal lattice pattern
hex_gds_path = output_dir / "hexagonal_pattern.gds"
mask_to_gds(
    mask=hex_mask,
    layout=shape_gen_hex.get_layout(),
    cell_name="HEX_LATTICE_CIRCLE",
    file_path=hex_gds_path,
    smooth=0.001
)
print(f"✓ Exported hexagonal lattice pattern to {hex_gds_path}")

# =============================================================================
# Part 6: Parameter Recommendations Summary
# =============================================================================

print("\n6. Parameter Recommendations")
print("-" * 40)
print("Smoothing parameter guidelines:")
print("  - Simple shapes (circles, rectangles): smooth=0.001-0.005")
print("  - Complex shapes with holes: smooth=0.1 or higher")
print("  - Sharp corners needed: smooth=0 (no smoothing)")
print("\nPadding guidelines:")
print("  - Shapes fully within bounds: padding=0")
print("  - Shapes near/crossing boundaries: padding=20-40")
print("\nConnectivity options:")
print("  - connectivity=1: 4-connectivity (better separation)")
print("  - connectivity=2: 8-connectivity (preserves thin connections)")

print("\n✓ Example completed successfully!")
print(f"✓ GDS files saved to: {output_dir}/")

# =============================================================================
# Visualization of all created masks
# =============================================================================

# Create a summary figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Plot all the masks we created
masks_to_plot = [
    (circle_mask, "Circle (r=0.5 μm)"),
    (rect_mask, "Rectangle (30° rotation)"),
    (complex_mask, "Complex topology"),
    (reconstructed_complex, f"Reconstructed (IoU={iou:.3f})"),
    (batch_masks[2], "Batch example (r=0.5 μm)"),
    (hex_mask, "Hexagonal lattice")
]

for idx, (mask, title) in enumerate(masks_to_plot):
    im = axes[idx].imshow(mask.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[idx].set_title(title)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "gds_export_summary.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Summary visualization saved to: {output_dir / 'gds_export_summary.png'}")