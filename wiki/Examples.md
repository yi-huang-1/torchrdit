# TorchRDIT Examples

This page contains examples showing how to use TorchRDIT for common electromagnetic simulation tasks. The official repository includes many examples in the `examples/` folder that demonstrate different aspects of the library.

## Example Categories

The examples are organized into several categories:

### Basic Usage Examples
- Basic planar structures
- Patterned layers
- Different API styles (fluent, function-based, and standard builder)

### Simulation Algorithms
- RCWA (Rigorous Coupled-Wave Analysis)
- R-DIT (Rigorous Diffraction Interface Theory) 

### Material Properties
- Homogeneous materials
- Dispersive materials with data loading
- Material permittivity fitting

### Advanced Topics
- Optimization techniques
- Automatic differentiation and gradient calculation
- Parameter tuning

## Running the Examples

To run any of the examples, navigate to the repository root and run:

```bash
python examples/example_gmrf_variable_optimize.py
```

Most examples generate visualization outputs automatically, which are saved to the same directory. The examples use relative imports, so they must be run from the repository root.

## Required Dependencies

The examples require the following dependencies:
- PyTorch
- NumPy
- Matplotlib
- tqdm (for progress bars in optimization examples)

Some examples also require the data files included in the `examples` directory:
- `Si_C-e.txt` - Silicon Carbide permittivity data
- `SiO2-e.txt` - Silicon Dioxide permittivity data

## Key Features Demonstrated

### Different API Styles

#### Standard Builder Pattern
```python
# Initialize the solver using the builder pattern
builder = get_solver_builder()
builder.with_algorithm(Algorithm.RCWA)
builder.with_precision(Precision.DOUBLE)
builder.with_real_dimensions([512, 512])
# ... configure more parameters
builder.add_material(material_sio)
builder.add_layer({
    "material": "SiO",
    "thickness": h1.item(),
    "is_homogeneous": False,
    "is_optimize": True
})
# Build the solver
dev1 = builder.build()
```

#### Fluent Builder Pattern
```python
# Initialize and configure the solver with fluent method chaining
dev1 = (get_solver_builder()
        .with_algorithm(Algorithm.RCWA)
        .with_precision(Precision.DOUBLE)
        .with_real_dimensions([512, 512])
        # ... configure more parameters
        .add_material(material_sio)
        .add_layer({
            "material": "SiO",
            "thickness": h1.item(),
            "is_homogeneous": False,
            "is_optimize": True
        })
        .build())
```

#### Function-Based Pattern
```python
# Define a builder configuration function
def configure_gmrf_solver(builder):
    return (builder
            .with_algorithm(Algorithm.RCWA)
            # ... configure more parameters
            )

# Create the solver using the configuration function
dev1 = create_solver_from_builder(configure_gmrf_solver)
```

### Structure Building

```python
# Using masks to create complex geometries
from torchrdit.shapes import ShapeGenerator

# Create a shape generator
shape_generator = ShapeGenerator.from_solver(device)

# Create a circular mask
c1 = shape_generator.generate_circle_mask(center=[0, b/2], radius=r)
c2 = shape_generator.generate_circle_mask(center=[0, -b/2], radius=r)
c3 = shape_generator.generate_circle_mask(center=[a/2, 0], radius=r)
c4 = shape_generator.generate_circle_mask(center=[-a/2, 0], radius=r)

# Combine masks using boolean operations
mask = shape_generator.combine_masks(mask1=c1, mask2=c2, operation='union')
mask = shape_generator.combine_masks(mask1=mask, mask2=c3, operation='union')
mask = shape_generator.combine_masks(mask1=mask, mask2=c4, operation='union')

# Update permittivity with mask
device.update_er_with_mask(mask=mask, layer_index=0)
```

### Dispersive Materials

```python
# Creating materials with dispersion from data files
material_sic = create_material(
    name='SiC', 
    dielectric_dispersion=True, 
    user_dielectric_file='Si_C-e.txt', 
    data_format='freq-eps', 
    data_unit='thz'
)

# Visualize fitted permittivity
display_fitted_permittivity(device, fig_ax=axes)
```

### Automatic Differentiation

```python
# Enable gradient tracking on the mask
mask.requires_grad = True

# Solve and calculate efficiencies
data = device.solve(src) # SolverResults object

# Compute backward pass for optimization
torch.sum(data.transmission[0]).backward()

# Access gradients
print(f"The gradient with respect to the mask is {torch.mean(mask.grad)}")
```

### Optimization

```python
# Define an objective function
def objective_GMRF(dev, src, radius):
    # ... calculation logic
    return loss

# Optimization loop
for epoch in trange(num_epochs):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    loss = objective_GMRF(dev, src, radius)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
```

These examples demonstrate the key capabilities of TorchRDIT, including differentiable simulation, optimization, and support for complex geometries and materials.

## Available Example Files


### example_gds_export

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

```python
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
import os

from torchrdit.shapes import ShapeGenerator
from torchrdit.gds import mask_to_gds, gds_to_mask, load_gds_vertices

# Create output directory for GDS files
output_dir = "gds_examples"
os.makedirs(output_dir, exist_ok=True)

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
circle_gds_path = os.path.join(output_dir, "circle.gds")
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

rect_gds_path = os.path.join(output_dir, "rectangle.gds")
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
complex_gds_path = os.path.join(output_dir, "complex_topology.gds")
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
json_path = complex_gds_path.replace('.gds', '.json')
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
plt.savefig(os.path.join(output_dir, "complex_topology_comparison.png"), dpi=150)
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
batch_gds_path = os.path.join(output_dir, "batch_circles.gds")
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
hex_gds_path = os.path.join(output_dir, "hexagonal_pattern.gds")
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
print(f"✓ GDS files saved to: {os.path.abspath(output_dir)}/")

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
plt.savefig(os.path.join(output_dir, "gds_export_summary.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Summary visualization saved to: {os.path.join(output_dir, 'gds_export_summary.png')}")
```

### example_gmrf_dispersive

# Example - GMRF with hexagonal unit cells with dispersive materials (Builder Pattern)

This example shows the simulation of the guided-mode resonance filter (GMRF) 
using the builder pattern with dispersive materials.

Keywords:
    dispersive material, permittivity, wavelength dependence, polynomial fitting,
    material characterization, complex permittivity, optical properties, material dispersion, R-DIT, builder

```python
"""
# Example - GMRF with hexagonal unit cells with dispersive materials (Builder Pattern)

This example shows the simulation of the guided-mode resonance filter (GMRF) 
using the builder pattern with dispersive materials.

Keywords:
    dispersive material, permittivity, wavelength dependence, polynomial fitting,
    material characterization, complex permittivity, optical properties, material dispersion, R-DIT, builder
"""

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torchrdit.solver import get_solver_builder
from torchrdit.shapes import ShapeGenerator
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm, Precision
from torchrdit.viz import display_fitted_permittivity, plot_layer

# units, normalizing all units to 'um'
um = 1
nm = 1e-3 * um
degrees = np.pi / 180

# angles of incident waves
theta = 0 * degrees
phi = 0 * degrees

# polarization
pte = 1
ptm = 0

# refractive index
n_SiO = 1.4496
n_SiN = 1.9360
n_fs = 1.5100

# dimensions of the cell
a = 1150 * nm
b = a * np.sqrt(3)

# radius of the holes on the top layer
r = 400 * nm

# thickness of each layer
h1 = torch.tensor(230 * nm, dtype=torch.float64)
h2 = torch.tensor(345 * nm, dtype=torch.float64)

# lattice vectors of the cell
t1 = torch.tensor([[a/2, -a*np.sqrt(3)/2]], dtype=torch.float64)
t2 = torch.tensor([[a/2, a*np.sqrt(3)/2]], dtype=torch.float64)

# Initialize the solver using the builder pattern
torchrdit_sim = (get_solver_builder()
    .with_algorithm(Algorithm.RDIT)
    .with_precision(Precision.DOUBLE)
    .with_real_dimensions([512, 512])
    .with_k_dimensions([21, 21])
    .with_wavelengths(np.array([1540 * nm, 1550 * nm, 1560 * nm, 1570 * nm]))
    .with_length_unit('um')
    .with_lattice_vectors(t1, t2)
    .build())

# creating materials
# all material objects should be added to the device after it's built
material_sic = create_material(name='SiC', dielectric_dispersion=True, 
                             user_dielectric_file=os.path.join(os.path.dirname(__file__), 'Si_C-e.txt'), 
                             data_format='freq-eps', data_unit='thz')
material_sio2 = create_material(name='SiO2', dielectric_dispersion=True, 
                              user_dielectric_file=os.path.join(os.path.dirname(__file__), 'SiO2-e.txt'), 
                              data_format='freq-eps', data_unit='thz')
material_sin = create_material(name='SiN', permittivity=n_SiN**2)
material_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

# Manually add materials to the device
torchrdit_sim.add_materials(material_list=[material_sic, material_sio2, material_sin, material_fs])

# update the material of the transmission layer
torchrdit_sim.update_trn_material(trn_material=material_fs)

# Add layers to the device
torchrdit_sim.add_layer(material_name=material_sic,
                      thickness=h1,
                      is_homogeneous=False,
                      is_optimize=True)

torchrdit_sim.add_layer(material_name=material_sin,
                      thickness=h2,
                      is_homogeneous=True,
                      is_optimize=False)

# build hexagonal unit cell
shape_generator = ShapeGenerator.from_solver(torchrdit_sim)

c1 = shape_generator.generate_circle_mask(center=[0, b/2], radius=r)
c2 = shape_generator.generate_circle_mask(center=[0, -b/2], radius=r)
c3 = shape_generator.generate_circle_mask(center=[a/2, 0], radius=r)
c4 = shape_generator.generate_circle_mask(center=[-a/2, 0], radius=r)

mask = shape_generator.combine_masks(mask1=c1, mask2=c2, operation='union')
mask = shape_generator.combine_masks(mask1=mask, mask2=c3, operation='union')
mask = shape_generator.combine_masks(mask1=mask, mask2=c4, operation='union')

mask = (1 - mask).to(torch.float64)

layer_index = 0
torchrdit_sim.update_er_with_mask(mask=mask, layer_index=layer_index)

# print layer information
torchrdit_sim.get_layer_structure()

# Plot the layer and save the figure
fig, axes = plt.subplots()
plot_layer(torchrdit_sim, layer_index=0, func='real', fig_ax=axes, cmap='BuGn', labels=('x (um)','y (um)'), title='layer 0')
script_dir = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_layer_0.png")
plt.savefig(output_filename, dpi=300)
plt.close(fig)

# Plot fitted permittivity
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
display_fitted_permittivity(torchrdit_sim, fig_ax=axes)
output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_fitted_dispersive.png")
plt.savefig(output_filename, dpi=300)
plt.close(fig)

# Create a source and solve
src = torchrdit_sim.add_source(theta=theta, phi=phi, pte=pte, ptm=ptm)
result = torchrdit_sim.solve(src)

# Print the efficiency at each wavelength
for i in range(len(torchrdit_sim.lam0)):
    print(f"The transmission efficiency at wavelength \t{torchrdit_sim.lam0[i] * 1e3} nm is \t{result.transmission[i] * 100}%")
    print(f"The reflection efficiency at wavelength \t{torchrdit_sim.lam0[i] * 1e3} nm is \t{result.reflection[i] * 100}%")

print("\n===== Demonstrating SolverResults API with Dispersive Materials =====\n")

# Example 1: Get zero-order field components
print("Example 1: Getting zero-order field components")
tx, ty, tz = result.get_zero_order_transmission()
rx, ry, rz = result.get_zero_order_reflection()

print(f"Zero-order transmission field amplitudes (first wavelength):")
tx_amplitude = torch.abs(tx[0])
ty_amplitude = torch.abs(ty[0])
tz_amplitude = torch.abs(tz[0])
print(f"  x-component amplitude: {tx_amplitude.item():.6f}")
print(f"  y-component amplitude: {ty_amplitude.item():.6f}")
print(f"  z-component amplitude: {tz_amplitude.item():.6f}")

# Example 2: Analyzing phase for dispersive materials
print("\nExample 2: Phase analysis across multiple wavelengths")
tx_phase_rad = torch.angle(tx)
ty_phase_rad = torch.angle(ty)

# Convert to degrees
rad_to_deg = 180.0 / np.pi

print("Phase of transmission x-component across wavelengths (in degrees):")
for i in range(len(torchrdit_sim.lam0)):
    wavelength = torchrdit_sim.lam0[i] * 1e3  # in nm
    print(f"  Wavelength {wavelength:.1f} nm: {tx_phase_rad[i].item() * rad_to_deg:.2f}°")

# Example 3: Phase differences vs wavelength
print("\nExample 3: Phase differences between x and y components across wavelengths")
tx_ty_phase_diff = tx_phase_rad - ty_phase_rad

# Normalize phase differences to [-180°, 180°]
for i in range(len(torchrdit_sim.lam0)):
    diff_deg = tx_ty_phase_diff[i].item() * rad_to_deg
    while diff_deg > 180:
        diff_deg -= 360
    while diff_deg < -180:
        diff_deg += 360
    print(f"  Wavelength {torchrdit_sim.lam0[i] * 1e3:.1f} nm: {diff_deg:.2f}°")

# Example 4: Diffraction efficiency vs wavelength
print("\nExample 4: Zero-order diffraction efficiency vs wavelength")
zero_order_t = result.get_order_transmission_efficiency(0, 0)
zero_order_r = result.get_order_reflection_efficiency(0, 0)

print("Zero-order efficiencies for each wavelength:")
for i in range(len(torchrdit_sim.lam0)):
    wavelength = torchrdit_sim.lam0[i] * 1e3  # in nm
    print(f"  Wavelength {wavelength:.1f} nm:")
    print(f"    Transmission: {zero_order_t[i].item() * 100:.4f}%")
    print(f"    Reflection: {zero_order_r[i].item() * 100:.4f}%")
    print(f"    Sum: {(zero_order_t[i].item() + zero_order_r[i].item()) * 100:.4f}%")

# Example 5: Analyzing propagating orders for each wavelength
print("\nExample 5: Propagating diffraction orders vs wavelength")
for i in range(len(torchrdit_sim.lam0)):
    wavelength = torchrdit_sim.lam0[i] * 1e3  # in nm
    prop_orders = result.get_propagating_orders(wavelength_idx=i)
    print(f"  Wavelength {wavelength:.1f} nm: {len(prop_orders)} propagating orders")
    print(f"    Orders: {prop_orders}")

# Example 6: Exploring scattering matrix for dispersive materials
print("\nExample 6: Structure scattering matrix analysis")
s11 = result.structure_matrix.S11
s12 = result.structure_matrix.S12
print(f"S11 shape: {s11.shape} - Shows matrix for each wavelength ({len(torchrdit_sim.lam0)} wavelengths)")
print(f"S-matrix values vary with wavelength due to material dispersion")

# Example 7: Effect of wavelength on wave vectors
print("\nExample 7: Wave vector analysis with dispersive materials")
print(f"Wave vectors are affected by wavelength-dependent material properties:")
print(f"kzref for different wavelengths:")
for i in range(len(torchrdit_sim.lam0)):
    wavelength = torchrdit_sim.lam0[i] * 1e3  # in nm
    # Get the central value (zero-order)
    idx = result.get_diffraction_order_indices(0, 0)
    kzref_val = result.wave_vectors.kzref[i].reshape(result.reflection_diffraction.shape[1:3])[idx]
    print(f"  Wavelength {wavelength:.1f} nm: kzref = {torch.abs(kzref_val).item():.6f}")

# Calculate and display energy conservation for the dispersive system
print("\nEnergy conservation check for dispersive system:")
for i in range(len(torchrdit_sim.lam0)):
    wavelength = torchrdit_sim.lam0[i] * 1e3  # in nm
    total = result.transmission[i].item() + result.reflection[i].item()
    print(f"  Wavelength {wavelength:.1f} nm: {total * 100:.4f}% (T+R)")

```

### example_gmrf_rdit

# Example - GMRF with hexagonal unit cells using R-DIT

This example shows the simulation of the guided-mode resonance filter (GMRF) using torchrdit with differentiable R-DIT algorithm. The device is composed of a SiO hexagonal grating layer, a SiN waveguide layer and a fused silica substrate.

The GMRF can be found in the following references:

- A. A. Mehta, R. C. Rumpf, Z. A. Roth, and E. G. Johnson, "Guided mode resonance filter as a spectrally selective feedback element in a double-cladding optical fiber laser," IEEE Photonics Technology Letters, vol. 19, pp. 2030-2032, 12 2007.

Keywords: R-DIT, guided-mode resonance filter (GMRF), torchrdit, torch, efficiency, simulation, tutorial, optical

```python
"""
# Example - GMRF with hexagonal unit cells using R-DIT

This example shows the simulation of the guided-mode resonance filter (GMRF) using torchrdit with differentiable R-DIT algorithm. The device is composed of a SiO hexagonal grating layer, a SiN waveguide layer and a fused silica substrate.

The GMRF can be found in the following references:

- A. A. Mehta, R. C. Rumpf, Z. A. Roth, and E. G. Johnson, "Guided mode resonance filter as a spectrally selective feedback element in a double-cladding optical fiber laser," IEEE Photonics Technology Letters, vol. 19, pp. 2030-2032, 12 2007.

Keywords: R-DIT, guided-mode resonance filter (GMRF), torchrdit, torch, efficiency, simulation, tutorial, optical
"""

import numpy as np
import torch
import os

from torchrdit.solver import get_solver_builder
from torchrdit.shapes import ShapeGenerator
from torchrdit.utils import create_material
from torchrdit.constants import Algorithm, Precision
from torchrdit.viz import plot_layer

import matplotlib.pyplot as plt

# units, normalizing all units to 'um'
um = 1
nm = 1e-3 * um
degrees = np.pi / 180

# angles of incident waves
theta = 0 * degrees
phi = 0 * degrees

# polarization
pte = 1
ptm = 0

# refractive index
n_SiO = 1.4496
n_SiN = 1.9360
n_fs = 1.5100

# dimensions of the cell
a = 1150 * nm
b = a * np.sqrt(3)

# radius of the holes on the top layer
r = 400 * nm

# thickness of each layer
h1 = torch.tensor(230 * nm, dtype=torch.float32)
h2 = torch.tensor(345 * nm, dtype=torch.float32)

# lattice vectors of the cell
t1 = torch.tensor([a/2, -a*np.sqrt(3)/2], dtype=torch.float32)
t2 = torch.tensor([a/2, a*np.sqrt(3)/2], dtype=torch.float32)

# creating materials
material_sio = create_material(name='SiO', permittivity=n_SiO**2)
material_sin = create_material(name='SiN', permittivity=n_SiN**2)
material_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

# Initialize the solver using the builder pattern
builder = get_solver_builder()

# Configure the builder with all necessary parameters
builder.with_algorithm(Algorithm.RDIT)
builder.with_precision(Precision.DOUBLE)
builder.with_real_dimensions([512, 512])
builder.with_k_dimensions([9, 9])
builder.with_wavelengths(np.array([1540 * nm, 1550 * nm, 1560 * nm, 1570 * nm]))
builder.with_length_unit('um')
builder.with_lattice_vectors(t1, t2)

# Add materials
builder.add_material(material_sio)
builder.add_material(material_sin)
builder.add_material(material_fs)

# Configure transmission material
builder.with_trn_material(material_fs)

# Add layers to the builder
# First layer: grating layer (SiO)
builder.add_layer({
    "material": "SiO",
    "thickness": h1.item(),
    "is_homogeneous": False,
    "is_optimize": True
})

# Second layer: homogeneous layer (SiN)
builder.add_layer({
    "material": "SiN",
    "thickness": h2.item(),
    "is_homogeneous": True,
    "is_optimize": False
})

# Build the solver
dev1 = builder.build()

# print layer information
dev1.get_layer_structure()

# create a source object
src1 = dev1.add_source(theta = theta,
                 phi = phi,
                 pte = pte,
                 ptm = ptm)

shapegen = ShapeGenerator.from_solver(dev1)

# build hexagonal unit cell
c1 = shapegen.generate_circle_mask(center=[0, b/2], radius=r)
c2 = shapegen.generate_circle_mask(center=[0, -b/2], radius=r)
c3 = shapegen.generate_circle_mask(center=[a/2, 0], radius=r)
c4 = shapegen.generate_circle_mask(center=[-a/2, 0], radius=r)

mask = shapegen.combine_masks(mask1=c1, mask2=c2, operation='union')
mask = shapegen.combine_masks(mask1=mask, mask2=c3, operation='union')
mask = shapegen.combine_masks(mask1=mask, mask2=c4, operation='union')

mask = (1 - mask).to(torch.float32)

mask.requires_grad = True

layer_index = 0

dev1.update_er_with_mask(mask=mask, layer_index=layer_index)

# plot the layer and save the figure
fig, axes = plt.subplots(figsize=(5, 5))
plot_layer(dev1, layer_index=layer_index, func='real', fig_ax=axes, cmap='BuGn', labels=('x (um)','y (um)'), title=f'layer {layer_index}')
script_dir = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_layer_{layer_index}.png")
plt.savefig(output_filename, dpi=300)
plt.close(fig)

data = dev1.solve(src1)

# Print the Efficiency each wavelength
for i in range(len(dev1.lam0)):
    print(f"The transmission efficiency at wavelength \t{dev1.lam0[i] * 1e3} nm is \t{data.transmission[i] * 100}%")
    print(f"The reflection efficiency at wavelength \t{dev1.lam0[i] * 1e3} nm is \t{data.reflection[i] * 100}%")

print("\n===== Demonstrating SolverResults API capabilities =====\n")

# Example 1: Get specific diffraction order indices
print("Example 1: Getting indices for specific diffraction orders")
try:
    zero_order_idx = data.get_diffraction_order_indices(0, 0)
    first_order_idx = data.get_diffraction_order_indices(1, 0)
    print(f"Zero order (0,0) indices: {zero_order_idx}")
    print(f"First order (1,0) indices: {first_order_idx}")
except ValueError as e:
    print(f"Error getting order indices: {e}")

# Example 2: Get zero-order field components
print("\nExample 2: Getting zero-order field components")
tx, ty, tz = data.get_zero_order_transmission()
print(f"Zero-order transmission field components (first wavelength):")
print(f"  x-component: {tx[0].item():.6f}")
print(f"  y-component: {ty[0].item():.6f}")
print(f"  z-component: {tz[0].item():.6f}")

rx, ry, rz = data.get_zero_order_reflection()
print(f"Zero-order reflection field components (first wavelength):")
print(f"  x-component: {rx[0].item():.6f}")
print(f"  y-component: {ry[0].item():.6f}")
print(f"  z-component: {rz[0].item():.6f}")

# Example 3: Get efficiency for specific diffraction orders
print("\nExample 3: Getting efficiency for specific diffraction orders")
zero_order_t = data.get_order_transmission_efficiency(0, 0)
zero_order_r = data.get_order_reflection_efficiency(0, 0)
print(f"Zero-order transmission efficiency across wavelengths: {zero_order_t.detach().numpy()}")
print(f"Zero-order reflection efficiency across wavelengths: {zero_order_r.detach().numpy()}")

# Try to get a higher-order if available in the simulation
try:
    first_order_t = data.get_order_transmission_efficiency(1, 0)
    print(f"First-order (1,0) transmission efficiency: {first_order_t.detach().numpy()}")
except ValueError as e:
    print(f"Could not get first-order efficiency: {e}")

# Example 4: Get all available diffraction orders
print("\nExample 4: Getting all available diffraction orders")
all_orders = data.get_all_diffraction_orders()
print(f"All available diffraction orders (total: {len(all_orders)}):")
print(all_orders[:10])  # Print first 10 for brevity if there are many

# Example 5: Get propagating orders for a specific wavelength
print("\nExample 5: Getting propagating orders for the first wavelength")
prop_orders = data.get_propagating_orders(wavelength_idx=0)
print(f"Propagating orders for wavelength {dev1.lam0[0] * 1e3} nm (total: {len(prop_orders)}):")
print(prop_orders)

# Example 6: Accessing raw scattering matrix data
print("\nExample 6: Accessing scattering matrix components")
s11_shape = data.structure_matrix.S11.shape
s12_shape = data.structure_matrix.S12.shape
print(f"Structure scattering matrix S11 shape: {s11_shape}")
print(f"Structure scattering matrix S12 shape: {s12_shape}")

# Example 7: Accessing wave vector information
print("\nExample 7: Accessing wave vector information")
print(f"kx shape: {data.wave_vectors.kx.shape}")
print(f"ky shape: {data.wave_vectors.ky.shape}")
print(f"Incident wave vector (kinc) shape: {data.wave_vectors.kinc.shape}")
print(f"Incident wave vector for first wavelength: {data.wave_vectors.kinc[0]}")

# Example 8: Extracting phase information
print("\nExample 8: Extracting phase information from field components")

# Get zero-order field components (these are complex values)
tx, ty, tz = data.get_zero_order_transmission()
rx, ry, rz = data.get_zero_order_reflection()

# Calculate phase in radians using torch.angle()
tx_phase_rad = torch.angle(tx)
ty_phase_rad = torch.angle(ty)
tz_phase_rad = torch.angle(tz)

rx_phase_rad = torch.angle(rx)
ry_phase_rad = torch.angle(ry)
rz_phase_rad = torch.angle(rz)

# Convert to degrees for easier interpretation
rad_to_deg = 180.0 / np.pi

print("Phase of transmission field components (in degrees):")
for i in range(len(dev1.lam0)):
    wavelength = dev1.lam0[i] * 1e3  # in nm
    print(f"  Wavelength {wavelength:.1f} nm:")
    print(f"    x-component: {tx_phase_rad[i].item() * rad_to_deg:.2f}°")
    print(f"    y-component: {ty_phase_rad[i].item() * rad_to_deg:.2f}°")
    print(f"    z-component: {tz_phase_rad[i].item() * rad_to_deg:.2f}°")

print("\nPhase of reflection field components (in degrees):")
for i in range(len(dev1.lam0)):
    wavelength = dev1.lam0[i] * 1e3  # in nm
    print(f"  Wavelength {wavelength:.1f} nm:")
    print(f"    x-component: {rx_phase_rad[i].item() * rad_to_deg:.2f}°")
    print(f"    y-component: {ry_phase_rad[i].item() * rad_to_deg:.2f}°")
    print(f"    z-component: {rz_phase_rad[i].item() * rad_to_deg:.2f}°")

# Example 9: Phase difference between field components
print("\nExample 9: Phase difference between field components")
tx_ty_phase_diff = tx_phase_rad - ty_phase_rad
print("Phase difference between x and y components of transmitted field (in degrees):")
for i in range(len(dev1.lam0)):
    wavelength = dev1.lam0[i] * 1e3  # in nm
    diff_deg = tx_ty_phase_diff[i].item() * rad_to_deg
    # Normalize to [-180, 180] range
    while diff_deg > 180:
        diff_deg -= 360
    while diff_deg < -180:
        diff_deg += 360
    print(f"  Wavelength {wavelength:.1f} nm: {diff_deg:.2f}°")

# Example 10: Phase of off-axis diffraction orders (if available)
print("\nExample 10: Phase of off-axis diffraction orders")
try:
    # Get indices for the (1,0) diffraction order
    idx_1_0 = data.get_diffraction_order_indices(1, 0)
    
    # Extract the field components for this order
    tx_1_0 = data.transmission_field.x[:, idx_1_0[0], idx_1_0[1]]
    ty_1_0 = data.transmission_field.y[:, idx_1_0[0], idx_1_0[1]]
    
    # Calculate phase
    tx_1_0_phase = torch.angle(tx_1_0) * rad_to_deg
    ty_1_0_phase = torch.angle(ty_1_0) * rad_to_deg
    
    print(f"Phase of (1,0) order transmission field (first wavelength):")
    print(f"  x-component: {tx_1_0_phase[0].item():.2f}°")
    print(f"  y-component: {ty_1_0_phase[0].item():.2f}°")
except ValueError as e:
    print(f"Could not analyze (1,0) order: {e}")
```

### example_gmrf_variable_optimize

# Example - Optimization of a guided-mode resonance filter (GMRF)

This example demonstrates how to optimize a guided-mode resonance filter (GMRF) using
differentiable RDIT algorithm in torchrdit. The device consists of a SiO hexagonal
grating layer, a SiN waveguide layer, and a fused silica substrate.

The optimization goal is to shift the resonance peak to a target wavelength (1537 nm)
using gradient descent to adjust the radius of the hexagonal pattern.

The GMRF design is based on:
- A. A. Mehta, R. C. Rumpf, Z. A. Roth, and E. G. Johnson, "Guided mode resonance filter
  as a spectrally selective feedback element in a double-cladding optical fiber laser,"
  IEEE Photonics Technology Letters, vol. 19, pp. 2030-2032, 12 2007.

Keywords:
    guided-mode resonance filter, GMRF, optimization, gradient descent,
    guided mode resonance, guided mode resonance filter, R-DIT, builder,
    automatic differentiation, optimizer, scheduler

```python
"""
# Example - Optimization of a guided-mode resonance filter (GMRF)

This example demonstrates how to optimize a guided-mode resonance filter (GMRF) using
differentiable RDIT algorithm in torchrdit. The device consists of a SiO hexagonal
grating layer, a SiN waveguide layer, and a fused silica substrate.

The optimization goal is to shift the resonance peak to a target wavelength (1537 nm)
using gradient descent to adjust the radius of the hexagonal pattern.

The GMRF design is based on:
- A. A. Mehta, R. C. Rumpf, Z. A. Roth, and E. G. Johnson, "Guided mode resonance filter
  as a spectrally selective feedback element in a double-cladding optical fiber laser,"
  IEEE Photonics Technology Letters, vol. 19, pp. 2030-2032, 12 2007.

Keywords:
    guided-mode resonance filter, GMRF, optimization, gradient descent,
    guided mode resonance, guided mode resonance filter, R-DIT, builder,
    automatic differentiation, optimizer, scheduler
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os

from torchrdit.solver import get_solver_builder
from torchrdit.utils import create_material
from tqdm import trange
from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator

# Global constants
UM = 1
NM = 1e-3 * UM
DEGREES = np.pi / 180

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def GMRF_simulator(radius, lams, rdit_orders=10, kdims=9, is_showinfo=False):
    """
    Simulate the GMRF device with the given parameters.

    Args:
        radius: Radius of the holes in the hexagonal pattern
        lams: Array of wavelengths to simulate
        rdit_orders: Number of orders for the RDIT algorithm
        kdims: Dimensions of the k-space grid
        is_showinfo: Whether to show additional info during simulation

    Returns:
        SolverResults: Results object with transmission, reflection, and field data
    """
    # Setup units and angles
    degrees = DEGREES

    theta = 0 * degrees
    phi = 0 * degrees
    pte = 1
    ptm = 0

    # Device parameters
    n_SiO = 1.4496
    n_SiN = 1.9360
    n_fs = 1.5100

    a = 1150 * NM
    b = a * np.sqrt(3)

    h1 = torch.tensor(230 * NM, dtype=torch.float32, device=device)
    h2 = torch.tensor([345 * NM], dtype=torch.float32, device=device)

    t1 = torch.tensor([[a / 2, -a * np.sqrt(3) / 2]], dtype=torch.float32, device=device)
    t2 = torch.tensor([[a / 2, a * np.sqrt(3) / 2]], dtype=torch.float32, device=device)

    # Create materials
    material_sio = create_material(name="SiO", permittivity=n_SiO**2)
    material_sin = create_material(name="SiN", permittivity=n_SiN**2)
    material_fs = create_material(name="FusedSilica", permittivity=n_fs**2)

    # Create and configure solver using Builder pattern
    builder = get_solver_builder()

    # Configure the builder with all necessary parameters
    builder.with_algorithm(Algorithm.RDIT)
    builder.with_precision(Precision.DOUBLE)
    builder.with_real_dimensions([512, 512])
    builder.with_k_dimensions([kdims, kdims])
    builder.with_wavelengths(lams)
    builder.with_length_unit("um")
    builder.with_lattice_vectors(t1, t2)
    builder.with_device(device=device)

    # Add materials to builder
    builder.add_material(material_sio)
    builder.add_material(material_sin)
    builder.add_material(material_fs)

    # Build the solver
    gmrf_sim = builder.build()

    gmrf_sim.set_rdit_order(rdit_orders)
    gmrf_sim.update_trn_material(trn_material=material_fs)

    # Add layers
    gmrf_sim.add_layer(material_name="SiO", thickness=h1, is_homogeneous=False, is_optimize=True)

    gmrf_sim.add_layer(material_name="SiN", thickness=h2, is_homogeneous=True, is_optimize=False)

    # Create source
    src = gmrf_sim.add_source(theta=theta, phi=phi, pte=pte, ptm=ptm)

    # Create mask and update permittivity
    shapegen = ShapeGenerator.from_solver(gmrf_sim)
    c1 = shapegen.generate_circle_mask(center=[0, b / 2], radius=radius)
    c2 = shapegen.generate_circle_mask(center=[0, -b / 2], radius=radius)
    c3 = shapegen.generate_circle_mask(center=[a / 2, 0], radius=radius)
    c4 = shapegen.generate_circle_mask(center=[-a / 2, 0], radius=radius)

    mask = shapegen.combine_masks(mask1=c1, mask2=c2, operation="union")
    mask = shapegen.combine_masks(mask1=mask, mask2=c3, operation="union")
    mask = shapegen.combine_masks(mask1=mask, mask2=c4, operation="union")

    mask = 1 - mask
    gmrf_sim.update_er_with_mask(mask=mask, layer_index=0)

    # Solve and return results
    if is_showinfo:
        gmrf_sim.get_layer_structure()

    data = gmrf_sim.solve(src)

    return data


def plot_spectrum(lamswp0, data_rdit):
    """
    Plot the transmission and reflection spectra.

    Args:
        lamswp0: Wavelength array
        data_rdit: Simulation results from RDIT

    Returns:
        tuple: Figure and axes objects
    """
    fig_size = (4, 3)
    markeverypoints = 4
    nlam = len(lamswp0)
    nref_gmrf_rdit = np.zeros(nlam)
    ntrn_gmrf_rdit = np.zeros(nlam)
    ncon_gmrf_rdit = np.zeros(nlam)

    for ilam, elem in enumerate(lamswp0):
        nref_gmrf_rdit[ilam] = data_rdit.reflection[ilam].detach().clone().cpu().numpy()
        ntrn_gmrf_rdit[ilam] = data_rdit.transmission[ilam].detach().clone().cpu().numpy()
        ncon_gmrf_rdit[ilam] = nref_gmrf_rdit[ilam] + ntrn_gmrf_rdit[ilam]

    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams.update({"font.size": 10})
    fig_gmrf, ax_gmrf = plt.subplots(figsize=fig_size)
    ax_gmrf.set_xlim(lamswp0[0], lamswp0[-1])

    ax_gmrf.plot(
        lamswp0,
        nref_gmrf_rdit,
        color="blue",
        marker="+",
        linestyle="-",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="Ref-RDIT",
    )
    ax_gmrf.plot(
        lamswp0,
        ntrn_gmrf_rdit,
        color="m",
        marker="x",
        linestyle="-",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="Trn-RDIT",
    )

    ax_gmrf.legend()
    ax_gmrf.set_xlabel("Wavelength (um)")
    ax_gmrf.set_ylabel("Transmission/Reflection Efficiency")
    ax_gmrf.set_ylim([0, 1.0])
    ax_gmrf.grid("on")

    # Save the figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_spectrum.png")
    plt.savefig(output_filename, dpi=300)

    return fig_gmrf, ax_gmrf


def plot_spectrum_compare_opt(lamswp0, data_org, data_opt):
    """
    Plot and compare the transmission and reflection spectra before and after optimization.

    Args:
        lamswp0: Wavelength array
        data_org: Original simulation results
        data_opt: Optimized simulation results

    Returns:
        tuple: Figure and axes objects
    """
    fig_size = (6, 4)
    markeverypoints = 4
    nlam = len(lamswp0)
    nref_gmrf_org = np.zeros(nlam)
    ntrn_gmrf_org = np.zeros(nlam)
    ncon_gmrf_org = np.zeros(nlam)

    nref_gmrf_opt = np.zeros(nlam)
    ntrn_gmrf_opt = np.zeros(nlam)
    ncon_gmrf_opt = np.zeros(nlam)

    for ilam, elem in enumerate(lamswp0):
        nref_gmrf_org[ilam] = data_org.reflection[ilam].detach().clone().cpu().numpy()
        ntrn_gmrf_org[ilam] = data_org.transmission[ilam].detach().clone().cpu().numpy()
        ncon_gmrf_org[ilam] = nref_gmrf_org[ilam] + ntrn_gmrf_org[ilam]

        nref_gmrf_opt[ilam] = data_opt.reflection[ilam].detach().clone().cpu().numpy()
        ntrn_gmrf_opt[ilam] = data_opt.transmission[ilam].detach().clone().cpu().numpy()
        ncon_gmrf_opt[ilam] = nref_gmrf_opt[ilam] + ntrn_gmrf_opt[ilam]

    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams.update({"font.size": 10})

    fig_gmrf, ax_gmrf = plt.subplots(figsize=fig_size)
    ax_gmrf.set_xlim(lamswp0[0], lamswp0[-1])

    # Plot original data
    ax_gmrf.plot(
        lamswp0,
        nref_gmrf_org,
        color="red",
        marker="",
        linestyle="-.",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="R-Init",
    )
    ax_gmrf.plot(
        lamswp0,
        ntrn_gmrf_org,
        color="green",
        marker="",
        linestyle="-",
        linewidth=1,
        markersize=8,
        markevery=markeverypoints,
        label="T-Init",
    )

    # Plot optimized data
    ax_gmrf.plot(
        lamswp0,
        nref_gmrf_opt,
        color="blue",
        marker="x",
        linestyle="-",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="R-Opt",
    )
    ax_gmrf.plot(
        lamswp0,
        ntrn_gmrf_opt,
        color="m",
        marker="",
        linestyle="--",
        linewidth=1,
        markersize=4,
        markevery=markeverypoints,
        label="T-Opt",
    )

    ax_gmrf.legend(loc="center left", bbox_to_anchor=(0.6, 0.5), frameon=False)
    ax_gmrf.set_xlabel("Wavelength [um]")
    ax_gmrf.set_ylabel("T/R")
    ax_gmrf.set_ylim([0, 1.0])

    # Save the figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_comparison.png")
    plt.savefig(output_filename, dpi=300)

    return fig_gmrf, ax_gmrf


def objective_GMRF(dev, src, radius):
    """
    Objective function for GMRF optimization.

    Args:
        dev: The solver device
        src: The source configuration
        radius: The radius of holes to optimize

    Returns:
        torch.Tensor: The transmission efficiency (to be minimized)
    """
    a = 1150 * NM
    b = a * np.sqrt(3)

    shapegen = ShapeGenerator.from_solver(dev)
    c1 = shapegen.generate_circle_mask(center=[0, b / 2], radius=radius)
    c2 = shapegen.generate_circle_mask(center=[0, -b / 2], radius=radius)
    c3 = shapegen.generate_circle_mask(center=[a / 2, 0], radius=radius)
    c4 = shapegen.generate_circle_mask(center=[-a / 2, 0], radius=radius)

    mask = shapegen.combine_masks(mask1=c1, mask2=c2, operation="union")
    mask = shapegen.combine_masks(mask1=mask, mask2=c3, operation="union")
    mask = shapegen.combine_masks(mask1=mask, mask2=c4, operation="union")

    mask = 1 - mask
    dev.update_er_with_mask(mask=mask, layer_index=0)

    data = dev.solve(src)

    return data.transmission[0] * 1e2  # return transmission efficiency as FoM to be minimized


def setup_gmrf_solver(lam_opt):
    """
    Set up the GMRF solver for optimization.

    Args:
        lam_opt: Target wavelength for optimization

    Returns:
        tuple: The solver and source objects
    """
    # Device parameters
    n_SiO = 1.4496
    n_SiN = 1.9360
    n_fs = 1.5100

    a = 1150 * NM

    h1 = torch.tensor(230 * NM, dtype=torch.float32, device=device)
    h2 = torch.tensor([345 * NM], dtype=torch.float32, device=device)

    t1 = torch.tensor([[a / 2, -a * np.sqrt(3) / 2]], dtype=torch.float32, device=device)
    t2 = torch.tensor([[a / 2, a * np.sqrt(3) / 2]], dtype=torch.float32, device=device)

    material_sio = create_material(name="SiO", permittivity=n_SiO**2)
    material_sin = create_material(name="SiN", permittivity=n_SiN**2)
    material_fs = create_material(name="FusedSilica", permittivity=n_fs**2)

    r_dit_order = 10
    kdims = 9

    # Create and configure solver using Builder pattern
    builder = get_solver_builder()

    # Configure the builder with all necessary parameters
    builder.with_algorithm(Algorithm.RDIT)
    builder.with_precision(Precision.DOUBLE)
    builder.with_real_dimensions([512, 512])
    builder.with_k_dimensions([kdims, kdims])
    builder.with_wavelengths(lam_opt)
    builder.with_length_unit("um")
    builder.with_lattice_vectors(t1, t2)
    builder.with_device(device=device)
    # Add materials to builder
    builder.add_material(material_sio)
    builder.add_material(material_sin)
    builder.add_material(material_fs)

    # Build the solver
    gmrf_sim_rdit = builder.build()

    gmrf_sim_rdit.set_rdit_order(r_dit_order)
    gmrf_sim_rdit.update_trn_material(trn_material="FusedSilica")

    # Add layers
    gmrf_sim_rdit.add_layer(material_name="SiO", thickness=h1, is_homogeneous=False, is_optimize=True)

    gmrf_sim_rdit.add_layer(material_name="SiN", thickness=h2, is_homogeneous=True, is_optimize=False)

    # Create source
    src_rdit = gmrf_sim_rdit.add_source(theta=0 * DEGREES, phi=0 * DEGREES, pte=1, ptm=0)

    return gmrf_sim_rdit, src_rdit


def optimize_radius(gmrf_sim_rdit, src_rdit, initial_radius, num_epochs=10):
    """
    Optimize the radius of the GMRF holes using gradient descent.

    Args:
        gmrf_sim_rdit: The GMRF solver
        src_rdit: The source configuration
        initial_radius: Initial radius to start optimization
        num_epochs: Number of optimization epochs

    Returns:
        torch.Tensor: The optimized radius
    """
    r_opt = torch.tensor(initial_radius)
    r_opt.requires_grad = True

    # Learning rate
    lr_rate = 5e-3

    # Define the optimizer
    optimizer = torch.optim.Adam([r_opt], lr=lr_rate, eps=1e-2)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.9)

    # Optimization loop
    t1 = time.perf_counter()

    for epoch in trange(num_epochs):
        optimizer.zero_grad()
        loss = objective_GMRF(gmrf_sim_rdit, src_rdit, r_opt)
        loss.backward()

        optimizer.step()
        scheduler.step()

    t2 = time.perf_counter()
    print(f"Optimization time = {(t2 - t1) * 1000:.2f} ms")

    return r_opt


def main():
    """Main function to demonstrate GMRF optimization."""
    print("Example 4: Optimization of a Guided-Mode Resonance Filter (GMRF)")
    print("=" * 70)

    # Part 1: Basic GMRF simulation with gradient calculation
    print("\nPart 1: Basic GMRF simulation with gradient calculation")
    print("-" * 70)

    # Setup initial radius with gradient tracking
    r0_rdit = torch.tensor(400 * NM, device=device)
    r0_rdit.requires_grad = True

    # Simulate at a single wavelength
    lam00 = np.array([1540 * NM])
    data_rdit = GMRF_simulator(r0_rdit, lam00, rdit_orders=10, kdims=15, is_showinfo=False)

    # Print efficiency and calculate gradient
    print(f"The T efficiency (R-DIT) is {data_rdit.transmission[0].to('cpu') * 100:.2f}%")
    print(f"The R efficiency (R-DIT) is {data_rdit.reflection[0].to('cpu') * 100:.2f}%")

    torch.sum(data_rdit.transmission[0]).backward()
    print(f"The derivative of transmission w.r.t. radius: {r0_rdit.grad}")

    # Part 2: Spectrum calculation
    print("\nPart 2: Spectrum calculation")
    print("-" * 70)

    r1_rdit = torch.tensor(400 * NM)
    r1_rdit.requires_grad = False

    # Calculate spectrum over wavelength range
    nlam = 200
    lam1 = 1530 * NM
    lam2 = 1550 * NM
    lamswp_gmrf = np.linspace(lam1, lam2, nlam, endpoint=True)

    data_gmrfswp_rdit = GMRF_simulator(r1_rdit, lamswp_gmrf, rdit_orders=10, kdims=11)

    # Plot initial spectrum
    fig, ax = plot_spectrum(lamswp0=lamswp_gmrf, data_rdit=data_gmrfswp_rdit)
    plt.close(fig)  # Close figure to avoid display in non-interactive mode
    print("Initial spectrum calculated and saved")

    # Part 3: Optimization
    print("\nPart 3: Optimization")
    print("-" * 70)

    # Target wavelength for optimization
    lam_opt = np.array([1537 * NM])

    # Setup solver for optimization
    gmrf_sim_rdit, src_rdit = setup_gmrf_solver(lam_opt)

    # Optimize radius
    initial_radius = 400 * NM
    r_optimized = optimize_radius(gmrf_sim_rdit, src_rdit, initial_radius)

    print(f"The optimal radius is {r_optimized.item() * 1e3:.2f} nm")

    # Part 4: Compare before and after optimization
    print("\nPart 4: Compare before and after optimization")
    print("-" * 70)

    # Simulate optimized design
    data_optimized_rdit = GMRF_simulator(r_optimized.detach(), lamswp_gmrf, rdit_orders=10, kdims=9, is_showinfo=True)

    # Plot comparison
    fig, ax = plot_spectrum_compare_opt(lamswp0=lamswp_gmrf, data_org=data_gmrfswp_rdit, data_opt=data_optimized_rdit)

    # Find the transmission dip for both the original and optimized designs
    orig_idx = np.argmin(data_gmrfswp_rdit.transmission.detach().to("cpu").numpy())
    opt_idx = np.argmin(data_optimized_rdit.transmission.detach().to("cpu").numpy())

    print(f"Original resonance wavelength: {lamswp_gmrf[orig_idx]:.4f} um")
    print(f"Optimized resonance wavelength: {lamswp_gmrf[opt_idx]:.4f} um")
    print(f"Target wavelength: {lam_opt[0]:.4f} um")

    print("\nExample completed successfully!")
    print("Plots saved in the current directory.")


if __name__ == "__main__":
    main()

```

### example_source_batching

Example: Source Batching for Efficient Multi-Angle and Multi-Polarization Simulations

This example demonstrates how to use the source batching feature in TorchRDIT to
efficiently simulate multiple incident conditions simultaneously. Source batching
provides significant performance improvements when analyzing structures under
various illumination conditions.

Key features demonstrated:
1. Angle sweep simulations
2. Polarization state analysis
3. Parameter optimization with multiple sources
4. Performance comparison between sequential and batched processing
5. Visualization of batched results

```python
"""
Example: Source Batching for Efficient Multi-Angle and Multi-Polarization Simulations

This example demonstrates how to use the source batching feature in TorchRDIT to
efficiently simulate multiple incident conditions simultaneously. Source batching
provides significant performance improvements when analyzing structures under
various illumination conditions.

Key features demonstrated:
1. Angle sweep simulations
2. Polarization state analysis
3. Parameter optimization with multiple sources
4. Performance comparison between sequential and batched processing
5. Visualization of batched results
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from pathlib import Path

from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator


def example_angle_sweep():
    """Demonstrate angle sweep with batched sources."""
    print("\n=== Example 1: Angle Sweep ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),  # 1.55 μm wavelength
        rdim=[512, 512],
        kdim=[5, 5],
        device='cpu'
    )
    
    # Define materials
    si = create_material(name="Si", permittivity=12.25)  # n=3.5
    sio2 = create_material(name="SiO2", permittivity=2.25)  # n=1.5
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, sio2, air])
    
    # Create a simple grating structure
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    
    # Create grating pattern
    mask = torch.zeros(512, 512)
    for i in range(0, 512, 128):
        mask[:, i:i+64] = 1.0
    solver.update_er_with_mask(mask=mask, layer_index=1, bg_material="SiO2")
    
    solver.add_layer(material_name="SiO2", thickness=1.0, is_homogeneous=True)
    
    # Create multiple sources for angle sweep
    deg = np.pi / 180
    angles = np.linspace(0, 60, 13) * deg  # 0° to 60° in 5° steps
    
    sources = []
    for theta in angles:
        source = solver.add_source(
            theta=theta,
            phi=0,
            pte=1.0,  # TE polarization
            ptm=0.0
        )
        sources.append(source)
    
    # Time batched vs sequential processing
    print(f"\nComparing performance for {len(sources)} sources:")
    
    # Sequential processing
    start_time = time.time()
    sequential_results = []
    for source in sources:
        result = solver.solve(source)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"Sequential processing: {sequential_time:.3f} seconds")
    
    # Batched processing
    start_time = time.time()
    batched_results = solver.solve(sources)
    batched_time = time.time() - start_time
    print(f"Batched processing: {batched_time:.3f} seconds")
    print(f"Speedup: {sequential_time/batched_time:.2f}x")
    
    # Extract and plot results
    angles_deg = angles * 180 / np.pi
    
    # Get transmission for all angles
    transmission_te = batched_results.transmission[:, 0].detach().numpy()
    reflection_te = batched_results.reflection[:, 0].detach().numpy()
    
    # Plot angle dependence
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(angles_deg, transmission_te, 'b-', linewidth=2, label='Transmission')
    plt.plot(angles_deg, reflection_te, 'r-', linewidth=2, label='Reflection')
    plt.plot(angles_deg, transmission_te + reflection_te, 'k--', 
             linewidth=1, label='Total', alpha=0.5)
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Efficiency')
    plt.title('TE Polarization Response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    plt.ylim(0, 1.05)
    
    # Find and mark critical angle features
    best_trans_idx = batched_results.find_optimal_source('max_transmission')
    worst_trans_idx = batched_results.find_optimal_source('min_reflection')
    
    plt.plot(angles_deg[best_trans_idx], transmission_te[best_trans_idx], 
             'go', markersize=8, label=f'Max T @ {angles_deg[best_trans_idx]:.1f}°')
    plt.plot(angles_deg[worst_trans_idx], reflection_te[worst_trans_idx], 
             'rs', markersize=8, label=f'Min R @ {angles_deg[worst_trans_idx]:.1f}°')
    
    # Diffraction efficiency map
    plt.subplot(1, 2, 2)
    # Get zero-order transmission for all angles
    zero_order_trans = batched_results.transmission_diffraction[:, 0, 2, 2].detach().numpy()
    
    plt.plot(angles_deg, zero_order_trans, 'b-', linewidth=2, label='Zero Order')
    plt.plot(angles_deg, transmission_te, 'k--', linewidth=1, label='Total', alpha=0.5)
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Transmission Efficiency')
    plt.title('Diffraction Order Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('example_angle_sweep.png', dpi=150)
    plt.show()
    
    return batched_results


def example_polarization_sweep():
    """Demonstrate polarization state analysis with batched sources."""
    print("\n=== Example 2: Polarization State Analysis ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5],
        device='cpu'
    )
    
    # Define materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Create anisotropic structure (elliptical pillar array)
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    
    # Create elliptical pattern
    shape_gen = ShapeGenerator.from_solver(solver)
    # Create an elongated pattern using a rectangle to simulate ellipse
    mask = shape_gen.generate_rectangle_mask(center=(0, 0), width=0.3, height=0.5, angle=0)
    solver.update_er_with_mask(mask=mask, layer_index=1)
    
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    
    # Create sources with different polarization states
    theta = 30 * np.pi / 180  # 30° incidence
    
    # Define polarization states
    polarizations = [
        {"name": "TE", "pte": 1.0, "ptm": 0.0},
        {"name": "TM", "pte": 0.0, "ptm": 1.0},
        {"name": "45° Linear", "pte": 0.7071, "ptm": 0.7071},
        {"name": "-45° Linear", "pte": 0.7071, "ptm": -0.7071},
        # Note: Complex polarizations not supported in current implementation
        # Using 45° elliptical approximations instead
        {"name": "RCP-like", "pte": 0.7071, "ptm": 0.5},
        {"name": "LCP-like", "pte": 0.7071, "ptm": -0.5},
    ]
    
    sources = []
    for pol in polarizations:
        source = solver.add_source(
            theta=theta,
            phi=0,
            pte=pol["pte"],
            ptm=pol["ptm"]
        )
        sources.append(source)
    
    # Solve for all polarizations
    results = solver.solve(sources)
    
    # Extract results
    transmission = results.transmission[:, 0].detach().numpy()
    reflection = results.reflection[:, 0].detach().numpy()
    
    # Visualize polarization response
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of transmission/reflection
    x = np.arange(len(polarizations))
    width = 0.35
    
    ax1.bar(x - width/2, transmission, width, label='Transmission', color='blue', alpha=0.7)
    ax1.bar(x + width/2, reflection, width, label='Reflection', color='red', alpha=0.7)
    
    ax1.set_xlabel('Polarization State')
    ax1.set_ylabel('Efficiency')
    ax1.set_title('Polarization-Dependent Response')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p["name"] for p in polarizations], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)
    
    # Polarization ellipse visualization
    ax2.set_aspect('equal')
    
    # Plot Poincaré sphere projection
    for i, (pol, trans) in enumerate(zip(polarizations, transmission)):
        # Calculate Stokes parameters
        Ex = pol["pte"]
        Ey = pol["ptm"]
        
        S0 = abs(Ex)**2 + abs(Ey)**2
        S1 = abs(Ex)**2 - abs(Ey)**2
        S2 = 2 * Ex * Ey  # For real polarizations
        S3 = 0  # For real polarizations, imaginary part is 0
        
        # Normalize
        if S0 > 0:
            s1 = S1 / S0
            s2 = S2 / S0
            s3 = S3 / S0
            
            # Plot on unit circle (equatorial projection)
            color = plt.cm.viridis(trans)
            ax2.scatter(s1, s2, s=200, c=[color], alpha=0.8, edgecolors='black', linewidth=1)
            ax2.annotate(pol["name"], (s1, s2), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax2.add_artist(circle)
    
    # Draw axes
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel('S₁/S₀')
    ax2.set_ylabel('S₂/S₀')
    ax2.set_title('Polarization States on Poincaré Sphere (Equatorial View)')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for transmission
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Transmission')
    
    plt.tight_layout()
    plt.savefig('example_polarization_sweep.png', dpi=150)
    plt.show()
    
    return results


def example_optimization_with_batched_sources():
    """Demonstrate optimization with multiple incident conditions."""
    print("\n=== Example 3: Multi-Angle Optimization ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5],
        device='cpu'
    )
    
    # Define materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Create structure with optimizable layer
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.3, is_homogeneous=False)
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    
    # Create initial pattern (to be optimized)
    radius_init = 0.2
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_init)
    mask = mask.to(torch.float32)
    mask.requires_grad = True
    
    solver.update_er_with_mask(mask=mask, layer_index=1)
    
    # Define target angles for optimization
    deg = np.pi / 180
    target_angles = [0*deg, 15*deg, 30*deg]  # Optimize for these angles
    
    sources = []
    for theta in target_angles:
        source = solver.add_source(theta=theta, phi=0, pte=1.0, ptm=0.0)
        sources.append(source)
    
    # Optimization loop
    optimizer = torch.optim.Adam([mask], lr=0.02)
    
    history = {
        'loss': [],
        'transmission': []
    }
    
    print("\nOptimizing for uniform response across multiple angles...")
    n_iterations = 50
    
    for iter in range(n_iterations):
        optimizer.zero_grad()
        
        # Solve for all sources
        results = solver.solve(sources)
        
        # Objective: Maximize average transmission while minimizing variance
        trans_values = results.transmission[:, 0]
        avg_trans = trans_values.mean()
        var_trans = trans_values.var()
        
        # Combined loss
        loss = -avg_trans + 0.5 * var_trans  # Maximize avg, minimize variance
        
        loss.backward()
        optimizer.step()
        
        # Clamp mask values to [0, 1]
        with torch.no_grad():
            mask.data = torch.clamp(mask.data, 0, 1)
        
        # Update structure
        solver.update_er_with_mask(mask=mask, layer_index=1)
        
        # Store history
        history['loss'].append(loss.item())
        history['transmission'].append(trans_values.detach().numpy())
        
        if iter % 10 == 0:
            print(f"Iteration {iter}: Loss = {loss.item():.4f}, "
                  f"Avg T = {avg_trans.item():.4f}, "
                  f"Std T = {var_trans.sqrt().item():.4f}")
    
    # Plot optimization results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss history
    ax = axes[0, 0]
    ax.plot(history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Optimization Loss')
    ax.grid(True, alpha=0.3)
    
    # Transmission evolution
    ax = axes[0, 1]
    trans_history = np.array(history['transmission'])
    for i, angle in enumerate(target_angles):
        ax.plot(trans_history[:, i], linewidth=2, 
                label=f'{angle*180/np.pi:.0f}°')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Transmission')
    ax.set_title('Transmission Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Initial pattern
    ax = axes[1, 0]
    initial_mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_init)
    im = ax.imshow(initial_mask.numpy(), cmap='binary')
    ax.set_title('Initial Pattern')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Optimized pattern
    ax = axes[1, 1]
    im = ax.imshow(mask.detach().numpy(), cmap='binary')
    ax.set_title('Optimized Pattern')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('example_optimization_batched.png', dpi=150)
    plt.show()
    
    # Verify final performance with more angles
    print("\nFinal performance verification:")
    test_angles = np.linspace(0, 45, 10) * deg
    test_sources = [solver.add_source(theta=theta, phi=0, pte=1.0, ptm=0.0) 
                    for theta in test_angles]
    
    final_results = solver.solve(test_sources)
    final_trans = final_results.transmission[:, 0].detach().numpy()
    
    print(f"Average transmission: {final_trans.mean():.4f}")
    print(f"Standard deviation: {final_trans.std():.4f}")
    print(f"Min/Max transmission: {final_trans.min():.4f} / {final_trans.max():.4f}")
    
    return mask, final_results


def example_wavelength_and_angle_sweep():
    """Demonstrate combined wavelength and angle parameter sweep."""
    print("\n=== Example 4: Combined Wavelength and Angle Sweep ===")
    
    # Create solver with multiple wavelengths
    wavelengths = np.linspace(1.5, 1.6, 5)  # 5 wavelengths
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=wavelengths,
        rdim=[256, 256],
        kdim=[5, 5],
        device='cpu'
    )
    
    # Define materials
    si = create_material(name="Si", permittivity=12.25)
    sio2 = create_material(name="SiO2", permittivity=2.25)
    solver.add_materials([si, sio2])
    
    # Create thin film stack
    solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.1, is_homogeneous=True)
    solver.add_layer(material_name="SiO2", thickness=0.2, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.1, is_homogeneous=True)
    solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)
    
    # Create sources for angle sweep
    deg = np.pi / 180
    angles = np.linspace(0, 60, 7) * deg
    
    sources = []
    for theta in angles:
        source = solver.add_source(theta=theta, phi=0, pte=1.0, ptm=0.0)
        sources.append(source)
    
    # Solve
    results = solver.solve(sources)
    
    # Extract 2D data: (n_angles, n_wavelengths)
    transmission = results.transmission.detach().numpy()
    reflection = results.reflection.detach().numpy()
    
    # Create 2D plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Transmission map
    im1 = ax1.imshow(transmission, aspect='auto', origin='lower', 
                     extent=[wavelengths[0], wavelengths[-1], 0, 60],
                     cmap='viridis', vmin=0, vmax=1)
    ax1.set_xlabel('Wavelength (μm)')
    ax1.set_ylabel('Incident Angle (degrees)')
    ax1.set_title('Transmission Map')
    plt.colorbar(im1, ax=ax1)
    
    # Reflection map
    im2 = ax2.imshow(reflection, aspect='auto', origin='lower',
                     extent=[wavelengths[0], wavelengths[-1], 0, 60],
                     cmap='plasma', vmin=0, vmax=1)
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_ylabel('Incident Angle (degrees)')
    ax2.set_title('Reflection Map')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('example_wavelength_angle_sweep.png', dpi=150)
    plt.show()
    
    # Find optimal operating point
    best_idx = results.find_optimal_source('max_transmission')
    best_angle = angles[best_idx] * 180 / np.pi
    
    # Find best wavelength for this angle
    best_wl_idx = np.argmax(transmission[best_idx, :])
    best_wl = wavelengths[best_wl_idx]
    
    print(f"\nOptimal operating point:")
    print(f"  Angle: {best_angle:.1f}°")
    print(f"  Wavelength: {best_wl:.3f} μm")
    print(f"  Transmission: {transmission[best_idx, best_wl_idx]:.4f}")
    
    return results


def main():
    """Run all examples."""
    print("TorchRDIT Source Batching Examples")
    print("==================================")
    
    # Run examples
    example_angle_sweep()
    example_polarization_sweep()
    example_optimization_with_batched_sources()
    example_wavelength_and_angle_sweep()
    
    print("\nAll examples completed!")
    print("Generated plots: ")
    print("  - example_angle_sweep.png")
    print("  - example_polarization_sweep.png")
    print("  - example_optimization_batched.png")
    print("  - example_wavelength_angle_sweep.png")


if __name__ == "__main__":
    main()
```

### source_batching_advanced

Source Batching Advanced Examples for TorchRDIT v0.1.22

This script demonstrates advanced optimization techniques using source batching,
including multi-angle optimization, robust design, and gradient-based inverse design.

Key features demonstrated:
- Optimization with multiple incident conditions
- Robust design across angle/polarization variations
- Memory-efficient chunking for large sweeps
- Gradient preservation in batched mode

```python
"""
Source Batching Advanced Examples for TorchRDIT v0.1.22

This script demonstrates advanced optimization techniques using source batching,
including multi-angle optimization, robust design, and gradient-based inverse design.

Key features demonstrated:
- Optimization with multiple incident conditions
- Robust design across angle/polarization variations
- Memory-efficient chunking for large sweeps
- Gradient preservation in batched mode
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator
import time
from pathlib import Path

def example_multi_angle_optimization():
    """Optimize a structure for uniform response across multiple angles."""
    print("\n=== Example 1: Multi-Angle Optimization ===")
    
    # Create optimizer solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[7, 7],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Add layer
    solver.add_layer(material_name="Si", thickness=0.6, is_homogeneous=False)
    
    # Parameterized structure - optimizable radius
    radius_param = torch.tensor(0.25, requires_grad=True, device=solver.device)
    optimizer = torch.optim.Adam([radius_param], lr=0.02)
    
    # Target multiple angles for robust design
    deg = np.pi / 180
    target_angles = np.array([0, 15, 30]) * deg
    sources = [
        solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
        for angle in target_angles
    ]
    
    # Optimization loop
    n_epochs = 50
    history = {'loss': [], 'radius': [], 'transmissions': []}
    
    print("Starting optimization...")
    print("Target: Maximize average transmission across angles while minimizing variance")
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Update structure with current radius
        shape_gen = ShapeGenerator.from_solver(solver)
        mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_param)
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Batch solve for all angles
        results = solver.solve(sources)
        
        # Loss: maximize average transmission across all angles
        trans_values = results.transmission[:, 0]
        avg_transmission = trans_values.mean()
        variance = trans_values.var()
        
        # Combined loss: maximize average, minimize variance
        loss = -avg_transmission + 0.5 * variance
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Clamp radius to valid range
        with torch.no_grad():
            radius_param.clamp_(0.1, 0.45)
        
        # Record history
        history['loss'].append(loss.item())
        history['radius'].append(radius_param.item())
        history['transmissions'].append(trans_values.detach().cpu().numpy())
        
        if epoch % 10 == 0:
            trans_np = trans_values.detach().cpu().numpy()
            print(f"Epoch {epoch:3d}: Loss = {loss.item():7.4f}, "
                  f"Radius = {radius_param.item():.4f}, "
                  f"T = [{trans_np[0]:.3f}, {trans_np[1]:.3f}, {trans_np[2]:.3f}]")
    
    print("\nOptimization complete!")
    print(f"Final radius: {radius_param.item():.4f}")
    print(f"Final average transmission: {-history['loss'][-1]:.4f}")
    
    # Plot optimization history
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Loss history
    ax1.plot(-np.array(history['loss']), 'b-', linewidth=2)
    ax1.set_ylabel('Average Transmission')
    ax1.set_title('Multi-Angle Optimization Progress')
    ax1.grid(True, alpha=0.3)
    
    # Transmission evolution
    trans_history = np.array(history['transmissions'])
    for i, angle in enumerate(target_angles):
        ax2.plot(trans_history[:, i], linewidth=2, 
                 label=f'θ = {angle*180/np.pi:.0f}°')
    ax2.plot(-np.array(history['loss']), 'k--', linewidth=2, 
             label='Average', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Transmission')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'multi_angle_optimization.png', dpi=150)
    plt.close(fig)


def example_robust_design():
    """Design a structure robust to both angle and polarization variations."""
    print("\n=== Example 2: Robust Design for Angle and Polarization ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[7, 7],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)
    sio2 = create_material(name="SiO2", permittivity=2.25)
    solver.add_materials([si, sio2])
    
    # Add optimizable layer
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    
    # Create parametric structure - two circles
    radius1 = torch.tensor(0.15, requires_grad=True, device=solver.device)
    radius2 = torch.tensor(0.15, requires_grad=True, device=solver.device)
    sep = torch.tensor(0.3, requires_grad=True, device=solver.device)
    
    optimizer = torch.optim.Adam([radius1, radius2, sep], lr=0.01)
    
    # Create sources for robustness - angles and polarizations
    deg = np.pi / 180
    sources = []
    
    # Multiple angles with TE
    for angle in np.array([0, 20, 40]) * deg:
        sources.append(solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0))
    
    # Multiple polarizations at 30°
    for pte, ptm in [(1.0, 0.0), (0.0, 1.0), (0.707, 0.707)]:
        sources.append(solver.add_source(theta=30*deg, phi=0, pte=pte, ptm=ptm))
    
    print(f"Optimizing for {len(sources)} different incident conditions")
    
    # Optimization
    n_epochs = 40
    history = {'loss': [], 'params': [], 'mean_trans': [], 'std_trans': []}
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Create two-circle pattern
        shape_gen = ShapeGenerator.from_solver(solver)
        mask1 = shape_gen.generate_circle_mask(center=(-sep/2, 0), radius=radius1)
        mask2 = shape_gen.generate_circle_mask(center=(sep/2, 0), radius=radius2)
        mask = torch.maximum(mask1, mask2)
        
        solver.update_er_with_mask(mask=mask, layer_index=0, bg_material="SiO2")
        
        # Batch solve
        results = solver.solve(sources)
        
        # Robust objective: high mean, low variance
        trans = results.transmission[:, 0]
        mean_trans = trans.mean()
        std_trans = trans.std()
        
        # Penalize large structures
        size_penalty = 0.1 * (radius1**2 + radius2**2)
        
        loss = -mean_trans + 2.0 * std_trans + size_penalty
        
        loss.backward()
        optimizer.step()
        
        # Constraints
        with torch.no_grad():
            radius1.clamp_(0.05, 0.25)
            radius2.clamp_(0.05, 0.25)
            sep.clamp_(0.1, 0.6)
        
        # Record
        history['loss'].append(loss.item())
        history['params'].append([radius1.item(), radius2.item(), sep.item()])
        history['mean_trans'].append(mean_trans.item())
        history['std_trans'].append(std_trans.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Mean T = {mean_trans.item():.4f}, "
                  f"Std T = {std_trans.item():.4f}, "
                  f"r1={radius1.item():.3f}, r2={radius2.item():.3f}, sep={sep.item():.3f}")
    
    # Visualize results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss evolution
    ax1.plot(history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Optimization Loss')
    ax1.grid(True, alpha=0.3)
    
    # Mean vs Std transmission
    ax2.plot(history['mean_trans'], 'g-', linewidth=2, label='Mean T')
    ax2.plot(history['std_trans'], 'r-', linewidth=2, label='Std T')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Transmission')
    ax2.set_title('Transmission Statistics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Parameter evolution
    params = np.array(history['params'])
    ax3.plot(params[:, 0], label='Radius 1', linewidth=2)
    ax3.plot(params[:, 1], label='Radius 2', linewidth=2)
    ax3.plot(params[:, 2], label='Separation', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Structure Parameters')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final structure
    ax4.imshow(mask.detach().cpu().numpy(), cmap='hot', origin='lower')
    ax4.set_title('Final Optimized Structure')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'robust_design_optimization.png', dpi=150)
    plt.close(fig)
    
    print("\nFinal design:")
    print(f"  Mean transmission: {history['mean_trans'][-1]:.4f}")
    print(f"  Std transmission: {history['std_trans'][-1]:.4f}")
    print(f"  Parameters: r1={radius1.item():.3f}, r2={radius2.item():.3f}, sep={sep.item():.3f}")


def example_large_sweep_chunking():
    """Demonstrate memory-efficient processing of large parameter sweeps."""
    print("\n=== Example 3: Memory-Efficient Large Sweep Processing ===")
    
    def process_large_sweep(solver, angles, chunk_size=100):
        """Process large angle sweep in memory-efficient chunks."""
        all_transmissions = []
        all_reflections = []
        
        n_chunks = (len(angles) + chunk_size - 1) // chunk_size
        print(f"Processing {len(angles)} angles in {n_chunks} chunks...")
        
        for i in range(0, len(angles), chunk_size):
            # Get chunk of angles
            chunk_angles = angles[i:i+chunk_size]
            
            # Create sources for chunk
            sources = [
                solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
                for angle in chunk_angles
            ]
            
            # Process chunk
            results = solver.solve(sources)
            all_transmissions.extend(results.transmission[:, 0].detach().cpu().numpy())
            all_reflections.extend(results.reflection[:, 0].detach().cpu().numpy())
            
            # Clear GPU memory if needed
            if str(solver.device).startswith('cuda'):
                torch.cuda.empty_cache()
            
            # Progress
            print(f"  Processed chunk {i//chunk_size + 1}/{n_chunks}")
        
        return np.array(all_transmissions), np.array(all_reflections)
    
    # Create solver with simple structure
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5]
    )
    
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_rectangle_mask(width=0.4, height=0.2, angle=45)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Process 1000 angles
    deg = np.pi / 180
    many_angles = np.linspace(0, 89, 1000) * deg
    
    start = time.time()
    trans_large, refl_large = process_large_sweep(solver, many_angles, chunk_size=100)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results shape: {trans_large.shape}")
    print(f"Average transmission: {trans_large.mean():.4f}")
    print("Peak memory usage avoided by chunking!")
    
    # Plot subset of results
    stride = 10  # Plot every 10th point
    angles_deg = many_angles[::stride] * 180 / np.pi
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, trans_large[::stride], 'b-', linewidth=1, label='Transmission')
    plt.plot(angles_deg, refl_large[::stride], 'r-', linewidth=1, label='Reflection')
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Efficiency')
    plt.title('Large Angle Sweep (1000 angles)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 89)
    plt.ylim(0, 1.05)
    plt.savefig(Path(__file__).parent / 'large_sweep_chunking.png', dpi=150)
    plt.close(plt.gcf())


def example_gradient_validation():
    """Validate gradient consistency between sequential and batched solving."""
    print("\n=== Example 4: Gradient Validation for Batched Solving ===")
    
    # Create two identical solvers
    def create_test_solver():
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[128, 128],
            kdim=[5, 5],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
        return solver
    
    solver_seq = create_test_solver()
    solver_batch = create_test_solver()
    
    # Create identical optimizable radius parameters
    radius_seq = torch.tensor(0.3, requires_grad=True, device=solver_seq.device)
    radius_batch = torch.tensor(0.3, requires_grad=True, device=solver_batch.device)
    
    # Create sources
    deg = np.pi / 180
    angles = np.array([0, 30, 60]) * deg
    sources = [
        solver_seq.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
        for angle in angles
    ]
    
    # Sequential solving and gradient
    print("Computing sequential gradients...")
    trans_seq = []
    for source in sources:
        # Update mask for each source
        shape_gen = ShapeGenerator.from_solver(solver_seq)
        mask_seq = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_seq)
        solver_seq.update_er_with_mask(mask=mask_seq, layer_index=0)
        
        result = solver_seq.solve(source)
        trans_seq.append(result.transmission[0])
    
    loss_seq = -torch.stack(trans_seq).mean()
    loss_seq.backward()
    grad_seq = radius_seq.grad.clone()
    
    # Batched solving and gradient
    print("Computing batched gradients...")
    # Update mask once for batched solve
    shape_gen = ShapeGenerator.from_solver(solver_batch)
    mask_batch = shape_gen.generate_circle_mask(center=(0, 0), radius=radius_batch)
    solver_batch.update_er_with_mask(mask=mask_batch, layer_index=0)
    
    results_batch = solver_batch.solve(sources)
    loss_batch = -results_batch.transmission[:, 0].mean()
    loss_batch.backward()
    grad_batch = radius_batch.grad
    
    # Compare gradients
    grad_diff = torch.abs(grad_seq - grad_batch).item()
    grad_rel_diff = grad_diff / (torch.abs(grad_seq).item() + 1e-8)
    
    print("\nGradient comparison:")
    print(f"  Sequential gradient: {grad_seq.item():.6f}")
    print(f"  Batched gradient: {grad_batch.item():.6f}")
    print(f"  Absolute difference: {grad_diff:.2e}")
    print(f"  Relative difference: {grad_rel_diff:.2e}")
    print(f"  Gradients match: {'YES' if grad_rel_diff < 1e-5 else 'NO'}")
    
    # Visualize radius optimization
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Show the gradient values as a bar chart
    gradients = [grad_seq.item(), grad_batch.item()]
    labels = ['Sequential', 'Batched']
    colors = ['blue', 'orange']
    
    bars = ax.bar(labels, gradients, color=colors)
    ax.set_ylabel('Gradient Value')
    ax.set_title('Gradient Comparison: Sequential vs Batched')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, grad in zip(bars, gradients):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{grad:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'gradient_validation.png', dpi=150)
    plt.close(fig)


def main():
    """Run all advanced examples."""
    print("TorchRDIT Source Batching - Advanced Examples")
    print("=============================================")
    
    # Run examples
    example_multi_angle_optimization()
    example_robust_design()
    example_large_sweep_chunking()
    example_gradient_validation()
    
    print("\nAll advanced examples completed!")
    print("Generated plots:")
    print("  - multi_angle_optimization.png")
    print("  - robust_design_optimization.png")
    print("  - large_sweep_chunking.png")
    print("  - gradient_validation.png")


if __name__ == "__main__":
    main()
```

### source_batching_basic

Source Batching Basic Usage Examples for TorchRDIT v0.1.22

This script demonstrates the basic usage of source batching feature that allows 
you to process multiple incident angles and polarizations simultaneously.

Key features demonstrated:
- Single vs batched source comparison
- Angle sweep simulations
- Polarization state analysis
- Basic performance comparison

```python
"""
Source Batching Basic Usage Examples for TorchRDIT v0.1.22

This script demonstrates the basic usage of source batching feature that allows 
you to process multiple incident angles and polarizations simultaneously.

Key features demonstrated:
- Single vs batched source comparison
- Angle sweep simulations
- Polarization state analysis
- Basic performance comparison
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator
import time


def example_single_vs_batched():
    """Compare traditional single-source approach with new batched processing."""
    print("\n=== Example 1: Single vs Batched Processing ===")
    
    # Create a solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),  # 1.55 μm wavelength
        rdim=[512, 512],        # Real-space dimensions
        kdim=[7, 7],            # k-space dimensions
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)  # n=3.5
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Add a simple structure - silicon cylinder in air
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    print(f"Solver created with device: {solver.device}")
    
    # Single source example (backward compatible)
    print("\n--- Single Source Processing ---")
    source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
    result = solver.solve(source)
    
    print(f"Single source result:")
    print(f"  Transmission: {result.transmission[0].item():.4f}")
    print(f"  Reflection: {result.reflection[0].item():.4f}")
    print(f"  Result type: {type(result).__name__}")
    
    # Multiple sources example (new feature)
    print("\n--- Batched Source Processing ---")
    deg = np.pi / 180
    sources = [
        solver.add_source(theta=0*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=30*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=45*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=60*deg, phi=0, pte=1.0, ptm=0.0)
    ]
    
    # Batch solve
    results = solver.solve(sources)
    
    print(f"Batched results:")
    print(f"  Result type: {type(results).__name__}")
    print(f"  Number of sources: {results.n_sources}")
    print(f"  Transmission shape: {results.transmission.shape}")
    print(f"\nTransmission values:")
    for i, trans in enumerate(results.transmission[:, 0]):
        angle = results.source_parameters[i]['theta'] * 180 / np.pi
        print(f"  θ={angle:3.0f}°: {trans.item():.4f}")
    
    # Demonstrate accessing individual results
    print("\n--- Accessing Individual Results ---")
    result_45deg = results[2]  # Returns SolverResults for 45° incidence
    print(f"Result at 45°: {type(result_45deg).__name__}")
    print(f"  Transmission: {result_45deg.transmission[0].item():.4f}")
    
    # Iterate through all results
    print("\nIterating through results:")
    for i, single_result in enumerate(results):
        angle = results.source_parameters[i]['theta'] * 180 / np.pi
        print(f"  θ={angle:3.0f}°: T={single_result.transmission[0].item():.4f}")


def example_angle_sweep():
    """Demonstrate efficient angle sweep using source batching."""
    print("\n=== Example 2: Angle Sweep with Batched Sources ===")
    
    # Create a more complex structure - grating
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[512, 512],
        kdim=[11, 11]
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Add grating structure
    solver.add_layer(material_name="Si", thickness=0.6, is_homogeneous=False)
    
    # Create grating pattern
    mask = torch.zeros(512, 512)
    period = 128  # pixels
    duty_cycle = 0.5
    for i in range(0, 512, period):
        mask[:, i:i+int(period*duty_cycle)] = 1.0
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Angle sweep from -60° to 60°
    deg = np.pi / 180
    angles = np.linspace(-60, 60, 121) * deg
    sources = [
        solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
        for angle in angles
    ]
    
    # Compare sequential vs batched processing
    print(f"\nComparing performance for {len(sources)} sources:")
    
    # Sequential processing (for comparison)
    start_time = time.time()
    sequential_results = []
    for i in range(min(5, len(sources))):  # Only do a few for time comparison
        result = solver.solve(sources[i])
        sequential_results.append(result)
    sequential_time = (time.time() - start_time) * len(sources) / 5  # Extrapolate
    print(f"Sequential processing (estimated): {sequential_time:.2f}s")
    
    # Batched processing
    start_time = time.time()
    results = solver.solve(sources)
    batched_time = time.time() - start_time
    print(f"Batched processing: {batched_time:.2f}s")
    print(f"Speedup: {sequential_time/batched_time:.1f}x")
    
    # Extract and plot results
    transmissions = results.transmission[:, 0].detach().cpu().numpy()
    reflections = results.reflection[:, 0].detach().cpu().numpy()
    angles_deg = angles * 180 / np.pi
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, transmissions, 'b-', linewidth=2, label='Transmission')
    plt.plot(angles_deg, reflections, 'r-', linewidth=2, label='Reflection')
    plt.plot(angles_deg, transmissions + reflections, 'k--', 
             linewidth=1, label='Sum (Conservation)', alpha=0.5)
    plt.xlabel('Incident Angle (degrees)')
    plt.ylabel('Efficiency')
    plt.title('Angular Response of Silicon Grating')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-60, 60)
    plt.ylim(0, 1.05)
    plt.savefig('angle_sweep_basic.png', dpi=150)
    plt.show()
    
    # Find optimal angle
    best_idx = results.find_optimal_source(metric="max_transmission")
    best_params = results.source_parameters[best_idx]
    print(f"\nMaximum transmission: {transmissions[best_idx]:.4f}")
    print(f"Occurs at angle: {best_params['theta']*180/np.pi:.1f}°")


def example_polarization_analysis():
    """Demonstrate polarization state analysis with batched sources."""
    print("\n=== Example 3: Polarization State Analysis ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[7, 7]
    )
    
    # Add materials
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    # Create anisotropic structure (elliptical pillar)
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    
    # Create elliptical pattern
    shape_gen = ShapeGenerator.from_solver(solver)
    # Create an elongated pattern using a rectangle to simulate ellipse
    mask = shape_gen.generate_rectangle_mask(center=(0, 0), width=0.3, height=0.5, angle=0)
    solver.update_er_with_mask(mask=mask, layer_index=1)
    
    solver.add_layer(material_name="air", thickness=0.5, is_homogeneous=True)
    
    # Create sources with different polarization states
    theta = 30 * np.pi / 180  # 30° incidence
    
    # Define polarization states
    polarizations = [
        {"name": "TE", "pte": 1.0, "ptm": 0.0},
        {"name": "TM", "pte": 0.0, "ptm": 1.0},
        {"name": "45° Linear", "pte": 0.7071, "ptm": 0.7071},
        {"name": "-45° Linear", "pte": 0.7071, "ptm": -0.7071},
        # Note: Complex polarizations not supported in current implementation
        # Using elliptical approximations instead
        {"name": "RCP-like", "pte": 0.7071, "ptm": 0.5},
        {"name": "LCP-like", "pte": 0.7071, "ptm": -0.5},
    ]
    
    sources = []
    for pol in polarizations:
        source = solver.add_source(
            theta=theta,
            phi=0,
            pte=pol["pte"],
            ptm=pol["ptm"]
        )
        sources.append(source)
    
    # Solve for all polarizations
    results = solver.solve(sources)
    
    # Extract results
    transmission = results.transmission[:, 0].detach().cpu().numpy()
    reflection = results.reflection[:, 0].detach().cpu().numpy()
    
    # Visualize polarization response
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Bar plot of transmission/reflection
    x = np.arange(len(polarizations))
    width = 0.35
    
    ax.bar(x - width/2, transmission, width, label='Transmission', color='blue', alpha=0.7)
    ax.bar(x + width/2, reflection, width, label='Reflection', color='red', alpha=0.7)
    
    ax.set_xlabel('Polarization State')
    ax.set_ylabel('Efficiency')
    ax.set_title('Polarization-Dependent Response at 30° Incidence')
    ax.set_xticks(x)
    ax.set_xticklabels([p["name"] for p in polarizations], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('polarization_analysis_basic.png', dpi=150)
    plt.show()
    
    print("\nPolarization analysis results:")
    for i, pol in enumerate(polarizations):
        print(f"{pol['name']:15s}: T={transmission[i]:.4f}, R={reflection[i]:.4f}")


def example_parameter_sweep():
    """Demonstrate mixed parameter sweeps in a single batch."""
    print("\n=== Example 4: Mixed Parameter Sweep ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5]
    )
    
    # Add materials and simple structure
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    solver.add_layer(material_name="Si", thickness=0.4, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.25)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Create mixed parameter sweep
    deg = np.pi / 180
    mixed_sources = [
        # Vary angle at TE polarization
        solver.add_source(theta=0*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=30*deg, phi=0, pte=1.0, ptm=0.0),
        solver.add_source(theta=60*deg, phi=0, pte=1.0, ptm=0.0),
        # Vary polarization at fixed angle
        solver.add_source(theta=45*deg, phi=0, pte=0.0, ptm=1.0),
        solver.add_source(theta=45*deg, phi=0, pte=0.707, ptm=0.707),
        # Vary azimuthal angle
        solver.add_source(theta=30*deg, phi=45*deg, pte=1.0, ptm=0.0),
        solver.add_source(theta=30*deg, phi=90*deg, pte=1.0, ptm=0.0),
    ]
    
    # Solve batch
    results = solver.solve(mixed_sources)
    
    # Extract parameter sweep data
    theta_data, theta_values = results.get_parameter_sweep_data('theta', 'transmission')
    print("\nUnique theta values (degrees):", np.unique(theta_values.numpy()) * 180 / np.pi)
    
    # Display results
    print("\nMixed parameter sweep results:")
    for i, params in enumerate(results.source_parameters):
        t = results.transmission[i, 0].item()
        print(f"  θ={params['theta']*180/np.pi:5.1f}°, "
              f"φ={params['phi']*180/np.pi:5.1f}°, "
              f"pte={params['pte']:.3f}, ptm={params['ptm']:.3f} "
              f"→ T={t:.4f}")


def main():
    """Run all basic examples."""
    print("TorchRDIT Source Batching - Basic Examples")
    print("==========================================")
    
    # Run examples
    example_single_vs_batched()
    example_angle_sweep()
    example_polarization_analysis()
    example_parameter_sweep()
    
    print("\nAll basic examples completed!")
    print("Generated plots:")
    print("  - angle_sweep_basic.png")
    print("  - polarization_analysis_basic.png")


if __name__ == "__main__":
    main()
```

### source_batching_performance

Source Batching Performance Comparison for TorchRDIT v0.1.22

This script provides comprehensive performance benchmarks comparing sequential
vs batched source processing across different scenarios.

Key benchmarks:
- Scaling with number of sources
- Impact of structure complexity
- Memory usage comparison
- GPU vs CPU performance
- Different batch sizes

```python
"""
Source Batching Performance Comparison for TorchRDIT v0.1.22

This script provides comprehensive performance benchmarks comparing sequential
vs batched source processing across different scenarios.

Key benchmarks:
- Scaling with number of sources
- Impact of structure complexity
- Memory usage comparison
- GPU vs CPU performance
- Different batch sizes
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material
from torchrdit.shapes import ShapeGenerator
import time
import gc
from pathlib import Path


def benchmark_scaling():
    """Benchmark performance scaling with number of sources."""
    print("\n=== Benchmark 1: Performance Scaling with Number of Sources ===")
    
    # Create solver with moderate complexity
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[7, 7],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials and structure
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Test different numbers of sources
    n_sources_list = [1, 2, 5, 10, 20, 40, 80]
    sequential_times = []
    batched_times = []
    speedups = []
    
    deg = np.pi / 180
    
    for n_sources in n_sources_list:
        # Create sources
        angles = np.linspace(0, 60, n_sources) * deg
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]
        
        # Clear cache
        gc.collect()
        if str(solver.device).startswith('cuda'):
            torch.cuda.empty_cache()
        
        # Sequential processing
        start = time.time()
        for source in sources:
            _ = solver.solve(source)
        seq_time = time.time() - start
        sequential_times.append(seq_time)
        
        # Clear cache again
        gc.collect()
        if str(solver.device).startswith('cuda'):
            torch.cuda.empty_cache()
        
        # Batched processing
        start = time.time()
        _ = solver.solve(sources)
        batch_time = time.time() - start
        batched_times.append(batch_time)
        
        speedup = seq_time / batch_time if batch_time > 0 else 1.0
        speedups.append(speedup)
        
        print(f"N={n_sources:3d}: Sequential={seq_time:6.3f}s, "
              f"Batched={batch_time:6.3f}s, Speedup={speedup:.2f}x")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Timing comparison
    ax1.plot(n_sources_list, sequential_times, 'ro-', linewidth=2, 
             markersize=8, label='Sequential')
    ax1.plot(n_sources_list, batched_times, 'bo-', linewidth=2, 
             markersize=8, label='Batched')
    ax1.set_xlabel('Number of Sources')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Processing Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Speedup factor
    ax2.plot(n_sources_list, speedups, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Sources')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Batched Processing Speedup')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(speedups) * 1.2)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'performance_scaling.png', dpi=150)
    plt.close()
    
    print(f"\nAverage speedup for N>1: {np.mean(speedups[1:]):.2f}x")


def benchmark_complexity():
    """Benchmark performance with different structure complexities."""
    print("\n=== Benchmark 2: Impact of Structure Complexity ===")
    
    kdim_values = [3, 5, 7, 9, 11]
    speedups = []
    
    # Fixed number of sources
    n_sources = 20
    deg = np.pi / 180
    angles = np.linspace(0, 60, n_sources) * deg
    
    for kdim in kdim_values:
        # Create solver with varying complexity
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[256, 256],
            kdim=[kdim, kdim],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Add simple structure
        si = create_material(name="Si", permittivity=12.25)
        air = create_material(name="air", permittivity=1.0)
        solver.add_materials([si, air])
        
        solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
        shape_gen = ShapeGenerator.from_solver(solver)
        mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)
        solver.update_er_with_mask(mask=mask, layer_index=0)
        
        # Create sources
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]
        
        # Time sequential (sample only)
        start = time.time()
        for i in range(min(3, n_sources)):
            _ = solver.solve(sources[i])
        seq_time = (time.time() - start) * n_sources / 3
        
        # Time batched
        start = time.time()
        _ = solver.solve(sources)
        batch_time = time.time() - start
        
        speedup = seq_time / batch_time
        speedups.append(speedup)
        
        print(f"kdim={kdim}x{kdim}: Sequential≈{seq_time:.3f}s, "
              f"Batched={batch_time:.3f}s, Speedup={speedup:.2f}x")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(kdim_values, speedups, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Fourier Harmonics (kdim)')
    plt.ylabel('Speedup Factor')
    plt.title(f'Speedup vs Structure Complexity ({n_sources} sources)')
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(__file__).parent / 'complexity_impact.png', dpi=150)
    plt.close()


def benchmark_wavelength_angle_sweep():
    """Benchmark combined wavelength and angle sweeps."""
    print("\n=== Benchmark 3: Combined Wavelength and Angle Sweep ===")
    
    # Create solver with multiple wavelengths
    wavelengths = np.linspace(1.5, 1.6, 5)
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=wavelengths,
        rdim=[256, 256],
        kdim=[5, 5],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add thin film stack
    si = create_material(name="Si", permittivity=12.25)
    sio2 = create_material(name="SiO2", permittivity=2.25)
    solver.add_materials([si, sio2])
    
    solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.1, is_homogeneous=True)
    solver.add_layer(material_name="SiO2", thickness=0.2, is_homogeneous=True)
    solver.add_layer(material_name="Si", thickness=0.1, is_homogeneous=True)
    solver.add_layer(material_name="SiO2", thickness=0.5, is_homogeneous=True)
    
    # Create sources for angle sweep
    deg = np.pi / 180
    angles = np.linspace(0, 60, 13) * deg
    sources = [
        solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
        for angle in angles
    ]
    
    print(f"Processing {len(wavelengths)} wavelengths × {len(sources)} angles = "
          f"{len(wavelengths) * len(sources)} total calculations")
    
    # Sequential timing (estimate)
    start = time.time()
    for i in range(min(3, len(sources))):
        _ = solver.solve(sources[i])
    est_seq_time = (time.time() - start) * len(sources) / 3
    
    # Batched timing
    start = time.time()
    results = solver.solve(sources)
    batch_time = time.time() - start
    
    speedup = est_seq_time / batch_time
    print(f"\nEstimated sequential: {est_seq_time:.2f}s")
    print(f"Batched processing: {batch_time:.2f}s")
    print(f"Speedup factor: {speedup:.1f}x")
    
    # Visualize results as 2D map
    transmission = results.transmission.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(transmission, aspect='auto', origin='lower',
                    extent=[wavelengths[0], wavelengths[-1], 0, 60],
                    cmap='viridis', vmin=0, vmax=1)
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('Incident Angle (degrees)')
    plt.title('Transmission Map (Wavelength × Angle)')
    plt.colorbar(im, label='Transmission')
    plt.savefig(Path(__file__).parent / 'wavelength_angle_map.png', dpi=150)
    plt.close()


def benchmark_memory_usage():
    """Compare memory usage patterns between sequential and batched processing."""
    print("\n=== Benchmark 4: Memory Usage Comparison ===")
    
    if not torch.cuda.is_available():
        print("GPU not available, skipping memory benchmark")
        return
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[512, 512],
        kdim=[9, 9],
        device='cuda'
    )
    
    # Add structure
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.3)
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Test configurations
    n_sources_list = [10, 20, 40, 80]
    peak_memory_seq = []
    peak_memory_batch = []
    
    deg = np.pi / 180
    
    for n_sources in n_sources_list:
        # Create sources
        angles = np.linspace(0, 60, n_sources) * deg
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]
        
        # Sequential memory usage
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        for source in sources:
            _ = solver.solve(source)
        
        peak_seq = torch.cuda.max_memory_allocated() / 1024**2  # MB
        peak_memory_seq.append(peak_seq)
        
        # Batched memory usage
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        _ = solver.solve(sources)
        
        peak_batch = torch.cuda.max_memory_allocated() / 1024**2  # MB
        peak_memory_batch.append(peak_batch)
        
        print(f"N={n_sources:3d}: Sequential={peak_seq:7.1f}MB, "
              f"Batched={peak_batch:7.1f}MB, "
              f"Ratio={peak_batch/peak_seq:.2f}x")
    
    # Plot memory usage
    plt.figure(figsize=(8, 6))
    plt.plot(n_sources_list, peak_memory_seq, 'ro-', linewidth=2, 
             markersize=8, label='Sequential')
    plt.plot(n_sources_list, peak_memory_batch, 'bo-', linewidth=2, 
             markersize=8, label='Batched')
    plt.xlabel('Number of Sources')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(__file__).parent / 'memory_usage.png', dpi=150)
    plt.close()


def objective_function_radius(solver, sources, radius):
    """
    Objective function for radius-based optimization.
    
    Args:
        solver: The solver instance
        sources: List of source configurations
        radius: The radius tensor to optimize
        
    Returns:
        torch.Tensor: The loss value (negative mean transmission)
    """
    # Generate circle mask using the radius
    shape_gen = ShapeGenerator.from_solver(solver)
    mask = shape_gen.generate_circle_mask(center=(0, 0), radius=radius)
    
    # Update solver with the new mask
    solver.update_er_with_mask(mask=mask, layer_index=0)
    
    # Solve for all sources
    results = solver.solve(sources)
    
    # Return loss (negative transmission to minimize)
    return -results.transmission.mean()


def benchmark_optimization_overhead():
    """Benchmark overhead in gradient computation for optimization."""
    print("\n=== Benchmark 5: Optimization Overhead ===")
    
    # Create solver
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        rdim=[256, 256],
        kdim=[5, 5],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Add materials and layer
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    solver.add_layer(material_name="Si", thickness=0.5, is_homogeneous=False)
    
    # Test with different numbers of sources
    n_sources_list = [1, 5, 10, 20]
    forward_times = []
    backward_times = []
    
    deg = np.pi / 180
    
    for n_sources in n_sources_list:
        # Create sources
        angles = np.linspace(0, 60, n_sources) * deg
        sources = [
            solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
            for angle in angles
        ]
        
        # Create radius tensor as optimization variable
        radius = torch.tensor(0.3, dtype=torch.float32, device=solver.device)
        radius.requires_grad = True
        
        # Forward pass timing
        start = time.time()
        loss = objective_function_radius(solver, sources, radius)
        forward_time = time.time() - start
        forward_times.append(forward_time)
        
        # Backward pass timing
        start = time.time()
        loss.backward()
        backward_time = time.time() - start
        backward_times.append(backward_time)
        
        print(f"N={n_sources:2d}: Forward={forward_time:.3f}s, "
              f"Backward={backward_time:.3f}s, "
              f"Ratio={backward_time/forward_time:.2f}x")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Timing breakdown
    width = 0.35
    x = np.arange(len(n_sources_list))
    ax1.bar(x - width/2, forward_times, width, label='Forward', alpha=0.7)
    ax1.bar(x + width/2, backward_times, width, label='Backward', alpha=0.7)
    ax1.set_xlabel('Number of Sources')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Forward vs Backward Pass Timing')
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_sources_list)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Backward/Forward ratio
    ratios = [b/f for b, f in zip(backward_times, forward_times)]
    ax2.plot(n_sources_list, ratios, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Sources')
    ax2.set_ylabel('Backward/Forward Time Ratio')
    ax2.set_title('Gradient Computation Overhead')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'optimization_overhead.png', dpi=150)
    plt.close()


def main():
    """Run all performance benchmarks."""
    print("TorchRDIT Source Batching - Performance Benchmarks")
    print("=================================================")
    
    # Run benchmarks
    benchmark_scaling()
    benchmark_complexity()
    benchmark_wavelength_angle_sweep()
    benchmark_memory_usage()
    benchmark_optimization_overhead()
    
    print("\nAll performance benchmarks completed!")
    print("Generated plots:")
    print("  - performance_scaling.png")
    print("  - complexity_impact.png")
    print("  - wavelength_angle_map.png")
    print("  - memory_usage.png")
    print("  - optimization_overhead.png")


if __name__ == "__main__":
    main()
```
