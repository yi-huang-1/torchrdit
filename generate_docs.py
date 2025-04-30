#!/usr/bin/env python3
"""
Documentation generator for TorchRDIT project.
This script uses pydoc-markdown to generate structured documentation
based on the configuration in pydoc-markdown.yml.
"""

import os
import yaml
import subprocess
from pathlib import Path

def main():
    # Create wiki directory if it doesn't exist
    wiki_dir = Path("wiki")
    wiki_dir.mkdir(exist_ok=True)
    
    # Load configuration
    with open("pydoc-markdown.yml", "r") as f:
        config = yaml.safe_load(f)
    
    # Extract docs directory from config
    docs_dir = Path(config.get("docs_directory", "wiki"))
    docs_dir.mkdir(exist_ok=True)
    
    # Remove all existing files in the wiki directory
    print("Cleaning wiki directory...")
    file_count = 0
    for file_path in docs_dir.glob('*'):
        if file_path.is_file():
            file_path.unlink()
            file_count += 1
    print(f"  -> Removed {file_count} files")
    
    # Generate documentation for each module
    for module in config.get("modules", []):
        module_name = module["name"]
        output_file = docs_dir / module["output_file"]
        
        print(f"Generating documentation for {module_name}...")
        
        # Run pydoc-markdown for this module
        cmd = ["pydoc-markdown", "-m", module_name, "--render-toc"]
        
        try:
            with open(output_file, "w") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
                
            if result.returncode != 0:
                print(f"  -> Warning: Failed to generate documentation for {module_name}")
                # Create a minimal documentation file when module import fails
                with open(output_file, "w") as f:
                    f.write(f"""# {module_name}

**Note: Documentation generation for this module failed.**

This could be due to import errors or other issues. Please ensure the module is properly installed and importable.

## Module Structure

This is a placeholder for {module_name} documentation.
""")
                print(f"  -> Created placeholder documentation for {module_name}")
            else:
                print(f"  -> Written to {output_file}")
        except Exception as e:
            print(f"  -> Error generating documentation for {module_name}: {str(e)}")
            # Create an error documentation file
            with open(output_file, "w") as f:
                f.write(f"""# {module_name}

**Error: Documentation generation for this module encountered an error: {str(e)}**

Please check that the module exists and is properly installed.
""")
            print(f"  -> Created error documentation for {module_name}")
    
    # Create Home page
    create_home_page(docs_dir)
    
    # Create a sidebar for navigation
    create_sidebar(docs_dir)
    
    # Create a Getting Started guide
    create_getting_started_guide(docs_dir)
    
    # Create an Examples page
    create_examples_page(docs_dir)
    
    # Create a Shapes page
    create_shapes_page(docs_dir)
    
    # Create an Observers page
    create_observers_page(docs_dir)
    
    # Create a README file for the wiki directory
    create_readme(docs_dir)

    print("Documentation generation complete!")

def create_home_page(docs_dir):
    """Create the Home page for the wiki."""
    with open(docs_dir / "Home.md", "w") as f:
        f.write("""# TorchRDIT Documentation

Welcome to the `TorchRDIT` documentation. `TorchRDIT` is an advanced software package designed for the inverse design of meta-optics devices, utilizing an eigendecomposition-free implementation of Rigorous Diffraction Interface Theory (R-DIT). It provides a GPU-accelerated and fully differentiable framework powered by PyTorch, enabling the efficient optimization of photonic structures.

## Key Features

- **Differentiable**: Built on PyTorch for seamless integration with deep learning and optimization
- **High-Performance**: Efficient implementations of RCWA and RDIT algorithms
- **Flexible**: Support for complex geometries, dispersive materials, and various layer structures
- **Extensible**: Easy to integrate into machine learning and inverse design workflows

## Documentation

### User Guide

- [Getting Started](Getting-Started) - Installation and basic usage
- [Examples](Examples) - Detailed examples showing how to use TorchRDIT

### API Reference

- [API Overview](API-Overview)
- [Algorithm Module](Algorithm) - Implementation of electromagnetic solvers
- [Builder Module](Builder) - Fluent API for creating simulations
- [Cell Module](Cell) - Cell geometry definitions
- [Layers Module](Layers) - Layer definitions and operations
- [Materials Module](Materials) - Material property definitions
- [Observers Module](Observers) - Progress tracking and reporting
- [Shapes Module](Shapes) - Shape generation for photonic structures
- [Solver Module](Solver) - Core solver functionality
- [Utils](Utils) - Utility functions
- [Visualization](Visualization) - Tools for visualizing results

## Examples

TorchRDIT comes with several example files in the `examples/` directory

For more detailed explanations of each example, see the [Examples](Examples) page.
""")
        print("  -> Created Home.md")

def create_sidebar(docs_dir):
    """Create the sidebar navigation for the wiki."""
    with open(docs_dir / "_Sidebar.md", "w") as f:
        f.write("""# TorchRDIT Docs

- [Home](Home)
- User Guide
  - [Getting Started](Getting-Started)
  - [Examples](Examples)
    - [Basic Usage](Examples#demo-01-basic-usage)
    - [Parametric Sweeps](Examples#demo-02-parametric-sweeps)
    - [Dispersive Materials](Examples#demo-03-dispersive-materials)
    - [Performance](Examples#demo-04-performance-benchmark)
- API Reference
  - [Overview](API-Overview)
  - [Algorithm](Algorithm)
  - [Builder](Builder)
  - [Cell](Cell)
  - [Layers](Layers)
  - [Materials](Materials)
  - [Observers](Observers)
  - [Shapes](Shapes)
  - [Solver](Solver)
  - [Utils](Utils)
  - [Visualization](Visualization)
""")
        print("  -> Created _Sidebar.md")

def create_getting_started_guide(docs_dir):
    """Create the Getting Started guide for the wiki."""
    with open(docs_dir / "Getting-Started.md", "w") as f:
        f.write("""# Getting Started with TorchRDIT

This guide will help you set up and run your first electromagnetic simulation with TorchRDIT, a PyTorch-based library for rigorous coupled-wave analysis (RCWA) and related differential methods.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib (for visualization)

### Installation from PyPI

```bash
pip install torchrdit
```

### Installation from Source

```bash
git clone https://github.com/username/torchrdit.git
cd torchrdit
pip install -e .
```

## Basic Usage

TorchRDIT provides a builder pattern for configuring and running simulations. Here's a simple example:

```python
import torch
import numpy as np
import torchrdit as tr
from torchrdit.constants import Algorithm, Precision
from torchrdit.solver import get_solver_builder
from torchrdit.utils import create_material

# Set units
um = 1
nm = 1e-3 * um

# Create a builder and configure the simulation
builder = get_solver_builder()
builder.with_algorithm(Algorithm.RCWA)
builder.with_precision(Precision.DOUBLE)
builder.with_real_dimensions([512, 512])
builder.with_k_dimensions([9, 9])
builder.with_wavelengths(np.array([1550 * nm]))

# Create materials
material_si = create_material(name='Silicon', permittivity=12.25)  # n=3.5 for silicon
material_sio2 = create_material(name='SiO2', permittivity=2.25)    # n=1.5 for SiO2

# Add materials to the simulation
builder.add_material(material_si)
builder.add_material(material_sio2)

# Add layers
builder.add_layer({
    "material": "Silicon",
    "thickness": 0.5 * um,
    "is_homogeneous": True,
    "is_optimize": False
})

# Build the solver
device = builder.build()

# Update the transmission medium
device.update_trn_material(trn_material=material_sio2)

# Create a source
source = device.add_source(
    theta=0,     # Normal incidence
    phi=0,       # No azimuthal angle
    pte=1,       # TE polarization
    ptm=0        # No TM polarization
)

# Run the simulation
results = device.solve(source)

# Extract the results
print(f"Transmission: {results['TRN'][0] * 100:.2f}%")
print(f"Reflection: {results['REF'][0] * 100:.2f}%")
```

## Using the Fluent Interface

TorchRDIT also supports a fluent interface style for configuring the simulation:

```python
import torchrdit as tr
from torchrdit.constants import Algorithm, Precision

# Configure and run in a fluent style
device = (tr.solver.get_solver_builder()
    .with_algorithm(Algorithm.RCWA)
    .with_precision(Precision.DOUBLE)
    .with_real_dimensions([512, 512])
    .with_k_dimensions([9, 9])
    .with_wavelengths([1550e-3])  # in μm
    .build())

# Continue with simulation setup...
```

## Working with Patterned Layers

For patterned layers, you can use masks to define the geometry:

```python
# Create a circular pattern
circle_mask = device.get_circle_mask(center=[0, 0], radius=0.5)

# Update a layer with the mask
device.update_er_with_mask(mask=circle_mask, layer_index=0)
```

## Automatic Differentiation

One of the key features of TorchRDIT is its support for automatic differentiation through PyTorch:

```python
# Make mask parameters differentiable
mask.requires_grad = True

# Run simulation
results = device.solve(source)

# Backpropagate to compute gradients
target_value = results['TRN'][0]
target_value.backward()

# Access gradients
gradient = mask.grad
```

For more detailed examples, see the [Examples](Examples) section.
""")
        print("  -> Created Getting-Started.md")

def create_examples_page(docs_dir):
    """Create the Examples page for the wiki."""
    with open(docs_dir / "Examples.md", "w") as f:
        f.write("""# TorchRDIT Examples

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
data = device.solve(src)

# Compute backward pass for optimization
torch.sum(data['TRN'][0]).backward()

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

These examples demonstrate the key capabilities of TorchRDIT, including differentiable simulation, optimization, and support for complex geometries and materials.""")
        print("  -> Created Examples.md")

def create_shapes_page(docs_dir):
    """Create a dedicated Shapes page for the wiki."""
    with open(docs_dir / "Shapes.md", "w") as f:
        f.write("""---
title: "Shape Generation"
category: "Core Components"
tags: ["shapes", "mask", "geometry", "photonics"]
related: ["Solver", "Layers"]
complexity: "intermediate"
---

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

## Detailed Description

The `torchrdit.shapes` module centers around the `ShapeGenerator` class, which provides methods for creating binary mask tensors. These masks represent the geometry of photonic structures and can be used to define the permittivity distribution in patterned layers.

The module supports both Cartesian and non-Cartesian coordinate systems through the use of lattice vectors. This allows for simulating structures with various symmetries, such as hexagonal lattices.

All operations in the module are performed using PyTorch tensors, ensuring compatibility with GPU acceleration and automatic differentiation. This makes the module suitable for gradient-based optimization of photonic structures.

## Main Components

### ShapeGenerator

The main class for generating and manipulating shape masks. It provides methods for creating common shapes and combining them using boolean operations.

```python
class ShapeGenerator:
    \"\"\"Class to generate binary shape masks for photonic structures with lattice vector support.\"\"\"
    
    def __init__(self, XO, YO, rdim, lattice_t1=None, lattice_t2=None):
        \"\"\"Initialize a shape generator with coordinate grids and lattice vectors.\"\"\"
        # ...
```

#### Initialization

The `ShapeGenerator` class can be initialized in three ways:

1. Directly with coordinate grids:
   ```python
   import torch
   import numpy as np
   from torchrdit.shapes import ShapeGenerator
   
   # Create coordinate grids
   x = torch.linspace(-1, 1, 128)
   y = torch.linspace(-1, 1, 128)
   X, Y = torch.meshgrid(x, y, indexing='ij')
   
   # Initialize shape generator
   shape_gen = ShapeGenerator(X, Y, (128, 128))
   ```

2. From an existing solver using the `from_solver` class method:
   ```python
   from torchrdit.solver import create_solver
   from torchrdit.constants import Algorithm
   from torchrdit.shapes import ShapeGenerator
   
   # Create a solver
   solver = create_solver(algorithm=Algorithm.RDIT, rdim=[512, 512], kdim=[7, 7])
   
   # Create a shape generator from the solver
   shape_gen = ShapeGenerator.from_solver(solver)
   ```

3. Using solver's parameter dictionary with `get_shape_generator_params`:
   ```python
   from torchrdit.solver import create_solver
   from torchrdit.constants import Algorithm
   from torchrdit.shapes import ShapeGenerator
   
   # Create a solver
   solver = create_solver(algorithm=Algorithm.RDIT, rdim=[512, 512], kdim=[7, 7])
   
   # Get parameters from solver and create a shape generator
   params = solver.get_shape_generator_params()
   shape_gen = ShapeGenerator(**params)
   
   # This method is useful when you need to customize parameters
   # before creating the shape generator
   params["tfloat"] = torch.float32  # Change precision if needed
   custom_shape_gen = ShapeGenerator(**params)
   ```

#### Shape Generation Methods

##### generate_circle_mask
```python
def generate_circle_mask(self, center=None, radius=0.1, soft_edge=0.001):
    \"\"\"Generate a mask for a circle in Cartesian coordinates.\"\"\"
    # ...
```

Creates a circular mask with the specified center, radius, and edge softness.

##### generate_rectangle_mask
```python
def generate_rectangle_mask(self, center=(0.0, 0.0), width=0.2, height=0.2, angle=0.0, soft_edge=0.001):
    \"\"\"Generate a mask for a rectangle in Cartesian coordinates.\"\"\"
    # ...
```

Creates a rectangular mask with the specified center, dimensions, rotation angle, and edge softness.

##### generate_polygon_mask
```python
def generate_polygon_mask(self, polygon_points, center=None, angle=None, invert=False, soft_edge=0.001):
    \"\"\"Generate a mask for a polygon in Cartesian coordinates.\"\"\"
    # ...
```

Creates a mask for an arbitrary polygon defined by its vertices.

#### Mask Combination

##### combine_masks
```python
def combine_masks(self, mask1, mask2, operation=\"union\"):
    \"\"\"Combine two masks using a specified operation.\"\"\"
    # ...
```

Combines two masks using a boolean operation. Supported operations are:
- `\"union\"`: Logical OR (maximum) of the masks
- `\"intersection\"`: Logical AND (minimum) of the masks
- `\"difference\"`: Absolute difference between the masks
- `\"subtract\"`: Remove mask2 from mask1

## Examples

### Creating a Circle Mask

```python
import torch
from torchrdit.shapes import ShapeGenerator

# Create coordinate grids
x = torch.linspace(-1, 1, 128)
y = torch.linspace(-1, 1, 128)
X, Y = torch.meshgrid(x, y, indexing='ij')
shape_gen = ShapeGenerator(X, Y, (128, 128))

# Generate a circle mask with hard edges
hard_circle = shape_gen.generate_circle_mask(
    center=(0.2, -0.3),
    radius=0.4,
    soft_edge=0
)

# Generate a circle mask with soft edges
soft_circle = shape_gen.generate_circle_mask(
    center=(0.2, -0.3),
    radius=0.4,
    soft_edge=0.02
)
```

### Creating a Rectangle Mask

```python
# Generate a rectangle mask
rectangle = shape_gen.generate_rectangle_mask(
    center=(0.1, 0.1),
    width=0.5,
    height=0.3,
    angle=45,
    soft_edge=0.01
)
```

### Creating a Polygon Mask

```python
# Create a triangle mask
triangle_points = [(-0.2, -0.2), (0.2, -0.2), (0.0, 0.2)]
triangle = shape_gen.generate_polygon_mask(
    polygon_points=triangle_points,
    soft_edge=0.01
)

# Create a hexagon mask
import numpy as np
n = 6  # hexagon
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
radius = 0.3
hexagon_points = [(radius*np.cos(a), radius*np.sin(a)) for a in angles]
hexagon = shape_gen.generate_polygon_mask(
    polygon_points=hexagon_points,
    center=(0.1, 0.1),
    angle=30
)
```

### Combining Masks

```python
# Create two circular masks
circle1 = shape_gen.generate_circle_mask(center=(-0.1, 0), radius=0.3)
circle2 = shape_gen.generate_circle_mask(center=(0.1, 0), radius=0.3)

# Combine masks using different operations
union = shape_gen.combine_masks(circle1, circle2, operation="union")
intersection = shape_gen.combine_masks(circle1, circle2, operation="intersection")
difference = shape_gen.combine_masks(circle1, circle2, operation="difference")
circle1_minus_circle2 = shape_gen.combine_masks(circle1, circle2, operation="subtract")
```

### Creating Complex Patterns

```python
# Create a pattern of four circles in a square arrangement
c1 = shape_gen.generate_circle_mask(center=[0.2, 0.2], radius=0.1)
c2 = shape_gen.generate_circle_mask(center=[0.2, -0.2], radius=0.1)
c3 = shape_gen.generate_circle_mask(center=[-0.2, 0.2], radius=0.1)
c4 = shape_gen.generate_circle_mask(center=[-0.2, -0.2], radius=0.1)

# Combine all circles
mask = c1
mask = shape_gen.combine_masks(mask, c2, operation='union')
mask = shape_gen.combine_masks(mask, c3, operation='union')
mask = shape_gen.combine_masks(mask, c4, operation='union')

# Add a square in the middle
square = shape_gen.generate_rectangle_mask(width=0.2, height=0.2, angle=0)
final_mask = shape_gen.combine_masks(mask, square, operation='union')
```

### Using Masks in Simulations

```python
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
import torch

# Create a solver
solver = create_solver(
    algorithm=Algorithm.RCWA,
    rdim=[512, 512],
    kdim=[7, 7]
)

# Create a shape generator
shape_gen = ShapeGenerator.from_solver(solver)

# Generate a mask
mask = shape_gen.generate_circle_mask(radius=0.3)

# Make the mask differentiable for optimization
mask = mask.clone().detach().requires_grad_(True)

# Apply the mask to a layer
solver.update_er_with_mask(mask=mask, layer_index=0)

# Run the simulation and compute gradients
source = solver.add_source(theta=0, phi=0, pte=1, ptm=0)
results = solver.solve(source)
transmission = results['TRN'][0]
transmission.backward()

# Access gradients for shape optimization
gradient = mask.grad
```

## Common Issues

### Memory Usage

When working with large grid sizes, memory usage can become significant. Consider these tips:
- Use smaller grid sizes during prototyping and increase only when needed
- Release unused tensors to free memory
- If using GPUs with limited memory, consider moving computations to CPU for large grids

### Edge Effects

When using soft edges, be careful with the `soft_edge` parameter:
- Too small values can lead to aliasing effects on coarse grids
- Too large values can cause shapes to lose their definition
- A general guideline is to use `soft_edge` values around 1-5% of the shape's characteristic dimension

### Coordinate Systems

When using non-Cartesian coordinate systems:
- Ensure that lattice vectors are properly defined
- Remember that shape parameters (center, radius, etc.) are in the Cartesian coordinate system
- The coordinate transformation is handled internally by the ShapeGenerator

## See Also
- [Solver Module](Solver) - Core solver functionality
- [Layers Module](Layers) - Layer definitions and operations
- [Utils Module](Utils) - Utility functions

## Keywords
shape, mask, binary mask, photonics, circle, rectangle, polygon, lattice, geometry, structure generation, boolean operations
""")
        print("  -> Created Shapes.md")

def create_observers_page(docs_dir):
    """Create a dedicated Observers page for the wiki."""
    content = """---
title: "Progress Tracking and Reporting"
category: "Utility Components"
tags: ["progress", "monitoring", "feedback", "observer pattern", "console", "progress bar"]
related: ["Solver"]
complexity: "beginner"
---

# Observer Module

## Overview
The `torchrdit.observers` module provides implementations of the Observer pattern for tracking and reporting progress during electromagnetic simulations. These observers can be attached to solvers to receive notifications about various stages of the calculation process, allowing for real-time feedback during potentially long-running simulations.

## Key Features
- Implementation of the Observer design pattern for solver progress tracking
- Console output for detailed progress information
- Progress bar visualization using tqdm (optional)
- Event-based notification system
- Compatible with all solver types (RCWA, RDIT)

## Usage
```python
from torchrdit.solver import create_solver
from torchrdit.observers import ConsoleProgressObserver, TqdmProgressObserver

# Create a solver
solver = create_solver()

# Add observers
verbose_observer = ConsoleProgressObserver(verbose=True)
progress_observer = TqdmProgressObserver()
solver.add_observer(verbose_observer)
solver.add_observer(progress_observer)

# Run the simulation - progress will be reported
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
results = solver.solve(source)
```

## Main Components

### ConsoleProgressObserver
Prints progress information to the console. Useful for detailed tracking of solver stages.

### TqdmProgressObserver
Displays a progress bar using the tqdm library (if available).

## Common Issues
- If tqdm is not installed, TqdmProgressObserver will not be available
- In Jupyter notebooks, console output may not update in real-time

## See Also
- [Solver Module](Solver) - Core solver functionality that generates the events
- [Utils Module](Utils) - Utility functions and tools

## Keywords
observer pattern, progress tracking, console output, progress bar, tqdm, simulation monitoring
"""
    
    with open(docs_dir / "Observers.md", "w") as f:
        f.write(content)
    
    print("  -> Created Observers.md")

def create_readme(docs_dir):
    """Create a README file for the wiki directory explaining how it's generated."""
    with open(docs_dir / "README.md", "w") as f:
        f.write("""# TorchRDIT Documentation

This directory contains automatically generated documentation for the TorchRDIT project.

## How to Update the Documentation

The documentation is generated automatically using pydoc-markdown. To update it:

1. Ensure your docstrings in the code are up-to-date
2. Run the documentation generator script: `python generate_docs.py`
3. The updated documentation will be created in this directory

## Documentation Structure

- **API-Overview.md**: Overview of the entire API
- **Algorithm.md**: Documentation for the algorithm module
- **Builder.md**: Documentation for the builder module
- **Cell.md**: Documentation for the cell module
- **Layers.md**: Documentation for the layers module
- **Materials.md**: Documentation for the materials module
- **Observers.md**: Documentation for the observers module
- **Shapes.md**: Documentation for the shapes module
- **Solver.md**: Documentation for the solver module
- **Utils.md**: Documentation for utility functions
- **Visualization.md**: Documentation for visualization tools
- **Getting-Started.md**: Guide for getting started with TorchRDIT
- **Examples.md**: Code examples showing how to use TorchRDIT
- **_Sidebar.md**: Navigation sidebar for the wiki

## How to Write Good Docstrings

For your documentation to be most effective, follow these guidelines for docstrings:

```python
def calculate_field(mesh, material_properties, frequency):
    \"\"\"
    Calculate electromagnetic fields on a given mesh.
    
    Parameters
    ----------
    mesh : Mesh
        The computational mesh on which to calculate fields.
    material_properties : dict
        Dictionary mapping mesh elements to material properties:
        - 'epsilon': relative permittivity
        - 'mu': relative permeability
        - 'sigma': conductivity
    frequency : float
        Operating frequency in Hz.
        
    Returns
    -------
    ndarray
        Complex E-field values at mesh nodes.
        
    Notes
    -----
    This function solves Maxwell's equations using the FDTD method.
    The governing equation is:
    
    ∇ x (∇ x E) - k₀²εᵣE = 0
    
    where k₀ is the wavenumber and εᵣ is the relative permittivity.
    
    Examples
    --------
    >>> mesh = create_rectangular_mesh(0.1, 10, 10, 10)
    >>> props = {'epsilon': 1.0, 'mu': 1.0, 'sigma': 0.0}
    >>> E = calculate_field(mesh, props, 1e9)
    \"\"\"
```

For classes, document the class purpose and all important methods.

## GitHub Wiki Integration

This documentation is designed to be automatically deployed to the GitHub wiki.
A GitHub Actions workflow in `.github/workflows/wiki.yml` handles this automatically
when changes are pushed to the main branch.

## Important Note

DO NOT modify these files directly. All files in this directory are automatically generated.
If you need to make changes, update the docstrings in the source code or modify the generator script.
""")
        print("  -> Created README.md")

if __name__ == "__main__":
    main() 