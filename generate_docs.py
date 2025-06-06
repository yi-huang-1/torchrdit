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

def create_material_proxy_page(docs_dir):
    """Create a MaterialProxy page for the wiki."""
    # First check if the auto-generated API file exists
    api_file = docs_dir / "MaterialProxy-API.md"
    
    # Create the custom intro content
    intro_content = """# Material Proxy Module

## Overview
The `torchrdit.material_proxy` module provides classes for loading, processing, and converting material data with appropriate unit handling for electromagnetic simulations. It serves as a foundation for the materials system in TorchRDIT.

## Key Components

### UnitConverter
A class for handling unit conversions for wavelength, frequency, and derived quantities:
- Converting between different length units
- Converting between different frequency units
- Converting between wavelength and frequency domains

### MaterialDataProxy
A class implementing the proxy pattern for material data handling:
- Loading material data from files with different formats (eps, n/k)
- Processing data with appropriate unit conversions
- Extracting material properties at specific wavelengths

## Usage Examples

```python
# Basic unit conversion
from torchrdit.material_proxy import UnitConverter
converter = UnitConverter()

# Convert from nanometers to micrometers
wavelength_um = converter.convert_length(1550, 'nm', 'um')
print(f"1550 nm = {wavelength_um} μm")  # 1550 nm = 1.55 μm

# Convert from frequency to wavelength
wavelength = converter.freq_to_wavelength(193.5, 'thz', 'um')
print(f"193.5 THz = {wavelength:.2f} μm")  # 193.5 THz = 1.55 μm

# Loading material data
from torchrdit.material_proxy import MaterialDataProxy
import numpy as np

# Initialize proxy
proxy = MaterialDataProxy()

# Load data from file (wavelength-permittivity format with units in um)
si_data = proxy.load_data('Si_data.txt', 'wl-eps', 'um')

# Extract permittivity at specific wavelengths
wavelengths = np.array([1.3, 1.55, 1.7])
eps_real, eps_imag = proxy.extract_permittivity(si_data, wavelengths)
```

## API Reference

Below is the complete API reference for the material_proxy module, automatically generated from the source code.

"""
    
    # Create the MaterialProxy.md file
    with open(docs_dir / "MaterialProxy.md", "w") as f:
        f.write(intro_content)
        
        # If the auto-generated API file exists, append its content (skipping the header)
        if api_file.exists():
            with open(api_file, "r") as api_f:
                api_content = api_f.read()
                
                # Find the first heading and skip everything before it
                import re
                match = re.search(r'^#', api_content, re.MULTILINE)
                if match:
                    api_content = api_content[match.start():]
                
                f.write(api_content)
        else:
            # If API file doesn't exist, include a warning
            f.write("""
**Note: Automatic API documentation could not be generated. 
Please ensure that the module is properly installed and importable.**
""")
    
    print("  -> Created MaterialProxy.md")
    
    # Try to remove the temporary API file to keep things clean
    if api_file.exists():
        try:
            api_file.unlink()
        except:
            pass

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
    
    # Create a Results page
    create_results_page(docs_dir)
    
    # Create a Constants page
    create_constants_page(docs_dir)
    
    # Create a MaterialProxy page
    create_material_proxy_page(docs_dir)
    
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
- [Constants Module](Constants) - Physical constants and enumerations
- [Layers Module](Layers) - Layer definitions and operations
- [Materials Module](Materials) - Material property definitions
- [Material Proxy Module](MaterialProxy) - Material data handling and unit conversion
- [Observers Module](Observers) - Progress tracking and reporting
- [Results Module](Results) - Structured data containers for simulation results
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
  - [Constants](Constants)
  - [Layers](Layers)
  - [Materials](Materials)
  - [MaterialProxy](MaterialProxy)
  - [Observers](Observers)
  - [Results](Results)
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
results = device.solve(source) # SolverResults object

# Extract the results
print(f"Transmission: {results.transmission * 100:.2f}%")
print(f"Reflection: {results.reflection * 100:.2f}%")
                
print("Phase of the transmission field x-component: ", torch.angle(results.transmission_field.x))
print("Phase of the reflection field x-component: ", torch.angle(results.reflection_field.x))
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
from torchrdit.shapes import ShapeGenerator

# Create a shape generator
shape_generator = ShapeGenerator.from_solver(device)

# Generate a circle mask
circle_mask = shape_generator.generate_circle_mask(center=[0, 0], radius=0.5)

# Update a layer with the mask
device.update_er_with_mask(mask=circle_mask, layer_index=0)
```

## Automatic Differentiation

One of the key features of TorchRDIT is its support for automatic differentiation through PyTorch:

```python
# Make mask parameters differentiable
mask.requires_grad = True

# Run simulation
results = device.solve(source) # SolverResults object

# Backpropagate to compute gradients
target_transmission = 1.0
loss = torch.abs(results.transmission[0] - target_transmission)
loss.backward()

# Access gradients
gradient = mask.grad
```

For more detailed examples, see the [Examples](Examples) section.
""")
        print("  -> Created Getting-Started.md")

def create_examples_page(docs_dir):
    """Create the Examples page for the wiki."""
    # Get the examples directory
    examples_dir = Path("examples")
    
    # Check if examples directory exists
    examples_exists = examples_dir.exists()
    example_files = []
    
    # If examples not found in root, try src directory structure
    if not examples_exists:
        examples_dir = Path("src/examples")
        examples_exists = examples_dir.exists()
    
    if examples_exists:
        # Find all Python example files
        example_files = sorted([f for f in examples_dir.glob("*.py") if f.is_file()])
    
    # Create base examples content
    content = """# TorchRDIT Examples

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

"""

    # If examples directory exists, add each example with a code block
    if examples_exists and example_files:
        for example_file in example_files:
            example_name = example_file.stem
            example_content = ""
            
            # Read the file content
            try:
                with open(example_file, 'r') as f:
                    example_content = f.read()
                
                # Extract docstring (if it exists)
                docstring = ""
                if example_content.strip().startswith('"""'):
                    start = example_content.find('"""') + 3
                    end = example_content.find('"""', start)
                    if end > start:
                        docstring = example_content[start:end].strip()
                
                # Add example section
                content += f"\n### {example_name}\n\n"
                
                # Add docstring if available
                if docstring:
                    content += f"{docstring}\n\n"
                
                # Add code block with the full content (no truncation)
                content += f"```python\n{example_content}\n```\n"
                
            except Exception as e:
                content += f"\n### {example_name}\n\n**Error loading example: {str(e)}**\n"
    else:
        content += "\n**No example files found in the examples directory.**\n"

    with open(docs_dir / "Examples.md", "w") as f:
        f.write(content)
        print("  -> Created Examples.md")

def create_shapes_page(docs_dir):
    """Create a dedicated Shapes page for the wiki."""
    # First check if the auto-generated API file exists
    api_file = docs_dir / "Shapes-API.md"
    
    # Create the custom intro content
    intro_content = """# Shape Generation Module

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

"""
    
    # Create the Shapes.md file
    with open(docs_dir / "Shapes.md", "w") as f:
        f.write(intro_content)
        
        # If the auto-generated API file exists, append its content (skipping the header)
        if api_file.exists():
            with open(api_file, "r") as api_f:
                api_content = api_f.read()
                
                # Find the first heading and skip everything before it
                import re
                match = re.search(r'^#', api_content, re.MULTILINE)
                if match:
                    api_content = api_content[match.start():]
                
                f.write(api_content)
        else:
            # If API file doesn't exist, include a warning
            f.write("""
**Note: Automatic API documentation could not be generated. 
Please ensure that the module is properly installed and importable.**
""")
    
    print("  -> Created Shapes.md")
    
    # Try to remove the temporary API file to keep things clean
    if api_file.exists():
        try:
            api_file.unlink()
        except:
            pass

def create_observers_page(docs_dir):
    """Create a dedicated Observers page for the wiki."""
    # First check if the auto-generated API file exists
    api_file = docs_dir / "Observers-API.md"
    
    # Create the custom intro content
    intro_content = """# Observer Module

## Overview
The `torchrdit.observers` module provides implementations of the Observer pattern for tracking and reporting progress during electromagnetic simulations.

## Key Components
The module includes several observer classes for monitoring simulation progress:

- **ConsoleProgressObserver**: Prints progress information to the console
- **TqdmProgressObserver**: Displays a progress bar using the tqdm library (if available)

## Usage Examples

```python
from torchrdit.solver import create_solver
from torchrdit.observers import ConsoleProgressObserver
from torchrdit.constants import Algorithm

# Create a solver and add a console observer
solver = create_solver(algorithm=Algorithm.RCWA)
observer = ConsoleProgressObserver(verbose=True)
solver.add_observer(observer)

# Run the solver - progress will be reported to the console
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
result = solver.solve(source)
```

## API Reference

Below is the complete API reference for the observers module, automatically generated from the source code.

"""
    
    # Create the Observers.md file
    with open(docs_dir / "Observers.md", "w") as f:
        f.write(intro_content)
        
        # If the auto-generated API file exists, append its content (skipping the header)
        if api_file.exists():
            with open(api_file, "r") as api_f:
                api_content = api_f.read()
                
                # Find the first heading and skip everything before it
                import re
                match = re.search(r'^#', api_content, re.MULTILINE)
                if match:
                    api_content = api_content[match.start():]
                
                f.write(api_content)
        else:
            # If API file doesn't exist, include a warning
            f.write("""
**Note: Automatic API documentation could not be generated. 
Please ensure that the module is properly installed and importable.**
""")
    
    print("  -> Created Observers.md")
    
    # Try to remove the temporary API file to keep things clean
    if api_file.exists():
        try:
            api_file.unlink()
        except:
            pass

def create_results_page(docs_dir):
    """Create the Results page for the wiki."""
    # First check if the auto-generated API file exists
    api_file = docs_dir / "Results-API.md"
    
    # Create the custom intro content
    intro_content = """# Results Module

## Overview

The `torchrdit.results` module provides structured data containers for electromagnetic simulation results. It defines dataclasses that organize field components, scattering matrices, wave vectors, and diffraction efficiencies, making simulation results more accessible and easier to work with.

## Key Components

The module consists of several dataclasses that organize different aspects of simulation results:

### ScatteringMatrix

Contains the four components of the scattering matrix:

```python
@dataclass
class ScatteringMatrix:
    \"\"\"Scattering matrix components for electromagnetic simulation\"\"\"
    S11: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
    S12: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
    S21: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
    S22: torch.Tensor  # (n_freqs, 2*n_harmonics, 2*n_harmonics)
```

### FieldComponents

Organizes the x, y, and z components of electromagnetic fields:

```python
@dataclass
class FieldComponents:
    \"\"\"Field components in x, y, z directions\"\"\"
    x: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
    y: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
    z: torch.Tensor  # (n_freqs, kdim[0], kdim[1])
```

### WaveVectors

Stores wave vector components for the simulation:

```python
@dataclass
class WaveVectors:
    \"\"\"Wave vector components for the simulation\"\"\"
    kx: torch.Tensor  # (kdim[0], kdim[1])
    ky: torch.Tensor  # (kdim[0], kdim[1])
    kinc: torch.Tensor  # (n_freqs, 3)
    kzref: torch.Tensor  # (n_freqs, kdim[0]*kdim[1])
    kztrn: torch.Tensor  # (n_freqs, kdim[0]*kdim[1])
```

### SolverResults

The main results container that includes reflection and transmission data, field components, and methods for analyzing diffraction orders.

## Usage Examples

### Basic Usage

```python
import torch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm

# Create a solver
solver = create_solver(algorithm=Algorithm.RCWA)

# Set up the simulation
# ...

# Run the simulation
source = solver.add_source(theta=0, phi=0, pte=1, ptm=0)
results = solver.solve(source)  # Returns a SolverResults object

# Access overall efficiencies
print(f"Reflection: {results.reflection[0].item():.3f}")
print(f"Transmission: {results.transmission[0].item():.3f}")
```

### Accessing Field Components

```python
# Get field components for the zero-order diffraction
tx, ty, tz = results.get_zero_order_transmission()

# Calculate field amplitude and phase
amplitude = torch.abs(tx[0])  # Amplitude of x-component (first wavelength)
phase = torch.angle(tx[0])    # Phase in radians
```

### Analyzing Diffraction Orders

```python
# Get efficiency of specific diffraction order
order_1_1_efficiency = results.get_order_transmission_efficiency(order_x=1, order_y=1)

# Get all available diffraction orders
all_orders = results.get_all_diffraction_orders()

# Find all propagating orders
propagating = results.get_propagating_orders(wavelength_idx=0)
print(f"Propagating orders: {propagating}")
```

### Backward Compatibility

```python
# For code that expects the old dictionary format
data_dict = results.to_dict()

# Access data using the old dictionary keys
trn = data_dict['TRN']
ref = data_dict['REF']
```

## API Reference

Below is the complete API reference for the results module, automatically generated from the source code.

"""

    # Create the Results.md file
    with open(docs_dir / "Results.md", "w") as f:
        f.write(intro_content)
        
        # If the auto-generated API file exists, append its content (skipping the header)
        if api_file.exists():
            with open(api_file, "r") as api_f:
                # Skip the first line which is usually the module header
                api_content = api_f.read()
                
                # Find the first heading and skip everything before it
                import re
                match = re.search(r'^#', api_content, re.MULTILINE)
                if match:
                    api_content = api_content[match.start():]
                
                f.write(api_content)
        else:
            # If API file doesn't exist, include a warning
            f.write("""
**Note: Automatic API documentation could not be generated. 
Please ensure that the module is properly installed and importable.**

## Key Methods

### Factory and Conversion Methods

- `from_dict`: Create a SolverResults instance from a dictionary
- `to_dict`: Convert to dictionary for backward compatibility

### Diffraction Order Analysis

- `get_diffraction_order_indices`: Get indices for specific diffraction orders
- `get_zero_order_transmission/reflection`: Get field components for zero-order diffraction
- `get_order_transmission/reflection_efficiency`: Get efficiency for specific orders
- `get_all_diffraction_orders`: List all available diffraction orders
- `get_propagating_orders`: List only propagating diffraction orders
""")
    
    print("  -> Created Results.md")
    
    # Try to remove the temporary Results-API.md file to keep things clean
    if api_file.exists():
        try:
            api_file.unlink()
        except:
            pass

def create_constants_page(docs_dir):
    """Create a Constants page for the wiki."""
    # First check if the auto-generated API file exists
    api_file = docs_dir / "Constants-API.md"
    
    # Create the custom intro content
    intro_content = """# Constants Module

## Overview
The `torchrdit.constants` module defines physical constants, unit conversion factors, and enumerations used throughout the TorchRDIT package for electromagnetic simulations.

## Key Components

### Physical Constants
- `EPS_0`: Vacuum permittivity (8.85418782e-12 F/m)
- `MU_0`: Vacuum permeability (1.25663706e-6 H/m)
- `C_0`: Speed of light in vacuum (2.99792458e8 m/s)
- `ETA_0`: Vacuum impedance (376.730313668 Ω)
- `Q_E`: Elementary charge (1.602176634e-19 C)

### Unit Conversion Dictionaries
- `frequnit_dict`: Frequency unit conversion factors (Hz to PHz)
- `lengthunit_dict`: Length unit conversion factors (meter to angstrom)

### Enumerations
- `Algorithm`: Supported electromagnetic solver algorithms (RCWA, RDIT)
- `Precision`: Numerical precision options (SINGLE, DOUBLE)

## Usage Examples

```python
# Using physical constants
from torchrdit.constants import EPS_0, MU_0, C_0
# Calculate the refractive index from relative permittivity
epsilon_r = 2.25  # SiO2
n = (epsilon_r)**0.5
print(f"Refractive index: {n}")  # Refractive index: 1.5

# Converting between units
from torchrdit.constants import lengthunit_dict
# Convert 1550 nm to meters
wavelength_nm = 1550
wavelength_m = wavelength_nm * lengthunit_dict['nm']
print(f"Wavelength in meters: {wavelength_m}")  # Wavelength in meters: 1.55e-06

# Using algorithm enumeration
from torchrdit.constants import Algorithm
# Create a solver with specific algorithm
from torchrdit.solver import create_solver
solver = create_solver(algorithm=Algorithm.RCWA)
```

## API Reference

Below is the complete API reference for the constants module, automatically generated from the source code.

"""
    
    # Create the Constants.md file
    with open(docs_dir / "Constants.md", "w") as f:
        f.write(intro_content)
        
        # If the auto-generated API file exists, append its content (skipping the header)
        if api_file.exists():
            with open(api_file, "r") as api_f:
                api_content = api_f.read()
                
                # Find the first heading and skip everything before it
                import re
                match = re.search(r'^#', api_content, re.MULTILINE)
                if match:
                    api_content = api_content[match.start():]
                
                f.write(api_content)
        else:
            # If API file doesn't exist, include a warning
            f.write("""
**Note: Automatic API documentation could not be generated. 
Please ensure that the module is properly installed and importable.**
""")
    
    print("  -> Created Constants.md")
    
    # Try to remove the temporary API file to keep things clean
    if api_file.exists():
        try:
            api_file.unlink()
        except:
            pass

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
- **Constants.md**: Documentation for physical constants and enumerations
- **Layers.md**: Documentation for the layers module
- **Materials.md**: Documentation for the materials module
- **MaterialProxy.md**: Documentation for the material data proxy and unit conversion
- **Observers.md**: Documentation for the observers module
- **Results.md**: Documentation for the results module
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
    ```python
    mesh = create_rectangular_mesh(0.1, 10, 10, 10)
    props = {'epsilon': 1.0, 'mu': 1.0, 'sigma': 0.0}
    E = calculate_field(mesh, props, 1e9)
    ```
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