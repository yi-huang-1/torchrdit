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
- [Solver Module](Solver) - Core solver functionality
- [Utils](Utils) - Utility functions
- [Visualization](Visualization) - Tools for visualizing results

## Examples

TorchRDIT comes with several example files in the `examples/` directory:

### Basic Usage Examples (Demo-01)

- **Demo-01a.py**: GMRF with hexagonal unit cells using RCWA (standard builder pattern)
- **Demo-01a-fluent.py**: Same example using fluent builder pattern
- **Demo-01a-function.py**: Same example using function-based builder pattern
- **Demo-01b.py**: Multilayer structure configuration with standard builder
- **Demo-01b-fluent.py**: Multilayer structure with fluent API
- **Demo-01b-function.py**: Multilayer structure with function-based API

### R-DIT Algorithm Examples (Demo-02)

- **Demo-02a.py**: GMRF with hexagonal unit cells using R-DIT algorithm
- **Demo-02b.py**: Same example with fluent builder pattern

### Dispersive Materials (Demo-03)

- **Demo-03a.py**: GMRF with dispersive materials and permittivity fitting
- **Demo-03b.py**: Spectral analysis with dispersive materials

### Design Optimization (Demo-04)

- **Demo-04.py**: Demonstration of parameter optimization using automatic differentiation

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
builder.with_fff(True)  # Use Fast Fourier Factorization

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
    .with_fff(True)
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

## Example Files Overview

Below is a summary of key example files available in the `examples/` directory:

### Demo-01: Basic Usage Examples

* **Demo-01a.py**: GMRF with hexagonal unit cells using RCWA (standard builder pattern)
  * Demonstrates basic setup of a guided-mode resonance filter
  * Uses hexagonal unit cells with circular holes
  * Shows the standard builder pattern API
  * Demonstrates back-propagation for gradient calculation

* **Demo-01a-fluent.py**: Same example using fluent builder pattern
  * Same simulation as Demo-01a but using the fluent API with method chaining
  * Shows how to configure the solver in a single statement chain
  * Provides a more concise, readable approach for complex configurations

* **Demo-01a-function.py**: Same example using function-based builder pattern
  * Demonstrates the function-based approach to solver configuration
  * Shows how to define reusable configuration functions
  * Illustrates creating solvers from builder functions

* **Demo-01b.py**: Multilayer structure configuration with standard builder
  * Shows how to set up and convert from configuration dictionaries
  * Demonstrates the builder pattern for multilayer devices
  * Includes explicit configuration dictionary to builder conversion

* **Demo-01b-fluent.py** and **Demo-01b-function.py**: Multilayer structure with alternative API styles
  * Shows the same multilayer example in both fluent and function-based styles
  * Provides comparison of different API approaches

### Demo-02: R-DIT Algorithm Examples

* **Demo-02a.py**: GMRF with hexagonal unit cells using R-DIT algorithm
  * Similar to Demo-01a but uses the R-DIT algorithm instead of RCWA
  * Demonstrates the same geometry but with a different solver approach
  * Shows how to configure the R-DIT algorithm parameters

* **Demo-02b.py**: Same example with fluent builder pattern
  * Demonstrates the R-DIT algorithm using fluent API style
  * Shows the same structure as Demo-02a but with a different coding style

### Demo-03: Dispersive Materials

* **Demo-03a.py**: GMRF with dispersive materials and permittivity fitting
  * Shows how to load and use dispersive material data from files
  * Demonstrates permittivity fitting for realistic material models
  * Uses SiC and SiO2 data files with frequency-dependent properties
  * Includes visualization of fitted permittivity

* **Demo-03b.py**: Spectral analysis with dispersive materials
  * Performs wavelength sweep with realistic material models
  * Shows how dispersion affects resonance features
  * Calculates and visualizes spectra with material dispersion effects

### Demo-04: Design Optimization

* **Demo-04.py**: Demonstration of parameter optimization
  * Demonstrates how to set up and perform automatic design optimization
  * Uses gradient descent to tune the radius of the holes in a GMRF
  * Shows how to define an objective function for optimization
  * Provides a complete workflow for parameter tuning using automatic differentiation

## Running the Examples

To run any of the examples, navigate to the repository root and run:

```bash
python examples/Demo-01a.py
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
c1 = device.get_circle_mask(center=[0, b/2], radius=r)
c2 = device.get_circle_mask(center=[0, -b/2], radius=r)
c3 = device.get_circle_mask(center=[a/2, 0], radius=r)
c4 = device.get_circle_mask(center=[-a/2, 0], radius=r)

# Combine masks using boolean operations
mask = device.combine_masks(mask1=c1, mask2=c2, operation='union')
mask = device.combine_masks(mask1=mask, mask2=c3, operation='union')
mask = device.combine_masks(mask1=mask, mask2=c4, operation='union')

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
    
    ∇ × (∇ × E) - k₀²εᵣE = 0
    
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