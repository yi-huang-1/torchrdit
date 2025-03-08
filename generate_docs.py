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

TorchRDIT comes with a variety of examples in the `examples/` directory:

- Basic usage examples (Demo-01)
- Parametric sweeps (Demo-02)
- Dispersive materials (Demo-03)
- Performance benchmarks (Demo-04)

See the [Examples](Examples) page for more details.
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
- Fluent API vs Function-style API

### Guided-Mode Resonance Filters (GMRF)
- Simple GMRFs
- Hexagonal unit cells
- Angular sweep

### Material Properties
- Homogeneous materials
- Dispersive materials
- Material fitting

### Advanced Topics
- Optimization
- Automatic differentiation
- Performance comparison

## Example Files Overview

Below is a summary of key example files available in the `examples/` directory:

### Demo-01: Basic Usage

* **Demo-01a.py**: GMRF with hexagonal unit cells using RCWA
  * Demonstrates basic setup of a guided-mode resonance filter
  * Uses hexagonal unit cells with circular holes
  * Shows both function and fluent API styles

* **Demo-01b.py**: Simpler planar structure examples
  * Basic multilayer stack simulation
  * Shows reflection and transmission calculations

### Demo-02: Parametric Sweeps

* **Demo-02a.py**: Angle sweep simulation
  * Shows how to perform parameter sweeps over incident angles
  * Generates reflection vs angle plots

* **Demo-02b.py**: Wavelength sweep simulation
  * Demonstrates spectral response calculation
  * Shows how to analyze resonance features

### Demo-03: Dispersive Materials

* **Demo-03a.py**: GMRF with dispersive materials
  * Shows how to load and use dispersive material data
  * Demonstrates permittivity fitting
  * Uses SiC and SiO2 data files

* **Demo-03b.py**: Spectral analysis with dispersive materials
  * Performs wavelength sweep with realistic material models
  * Shows how dispersion affects resonance features

### Demo-04: Performance Benchmark

* **Demo-04.py**: Performance comparison
  * Compares different algorithm configurations
  * Provides timing and accuracy benchmarks
  * Shows scaling with problem size

## Running the Examples

To run any of the examples, navigate to the repository root and run:

```bash
python examples/Demo-01a.py
```

Most examples generate visualization outputs automatically, which are saved to the same directory.

## Key Features Demonstrated

### Structure Building

```python
# Using masks to create complex geometries
c1 = device.get_circle_mask(center=[0, b/2], radius=r)
c2 = device.get_circle_mask(center=[0, -b/2], radius=r)
c3 = device.get_circle_mask(center=[a/2, 0], radius=r)
c4 = device.get_circle_mask(center=[-a/2, 0], radius=r)

mask = device.combine_masks(mask1=c1, mask2=c2, operation='union')
mask = device.combine_masks(mask1=mask, mask2=c3, operation='union')
mask = device.combine_masks(mask1=mask, mask2=c4, operation='union')
```

### Dispersive Materials

```python
# Creating materials with dispersion
material_sic = create_material(
    name='SiC', 
    dielectric_dispersion=True, 
    user_dielectric_file='Si_C-e.txt', 
    data_format='freq-eps', 
    data_unit='thz'
)
```

### Visualization

```python
# Visualizing layer patterns
plot_layer(device, layer_index=0, func='real', fig_ax=axes, cmap='BuGn', 
           labels=('x (um)','y (um)'), title='layer 0')

# Visualizing fitted permittivity for dispersive materials
display_fitted_permittivity(device, fig_ax=axes)
```

### Optimization

```python
# Enable gradient calculation for optimization
mask.requires_grad = True

# Run simulation
data = device.solve(source)

# Backpropagation
torch.sum(data['TRN'][0]).backward()

# Update design based on gradients
# ...
```

## Example Results

Many examples produce visualizations of the simulated structures and their optical responses. Here are some examples of what you might see:

- Layer permittivity patterns
- Reflection and transmission spectra
- Field distributions
- Fitted material dispersion curves

To see these examples in action, refer to the images in the `examples/` directory, such as:
- `Demo-01a_layer_0.png`
- `Demo-03a_fitted_dispersive.png`
- `Demo-03b_spectrum.png`
""")
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