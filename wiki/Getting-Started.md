# Getting Started with TorchRDIT

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
