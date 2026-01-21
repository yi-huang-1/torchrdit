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
    "is_optimize": True,
    "slice_count": 3,  # Optional: reuse a sub-slice scattering block three times
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
            "is_optimize": True,
            "slice_count": 3,
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

> **Tip:** Every `add_layer` call accepts an optional `slice_count` field. When greater than one, TorchRDIT computes the scattering response for a single sub-slice and reuses it `slice_count` times via the Redheffer product. Non-integer or non-positive inputs fall back to `1`, so existing configurations continue to work without changes.

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
# Dispersive materials are typically specified via `dielectric_file` in a spec/config.
# When running through `tr.simulate(spec)` / `tr.optimize(spec, ...)` or loading a JSON
# config/spec file, relative `dielectric_file` paths resolve automatically:
# spec/config directory → caller script directory → current working directory.
#
# Recommended usage (portable, no `base_path` required):
import torchrdit as tr

spec = {
    "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [64, 64], "harmonics": [3, 3]},
    "materials": {
        "SiC": {"dielectric_dispersion": True, "dielectric_file": "Si_C-e.txt", "data_format": "freq-eps", "data_unit": "thz"},
    },
    "layers": [{"material": "SiC", "thickness": 0.1, "is_homogeneous": True}],
    "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
}
results = tr.simulate(spec)           # dict spec
results2 = tr.simulate("spec.json")   # JSON spec (recommended for portability)
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


**No example files found in the examples directory.**
