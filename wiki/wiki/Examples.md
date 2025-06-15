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

    h1 = torch.tensor(230 * NM, dtype=torch.float32)
    h2 = torch.tensor([345 * NM], dtype=torch.float32)

    t1 = torch.tensor([[a/2, -a*np.sqrt(3)/2]], dtype=torch.float32)
    t2 = torch.tensor([[a/2, a*np.sqrt(3)/2]], dtype=torch.float32)

    # Create materials
    material_sio = create_material(name='SiO', permittivity=n_SiO**2)
    material_sin = create_material(name='SiN', permittivity=n_SiN**2)
    material_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

    # Create and configure solver using Builder pattern
    builder = get_solver_builder()
    
    # Configure the builder with all necessary parameters
    builder.with_algorithm(Algorithm.RDIT)
    builder.with_precision(Precision.DOUBLE)
    builder.with_real_dimensions([512, 512])
    builder.with_k_dimensions([kdims, kdims])
    builder.with_wavelengths(lams)
    builder.with_length_unit('um')
    builder.with_lattice_vectors(t1, t2)
    
    # Add materials to builder
    builder.add_material(material_sio)
    builder.add_material(material_sin)
    builder.add_material(material_fs)
    
    # Build the solver
    gmrf_sim = builder.build()
    
    gmrf_sim.set_rdit_order(rdit_orders)
    gmrf_sim.update_trn_material(trn_material=material_fs)

    # Add layers
    gmrf_sim.add_layer(material_name='SiO',
                  thickness=h1,
                  is_homogeneous=False,
                  is_optimize=True)

    gmrf_sim.add_layer(material_name='SiN',
                  thickness=h2,
                  is_homogeneous=True,
                  is_optimize=False)

    # Create source
    src = gmrf_sim.add_source(theta=theta,
                     phi=phi,
                     pte=pte,
                     ptm=ptm)

    # Create mask and update permittivity
    shapegen = ShapeGenerator.from_solver(gmrf_sim)
    c1 = shapegen.generate_circle_mask(center=[0, b/2], radius=radius)
    c2 = shapegen.generate_circle_mask(center=[0, -b/2], radius=radius)
    c3 = shapegen.generate_circle_mask(center=[a/2, 0], radius=radius)
    c4 = shapegen.generate_circle_mask(center=[-a/2, 0], radius=radius)

    mask = shapegen.combine_masks(mask1=c1, mask2=c2, operation='union')
    mask = shapegen.combine_masks(mask1=mask, mask2=c3, operation='union')
    mask = shapegen.combine_masks(mask1=mask, mask2=c4, operation='union')

    mask = (1 - mask)
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
        nref_gmrf_rdit[ilam] = data_rdit.reflection[ilam].detach().clone()
        ntrn_gmrf_rdit[ilam] = data_rdit.transmission[ilam].detach().clone()
        ncon_gmrf_rdit[ilam] = nref_gmrf_rdit[ilam] + ntrn_gmrf_rdit[ilam]

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    fig_gmrf, ax_gmrf = plt.subplots(figsize=fig_size)
    ax_gmrf.set_xlim(lamswp0[0], lamswp0[-1])

    ax_gmrf.plot(lamswp0, nref_gmrf_rdit, color='blue', marker='+', linestyle='-', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='Ref-RDIT')
    ax_gmrf.plot(lamswp0, ntrn_gmrf_rdit, color='m', marker='x', linestyle='-', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='Trn-RDIT')

    ax_gmrf.legend()
    ax_gmrf.set_xlabel('Wavelength (um)')
    ax_gmrf.set_ylabel('Transmission/Reflection Efficiency')
    ax_gmrf.set_ylim([0, 1.0])
    ax_gmrf.grid('on')
    
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
        nref_gmrf_org[ilam] = data_org.reflection[ilam].detach().clone()
        ntrn_gmrf_org[ilam] = data_org.transmission[ilam].detach().clone()
        ncon_gmrf_org[ilam] = nref_gmrf_org[ilam] + ntrn_gmrf_org[ilam]

        nref_gmrf_opt[ilam] = data_opt.reflection[ilam].detach().clone()
        ntrn_gmrf_opt[ilam] = data_opt.transmission[ilam].detach().clone()
        ncon_gmrf_opt[ilam] = nref_gmrf_opt[ilam] + ntrn_gmrf_opt[ilam]

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 10})
    
    fig_gmrf, ax_gmrf = plt.subplots(figsize=fig_size)
    ax_gmrf.set_xlim(lamswp0[0], lamswp0[-1])
    
    # Plot original data
    ax_gmrf.plot(lamswp0, nref_gmrf_org, color='red', marker='', linestyle='-.', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='R-Init')
    ax_gmrf.plot(lamswp0, ntrn_gmrf_org, color='green', marker='', linestyle='-', 
                 linewidth=1, markersize=8, markevery=markeverypoints, label='T-Init')
    
    # Plot optimized data
    ax_gmrf.plot(lamswp0, nref_gmrf_opt, color='blue', marker='x', linestyle='-', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='R-Opt')
    ax_gmrf.plot(lamswp0, ntrn_gmrf_opt, color='m', marker='', linestyle='--', 
                 linewidth=1, markersize=4, markevery=markeverypoints, label='T-Opt')

    ax_gmrf.legend(loc='center left', bbox_to_anchor=(0.6, 0.5), frameon=False)
    ax_gmrf.set_xlabel('Wavelength [um]')
    ax_gmrf.set_ylabel('T/R')
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
    c1 = shapegen.generate_circle_mask(center=[0, b/2], radius=radius)
    c2 = shapegen.generate_circle_mask(center=[0, -b/2], radius=radius)
    c3 = shapegen.generate_circle_mask(center=[a/2, 0], radius=radius)
    c4 = shapegen.generate_circle_mask(center=[-a/2, 0], radius=radius)

    mask = shapegen.combine_masks(mask1=c1, mask2=c2, operation='union')
    mask = shapegen.combine_masks(mask1=mask, mask2=c3, operation='union')
    mask = shapegen.combine_masks(mask1=mask, mask2=c4, operation='union')

    mask = (1 - mask)
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
    b = a * np.sqrt(3)

    h1 = torch.tensor(230 * NM, dtype=torch.float32)
    h2 = torch.tensor([345 * NM], dtype=torch.float32)

    t1 = torch.tensor([[a/2, -a*np.sqrt(3)/2]], dtype=torch.float32)
    t2 = torch.tensor([[a/2, a*np.sqrt(3)/2]], dtype=torch.float32)

    material_sio = create_material(name='SiO', permittivity=n_SiO**2)
    material_sin = create_material(name='SiN', permittivity=n_SiN**2)
    material_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

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
    builder.with_length_unit('um')
    builder.with_lattice_vectors(t1, t2)
    
    # Add materials to builder
    builder.add_material(material_sio)
    builder.add_material(material_sin)
    builder.add_material(material_fs)
    
    # Build the solver
    gmrf_sim_rdit = builder.build()

    gmrf_sim_rdit.set_rdit_order(r_dit_order)
    gmrf_sim_rdit.update_trn_material(trn_material='FusedSilica')

    # Add layers
    gmrf_sim_rdit.add_layer(material_name='SiO',
                thickness=h1,
                is_homogeneous=False,
                is_optimize=True)

    gmrf_sim_rdit.add_layer(material_name='SiN',
                thickness=h2,
                is_homogeneous=True,
                is_optimize=False)

    # Create source
    src_rdit = gmrf_sim_rdit.add_source(theta=0 * DEGREES,
                    phi=0 * DEGREES,
                    pte=1,
                    ptm=0)
                    
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150], gamma=0.9)

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
    r0_rdit = torch.tensor(400 * NM)
    r0_rdit.requires_grad = True
    
    # Simulate at a single wavelength
    lam00 = np.array([1540 * NM])
    data_rdit = GMRF_simulator(r0_rdit, lam00, rdit_orders=10, kdims=15, is_showinfo=False)

    # Print efficiency and calculate gradient
    print(f"The T efficiency (R-DIT) is {data_rdit.transmission[0] * 100:.2f}%")
    print(f"The R efficiency (R-DIT) is {data_rdit.reflection[0] * 100:.2f}%")

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
    data_optimized_rdit = GMRF_simulator(
        r_optimized.detach(), lamswp_gmrf, rdit_orders=10, kdims=9, is_showinfo=True)
    
    # Plot comparison
    fig, ax = plot_spectrum_compare_opt(
        lamswp0=lamswp_gmrf, data_org=data_gmrfswp_rdit, data_opt=data_optimized_rdit)
    
    # Find the transmission dip for both the original and optimized designs
    orig_idx = np.argmin(data_gmrfswp_rdit.transmission.detach().numpy())
    opt_idx = np.argmin(data_optimized_rdit.transmission.detach().numpy())
    
    print(f"Original resonance wavelength: {lamswp_gmrf[orig_idx]:.4f} um")
    print(f"Optimized resonance wavelength: {lamswp_gmrf[opt_idx]:.4f} um")
    print(f"Target wavelength: {lam_opt[0]:.4f} um")
    
    print("\nExample completed successfully!")
    print("Plots saved in the current directory.")

if __name__ == "__main__":
    main()
```
