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