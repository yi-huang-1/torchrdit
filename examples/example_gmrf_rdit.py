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
    print(f"The transmission efficiency at wavelength \t{dev1.lam0[i] * 1e3} nm is \t{data['TRN'][i] * 100}%")
    print(f"The reflection efficiency at wavelength \t{dev1.lam0[i] * 1e3} nm is \t{data['REF'][i] * 100}%")
