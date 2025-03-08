"""
# Example 2 - GMRF with hexagonal unit cells using R-DIT (Fluent Builder Version)

This example shows the simulation of the guided-mode resonance filter (GMRF) using torchrdit with differentiable RDIT algorithm. The device is composed of a SiO hexagonal grating layer, a SiN waveguide layer and a fused silica substrate.

The GMRF can be found in the following references:

- A. A. Mehta, R. C. Rumpf, Z. A. Roth, and E. G. Johnson, "Guided mode resonance filter as a spectrally selective feedback element in a double-cladding optical fiber laser," IEEE Photonics Technology Letters, vol. 19, pp. 2030–2032, 12 2007.
"""
import numpy as np
import torch
import os

from torchrdit.solver import get_solver_builder
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
h1 = torch.tensor(230 * nm, dtype=torch.float64)
h2 = torch.tensor(345 * nm, dtype=torch.float64)

# lattice vectors of the cell
t1 = torch.tensor([[a/2, -a*np.sqrt(3)/2]], dtype=torch.float64)
t2 = torch.tensor([[a/2, a*np.sqrt(3)/2]], dtype=torch.float64)

# creating materials
material_sio = create_material(name='SiO', permittivity=n_SiO**2)
material_sin = create_material(name='SiN', permittivity=n_SiN**2)
material_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

# Initialize and configure the solver using the fluent Builder pattern with method chaining
dev1rdit = (get_solver_builder()
    .with_algorithm(Algorithm.RDIT)
    .with_precision(Precision.DOUBLE)
    .with_real_dimensions([512, 512])
    .with_k_dimensions([11, 11])
    .with_wavelengths(np.array([1540 * nm, 1550 * nm, 1560 * nm, 1570 * nm]))
    .with_length_unit('um')
    .with_lattice_vectors(t1, t2)
    .with_fff(True)
    .add_material(material_sio)
    .add_material(material_sin)
    .add_material(material_fs)
    .add_layer({
        "material": "SiO",
        "thickness": h1.item(),
        "is_homogeneous": False,
        "is_optimize": True
    })
    .add_layer({
        "material": "SiN",
        "thickness": h2.item(),
        "is_homogeneous": True,
        "is_optimize": False
    })
    .build())

# Update the transmission material
dev1rdit.update_trn_material(trn_material=material_fs)

# print layer information
dev1rdit.get_layer_structure()

src1 = dev1rdit.add_source(theta = theta,
                 phi = phi,
                 pte = pte,
                 ptm = ptm)

# build hexagonal unit cell
c1 = dev1rdit.get_circle_mask(center=[0, b/2], radius=r)
c2 = dev1rdit.get_circle_mask(center=[0, -b/2], radius=r)
c3 = dev1rdit.get_circle_mask(center=[a/2, 0], radius=r)
c4 = dev1rdit.get_circle_mask(center=[-a/2, 0], radius=r)

mask = dev1rdit.combine_masks(mask1=c1, mask2=c2, operation='union')
mask = dev1rdit.combine_masks(mask1=mask, mask2=c3, operation='union')
mask = dev1rdit.combine_masks(mask1=mask, mask2=c4, operation='union')

mask = (1 - mask).to(torch.float64)

mask.requires_grad = True

dev1rdit.update_er_with_mask(mask=mask, layer_index=0)

# plot the layer and save the figure
fig, axes = plt.subplots()
plot_layer(dev1rdit, layer_index=0, func='real', fig_ax=axes, cmap='BuGn', labels=('x (um)','y (um)'), title='layer 0')
script_dir = os.path.dirname(os.path.abspath(__file__))
output_filename = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_layer_0.png")
plt.savefig(output_filename, dpi=300)
plt.close(fig)

data = dev1rdit.solve(src1)

print(f"The transmission efficiency is {data['TRN'][0] * 100}%")
print(f"The reflection efficiency is {data['REF'][0] * 100}%")

# Start back propagation
torch.sum(data['TRN'][0]).backward()

print(f"The gradient with respect to the mask is {torch.mean(mask.grad)}")
