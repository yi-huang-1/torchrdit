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
    print(f"The transmission efficiency at wavelength \t{torchrdit_sim.lam0[i] * 1e3} nm is \t{result['TRN'][i] * 100}%")
    print(f"The reflection efficiency at wavelength \t{torchrdit_sim.lam0[i] * 1e3} nm is \t{result['REF'][i] * 100}%")
