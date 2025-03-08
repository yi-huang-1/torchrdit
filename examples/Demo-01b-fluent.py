"""
# Updated Example 1 - GMRF with hexagonal unit cells using RCWA (Fluent Builder Pattern)

This example shows the simulation of the guided-mode resonance filter (GMRF) 
using torchrdit with the fluent Builder pattern and method chaining.
"""

import numpy as np
import torch
from torchrdit.solver import get_solver_builder
from torchrdit.constants import Algorithm, Precision
from torchrdit.utils import create_material

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

# Initialize the simulation using the fluent builder pattern with method chaining
torchrdit_sim = (get_solver_builder()
    # Configure the solver parameters
    .with_algorithm(Algorithm.RCWA)
    .with_precision(Precision.DOUBLE)
    .with_wavelengths(np.array([1540e-3, 1550e-3, 1560e-3, 1570e-3]))
    .with_length_unit("um")
    .with_real_dimensions([512, 512])
    .with_k_dimensions([9, 9])
    .with_fff(True)
    
    # Set lattice vectors
    .with_lattice_vectors(
        torch.tensor([a/2, -a*np.sqrt(3)/2], dtype=torch.float32),
        torch.tensor([a/2, a*np.sqrt(3)/2], dtype=torch.float32)
    )
    
    # Create and add materials
    .add_material(create_material(name='SiO', permittivity=n_SiO**2))
    .add_material(create_material(name='SiN', permittivity=n_SiN**2))
    .add_material(create_material(name='FusedSilica', permittivity=n_fs**2))
    
    # Add layers
    .add_layer({
        "material": "SiO",
        "thickness": 230e-3,
        "is_homogeneous": False,
        "is_optimize": True
    })
    .add_layer({
        "material": "SiN",
        "thickness": 345e-3,
        "is_homogeneous": True,
        "is_optimize": False
    })
    
    # Build the solver
    .build()
)

# Update the transmission material
torchrdit_sim.update_trn_material(trn_material="FusedSilica")

# build hexagonal unit cell
c1 = torchrdit_sim.get_circle_mask(center=[0, b/2], radius=r)
c2 = torchrdit_sim.get_circle_mask(center=[0, -b/2], radius=r)
c3 = torchrdit_sim.get_circle_mask(center=[a/2, 0], radius=r)
c4 = torchrdit_sim.get_circle_mask(center=[-a/2, 0], radius=r)

mask = torchrdit_sim.combine_masks(mask1=c1, mask2=c2, operation='union')
mask = torchrdit_sim.combine_masks(mask1=mask, mask2=c3, operation='union')
mask = torchrdit_sim.combine_masks(mask1=mask, mask2=c4, operation='union')

mask = (1 - mask).to(torch.float32)

mask.requires_grad = True

layer_index = 0
torchrdit_sim.update_er_with_mask(mask=mask, layer_index=layer_index)

# print layer information
torchrdit_sim.get_layer_structure()

# create a source object and solve
result = torchrdit_sim.solve({'theta': theta, 'phi': phi, 'pte': pte, 'ptm': ptm})

# Print full simulation data
print(f"The transmission efficiency is {result['TRN'][0] * 100}%")
print(f"The reflection efficiency is {result['REF'][0] * 100}%")

# Start back propagation
torch.sum(result['TRN'][0]).backward()

print(f"The gradient with respect to the mask is {torch.mean(mask.grad)}") 