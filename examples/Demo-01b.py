"""
# Updated Example 1 - GMRF with hexagonal unit cells using RCWA (Builder Pattern)

This updated example shows the simulation of the guided-mode resonance filter (GMRF) 
using torchrdit with the Builder pattern, converting from a dictionary-based configuration.
"""

import numpy as np
import torch
from torchrdit.solver import get_solver_builder  # Import the builder function
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

# Define configuration dictionary (for reference)
config_dict = {
    "device": "cpu",
    "solver": {
        "algorithm": "RCWA",
        "precision": "DOUBLE",
        "wavelengths": [1540e-3, 1550e-3, 1560e-3, 1570e-3],
        "lengthunit": "um",
        "rdim": [512, 512],
        "kdim": [9, 9],
        "is_use_FFF": True
    },
    "lattice": {
        "t1": [a/2, -a*np.sqrt(3)/2],
        "t2": [a/2, a*np.sqrt(3)/2]
    },
    "materials": {
        "SiO": {
            "permittivity": n_SiO**2
        },
        "SiN": {
            "permittivity": n_SiN**2
        },
        "FusedSilica": {
            "permittivity": n_fs**2
        }
    },
    "layers": [
        {
            "material": "SiO",
            "thickness": 230e-3,
            "is_homogeneous": False,
            "is_optimize": True
        },
        {
            "material": "SiN",
            "thickness": 345e-3,
            "is_homogeneous": True,
            "is_optimize": False
        }
    ],
    "trn_material": "FusedSilica",
}

# Method 1: Convert configuration to builder approach
# Initialize the builder
builder = get_solver_builder()

# Transfer solver configuration
builder.with_algorithm(Algorithm.RCWA)
builder.with_precision(Precision.DOUBLE)
builder.with_wavelengths(np.array([1540e-3, 1550e-3, 1560e-3, 1570e-3]))
builder.with_length_unit("um")
builder.with_real_dimensions([512, 512])
builder.with_k_dimensions([9, 9])
builder.with_fff(True)

# Set lattice vectors
builder.with_lattice_vectors(
    torch.tensor([a/2, -a*np.sqrt(3)/2], dtype=torch.float32),
    torch.tensor([a/2, a*np.sqrt(3)/2], dtype=torch.float32)
)

# Create materials
material_sio = create_material(name='SiO', permittivity=n_SiO**2)
material_sin = create_material(name='SiN', permittivity=n_SiN**2)
material_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

# Add materials to builder
builder.add_material(material_sio)
builder.add_material(material_sin)
builder.add_material(material_fs)

# Add layers from config
builder.add_layer({
    "material": "SiO",
    "thickness": 230e-3,
    "is_homogeneous": False,
    "is_optimize": True
})
builder.add_layer({
    "material": "SiN",
    "thickness": 345e-3,
    "is_homogeneous": True,
    "is_optimize": False
})

# Build the solver
torchrdit_sim = builder.build()

# Update the transmission material
torchrdit_sim.update_trn_material(trn_material="FusedSilica")

# Method 2: Alternative - directly convert config to builder in one step
# Commented out as we're using Method 1 above
"""
# This would be another way to convert from config to builder
builder = get_solver_builder()
   
# Transfer all config settings at once
cfg = config_dict
builder.with_algorithm(Algorithm[cfg["solver"]["algorithm"]])
   .with_precision(Precision[cfg["solver"]["precision"]])
   .with_wavelengths(np.array(cfg["solver"]["wavelengths"]))
   # ... and so on
"""

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