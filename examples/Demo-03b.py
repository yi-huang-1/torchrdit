"""
# Example 3 - GMRF with hexagonal unit cells with dispersive materials

This updated example shows the simulation of the guided-mode resonance filter (GMRF) 
using torchrdit with a JSON-based configuration system.
"""

import numpy as np
import os
from torchrdit.solver import TorchrditConfig  # Import the newly created configuration class

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

# Initialize and run the simulation using the configuration
sovler_config = TorchrditConfig()
config_file = os.path.join(os.path.dirname(__file__), "config_demo_03.json")  # Ensure this file exists with the proper configuration
torchrdit_sim = sovler_config.create_solver(config_file)

# build hexagonal unit cell
c1 = torchrdit_sim.get_circle_mask(center=[0, b/2], radius=r)
c2 = torchrdit_sim.get_circle_mask(center=[0, -b/2], radius=r)
c3 = torchrdit_sim.get_circle_mask(center=[a/2, 0], radius=r)
c4 = torchrdit_sim.get_circle_mask(center=[-a/2, 0], radius=r)

mask = torchrdit_sim.combine_masks(mask1=c1, mask2=c2, operation='union')
mask = torchrdit_sim.combine_masks(mask1=mask, mask2=c3, operation='union')
mask = torchrdit_sim.combine_masks(mask1=mask, mask2=c4, operation='union')

mask = 1 - mask

layer_index = 0
torchrdit_sim.update_er_with_mask(mask=mask, layer_index=layer_index, set_grad = False)

# print layer information
torchrdit_sim.get_layer_structure()

result = torchrdit_sim.solve({'theta': theta, 'phi': phi, 'pte': pte, 'ptm': ptm})

# Print full simulation data
print(f"The transmission efficiency is {result['TRN'][0] * 100}%")
print(f"The reflection efficiency is {result['REF'][0] * 100}%")
