# Table of Contents

* [torchrdit](#torchrdit)

<a id="torchrdit"></a>

# torchrdit

TorchRDIT: GPU-accelerated electromagnetic solver for meta-optics design.

TorchRDIT is an advanced software package designed for the inverse design of
meta-optics devices, utilizing an eigendecomposition-free implementation of
Rigorous Diffraction Interface Theory (R-DIT). It provides a GPU-accelerated
and fully differentiable framework powered by PyTorch, enabling efficient
optimization of photonic structures.

This package achieves up to 16.2x speedup compared to traditional inverse design
methods based on Rigorous Coupled-Wave Analysis (RCWA). By integrating differentiable
R-DIT with topology optimization techniques and neural networks, TorchRDIT facilitates
the design of complex meta-optics devices.

Key modules:
- solver: Core electromagnetic solvers (RCWA and R-DIT)
- algorithm: Algorithm implementations for field calculations
- materials: Material property definitions and management
- material_proxy: Material data loading and processing
- constants: Physical constants and unit conversion utilities
- cell: Geometric cell definitions for simulations
- layers: Layer management for multilayer structures
- utils: Utility functions for calculations
- viz: Visualization tools for results
- builder: Builder pattern implementation for solver creation
- shapes: Shape generation for photonic structures
- observers: Observer pattern implementation for progress tracking
- batched_results: Results containers for source batching
- gds: GDS file format support for mask import/export

New in v0.1.23: Fully Tensorized Source Batching
------------------------------------------------
TorchRDIT now supports efficient batched processing of multiple sources:

```python
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
import numpy as np

# Create solver
solver = create_solver(algorithm=Algorithm.RDIT, rdim=[512, 512], kdim=[7, 7])

# Create multiple sources for angle sweep
deg = np.pi / 180
sources = [
    solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
    for angle in np.linspace(0, 60, 13) * deg
]

# Batch solve - returns BatchedSolverResults
results = solver.solve(sources)

# Access results
print(f"Transmission for all angles: {results.transmission[:, 0]}")
best_idx = results.find_optimal_source('max_transmission')
print(f"Best angle: {sources[best_idx]['theta'] * 180/np.pi:.1f}°")
```

New in v0.1.24: GDS Integration
--------------------------------
TorchRDIT now supports GDS file format for mask import/export:

```python
from torchrdit import mask_to_gds, gds_to_mask
from torchrdit.shapes import ShapeGenerator

# Create shapes and export to GDS
shape_gen = ShapeGenerator(X, Y, rdim)
mask = shape_gen.generate_circle_mask(radius=0.5)
mask_to_gds(mask, shape_gen, "DEVICE", "output.gds")

# Import from GDS
reconstructed = gds_to_mask("output.json", shape_gen)
```

For more information, see:
- Huang et al., "Eigendecomposition-free inverse design of meta-optics devices,"
  Opt. Express 32, 13986-13997 (2024)
- Huang et al., "Inverse Design of Photonic Structures Using Automatic Differentiable
  Rigorous Diffraction Interface Theory," CLEO (2023)

