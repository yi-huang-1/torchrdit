"""TorchRDIT: GPU-accelerated electromagnetic solver for meta-optics design.

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

New in v0.1.25: gdstk Migration
--------------------------------
TorchRDIT now uses gdstk instead of gdspy for improved platform compatibility:

- Better build support across different platforms
- Modern C++ backend for improved performance
- Full backward compatibility maintained
- No API changes for end users

Note: gdstk may require system dependencies (zlib and qhull).
See README for installation details.

New in v0.1.26: Enhanced Numerical Stability
---------------------------------------------
TorchRDIT now includes automatic protection for numerical edge cases:

1. **Plasmonic Material Stabilization**:
   - Automatic stabilization for materials near plasmon resonance (ε ≈ -1)
   - Configurable parameters for custom applications
   
   ```python
   from torchrdit.materials import MaterialClass
   
   # Automatic stabilization with default parameters
   plasmon = MaterialClass(name="plasmon", permittivity=-1.0 + 0j)
   
   # Custom stabilization parameters
   custom_plasmon = MaterialClass(
       name="custom_plasmon",
       permittivity=-1.0 + 0j,
       stabilization_params={
           'min_loss': 1e-4,     # Larger minimum loss
           'threshold': 0.05     # Wider detection window
       }
   )
   ```

2. **Extreme Grazing Incidence Protection**:
   - Automatic protection for numerical underflow at θ > 89.5°
   - Configurable threshold via `solver.min_kinc_z`
   
3. **Differentiable Alternatives**:
   - Smooth, differentiable operations for gradient-based optimization
   - Replaces non-differentiable torch.where with softplus alternatives

For more information, see:
- Huang et al., "Eigendecomposition-free inverse design of meta-optics devices,"
  Opt. Express 32, 13986-13997 (2024)
- Huang et al., "Inverse Design of Photonic Structures Using Automatic Differentiable
  Rigorous Diffraction Interface Theory," CLEO (2023)
"""

__version__ = "0.1.26"

# Import core functionality
from .batched_results import BatchedSolverResults, BatchedFieldComponents
from .gds import mask_to_gds, gds_to_mask, load_gds_vertices

__all__ = ["BatchedSolverResults", "BatchedFieldComponents", "mask_to_gds", "gds_to_mask", "load_gds_vertices"]
