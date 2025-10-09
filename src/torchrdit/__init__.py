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
- results: Results containers with unified single/batched interface
- gds: GDS file format support for mask import/export

"""

__version__ = "0.1.27"

# Import core functionality
from .gds import mask_to_gds, gds_to_mask, load_gds_vertices

__all__ = [
    "mask_to_gds", "gds_to_mask", "load_gds_vertices",
]
