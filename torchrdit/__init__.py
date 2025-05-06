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

For more information, see:
- Huang et al., "Eigendecomposition-free inverse design of meta-optics devices," 
  Opt. Express 32, 13986-13997 (2024)
- Huang et al., "Inverse Design of Photonic Structures Using Automatic Differentiable 
  Rigorous Diffraction Interface Theory," CLEO (2023)
"""
from . import cell
from . import utils
from . import solver
from . import materials
from . import constants
from . import layers
from . import viz
from . import algorithm
from . import builder
from . import shapes
from . import results
from . import observers
from . import material_proxy

__version__ = "0.1.17"
