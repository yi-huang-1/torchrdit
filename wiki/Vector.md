# Tangent Vector Field Module

## Overview
The `torchrdit.vector` module provides Torch-native tangent vector field generation used by Fourier-factorized solvers.

This module reproduces the tangent-field utilities from the fmmax project
(``https://github.com/facebookresearch/fmmax``). The implementation follows the derivations
of the following papers:
- M. F. Schubert and A. M. Hammond, "Fourier modal method for inverse design of metasurface-enhanced micro-LEDs," Opt. Express 31, 42945 (2023).
- V. Liu and S. Fan, "S4 : A free electromagnetic solver for layered periodic structures," Computer Physics Communications 183, 2233–2244 (2012).

## Key Components
- `LatticeVectors`: stores primitive lattice vectors and constructs their reciprocal basis.
- `FourierExpansionManager`: manages Fourier order selection, projection, and reconstruction.
- `TangentFieldGenerator`: optimizes smooth tangent fields with configurable penalties.
- `compute_tangent_field`: convenience wrapper that returns the field components for a given scheme.

## Supported Schemes
- `POL`: normalized tangent directions suitable for standard Fourier factorization.
- `NORMAL`: elementwise-normalized tangential vectors for unit-magnitude fields.
- `JONES`: tangent field mapped to Jones vectors for ellipse visualizations.
- `JONES_DIRECT`: direct Jones-space optimization without post-processing.

## Usage Example
```python
import torch
from torchrdit.vector import TangentFieldGenerator

grid_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask = torch.rand(grid_size, grid_size, dtype=torch.float32, device=device)
coords = torch.linspace(-0.5, 0.5, grid_size, device=device)
YO, XO = torch.meshgrid(coords, coords, indexing="ij")

generator = TangentFieldGenerator(
    lattice_t1=torch.tensor([1.0, 0.0], dtype=mask.dtype, device=device),
    lattice_t2=torch.tensor([0.0, 1.0], dtype=mask.dtype, device=device),
    kdim=(9, 9),
    fourier_loss_weight=1e-2,
    smoothness_loss_weight=1e-3,
    steps=2,
)

tx, ty = generator.compute(mask, XO, YO, scheme="JONES")
```

## API Reference

The API reference below is generated automatically from the source.

# torchrdit.vector

**Error: Documentation generation for `torchrdit.vector` encountered an error: [Errno 2] No such file or directory: 'pydoc-markdown'**
