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

# Table of Contents

* [torchrdit.vector](#torchrdit.vector)
  * [LatticeVectors](#torchrdit.vector.LatticeVectors)
    * [reciprocal](#torchrdit.vector.LatticeVectors.reciprocal)
    * [to](#torchrdit.vector.LatticeVectors.to)
    * [normalized\_basis](#torchrdit.vector.LatticeVectors.normalized_basis)
    * [inverse\_metric](#torchrdit.vector.LatticeVectors.inverse_metric)
  * [Expansion](#torchrdit.vector.Expansion)
    * [basis\_coefficients](#torchrdit.vector.Expansion.basis_coefficients)
    * [num\_terms](#torchrdit.vector.Expansion.num_terms)
  * [FourierExpansionManager](#torchrdit.vector.FourierExpansionManager)
    * [primitive\_lattice\_vectors](#torchrdit.vector.FourierExpansionManager.primitive_lattice_vectors)
    * [expansion](#torchrdit.vector.FourierExpansionManager.expansion)
    * [adapt\_to](#torchrdit.vector.FourierExpansionManager.adapt_to)
    * [min\_array\_shape](#torchrdit.vector.FourierExpansionManager.min_array_shape)
    * [project](#torchrdit.vector.FourierExpansionManager.project)
    * [reconstruct](#torchrdit.vector.FourierExpansionManager.reconstruct)
    * [fourier\_penalty\_weights](#torchrdit.vector.FourierExpansionManager.fourier_penalty_weights)
  * [generate\_expansion](#torchrdit.vector.generate_expansion)
  * [min\_array\_shape\_for\_expansion](#torchrdit.vector.min_array_shape_for_expansion)
  * [fft](#torchrdit.vector.fft)
  * [ifft](#torchrdit.vector.ifft)
  * [ConjugateGradientError](#torchrdit.vector.ConjugateGradientError)
  * [TangentFieldGenerator](#torchrdit.vector.TangentFieldGenerator)
    * [compute](#torchrdit.vector.TangentFieldGenerator.compute)
  * [compute\_tangent\_field](#torchrdit.vector.compute_tangent_field)

<a id="torchrdit.vector"></a>

# torchrdit.vector

Torch-native tangent vector field generation.

This module reproduces the tangent-field utilities from the fmmax project
(``https://github.com/facebookresearch/fmmax``). The implementation follows the derivations
of the following papers:
- M. F. Schubert and A. M. Hammond, "Fourier modal method for inverse design of metasurface-enhanced micro-LEDs," Opt. Express 31, 42945 (2023).
- V. Liu and S. Fan, "S4 : A free electromagnetic solver for layered periodic structures," Computer Physics Communications 183, 2233–2244 (2012).

<a id="torchrdit.vector.LatticeVectors"></a>

## LatticeVectors Objects

```python
@dataclass
class LatticeVectors()
```

Store primitive lattice vectors.

<a id="torchrdit.vector.LatticeVectors.reciprocal"></a>

#### reciprocal

```python
def reciprocal() -> LatticeVectors
```

Return the reciprocal lattice.

<a id="torchrdit.vector.LatticeVectors.to"></a>

#### to

```python
def to(*,
       device: torch.device | None = None,
       dtype: torch.dtype | None = None) -> LatticeVectors
```

Return a copy of the lattice on the requested device and dtype.

<a id="torchrdit.vector.LatticeVectors.normalized_basis"></a>

#### normalized\_basis

```python
def normalized_basis(*,
                     device: torch.device | None = None,
                     dtype: torch.dtype | None = None) -> torch.Tensor
```

Return normalized basis vectors on the requested device/dtype.

<a id="torchrdit.vector.LatticeVectors.inverse_metric"></a>

#### inverse\_metric

```python
def inverse_metric(*,
                   device: torch.device | None = None,
                   dtype: torch.dtype | None = None) -> torch.Tensor
```

Return the inverse lattice metric on the requested device/dtype.

<a id="torchrdit.vector.Expansion"></a>

## Expansion Objects

```python
@dataclass
class Expansion()
```

Hold Fourier basis coefficients.

<a id="torchrdit.vector.Expansion.basis_coefficients"></a>

#### basis\_coefficients

(num_terms, 2)

<a id="torchrdit.vector.Expansion.num_terms"></a>

#### num\_terms

```python
@property
def num_terms() -> int
```

Return the number of retained Fourier terms.

<a id="torchrdit.vector.FourierExpansionManager"></a>

## FourierExpansionManager Objects

```python
class FourierExpansionManager()
```

Manage Fourier bookkeeping for tangent-field calculations.

<a id="torchrdit.vector.FourierExpansionManager.primitive_lattice_vectors"></a>

#### primitive\_lattice\_vectors

```python
@property
def primitive_lattice_vectors() -> LatticeVectors
```

Return the primitive lattice vectors used for the expansion.

<a id="torchrdit.vector.FourierExpansionManager.expansion"></a>

#### expansion

```python
@property
def expansion() -> Expansion
```

Return the cached Fourier expansion.

<a id="torchrdit.vector.FourierExpansionManager.adapt_to"></a>

#### adapt\_to

```python
def adapt_to(*, device: torch.device, dtype: torch.dtype) -> None
```

Move internal tensors to the requested device and dtype.

<a id="torchrdit.vector.FourierExpansionManager.min_array_shape"></a>

#### min\_array\_shape

```python
def min_array_shape() -> Tuple[int, int]
```

Return the smallest spatial grid compatible with the expansion.

<a id="torchrdit.vector.FourierExpansionManager.project"></a>

#### project

```python
def project(x: torch.Tensor,
            *,
            axes: Tuple[int, int] = (-2, -1),
            centered_coordinates: bool = False) -> torch.Tensor
```

Project ``x`` onto the retained Fourier orders.

<a id="torchrdit.vector.FourierExpansionManager.reconstruct"></a>

#### reconstruct

```python
def reconstruct(y: torch.Tensor,
                *,
                shape: Tuple[int, int],
                axis: int = -1,
                centered_coordinates: bool = False) -> torch.Tensor
```

Reconstruct the spatial field from Fourier samples.

<a id="torchrdit.vector.FourierExpansionManager.fourier_penalty_weights"></a>

#### fourier\_penalty\_weights

```python
def fourier_penalty_weights() -> torch.Tensor
```

Return Fourier penalty weights on the current device/dtype.

<a id="torchrdit.vector.generate_expansion"></a>

#### generate\_expansion

```python
def generate_expansion(primitive_lattice_vectors: LatticeVectors,
                       approximate_num_terms: int) -> Expansion
```

Build a Fourier expansion for the lattice.

<a id="torchrdit.vector.min_array_shape_for_expansion"></a>

#### min\_array\_shape\_for\_expansion

```python
def min_array_shape_for_expansion(expansion: Expansion) -> Tuple[int, int]
```

Return the minimal grid dimensions for the expansion.

<a id="torchrdit.vector.fft"></a>

#### fft

```python
def fft(x: torch.Tensor,
        expansion: Expansion,
        axes: Tuple[int, int] = (-2, -1),
        centered_coordinates: bool = True) -> torch.Tensor
```

Project ``x`` onto the retained Fourier orders.

<a id="torchrdit.vector.ifft"></a>

#### ifft

```python
def ifft(y: torch.Tensor,
         expansion: Expansion,
         shape: Tuple[int, int],
         axis: int = -1,
         centered_coordinates: bool = True) -> torch.Tensor
```

Reconstruct ``y`` onto a spatial grid.

<a id="torchrdit.vector.ConjugateGradientError"></a>

## ConjugateGradientError Objects

```python
class ConjugateGradientError(RuntimeError)
```

Raised when the conjugate-gradient solver fails to converge.

<a id="torchrdit.vector.TangentFieldGenerator"></a>

## TangentFieldGenerator Objects

```python
class TangentFieldGenerator()
```

Generate tangent vector fields for patterned masks using Fourier expansions.

<a id="torchrdit.vector.TangentFieldGenerator.compute"></a>

#### compute

```python
def compute(mask: torch.Tensor,
            XO: torch.Tensor,
            YO: torch.Tensor,
            *,
            scheme: str = "POL",
            fourier_loss_weight: float | None = None,
            smoothness_loss_weight: float | None = None,
            steps: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]
```

Compute tangent fields for the requested scheme.

<a id="torchrdit.vector.compute_tangent_field"></a>

#### compute\_tangent\_field

```python
def compute_tangent_field(mask: torch.Tensor,
                          XO: torch.Tensor,
                          YO: torch.Tensor,
                          lattice_t1: torch.Tensor,
                          lattice_t2: torch.Tensor,
                          kdim: Tuple[int, int],
                          *,
                          scheme: str = "POL",
                          fourier_loss_weight: float = 1e-2,
                          smoothness_loss_weight: float = 1e-3,
                          steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]
```

Compute tangent fields via ``TangentFieldGenerator``.

