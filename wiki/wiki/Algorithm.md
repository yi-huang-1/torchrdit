# Table of Contents

* [torchrdit.algorithm](#torchrdit.algorithm)
  * [SolverAlgorithm](#torchrdit.algorithm.SolverAlgorithm)
    * [solve\_nonhomo\_layer](#torchrdit.algorithm.SolverAlgorithm.solve_nonhomo_layer)
    * [name](#torchrdit.algorithm.SolverAlgorithm.name)
  * [RCWAAlgorithm](#torchrdit.algorithm.RCWAAlgorithm)
    * [\_\_init\_\_](#torchrdit.algorithm.RCWAAlgorithm.__init__)
    * [name](#torchrdit.algorithm.RCWAAlgorithm.name)
    * [solve\_nonhomo\_layer](#torchrdit.algorithm.RCWAAlgorithm.solve_nonhomo_layer)
    * [set\_rdit\_order](#torchrdit.algorithm.RCWAAlgorithm.set_rdit_order)
  * [RDITAlgorithm](#torchrdit.algorithm.RDITAlgorithm)
    * [\_\_init\_\_](#torchrdit.algorithm.RDITAlgorithm.__init__)
    * [name](#torchrdit.algorithm.RDITAlgorithm.name)
    * [solve\_nonhomo\_layer](#torchrdit.algorithm.RDITAlgorithm.solve_nonhomo_layer)
    * [set\_rdit\_order](#torchrdit.algorithm.RDITAlgorithm.set_rdit_order)

<a id="torchrdit.algorithm"></a>

# torchrdit.algorithm

<a id="torchrdit.algorithm.SolverAlgorithm"></a>

## SolverAlgorithm Objects

```python
class SolverAlgorithm(ABC)
```

Abstract base class defining the interface for electromagnetic solver algorithms.

This class defines the common interface that all solver algorithm implementations
must adhere to. It follows the Strategy pattern, allowing different algorithms
(RCWA and R-DIT) to be used interchangeably with the same solver interface.

**Notes**:

  Users should not instantiate or use these algorithm classes directly.
  Instead, use the SolverBuilder interface which provides a more user-friendly
  way to configure and create solvers with the appropriate algorithm.
  

**Examples**:

```python
# Use SolverBuilder to configure and create a solver
from torchrdit.builder import SolverBuilder
from torchrdit.constants import Algorithm
builder = SolverBuilder()
# Configure with RCWA algorithm
solver_rcwa = builder.with_algorithm(Algorithm.RCWA).build()
# Configure with R-DIT algorithm
solver_rdit = builder.with_algorithm(Algorithm.RDIT).build()
# Check which algorithm a solver uses
print(f"Solver uses: {solver_rcwa.algorithm.name}")
print(f"Solver uses: {solver_rdit.algorithm.name}")
```
  
  
  Keywords:
  algorithm, solver, electromagnetic, strategy pattern, abstraction

<a id="torchrdit.algorithm.SolverAlgorithm.solve_nonhomo_layer"></a>

#### solve\_nonhomo\_layer

```python
@abstractmethod
def solve_nonhomo_layer(layer_index, p_mat, q_mat, w0_mat, v0_mat)
```

Solve equations for non-homogeneous layer.

This abstract method defines the interface for solving the electromagnetic
field equations within a non-homogeneous layer. Different algorithm
implementations (RCWA, R-DIT) will provide different approaches for
this calculation.

**Notes**:

  This method is called internally by the solver and should not be
  called directly by users.
  

**Arguments**:

- `layer_index` _int_ - Index of the layer to solve.
- `p_mat` _torch.Tensor_ - P matrix for the layer.
- `q_mat` _torch.Tensor_ - Q matrix for the layer.
- `w0_mat` _torch.Tensor_ - W0 matrix.
- `v0_mat` _torch.Tensor_ - V0 matrix.
  

**Returns**:

- `dict` - Results of the non-homogeneous layer calculation, typically a
  scattering matrix for the layer.
  
  Keywords:
  layer solving, non-homogeneous, electromagnetic, scattering matrix

<a id="torchrdit.algorithm.SolverAlgorithm.name"></a>

#### name

```python
@property
@abstractmethod
def name()
```

Return the name of the algorithm.

This property provides a human-readable identifier for the algorithm.

**Returns**:

- `str` - The name of the algorithm (e.g., 'RCWA', 'R-DIT')
  
  Keywords:
  algorithm name, identifier

<a id="torchrdit.algorithm.RCWAAlgorithm"></a>

## RCWAAlgorithm Objects

```python
class RCWAAlgorithm(SolverAlgorithm)
```

Rigorous Coupled Wave Analysis (RCWA) algorithm implementation.

This class implements the traditional RCWA method, which solves Maxwell's
equations in the frequency domain by expanding the electromagnetic fields
and material properties in Fourier series and matching boundary conditions
at layer interfaces.

While the TorchRDIT package primarily emphasizes the eigendecomposition-free
R-DIT method for improved performance, this RCWA implementation is provided
for comparison and for cases where RCWA may be preferred.

**Notes**:

  Users should not instantiate this class directly. Instead, use the
  SolverBuilder interface to create solvers with the appropriate algorithm.
  

**Attributes**:

- `solver` - Reference to the parent solver instance.
- `_rdit_order` _int_ - Order parameter for compatibility with R-DIT.
  

**Examples**:

```python
# Using SolverBuilder to create a solver with RCWA algorithm
from torchrdit.builder import SolverBuilder
from torchrdit.constants import Algorithm
builder = SolverBuilder()
solver = builder.with_algorithm(Algorithm.RCWA).build()
# Additional configuration can be applied through the builder
solver = builder.with_algorithm(Algorithm.RCWA) \
                .with_wavelengths(1.55) \
                .with_k_dimensions([5, 5]) \
                .build()
```
  
  Keywords:
  RCWA, electromagnetic, eigendecomposition, Fourier, Maxwell, scattering

<a id="torchrdit.algorithm.RCWAAlgorithm.__init__"></a>

#### \_\_init\_\_

```python
def __init__(solver)
```

Initialize the RCWA algorithm instance.

**Arguments**:

- `solver` - Reference to the parent solver that will use this algorithm.

<a id="torchrdit.algorithm.RCWAAlgorithm.name"></a>

#### name

```python
@property
def name()
```

Return the name of the algorithm.

**Returns**:

- `str` - The name of the algorithm ('RCWA')

<a id="torchrdit.algorithm.RCWAAlgorithm.solve_nonhomo_layer"></a>

#### solve\_nonhomo\_layer

```python
def solve_nonhomo_layer(layer_thickness, p_mat_i, q_mat_i, mat_w0, mat_v0,
                        kdim, k_0, **kwargs)
```

RCWA implementation for solving non-homogeneous layer.

This method implements the traditional approach for calculating the scattering
matrix of a non-homogeneous layer using eigenmode decomposition in the
Rigorous Coupled Wave Analysis (RCWA) algorithm.

**Notes**:

  This method is called internally by the solver and should not be
  called directly by users.
  

**Arguments**:

- `layer_thickness` _float_ - Thickness of the layer.
- `p_mat_i` _torch.Tensor_ - P matrix for the layer.
- `q_mat_i` _torch.Tensor_ - Q matrix for the layer.
- `mat_w0` _torch.Tensor_ - W0 matrix.
- `mat_v0` _torch.Tensor_ - V0 matrix.
- `kdim` _list_ - Dimensions in k-space [kheight, kwidth].
- `k_0` _torch.Tensor_ - Wave number.
- `**kwargs` - Additional parameters.
  

**Returns**:

- `dict` - Dictionary containing the scattering matrix for the layer
  with keys 'S11', 'S12', 'S21', 'S22'.
  
  Keywords:
  RCWA, scattering matrix, eigenmode, layer calculation, non-homogeneous

<a id="torchrdit.algorithm.RCWAAlgorithm.set_rdit_order"></a>

#### set\_rdit\_order

```python
def set_rdit_order(rdit_order)
```

Set R-DIT order for compatibility with RDITAlgorithm.

This method is provided for API compatibility with the R-DIT algorithm,
allowing for easy switching between algorithms.

**Notes**:

  Users should not call this method directly. RDIT order can be set
  using SolverBuilder.with_rdit_order() method.
  

**Arguments**:

- `rdit_order` _int_ - The order of the R-DIT algorithm. This parameter has
  minimal effect in the RCWA implementation.
  

**Examples**:

```python
# Setting RDIT order through the builder
from torchrdit.builder import SolverBuilder
from torchrdit.constants import Algorithm
solver = SolverBuilder() \
        .with_algorithm(Algorithm.RCWA) \
        .with_rdit_order(4) \
        .build()
```
  
  Keywords:
  RCWA, R-DIT, compatibility, order parameter

<a id="torchrdit.algorithm.RDITAlgorithm"></a>

## RDITAlgorithm Objects

```python
class RDITAlgorithm(SolverAlgorithm)
```

Rigorous Diffraction Interface Theory (R-DIT) algorithm implementation.

This class implements the eigendecomposition-free R-DIT algorithm, which offers
improved numerical stability and computational efficiency compared to the
traditional RCWA approach. The R-DIT method achieves up to 16.2× speedup
in inverse design applications.

**Notes**:

  Users should not instantiate this class directly. Instead, use the
  SolverBuilder interface to create solvers with the appropriate algorithm.
  

**Attributes**:

- `solver` - Reference to the parent solver instance.
- `_rdit_order` _int_ - Order parameter for the R-DIT algorithm approximation.
  

**Examples**:

```python
# Using SolverBuilder to create a solver with R-DIT algorithm
from torchrdit.builder import SolverBuilder
from torchrdit.constants import Algorithm
builder = SolverBuilder()
# Configure with R-DIT algorithm and set order
solver = builder.with_algorithm(Algorithm.RDIT) \
                .with_rdit_order(8) \
                .build()
# Or load configuration from a file
solver = SolverBuilder().from_config("config.json").build()
```
  
  Keywords:
  R-DIT, electromagnetic, eigendecomposition-free, optimization, speedup, scattering

<a id="torchrdit.algorithm.RDITAlgorithm.__init__"></a>

#### \_\_init\_\_

```python
def __init__(solver)
```

Initialize the R-DIT algorithm instance.

**Arguments**:

- `solver` - Reference to the parent solver that will use this algorithm.

<a id="torchrdit.algorithm.RDITAlgorithm.name"></a>

#### name

```python
@property
def name()
```

Return the name of the algorithm.

**Returns**:

- `str` - The name of the algorithm ('R-DIT')
  
  Keywords:
  algorithm name, R-DIT, identifier

<a id="torchrdit.algorithm.RDITAlgorithm.solve_nonhomo_layer"></a>

#### solve\_nonhomo\_layer

```python
def solve_nonhomo_layer(layer_thickness, p_mat_i, q_mat_i, mat_w0, mat_v0,
                        kdim, k_0, **kwargs)
```

R-DIT implementation for solving non-homogeneous layer.

This method implements the eigendecomposition-free approach for calculating
the scattering matrix of a non-homogeneous layer using the Rigorous Diffraction
Interface Theory algorithm.

**Notes**:

  This method is called internally by the solver and should not be
  called directly by users.
  

**Arguments**:

- `layer_thickness` _float_ - Thickness of the layer.
- `p_mat_i` _torch.Tensor_ - P matrix for the layer.
- `q_mat_i` _torch.Tensor_ - Q matrix for the layer.
- `mat_w0` _torch.Tensor_ - W0 matrix.
- `mat_v0` _torch.Tensor_ - V0 matrix.
- `kdim` _list_ - Dimensions in k-space [kheight, kwidth].
- `k_0` _torch.Tensor_ - Wave number.
- `**kwargs` - Additional parameters.
  

**Returns**:

- `dict` - Dictionary containing the scattering matrix for the layer
  with keys 'S11', 'S12', 'S21', 'S22'.
  
  Keywords:
  R-DIT, scattering matrix, eigendecomposition-free, layer calculation, non-homogeneous

<a id="torchrdit.algorithm.RDITAlgorithm.set_rdit_order"></a>

#### set\_rdit\_order

```python
def set_rdit_order(rdit_order)
```

Set the order of the R-DIT algorithm.

The R-DIT order determines the approximation used in the diffraction
interface theory. Higher orders generally provide better accuracy but
may be computationally more expensive.

**Notes**:

  Users should not call this method directly. RDIT order can be set
  using SolverBuilder.with_rdit_order() method.
  

**Arguments**:

- `rdit_order` _int_ - The order of the algorithm (typically 1-10). Higher values
  provide more accurate results at the cost of computational
  efficiency.
  

**Examples**:

```python
# Setting RDIT order through the builder
from torchrdit.builder import SolverBuilder
from torchrdit.constants import Algorithm
solver = SolverBuilder() \
                .with_algorithm(Algorithm.RDIT) \
                .with_rdit_order(8) \
                .build()
```
  
  Keywords:
  R-DIT, order, approximation, accuracy, performance, tradeoff

