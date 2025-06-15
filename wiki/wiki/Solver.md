# Table of Contents

* [torchrdit.solver](#torchrdit.solver)
  * [SolverObserver](#torchrdit.solver.SolverObserver)
    * [update](#torchrdit.solver.SolverObserver.update)
  * [SolverSubjectMixin](#torchrdit.solver.SolverSubjectMixin)
    * [\_\_init\_\_](#torchrdit.solver.SolverSubjectMixin.__init__)
    * [add\_observer](#torchrdit.solver.SolverSubjectMixin.add_observer)
    * [remove\_observer](#torchrdit.solver.SolverSubjectMixin.remove_observer)
    * [notify\_observers](#torchrdit.solver.SolverSubjectMixin.notify_observers)
  * [create\_solver\_from\_config](#torchrdit.solver.create_solver_from_config)
  * [create\_solver](#torchrdit.solver.create_solver)
  * [FourierBaseSolver](#torchrdit.solver.FourierBaseSolver)
    * [algorithm](#torchrdit.solver.FourierBaseSolver.algorithm)
    * [algorithm](#torchrdit.solver.FourierBaseSolver.algorithm)
    * [lam0](#torchrdit.solver.FourierBaseSolver.lam0)
    * [lam0](#torchrdit.solver.FourierBaseSolver.lam0)
    * [add\_source](#torchrdit.solver.FourierBaseSolver.add_source)
    * [solve](#torchrdit.solver.FourierBaseSolver.solve)
    * [update\_er\_with\_mask](#torchrdit.solver.FourierBaseSolver.update_er_with_mask)
    * [update\_er\_with\_mask\_extern\_NV](#torchrdit.solver.FourierBaseSolver.update_er_with_mask_extern_NV)
    * [update\_layer\_thickness](#torchrdit.solver.FourierBaseSolver.update_layer_thickness)
    * [expand\_dims](#torchrdit.solver.FourierBaseSolver.expand_dims)
  * [RCWASolver](#torchrdit.solver.RCWASolver)
    * [set\_rdit\_order](#torchrdit.solver.RCWASolver.set_rdit_order)
  * [RDITSolver](#torchrdit.solver.RDITSolver)
    * [set\_rdit\_order](#torchrdit.solver.RDITSolver.set_rdit_order)
  * [get\_solver\_builder](#torchrdit.solver.get_solver_builder)
  * [create\_solver\_from\_builder](#torchrdit.solver.create_solver_from_builder)

<a id="torchrdit.solver"></a>

# torchrdit.solver

TorchRDIT solver module for electromagnetic simulations and inverse design.

This module provides core solver implementations for electromagnetic wave propagation
through complex layered structures using both Rigorous Coupled-Wave Analysis (RCWA) and
eigendecomposition-free Rigorous Diffraction Interface Theory (R-DIT) methods.

TorchRDIT is an advanced software package designed for the inverse design of meta-optics
devices. It provides a GPU-accelerated and fully differentiable framework powered by
PyTorch, enabling efficient optimization of photonic structures. The R-DIT implementation
achieves up to 16.2x speedup compared to traditional RCWA-based inverse design methods.

Key components:
- FourierBaseSolver: Base class implementing common Fourier-based calculations
- RCWASolver: Implementation of the RCWA algorithm
- RDITSolver: Eigendecomposition-free implementation of the R-DIT algorithm
- SolverObserver: Interface for tracking solver progress
- create_solver: Factory function to create solver instances

The solvers handle:
- Multiple wavelength simulations
- Arbitrary incident angles
- Heterogeneous material layers
- Computation of reflection/transmission coefficients
- Field calculations
- Efficiency calculations
- Automatic differentiation for gradient-based optimization

**Examples**:

  Basic usage with RCWA:
  
```python
import numpy as np
import torch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material

# Create a solver with RCWA algorithm
solver = create_solver(
    algorithm=Algorithm.RCWA,
    lam0=np.array([1.55]),  # Wavelength in micrometers
    rdim=[512, 512],        # Real space grid dimensions
    kdim=[5, 5],            # Fourier space dimensions
    device="cuda"           # Use GPU acceleration
)

# Create materials
silicon = create_material(name="silicon", permittivity=11.7)
sio2 = create_material(name="sio2", permittivity=2.25)
solver.add_materials([silicon, sio2])

# Define layers
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.2))
solver.add_layer(material_name="sio2", thickness=torch.tensor(0.3))

# Add a source with normal incidence and TE polarization
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)

# Run the solver
result = solver.solve(source) # SolverResults object
print(f"Reflection: {result.reflection[0].item():.3f}, Transmission: {result.transmission[0].item():.3f}")  # reflection and transmission of the first wavelength
```
  
  Creating a patterned layer using a mask:
  
```python
# Add a patterned layer
solver.add_layer(material_name="silicon", thickness=torch.tensor(0.5), is_homogeneous=False)

# Create a circular pattern
mask = solver.get_circle_mask(center=(0, 0), radius=0.25)

# Use the mask to define material distribution
solver.update_er_with_mask(mask=mask, layer_index=2)

# Solve again with the patterned layer
result = solver.solve(source) # SolverResults object
```
  
  Inverse design with automatic differentiation:
  
```python
# Create a mask parameter with gradients enabled
mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
mask = mask.to(torch.float32)
mask.requires_grad = True

# Configure the patterned layer with the mask
solver.update_er_with_mask(mask=mask, layer_index=0)

# Solve and compute gradient to maximize transmission
result = solver.solve(source) # SolverResults object
(-result.transmission[0]).backward()  # Maximize transmission of the first wavelength

# Access gradients for optimization
grad = mask.grad
print(f"Mean gradient: {grad.abs().mean().item():.6f}")
```
  
  Keywords:
  electromagnetic solver, RCWA, R-DIT, photonics, meta-optics, inverse design,
  diffraction, Fourier optics, automatic differentiation, GPU acceleration,
  nanophotonics, layered media, periodic structures, reflection, transmission,
  gradient-based optimization, PyTorch, computational electromagnetics

<a id="torchrdit.solver.SolverObserver"></a>

## SolverObserver Objects

```python
class SolverObserver()
```

Interface for observers that track solver progress.

This class defines the Observer interface in the Observer design pattern
for monitoring the progress and state of solvers. Concrete observer
implementations should inherit from this class and implement the update
method to receive and process notifications from solvers.

The Observer pattern allows for loose coupling between solvers and progress
monitoring components, enabling flexible visualization and logging of solver
progress without modifying the solver implementation.

**Notes**:

  This is an abstract interface. Use concrete implementations like
  ConsoleProgressObserver or TqdmProgressObserver from the observers module.
  

**Attributes**:

  None
  

**Examples**:

```python
from torchrdit.solver import SolverObserver, create_solver
import time

class TimingObserver(SolverObserver):
    def __init__(self):
        self.start_times = {}
        self.end_times = {}

    def update(self, event_type, data):
        if event_type == "layer_started":
            layer = data.get("current", 0)
            self.start_times[layer] = time.time()
        elif event_type == "layer_completed":
            layer = data.get("current", 0)
            self.end_times[layer] = time.time()
            elapsed = self.end_times[layer] - self.start_times[layer]
            print(f"Layer {layer} processed in {elapsed:.3f} seconds")

# Create a solver with the custom observer
solver = create_solver()
solver.add_observer(TimingObserver())

# The observer will now receive notifications during solving
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
result = solver.solve(source) # SolverResults object
```
  
  Keywords:
  observer pattern, progress tracking, monitoring, notification, event handling,
  solver feedback, progress reporting, design pattern, interface

<a id="torchrdit.solver.SolverObserver.update"></a>

#### update

```python
def update(event_type: str, data: dict) -> None
```

Called when the solver notifies of an event.

This method is called by the solver when an event occurs. Concrete
observer implementations should override this method to process
notifications from the solver.

**Arguments**:

- `event_type` _str_ - The type of event that occurred. Common event types include:
  - "calculation_starting": When the solve process begins
  - "initializing_k_vectors": When setting up k-vectors
  - "setting_up_matrices": When matrices are being prepared
  - "processing_layers": When beginning to process all layers
  - "layer_started": When processing a specific layer begins
  - "layer_completed": When a layer's processing is finished
  - "connecting_external_regions": When connecting external regions
  - "calculating_fields": When calculating fields
  - "assembling_final_data": When preparing final results
  - "calculation_completed": When the solve process is finished
  
- `data` _dict_ - Additional data related to the event. Contents vary by event
  type but may include:
  - "mode": Solver mode (RCWA or RDIT)
  - "n_freqs": Number of frequencies being solved
  - "n_layers": Number of layers in the structure
  - "current": Current layer or iteration index
  - "total": Total number of items (e.g., layers)
  - "progress": Percentage complete (0-100)
  

**Examples**:

```python
# Example implementation in a concrete observer class
def update(self, event_type, data):
    if event_type == "calculation_starting":
        print(f"Starting calculation with {data.get('n_freqs')} wavelengths")
    elif event_type == "calculation_completed":
        print("Calculation finished!")
```
  

**Notes**:

  This is an abstract method that should be overridden by concrete observer
  classes. The default implementation does nothing.
  
  Keywords:
  event handling, notification, callback, progress update, observer method

<a id="torchrdit.solver.SolverSubjectMixin"></a>

## SolverSubjectMixin Objects

```python
class SolverSubjectMixin()
```

Mixin class that allows a solver to notify observers of progress.

This class implements the Subject role in the Observer design pattern.
It maintains a list of observers and provides methods to add, remove,
and notify observers of events. Classes that inherit from this mixin
can easily support observer notifications without reimplementing the
observer management logic.

The Observer pattern enables solvers to report their progress to multiple
observers (like console loggers, progress bars, or GUI components) without
tight coupling to these visualization components.

**Attributes**:

- `_observers` _list_ - List of observer objects registered to receive notifications
  

**Examples**:

```python
from torchrdit.solver import SolverSubjectMixin, SolverObserver

class MyProcessingClass(SolverSubjectMixin):
    def __init__(self):
        SolverSubjectMixin.__init__(self)  # Initialize the observer list

    def process_data(self, data):
        # Notify observers that processing is starting
        self.notify_observers('processing_started', {'data_size': len(data)})

        # Do some processing
        result = []
        for i, item in enumerate(data):
            # Process the item
            result.append(item * 2)

            # Notify observers of progress
            self.notify_observers('item_processed', {
                'current': i + 1,
                'total': len(data),
                'progress': (i + 1) / len(data) * 100
            })

        # Notify observers that processing is complete
        self.notify_observers('processing_completed', {'result_size': len(result)})
        return result

# Create a simple observer to monitor progress
class SimpleObserver(SolverObserver):
    def update(self, event_type, data):
        if event_type == 'processing_started':
            print(f"Starting to process {data['data_size']} items")
        elif event_type == 'item_processed':
            print(f"Progress: {data['progress']:.1f}%")
        elif event_type == 'processing_completed':
            print(f"Processing completed with {data['result_size']} results")

# Use the observer with our processing class
processor = MyProcessingClass()
processor.add_observer(SimpleObserver())
result = processor.process_data([1, 2, 3, 4, 5])
```
  
  Keywords:
  observer pattern, subject, notification, progress tracking, design pattern,
  event broadcasting, observer management, mixin, loose coupling

<a id="torchrdit.solver.SolverSubjectMixin.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

Initialize the observer list.

Creates an empty list to store observer objects. This method should be
called in the constructor of any class that inherits from this mixin.

**Examples**:

```python
class MyClass(SolverSubjectMixin):
    def __init__(self):
        SolverSubjectMixin.__init__(self)  # Initialize observer list
```
  
  Keywords:
  initialization, observer list, constructor

<a id="torchrdit.solver.SolverSubjectMixin.add_observer"></a>

#### add\_observer

```python
def add_observer(observer: SolverObserver) -> None
```

Add an observer to the notification list.

Registers an observer to receive notifications when events occur.
If the observer is already registered, it will not be added again.

**Arguments**:

- `observer` _SolverObserver_ - The observer object to add. Must implement
  the SolverObserver interface with an update method.
  

**Examples**:

```python
# Create a solver and add an observer
from torchrdit.solver import create_solver
from torchrdit.observers import ConsoleProgressObserver

solver = create_solver()
observer = ConsoleProgressObserver()
solver.add_observer(observer)
```
  
  Keywords:
  register observer, add listener, subscribe, event notification

<a id="torchrdit.solver.SolverSubjectMixin.remove_observer"></a>

#### remove\_observer

```python
def remove_observer(observer: SolverObserver) -> None
```

Remove an observer from the notification list.

Unregisters an observer so it no longer receives notifications.
If the observer is not in the list, no action is taken.

**Arguments**:

- `observer` _SolverObserver_ - The observer object to remove.
  

**Examples**:

```python
# Remove an observer from a solver
from torchrdit.solver import create_solver
from torchrdit.observers import ConsoleProgressObserver

solver = create_solver()
observer = ConsoleProgressObserver()
solver.add_observer(observer)

# Later, remove the observer
solver.remove_observer(observer)
```
  
  Keywords:
  unregister observer, remove listener, unsubscribe, notification management

<a id="torchrdit.solver.SolverSubjectMixin.notify_observers"></a>

#### notify\_observers

```python
def notify_observers(event_type: str, data: dict = None) -> None
```

Notify all observers of an event.

This method calls the update method on all registered observers,
passing the event type and data to each observer. This is the main
mechanism for broadcasting events to all observers.

**Arguments**:

- `event_type` _str_ - The type of event that occurred. This identifies
  the type of event, allowing observers to handle different
  events appropriately. Common event types include:
  - "calculation_starting": When the solve process begins
  - "layer_started": When beginning to process a layer
  - "layer_completed": When a layer has been processed
  - "calculation_completed": When solving is complete
  
- `data` _dict, optional_ - Additional data related to the event. This contains
  event-specific information that might be useful to observers.
  Default is an empty dictionary.
  

**Examples**:

```python
# Example usage within a solver method
def solve_layer(self, layer_index):
    # Notify observers that we're starting a layer
    self.notify_observers('layer_started', {
        'current': layer_index,
        'total': self.layer_manager.nlayer,
        'progress': layer_index / self.layer_manager.nlayer * 100
    })

    # Process the layer
    # ...

    # Notify observers that the layer is complete
    self.notify_observers('layer_completed', {
        'current': layer_index,
        'total': self.layer_manager.nlayer,
        'progress': (layer_index + 1) / self.layer_manager.nlayer * 100
    })
```
  
  Keywords:
  broadcast event, notify listeners, event dispatch, observer notification,
  progress reporting, event publishing

<a id="torchrdit.solver.create_solver_from_config"></a>

#### create\_solver\_from\_config

```python
def create_solver_from_config(
        config: Union[str, Dict[str, Any]],
        flip: bool = False) -> Union["RCWASolver", "RDITSolver"]
```

Create a solver from a configuration file or dictionary.

This function creates a solver instance based on a configuration specified
either as a dictionary or a path to a JSON/YAML configuration file. It uses
the SolverBuilder pattern internally to construct the solver.

The configuration can specify all solver parameters, including algorithm type,
wavelengths, dimensions, materials, and device options. This provides a convenient
way to store and reuse solver configurations across different simulations.

**Arguments**:

- `config` _Union[str, Dict[str, Any]]_ - Either a path to a configuration file (string)
  or a dictionary containing the configuration parameters. The configuration
  should specify all necessary parameters for the solver, including:
  - "algorithm": Algorithm type ("RCWA" or "RDIT")
  - "wavelengths": List of wavelengths to simulate
  - "lengthunit": Length unit (e.g., "um", "nm")
  - "rdim": Real space dimensions [height, width]
  - "kdim": Fourier space dimensions [kheight, kwidth]
  - "device": Device to use (e.g., "cpu", "cuda")
  
- `flip` _bool, optional_ - Whether to flip the coordinate system. When True,
  the solver will use a flipped coordinate system, which can be useful
  for certain types of simulations or to match other software conventions.
  Default is False.
  

**Returns**:

  Union[RCWASolver, RDITSolver]: A solver instance configured according
  to the provided parameters.
  

**Examples**:

  Creating a solver from a dictionary configuration:
  
```python
from torchrdit.solver import create_solver_from_config
# Define a configuration dictionary
config = {
    "algorithm": "RDIT",
    "wavelengths": [1.55],
    "length_unit": "um",
    "rdim": [512, 512],
    "kdim": [5, 5],
    "device": "cuda"
}
solver = create_solver_from_config(config)
```
  
  Creating a solver from a JSON configuration file:
  
```python
# Create a config file (config.json)
# {
#   "algorithm": "RCWA",
#   "wavelengths": [1.31, 1.55],
#   "length_unit": "um",
#   "rdim": [256, 256],
#   "kdim": [3, 3]
# }
solver = create_solver_from_config("config.json")
```
  

**Notes**:

  This function is particularly useful for reproducible simulations or
  when the simulation setup needs to be saved and restored later.
  
  Keywords:
  configuration, JSON, YAML, file configuration, solver creation, parameter loading,
  reproducible simulation, configuration file, solver setup

<a id="torchrdit.solver.create_solver"></a>

#### create\_solver

```python
def create_solver(
    algorithm: Algorithm = Algorithm.RDIT,
    precision: Precision = Precision.SINGLE,
    lam0: np.ndarray = np.array([1.0]),
    lengthunit: str = "um",
    rdim: List[int] = [512, 512],
    kdim: List[int] = [3, 3],
    materiallist: List[Any] = [],
    t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),
    t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),
    is_use_FFF: bool = False,
    device: Union[str,
                  torch.device] = "cpu") -> Union["RCWASolver", "RDITSolver"]
```

Create a solver with the given parameters.

This is the main factory function for creating solver instances with custom
parameters. It provides a convenient way to create a solver without having to
directly instantiate the solver classes or use the builder pattern.

The function creates either an RCWA solver (for traditional Rigorous Coupled-Wave
Analysis) or an R-DIT solver (the eigendecomposition-free Rigorous Diffraction
Interface Theory). By default, it creates an R-DIT solver, which offers significantly
better performance for inverse design applications.

**Arguments**:

- `algorithm` _Algorithm_ - The algorithm to use for solving. Options:
  - Algorithm.RCWA: Traditional RCWA method
  - Algorithm.RDIT: Eigendecomposition-free R-DIT method (default)
  R-DIT is recommended for most applications, especially inverse design,
  due to its superior computational efficiency and stable gradients.
  
- `precision` _Precision_ - Numerical precision to use. Options:
  - Precision.SINGLE: Single precision (float32/complex64)
  - Precision.DOUBLE: Double precision (float64/complex128, default)
  Use double precision for higher accuracy at the cost of memory usage.
  
- `lam0` _np.ndarray_ - Wavelength(s) to simulate, in the specified length unit.
  Can be a single value or an array of wavelengths. Default is [1.0].
  
- `lengthunit` _str_ - The unit of length used in the simulation. Common values:
  - 'um': Micrometers (default)
  - 'nm': Nanometers
  All dimensions (wavelengths, thicknesses) are interpreted in this unit.
  
- `rdim` _List[int]_ - Dimensions of the real space grid [height, width].
  Default is [512, 512]. Higher values provide more spatial resolution
  but require more memory and computation time.
  
- `kdim` _List[int]_ - Dimensions in Fourier space [kheight, kwidth]. Default is [3, 3].
  This determines the number of Fourier harmonics used in the simulation.
  Higher values improve accuracy but significantly increase computation time.
  
- `materiallist` _List[Any]_ - List of materials used in the simulation. Can include
  MaterialClass instances created with create_material().
  Default is an empty list.
  
- `t1` _torch.Tensor_ - First lattice vector defining the unit cell geometry.
  Default is [[1.0, 0.0]] (unit vector in x-direction).
  
- `t2` _torch.Tensor_ - Second lattice vector defining the unit cell geometry.
  Default is [[0.0, 1.0]] (unit vector in y-direction).
  
- `is_use_FFF` _bool_ - Whether to use Fast Fourier Factorization. Default is False.
  This improves convergence for high-contrast material interfaces
  but adds computational overhead.
  
- `device` _Union[str, torch.device]_ - The device to run the solver on.
  Options include 'cpu' (default) or 'cuda' for GPU acceleration.
  For optimal performance, especially with the R-DIT solver,
  using 'cuda' is highly recommended.
  

**Returns**:

  Union[RCWASolver, RDITSolver]: A solver instance configured according
  to the provided parameters. The returned solver is fully differentiable,
  enabling gradient-based optimization for inverse design.
  

**Examples**:

```python
import torch
import numpy as np
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm, Precision

# Create an R-DIT solver with GPU acceleration
solver = create_solver(
    algorithm=Algorithm.RDIT,
    lam0=np.array([1.55]),  # Wavelength (μm)
    rdim=[512, 512],        # Real space resolution
    kdim=[5, 5],            # Fourier space harmonics
    device='cuda'           # Use GPU acceleration
)
```
  
  Creating an RCWA solver with non-rectangular lattice:
```python
# Create an RCWA solver with a triangular lattice
solver = create_solver(
    algorithm=Algorithm.RCWA,
    precision=Precision.DOUBLE,  # Use double precision
    lam0=np.array([0.8, 1.0, 1.2]),  # Multiple wavelengths
    t1=torch.tensor([[1.0, 0.0]]),
    t2=torch.tensor([[0.5, 0.866]]),  # 60-degree lattice
    rdim=[1024, 1024],
    kdim=[7, 7],
    device='cuda'
)
```
  

**Notes**:

  To optimize memory usage and performance:
  1. Use the R-DIT algorithm (default) for inverse design applications
  2. Use GPU acceleration (device='cuda') when available
  3. Adjust rdim and kdim based on required accuracy and available memory
  4. Use single precision for large simulations where memory is a concern
  
  Keywords:
  solver creation, RCWA, R-DIT, factory function, electromagnetic simulation,
  photonics, meta-optics, GPU acceleration, lattice vectors, Fourier harmonics,
  numerical precision, wavelength, inverse design

<a id="torchrdit.solver.FourierBaseSolver"></a>

## FourierBaseSolver Objects

```python
class FourierBaseSolver(Cell3D, SolverSubjectMixin)
```

Base class for Fourier-based electromagnetic solvers.

This class serves as the foundation for Fourier-based electromagnetic solvers,
providing common functionality and interface for derived solver implementations
such as RCWASolver and RDITSolver.

FourierBaseSolver implements core computational methods for electromagnetic
simulations of layered structures with periodic boundary conditions. It handles
the setup of computational grids, material properties, layer stacks, and the
calculation of optical responses (reflection, transmission, field distributions)
using Fourier-based methods.

**Attributes**:

- `tcomplex` _torch.dtype_ - Complex data type used for calculations.
- `tfloat` _torch.dtype_ - Float data type used for calculations.
- `tint` _torch.dtype_ - Integer data type used for calculations.
- `nfloat` _np.dtype_ - NumPy float data type for compatibility.
- `device` _Union[str, torch.device]_ - Device used for computation ('cpu' or 'cuda').
- `rdim` _List[int]_ - Dimensions in real space [height, width].
- `kdim` _List[int]_ - Dimensions in k-space [kheight, kwidth].
- `shapes` _ShapeGenerator_ - Generator for creating shape masks.
- `layer_manager` _LayerManager_ - Manager for handling material layers.
  

**Notes**:

  This is an abstract base class and should not be instantiated directly.
  Use RCWASolver or RDITSolver for concrete implementations, or the
  create_solver() function to create an appropriate solver instance.
  

**Examples**:

```python
import numpy as np
import torch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import create_material

# Create a derived solver (RCWA)
solver = create_solver(
    algorithm=Algorithm.RCWA,
    lam0=np.array([1.55]),
    rdim=[512, 512],
    kdim=[5, 5]
)

# Add materials and layers
silicon = create_material(name="Si", permittivity=11.7)
solver.add_materials([silicon])
solver.add_layer(material_name="Si", thickness=torch.tensor(0.5))

# Set up a source and solve
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
result = solver.solve(source) # SolverResults object
```
  
  Keywords:
  base class, Fourier solver, RCWA, R-DIT, electromagnetic simulation,
  photonics, meta-optics, periodic structures, Fourier transform,
  diffraction, layered media

<a id="torchrdit.solver.FourierBaseSolver.algorithm"></a>

#### algorithm

```python
@property
def algorithm()
```

Get the current algorithm.

<a id="torchrdit.solver.FourierBaseSolver.algorithm"></a>

#### algorithm

```python
@algorithm.setter
def algorithm(algorithm)
```

Set the algorithm to use.

<a id="torchrdit.solver.FourierBaseSolver.lam0"></a>

#### lam0

```python
@property
def lam0() -> np.ndarray
```

Get the free space wavelength array.

**Returns**:

  Array of free space wavelengths used in the simulation

<a id="torchrdit.solver.FourierBaseSolver.lam0"></a>

#### lam0

```python
@lam0.setter
def lam0(value: Union[float, np.ndarray]) -> None
```

Set the free space wavelength array.

**Arguments**:

- `value` - New wavelength value(s) to set
  

**Raises**:

- `ValueError` - If the input type is not float or numpy.ndarray

<a id="torchrdit.solver.FourierBaseSolver.add_source"></a>

#### add\_source

```python
def add_source(theta: float,
               phi: float,
               pte: float,
               ptm: float,
               norm_te_dir: str = "y") -> dict
```

Configure the incident electromagnetic wave source.

This method creates and returns a source configuration for the incident
electromagnetic wave. The source is defined by its incident angles and
polarization components. The returned dictionary can be passed directly
to the solve() method.

The incident wave direction is specified using spherical coordinates:
- theta: Polar angle measured from the z-axis (0° = normal incidence)
- phi: Azimuthal angle measured in the xy-plane from the x-axis

The polarization is specified using TE (transverse electric) and TM
(transverse magnetic) components. These can be complex values to represent
phase differences between polarization components.

**Arguments**:

- `theta` _float_ - Incident angle (polar angle) in radians.
  0 corresponds to normal incidence along the z-axis.
  π/2 corresponds to grazing incidence in the xy-plane.
  
- `phi` _float_ - Azimuthal angle in radians, measured in the xy-plane.
  0 corresponds to the x-axis.
  π/2 corresponds to the y-axis.
  
- `pte` _float_ - TE polarization amplitude (can be complex).
  The TE component has its electric field perpendicular to the plane of incidence.
  
- `ptm` _float_ - TM polarization amplitude (can be complex).
  The TM component has its magnetic field perpendicular to the plane of incidence.
  
- `norm_te_dir` _str, optional_ - Direction of normal component for TE wave.
- `Options` - 'x', 'y', or 'z'.
  This defines the reference direction for the TE polarization.
  Default is 'y', meaning the TE component is perpendicular to
  the y-axis for normal incidence.
  

**Returns**:

- `dict` - Source configuration dictionary that can be passed to the solve() method,
  containing:
  - 'theta': Polar angle in radians
  - 'phi': Azimuthal angle in radians
  - 'pte': TE polarization amplitude
  - 'ptm': TM polarization amplitude
  - 'norm_te_dir': Direction of normal component for TE wave
  

**Examples**:

```python
import numpy as np
from torchrdit.solver import create_solver

solver = create_solver()
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
result = solver.solve(source) # SolverResults object
```
  
  45-degree incidence with mixed polarization:
```python
import numpy as np
# Convert degrees to radians
theta_rad = 45 * np.pi / 180
# Create source with 45-degree incidence and equal TE/TM components
source = solver.add_source(
    theta=theta_rad,
    phi=0,
    pte=0.7071,
    ptm=0.7071
)
```
  
  Normal incidence with circular polarization:
```python
import numpy as np
# Create source with circular polarization (90° phase shift between components)
source = solver.add_source(
    theta=0,
    phi=0,
    pte=1.0,
    ptm=1.0j  # 90° phase shift
)
```
  

**Notes**:

  The convention used in TorchRDIT defines the TE polarization as having
  the electric field perpendicular to the plane of incidence, and the TM
  polarization as having the magnetic field perpendicular to the plane of incidence.
  
  Keywords:
  source, incident wave, polarization, TE, TM, plane wave, incidence angle,
  azimuthal angle, polar angle, spherical coordinates, circular polarization

<a id="torchrdit.solver.FourierBaseSolver.solve"></a>

#### solve

```python
def solve(source: dict, **kwargs) -> SolverResults
```

Solve the electromagnetic problem for the configured structure.

This is the main entry point for running simulations. The function computes
the electromagnetic response of the configured structure for the specified
source and wavelengths. It calculates reflection/transmission coefficients,
diffraction efficiencies, and optionally field distributions.

The solver handles all aspects of the calculation, including:
- Setting up k-vectors based on incident angles
- Computing Fourier harmonics for structured layers
- Building the S-matrix for the entire structure
- Calculating fields and efficiencies

The implementation is fully differentiable, enabling gradient-based
optimization for inverse design of photonic structures. You can use
automatic differentiation through the solve operation to compute
gradients with respect to any tensor parameters (masks, thicknesses, etc.)
that have requires_grad=True.

**Arguments**:

- `source` _dict_ - Source configuration dictionary containing the incident wave
  parameters. This should be created using the add_source() method,
  with the following keys:
  - 'theta': Polar angle of incidence (in radians)
  - 'phi': Azimuthal angle of incidence (in radians)
  - 'pte': Complex amplitude of the TE polarization component
  - 'ptm': Complex amplitude of the TM polarization component
  - 'norm_te_dir': Direction of normal component for TE wave (default: 'y')
  
- `**kwargs` - Additional keyword arguments to customize the solution process:
  Default is False. If True, electric and magnetic field components
  will be computed and returned.
  - 'compute_modes' (bool): Whether to compute and return mode information
  Default is False.
  - 'store_matrices' (bool): Whether to store system matrices
  Default is False. Setting to True increases memory usage but
  can be useful for debugging.
  - 'return_all' (bool): Whether to return all intermediate results
  Default is False. Setting to True increases memory usage.
  

**Returns**:

- `SolverResults` - A dataclass containing the solution results with these main attributes:
  - reflection: Total reflection efficiency for each wavelength
  - transmission: Total transmission efficiency for each wavelength
  - reflection_diffraction: Reflection efficiencies for each diffraction order
  - transmission_diffraction: Transmission efficiencies for each diffraction order
  - reflection_field: Field components (x, y, z) in reflection region
  - transmission_field: Field components (x, y, z) in transmission region
  - structure_matrix: Scattering matrix for the entire structure
  - wave_vectors: Wave vector components (kx, ky, kinc, kzref, kztrn)
  
  The results also provide helper methods for extracting specific diffraction orders
  and analyzing propagating modes.
  

**Examples**:

```python
import numpy as np
import torch
from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm

# Create a solver and set up structure
solver = create_solver(algorithm=Algorithm.RCWA)
# ... add materials and layers here ...

# Define source with normal incidence and TE polarization
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)

# Solve and get results
result = solver.solve(source) # SolverResults object
print(f"Reflection: {result.reflection[0].item():.4f}")  # reflection efficiency of the first wavelength
print(f"Transmission: {result.transmission[0].item():.4f}")  # transmission efficiency of the first wavelength

# Access specific diffraction orders
zero_order_efficiency = result.get_order_transmission_efficiency(0, 0)

# Extract field components for zero-order
tx, ty, tz = result.get_zero_order_transmission()
rx, ry, rz = result.get_zero_order_reflection()

# Calculate field amplitudes and phases
tx_amplitude = torch.abs(tx[0])  # Amplitude of x component (first wavelength)
tx_phase = torch.angle(tx[0])    # Phase in radians

# Convert phase to degrees
tx_phase_degrees = tx_phase * 180 / np.pi

# Calculate phase difference between field components
phase_diff = torch.angle(tx[0]) - torch.angle(ty[0])

# Get all available diffraction orders
all_orders = result.get_all_diffraction_orders()

# Find propagating orders for specific wavelength
propagating = result.get_propagating_orders(wavelength_idx=0)

# Access scattering matrix components
s11 = result.structure_matrix.S11  # shape: (n_wavelengths, 2*n_harmonics_squared, 2*n_harmonics_squared)

# Access k-vectors information
kx = result.wave_vectors.kx
ky = result.wave_vectors.ky
kzref = result.wave_vectors.kzref
```
  
  Gradient-based optimization:
```python
import torch

# Create a mask parameter with gradients enabled
mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
mask = mask.to(torch.float32)
mask.requires_grad = True

# Configure the patterned layer with the mask
solver.update_er_with_mask(mask=mask, layer_index=0)

# Solve and compute gradient to maximize transmission
result = solver.solve(source) # SolverResults object
(-result.transmission[0]).backward()  # Maximize transmission

# Access gradients for optimization
grad = mask.grad

# Optimize with gradient descent
optimizer = torch.optim.Adam([mask], lr=0.01)

for i in range(100):
    optimizer.zero_grad()
    result = solver.solve(source) # SolverResults object
    loss = -result.transmission[0]  # Negative for maximization
    loss.backward()
    optimizer.step()
```
  

**Notes**:

  Before calling solve(), you should:
  1. Configure the solver with appropriate parameters
  2. Define materials using add_materials()
  3. Create layers using add_layer()
  4. Set up patterned layers using update_er_with_mask() if needed
  5. Create a source using add_source()
  
  Keywords:
  electromagnetic simulation, Maxwell's equations, diffraction, reflection,
  transmission, efficiency, automatic differentiation, gradient,
  inverse design, field computation, optimization, differentiable simulation

<a id="torchrdit.solver.FourierBaseSolver.update_er_with_mask"></a>

#### update\_er\_with\_mask

```python
def update_er_with_mask(mask: torch.Tensor,
                        layer_index: int,
                        bg_material: str = "air",
                        method: str = "FFT") -> None
```

Update the permittivity distribution in a layer using a binary mask.

This method allows you to define the spatial distribution of materials in a layer
using a binary mask. The mask defines regions where the foreground material (mask=1)
and background material (mask=0) are located. This is particularly useful for
defining complex patterns, gratings, or arbitrary shapes within a layer.

This functionality is a key component for meta-optics design, enabling the creation of:
- Parameter-constrained and free-form meta-atoms
- Reconfigurable photonic structures
- Meta-lenses and beam deflectors
- Complex diffractive elements

When combined with automatic differentiation, this method enables gradient-based
optimization for the inverse design of meta-optics devices, as demonstrated in
"Eigendecomposition-free inverse design of meta-optics devices" (Huang et al., 2024).

If the specified layer is currently homogeneous, it will be automatically
converted to a grating (patterned) layer. The method computes the Fourier
transform of the pattern to obtain the permittivity distribution in k-space,
which is required for the RCWA/R-DIT calculations.

The method supports two approaches for computing the Toeplitz matrix:
- 'FFT': Uses Fast Fourier Transform, suitable for all cell types
- 'Analytical': Uses analytical expressions, only available for Cartesian cells

**Arguments**:

- `mask` - Binary tensor representing the pattern mask, where:
  - 1 (or True) represents the foreground material (from the layer's material)
  - 0 (or False) represents the background material (specified by bg_material)
  The mask dimensions must match the real-space dimensions (rdim) of the solver.
  
  For inverse design applications, this mask can be a differentiable tensor
  generated from a neural network or optimization process.
  
- `layer_index` - Index of the layer to update. This should be a valid index in the
  layer stack (0 to nlayer-1).
  
- `bg_material` - Name of the background material to use where mask=0. Default is 'air'.
  This must be a valid material name that exists in the solver's material list.
  
- `method` - Method for computing the Toeplitz matrix:
  - 'FFT': Fast Fourier Transform method (works for all cell types)
  - 'Analytical': Analytical method (only works for Cartesian cells)
  Default is 'FFT'.
  

**Returns**:

  None
  

**Raises**:

- `ValueError` - If the mask dimensions don't match the solver's real-space dimensions.
- `KeyError` - If the specified background material doesn't exist in the material list.
  

**Examples**:

```python
# Basic usage: Create a binary mask for a simple grating pattern
mask = torch.zeros(512, 512)
mask[:, :256] = 1.0  # Half the domain has the foreground material

# Update layer 1 with the mask, using air as background
solver.update_er_with_mask(mask=mask, layer_index=1, bg_material='air')
```
  
  Inverse design usage:
```python
import torch.nn as nn

# Define a simple network to generate a mask
class MaskGenerator(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        # Create normalized coordinate grid
        x = torch.linspace(-1, 1, dim)
        y = torch.linspace(-1, 1, dim)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')

    def forward(self):
        # Stack coordinates
        coords = torch.stack([self.X.flatten(), self.Y.flatten()], dim=1)
        # Generate mask values
        mask_flat = self.layers(coords)
        # Reshape to 2D
        return mask_flat.reshape(self.X.shape)

# Create and use the mask generator
mask_gen = MaskGenerator(dim=512).to(device)
optimizer = torch.optim.Adam(mask_gen.parameters(), lr=0.001)

# Optimization loop
for i in range(100):
    optimizer.zero_grad()

    # Generate mask
    mask = mask_gen()

    # Update solver with mask
    solver.update_er_with_mask(mask, layer_index=1)

    # Solve and calculate loss
    result = solver.solve(source) # SolverResults object
    loss = -result.transmission[0]  # Maximize transmission

    # Backpropagate and update
    loss.backward()
    optimizer.step()
```
  

**Notes**:

  This method only updates the permittivity (epsilon) distribution. To update
  the permeability (mu) distribution, you would need to modify the layer's
  urmat property directly.
  

**References**:

  - Huang et al., "Eigendecomposition-free inverse design of meta-optics devices,"
  Opt. Express 32, 13986-13997 (2024)
  - Huang et al., "Inverse Design of Photonic Structures Using Automatic Differentiable
  Rigorous Diffraction Interface Theory," CLEO (2023)

<a id="torchrdit.solver.FourierBaseSolver.update_er_with_mask_extern_NV"></a>

#### update\_er\_with\_mask\_extern\_NV

```python
def update_er_with_mask_extern_NV(mask: torch.Tensor,
                                  nv_vectors: tuple,
                                  layer_index: int,
                                  bg_material: str = "air",
                                  method: str = "FFT") -> None
```

Update permittivity in a layer using a mask with external normal vectors.

This method is an advanced version of update_er_with_mask that allows you
to provide pre-computed normal vectors for Fast Fourier Factorization (FFF).
It's particularly useful for complex geometries where normal vectors need
to be computed using specialized algorithms or external tools.

The method creates a patterned material distribution in the specified layer,
where the mask defines the regions of foreground and background materials.
It then uses the provided normal vectors for improved accuracy in the
Fourier factorization process.

**Arguments**:

- `mask` _torch.Tensor_ - Binary pattern mask that defines the material distribution.
  Regions with mask=1 use the layer's material, while regions with
  mask=0 use the background material. Must match the solver's rdim dimensions.
  
- `nv_vectors` _tuple_ - Tuple containing two tensors (norm_vec_x, norm_vec_y) that
  represent the x and y components of the normal vectors at material
  interfaces. These vectors are used for the FFF algorithm.
  
- `layer_index` _int_ - Index of the layer to update. Must be a valid index
  in the range 0 to (number of layers - 1).
  
- `bg_material` _str, optional_ - Name of the background material to use
  where mask=0. Must be a valid material name in the solver's
  material list. Default is 'air'.
  
- `method` _str, optional_ - Method for computing the Toeplitz matrix:
  - 'FFT': Fast Fourier Transform method (works for all cell types)
  - 'Analytical': Analytical method (only works for Cartesian cells)
  Default is 'FFT'.
  

**Examples**:

```python
import torch
import numpy as np
from torchrdit.solver import create_solver

# Create a solver and add materials
solver = create_solver(is_use_FFF=True)  # Enable FFF
solver.add_materials(['Si', 'Air'])
solver.add_layer('Si', thickness=0.5, is_homogeneous=False)

# Create a mask and compute normal vectors externally
mask = torch.zeros(512, 512)
mask[200:300, 200:300] = 1.0  # Square pattern

# Pre-computed normal vectors (would normally come from specialized algorithm)
norm_x = torch.zeros_like(mask)
norm_y = torch.zeros_like(mask)
# ... compute normal vectors at material interfaces ...

# Update the layer with the mask and external normal vectors
solver.update_er_with_mask_extern_NV(
    mask=mask,
    nv_vectors=(norm_x, norm_y),
    layer_index=0,
    bg_material='Air'
)
```
  

**Notes**:

  This method is specifically designed for advanced users who need precise
  control over the Fast Fourier Factorization process. For most applications,
  the standard update_er_with_mask method is sufficient.

<a id="torchrdit.solver.FourierBaseSolver.update_layer_thickness"></a>

#### update\_layer\_thickness

```python
def update_layer_thickness(layer_index: int, thickness: torch.Tensor)
```

Update the thickness of a specific layer in the structure.

This method allows you to dynamically change the thickness of a layer
in the simulation structure. This is particularly useful for:
- Parametric sweeps over layer thicknesses
- Optimization of layer thicknesses for specific optical responses
- Dynamic adjustment of device structures during simulation

The thickness parameter can be a regular Tensor or one with gradients
enabled for automatic differentiation in optimization workflows.

**Arguments**:

- `layer_index` _int_ - Index of the layer to modify. Must be a valid index
  in the range 0 to (number of layers - 1).
- `thickness` _torch.Tensor_ - New thickness value for the layer.
  Must be a positive scalar Tensor in the solver's length units.
  Can have requires_grad=True for gradient-based optimization.
  

**Examples**:

```python
import torch
from torchrdit.solver import create_solver

# Create a solver and add some layers
solver = create_solver()
solver.add_layer(material_name="Si", thickness=torch.tensor(0.5))

# Update the thickness of the first layer
solver.update_layer_thickness(layer_index=0, thickness=torch.tensor(0.75))

# For optimization, use a thickness parameter with gradients enabled
thickness_param = torch.tensor(0.5, requires_grad=True)
solver.update_layer_thickness(layer_index=0, thickness=thickness_param)
```

<a id="torchrdit.solver.FourierBaseSolver.expand_dims"></a>

#### expand\_dims

```python
def expand_dims(mat: torch.Tensor) -> Optional[torch.Tensor]
```

Function that expands the input matrix to a standard output dimension without layer information:
(n_freqs, n_harmonics_squared, n_harmonics_squared)

**Arguments**:

- `mat` _torch.Tensor_ - input tensor matrxi

<a id="torchrdit.solver.RCWASolver"></a>

## RCWASolver Objects

```python
class RCWASolver(FourierBaseSolver)
```

Rigorous Coupled-Wave Analysis (RCWA) Solver.

This class implements the traditional Rigorous Coupled-Wave Analysis method for solving
electromagnetic wave propagation through periodic structures. RCWA is a
frequency-domain technique that expands the electromagnetic fields and
material properties in Fourier series and solves Maxwell's equations by
matching boundary conditions at layer interfaces.

While the TorchRDIT package primarily emphasizes the eigendecomposition-free R-DIT
method for improved performance, this RCWA implementation is provided for
comparison and for cases where RCWA may be preferred.

RCWA is particularly effective for:
- Periodic structures (gratings, photonic crystals, metasurfaces)
- Multilayer stacks with complex patterns
- Structures with sharp material interfaces
- Calculating diffraction efficiencies and field distributions

The implementation uses PyTorch for efficient computation and supports:
- GPU acceleration
- Automatic differentiation for gradient-based optimization
- Batch processing of multiple wavelengths
- Fast Fourier Factorization for improved convergence

Applications include:
- Analysis and design of diffractive optical elements
- Characterization of photonic crystals and metamaterials
- Inverse design of meta-optics (though R-DIT is recommended for better performance)

**Example**:

    ```python
    # Create an RCWA solver
    solver = RCWASolver(
        lam0=np.array([1.55]),  # Wavelength in micrometers
        rdim=[512, 512],        # Real space dimensions
        kdim=[5, 5],            # Fourier space dimensions
        device='cuda'           # Use GPU acceleration
    )

    # Add layers and configure the simulation
    solver.add_layer(0.5, 'Air')
    solver.add_layer(0.2, 'Si')
    solver.add_layer(1.0, 'SiO2')

    # Define a source and solve
    source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
    result = solver.solve(source) # SolverResults object
    ```

<a id="torchrdit.solver.RCWASolver.set_rdit_order"></a>

#### set\_rdit\_order

```python
def set_rdit_order(rdit_order)
```

Set the order of the R-DIT algorithm.

This method allows you to configure the order of the Rigorous Diffraction
Interface Theory algorithm. The R-DIT order affects the numerical stability
and accuracy of the solution, especially for thick or high-contrast layers.

**Arguments**:

- `rdit_order` _int_ - The order of the R-DIT algorithm. Higher orders generally
  provide better accuracy but may be computationally more expensive.
  

**Examples**:

```python
solver = RCWASolver(
    lam0=np.array([1.55]),
    device='cuda'
)

# Set RDIT order to 2 for a balance of accuracy and performance
solver.set_rdit_order(2)
```

<a id="torchrdit.solver.RDITSolver"></a>

## RDITSolver Objects

```python
class RDITSolver(FourierBaseSolver)
```

Eigendecomposition-free Rigorous Diffraction Interface Theory (R-DIT) Solver.

This class implements the eigendecomposition-free Rigorous Diffraction Interface Theory
method for solving electromagnetic wave propagation through periodic structures.
The R-DIT solver is specifically designed for inverse design of meta-optics devices,
achieving up to 16.2× speedup compared to traditional RCWA-based methods.

The R-DIT method:
- Reformulates the problem in terms of impedance matrices
- Avoids eigendecomposition, significantly improving computational efficiency
- Eliminates numerical instabilities associated with exponentially growing modes
- Provides better convergence for thick layers and high-contrast material systems
- Maintains the same accuracy as RCWA with better numerical properties
- Generates stable gradients for topology optimization

**Arguments**:

- `lam0` - Wavelength(s) to simulate, in the specified length unit. Default is [1.0].
- `lengthunit` - The unit of length used in the simulation (e.g., 'um', 'nm').
  Default is 'um' (micrometers).
- `rdim` - Dimensions of the real space grid [height, width]. Default is [512, 512].
- `kdim` - Dimensions in Fourier space [kheight, kwidth]. Default is [3, 3].
  This determines the number of Fourier harmonics used.
- `materiallist` - List of materials used in the simulation. Default is empty list.
- `t1` - First lattice vector. Default is [[1.0, 0.0]] (unit vector in x-direction).
- `t2` - Second lattice vector. Default is [[0.0, 1.0]] (unit vector in y-direction).
- `is_use_FFF` - Whether to use Fast Fourier Factorization. Default is True.
  This improves convergence for high-contrast material interfaces.
- `precision` - Numerical precision to use (SINGLE or DOUBLE). Default is SINGLE.
- `device` - The device to run the solver on ('cpu', 'cuda', etc.). Default is 'cpu'.
  For optimal performance, especially with the R-DIT solver, using 'cuda'
  is highly recommended for GPU acceleration.
  

**Returns**:

  A solver instance (either RCWASolver or RDITSolver) configured according
  to the provided parameters. The returned solver is fully differentiable,
  enabling gradient-based optimization for inverse design tasks.
  

**Example**:

```python
# Create an R-DIT solver
solver = RDITSolver(
    lam0=np.array([1.55]),  # Wavelength in micrometers
    rdim=[512, 512],        # Real space dimensions
    kdim=[5, 5],            # Fourier space dimensions
    device='cuda'           # Use GPU acceleration
)

# Set R-DIT order (optional)
solver.set_rdit_order(2)

# Add layers and configure the simulation
solver.add_layer(0.5, 'Air')
solver.add_layer(0.2, 'Si')
solver.add_layer(1.0, 'SiO2')

# Define a source and solve
source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
result = solver.solve(source) # SolverResults object
```
  

**References**:

  - Huang et al., "Eigendecomposition-free inverse design of meta-optics devices,"
  Opt. Express 32, 13986-13997 (2024)
  - Huang et al., "Inverse Design of Photonic Structures Using Automatic Differentiable
  Rigorous Diffraction Interface Theory," CLEO (2023)

<a id="torchrdit.solver.RDITSolver.set_rdit_order"></a>

#### set\_rdit\_order

```python
def set_rdit_order(rdit_order: int) -> None
```

Set the order of the R-DIT algorithm.

This method allows you to configure the order of the Rigorous Diffraction
Interface Theory algorithm. The R-DIT order affects the numerical stability
and accuracy of the solution, especially for thick or high-contrast layers.

The R-DIT order determines the approximation used in the diffraction
interface theory. Higher orders generally provide better accuracy but
may be computationally more expensive.

**Arguments**:

- `rdit_order` - The order of the R-DIT algorithm. Higher orders generally
  provide better accuracy but may be computationally more
  expensive. Typical values are 1, 2, or 3.
  - 1: First-order approximation (fastest, least accurate)
  - 2: Second-order approximation (good balance)
  - 3: Third-order approximation (most accurate, slowest)
  

**Returns**:

  None
  

**Example**:

```python
solver = RDITSolver(...)
solver.set_rdit_order(2)  # Set R-DIT order to 2 for good balance
```
  

**Notes**:

  The optimal R-DIT order depends on the specific problem. For most
  applications, order 2 provides a good balance between accuracy and
  computational efficiency. For highly demanding applications with
  thick layers or high material contrast, order 3 may be necessary.

<a id="torchrdit.solver.get_solver_builder"></a>

#### get\_solver\_builder

```python
def get_solver_builder()
```

Get a solver builder for creating a solver with a fluent interface.

This function returns a new instance of the SolverBuilder class, which provides
a fluent interface for configuring and creating solver instances. The builder
pattern allows for a more readable and flexible way to specify solver parameters
compared to passing all parameters to the constructor at once.

**Returns**:

- `SolverBuilder` - A new solver builder instance.
  

**Examples**:

```python
from torchrdit.solver import get_solver_builder

# Get a builder and configure it with method chaining
builder = get_solver_builder()
solver = (builder
          .with_algorithm("RDIT")
          .with_wavelengths([1.55])
          .with_device("cuda")
          .with_real_dimensions([512, 512])
          .build())
```
  

**See Also**:

- `create_solver_from_builder` - For an alternative approach using a configuration function.

<a id="torchrdit.solver.create_solver_from_builder"></a>

#### create\_solver\_from\_builder

```python
def create_solver_from_builder(
    builder_config: Callable[["SolverBuilder"], "SolverBuilder"]
) -> Union["RCWASolver", "RDITSolver"]
```

Create a solver from a builder configuration function.

This function provides an alternative way to create and configure a solver
using a configuration function that takes a builder and returns a configured
builder. This approach can be particularly useful when you want to encapsulate
solver configuration logic in reusable functions.

**Arguments**:

- `builder_config` _Callable_ - A function that takes a SolverBuilder instance,
  configures it, and returns the configured builder.
  

**Returns**:

  Union[RCWASolver, RDITSolver]: A solver instance configured according
  to the builder configuration.
  

**Examples**:

```python
from torchrdit.solver import create_solver_from_builder

# Define a configuration function
def configure_silicon_photonics_solver(builder):
    return (builder
            .with_algorithm("RDIT")
            .with_wavelengths([1.31, 1.55])
            .with_device("cuda")
            .with_real_dimensions([512, 512])
            .with_k_dimensions([7, 7]))

# Create a solver using the configuration function
solver = create_solver_from_builder(configure_silicon_photonics_solver)
```
  

**See Also**:

- `get_solver_builder` - For direct access to a builder instance.

