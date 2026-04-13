"""
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

Examples:
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
    grids=[512, 512],        # Real space grid dimensions
    harmonics=[5, 5],            # Fourier space dimensions
    device="cuda"           # Use GPU acceleration
)

# Create materials
silicon = create_material(name="silicon", permittivity=11.7)
sio2 = create_material(name="sio2", permittivity=2.25)
solver.add_materials([silicon, sio2])

# Dispersive material can be created from a data file or in-memory samples:
# gold = create_material(
#     name="gold",
#     dielectric_dispersion=True,
#     user_dielectric_wavelengths_um=[1.0, 1.5, 2.0],
#     user_dielectric_n=[0.45, 0.35, 0.30],
#     user_dielectric_k=[6.5, 7.0, 7.5],
# )

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
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .algorithm import RCWAAlgorithm, RDITAlgorithm, SolverAlgorithm
from .cell import Cell3D
from .external_regions import ExternalRegionsMixin
from .field_calculation import FieldCalculationMixin
from .layer_processing import LayerProcessingMixin
from .constants import Algorithm, Precision
from .materials import MaterialClass
from .numerics import safe_kz_reciprocal, softplus_floor, softplus_magnitude_floor
from .results import SolverResults, FieldComponents, ScatteringMatrix, WaveVectors

# BatchedSolverResults functionality is now integrated into SolverResults
from .utils import SMatrix, blockmat2x2, init_smatrix, redhstar, to_diag_util

if TYPE_CHECKING:
    from .builder import SolverBuilder


class SolverObserver:
    """Interface for observers that track solver progress.

    This class defines the Observer interface in the Observer design pattern
    for monitoring the progress and state of solvers. Concrete observer
    implementations should inherit from this class and implement the update
    method to receive and process notifications from solvers.

    The Observer pattern allows for loose coupling between solvers and progress
    monitoring components, enabling flexible visualization and logging of solver
    progress without modifying the solver implementation.

    Note:
        This is an abstract interface. Use concrete implementations like
        ConsoleProgressObserver or TqdmProgressObserver from the observers module.

    Attributes:
        None

    Examples:
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
    """

    def update(self, event_type: str, data: dict) -> None:
        """Called when the solver notifies of an event.

        This method is called by the solver when an event occurs. Concrete
        observer implementations should override this method to process
        notifications from the solver.

        Args:
            event_type (str): The type of event that occurred. Common event types include:
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

            data (dict): Additional data related to the event. Contents vary by event
                  type but may include:
                  - "mode": Solver mode (RCWA or RDIT)
                  - "n_freqs": Number of frequencies being solved
                  - "n_layers": Number of layers in the structure
                  - "current": Current layer or iteration index
                  - "total": Total number of items (e.g., layers)
                  - "progress": Percentage complete (0-100)

        Examples:
        ```python
        # Example implementation in a concrete observer class
        def update(self, event_type, data):
            if event_type == "calculation_starting":
                print(f"Starting calculation with {data.get('n_freqs')} wavelengths")
            elif event_type == "calculation_completed":
                print("Calculation finished!")
        ```

        Note:
            This is an abstract method that should be overridden by concrete observer
            classes. The default implementation does nothing.

        Keywords:
            event handling, notification, callback, progress update, observer method
        """
        pass


class SolverSubjectMixin:
    """Mixin class that allows a solver to notify observers of progress.

    This class implements the Subject role in the Observer design pattern.
    It maintains a list of observers and provides methods to add, remove,
    and notify observers of events. Classes that inherit from this mixin
    can easily support observer notifications without reimplementing the
    observer management logic.

    The Observer pattern enables solvers to report their progress to multiple
    observers (like console loggers, progress bars, or GUI components) without
    tight coupling to these visualization components.

    Attributes:
        _observers (list): List of observer objects registered to receive notifications

    Examples:
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
    """

    def __init__(self):
        """Initialize the observer list.

        Creates an empty list to store observer objects. This method should be
        called in the constructor of any class that inherits from this mixin.

        Examples:
        ```python
        class MyClass(SolverSubjectMixin):
            def __init__(self):
                SolverSubjectMixin.__init__(self)  # Initialize observer list
        ```

        Keywords:
            initialization, observer list, constructor
        """
        self._observers = []

    def add_observer(self, observer: SolverObserver) -> None:
        """Add an observer to the notification list.

        Registers an observer to receive notifications when events occur.
        If the observer is already registered, it will not be added again.

        Args:
            observer (SolverObserver): The observer object to add. Must implement
                      the SolverObserver interface with an update method.

        Examples:
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
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: SolverObserver) -> None:
        """Remove an observer from the notification list.

        Unregisters an observer so it no longer receives notifications.
        If the observer is not in the list, no action is taken.

        Args:
            observer (SolverObserver): The observer object to remove.

        Examples:
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
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self, event_type: str, data: dict = None) -> None:
        """Notify all observers of an event.

        This method calls the update method on all registered observers,
        passing the event type and data to each observer. This is the main
        mechanism for broadcasting events to all observers.

        Args:
            event_type (str): The type of event that occurred. This identifies
                      the type of event, allowing observers to handle different
                      events appropriately. Common event types include:
                      - "calculation_starting": When the solve process begins
                      - "layer_started": When beginning to process a layer
                      - "layer_completed": When a layer has been processed
                      - "calculation_completed": When solving is complete

            data (dict, optional): Additional data related to the event. This contains
                   event-specific information that might be useful to observers.
                   Default is an empty dictionary.

        Examples:
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
        """
        if data is None:
            data = {}

        for observer in self._observers:
            observer.update(event_type, data)


def create_solver_from_config(
    config: Union[str, Dict[str, Any]], flip: bool = False
) -> Union["RCWASolver", "RDITSolver"]:
    """Create a solver from a configuration file or dictionary.

    This function creates a solver instance based on a configuration specified
    either as a dictionary or a path to a JSON/YAML configuration file. It uses
    the SolverBuilder pattern internally to construct the solver.

    The configuration can specify all solver parameters, including algorithm type,
    wavelengths, dimensions, materials, and device options. This provides a convenient
    way to store and reuse solver configurations across different simulations.

    Args:
        config (Union[str, Dict[str, Any]]): Either a path to a configuration file (string)
                or a dictionary containing the configuration parameters. The configuration
                should specify all necessary parameters for the solver, including:
                - "algorithm": Algorithm type ("RCWA" or "RDIT")
                - "wavelengths": List of wavelengths to simulate
                - "lengthunit": Length unit (e.g., "um", "nm")
                - "grids": Real space dimensions [height, width]
                - "harmonics": Fourier space dimensions [kx, ky]
                - "device": Device to use (e.g., "cpu", "cuda")

        flip (bool, optional): Whether to flip the coordinate system. When True,
              the solver will use a flipped coordinate system, which can be useful
              for certain types of simulations or to match other software conventions.
              Default is False.

    Returns:
        Union[RCWASolver, RDITSolver]: A solver instance configured according
        to the provided parameters.

    Examples:
    Creating a solver from a dictionary configuration:

    ```python
    from torchrdit.solver import create_solver_from_config
    # Define a configuration dictionary
    config = {
        "algorithm": "RDIT",
        "wavelengths": [1.55],
        "length_unit": "um",
        "grids": [512, 512],
        "harmonics": [5, 5],
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
    #   "grids": [256, 256],
    #   "harmonics": [3, 3]
    # }
    solver = create_solver_from_config("config.json")
    ```

    Note:
        This function is particularly useful for reproducible simulations or
        when the simulation setup needs to be saved and restored later.

    Keywords:
        configuration, JSON, YAML, file configuration, solver creation, parameter loading,
        reproducible simulation, configuration file, solver setup
    """
    # Lazy import to avoid circular dependencies
    from .builder import SolverBuilder

    return SolverBuilder().from_config(config, flip).build()


def create_solver(
    algorithm: Algorithm = Algorithm.RDIT,
    precision: Precision = Precision.SINGLE,
    lam0: np.ndarray = np.array([1.0]),
    lengthunit: str = "um",
    grids: List[int] = [512, 512],
    harmonics: List[int] = [3, 3],
    materiallist: List[Any] = [],
    t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),
    t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),
    is_use_FFF: bool = False,
    fff_vector_scheme: str = "POL",
    fff_fourier_weight: float = 1e-2,
    fff_smoothness_weight: float = 1e-3,
    fff_vector_steps: int = 1,
    device: Union[str, torch.device] = "cpu",
    debug_batching: bool = False,
    debug_tensorization: bool = False,
    debug_unification: bool = False,
) -> Union["RCWASolver", "RDITSolver"]:
    """Create a solver with the given parameters.

    This is the main factory function for creating solver instances with custom
    parameters. It provides a convenient way to create a solver without having to
    directly instantiate the solver classes or use the builder pattern.

    The function creates either an RCWA solver (for traditional Rigorous Coupled-Wave
    Analysis) or an R-DIT solver (the eigendecomposition-free Rigorous Diffraction
    Interface Theory). By default, it creates an R-DIT solver, which offers significantly
    better performance for inverse design applications.

    Args:
        algorithm (Algorithm): The algorithm to use for solving. Options:
                 - Algorithm.RCWA: Traditional RCWA method
                 - Algorithm.RDIT: Eigendecomposition-free R-DIT method (default)
                 R-DIT is recommended for most applications, especially inverse design,
                 due to its superior computational efficiency and stable gradients.

        precision (Precision): Numerical precision to use. Options:
                  - Precision.SINGLE: Single precision (float32/complex64)
                  - Precision.DOUBLE: Double precision (float64/complex128, default)
                  Use double precision for higher accuracy at the cost of memory usage.

        lam0 (np.ndarray): Wavelength(s) to simulate, in the specified length unit.
               Can be a single value or an array of wavelengths. Default is [1.0].

        lengthunit (str): The unit of length used in the simulation. Common values:
                   - 'um': Micrometers (default)
                   - 'nm': Nanometers
                   All dimensions (wavelengths, thicknesses) are interpreted in this unit.

        grids (List[int]): Dimensions of the real space grid [height, width].
              Default is [512, 512]. Higher values provide more spatial resolution
              but require more memory and computation time.

        harmonics (List[int]): Dimensions in Fourier space [kx, ky]. Default is [3, 3].
              This determines the number of Fourier harmonics used in the simulation.
              Higher values improve accuracy but significantly increase computation time.

        materiallist (List[Any]): List of materials used in the simulation. Can include
                      MaterialClass instances created with create_material(), including
                      dispersive materials defined from a file or in-memory samples.
                      Default is an empty list.

        t1 (torch.Tensor): First lattice vector defining the unit cell geometry.
              Default is [[1.0, 0.0]] (unit vector in x-direction).

        t2 (torch.Tensor): Second lattice vector defining the unit cell geometry.
              Default is [[0.0, 1.0]] (unit vector in y-direction).

        is_use_FFF (bool): Whether to use Fast Fourier Factorization. Default is False.
                    This improves convergence for high-contrast material interfaces
                    but adds computational overhead.

        fff_vector_scheme (str): Tangent vector field scheme to use when generating
                Fourier-factorization normals. Choose among:

                * ``'POL'`` – polarization-preserving tangents normalized by the global maximum.
                * ``'NORMAL'`` – per-pixel unit tangents for geometry-driven factorization.
                * ``'JONES'`` – convert refined tangents into Jones vectors after the solve.
                * ``'JONES_DIRECT'`` – run the Newton refinement directly in Jones space
                  (fmmax/S4 style).

                Default is ``'POL'``.
        fff_fourier_weight (float): Weight applied to the Fourier-domain loss
                during tangent field refinement. Default is ``1e-2``.
        fff_smoothness_weight (float): Weight applied to the smoothness loss
                term for tangent field refinement. Default is ``1e-3``.
        fff_vector_steps (int): Number of Newton iterations used when refining the
                tangent field. Must be a positive integer. Default is ``1``.

        device (Union[str, torch.device]): The device to run the solver on.
                Options include 'cpu' (default) or 'cuda' for GPU acceleration.
                For optimal performance, especially with the R-DIT solver,
                using 'cuda' is highly recommended.

    Returns:
        Union[RCWASolver, RDITSolver]: A solver instance configured according
        to the provided parameters. The returned solver is fully differentiable,
        enabling gradient-based optimization for inverse design.

    Examples:
    ```python
    import torch
    import numpy as np
    from torchrdit.solver import create_solver
    from torchrdit.constants import Algorithm, Precision

    # Create an R-DIT solver with GPU acceleration
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),  # Wavelength (μm)
        grids=[512, 512],        # Real space resolution
        harmonics=[5, 5],            # Fourier space harmonics
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
        grids=[1024, 1024],
        harmonics=[7, 7],
        device='cuda'
    )
    ```

    Using source batching for efficient multi-angle simulations:
    ```python
    # Create solver and set up structure
    solver = create_solver(
        algorithm=Algorithm.RDIT,
        lam0=np.array([1.55]),
        grids=[512, 512],
        harmonics=[7, 7],
        device='cuda'
    )

    # Add materials and layers
    from torchrdit.utils import create_material
    si = create_material(name="Si", permittivity=12.25)
    air = create_material(name="air", permittivity=1.0)
    solver.add_materials([si, air])
    solver.add_layer(material_name="Si", thickness=0.6, is_homogeneous=False)

    # Create grating pattern
    import torch
    mask = torch.zeros(512, 512)
    period = int(512 * 0.8 / 1.0)  # Period in pixels
    duty_cycle = 0.5
    for i in range(0, 512, period):
        mask[:, i:i+int(period*duty_cycle)] = 1.0
    solver.update_er_with_mask(mask=mask, layer_index=0)

    # Create multiple sources for angle sweep
    deg = np.pi / 180
    sources = [
        solver.add_source(theta=angle, phi=0, pte=1.0, ptm=0.0)
        for angle in np.linspace(0, 60, 13) * deg
    ]

    # Batch solve - much faster than sequential processing
    results = solver.solve(sources)  # Returns unified SolverResults with batching support

    # Analyze results
    best_idx = results.find_optimal_source('max_transmission')
    print(f"Best angle: {sources[best_idx]['theta'] * 180/np.pi:.1f}°")
    ```

    Note:
        To optimize memory usage and performance:
        1. Use the R-DIT algorithm (default) for inverse design applications
        2. Use GPU acceleration (device='cuda') when available
        3. Adjust grids and harmonics based on required accuracy and available memory
        4. Use single precision for large simulations where memory is a concern

    Keywords:
        solver creation, RCWA, R-DIT, factory function, electromagnetic simulation,
        photonics, meta-optics, GPU acceleration, lattice vectors, Fourier harmonics,
        numerical precision, wavelength, inverse design
    """
    # Lazy import to avoid circular dependencies
    from .builder import SolverBuilder

    # Use the builder to create the solver
    builder = (
        SolverBuilder()
        .with_algorithm(algorithm)
        .with_precision(precision)
        .with_wavelengths(lam0)
        .with_length_unit(lengthunit)
        .with_real_dimensions(grids)
        .with_k_dimensions(harmonics)
        .with_materials(materiallist)
        .with_lattice_vectors(t1, t2)
        .with_fff(is_use_FFF)
        .with_fff_vector_options(
            scheme=fff_vector_scheme,
            fourier_weight=fff_fourier_weight,
            smoothness_weight=fff_smoothness_weight,
            steps=fff_vector_steps,
        )
        .with_device(device)
    )

    # Pass debug flags directly if the builder doesn't have methods for them
    solver = builder.build()
    if hasattr(solver, "debug_batching"):
        solver.debug_batching = debug_batching
    if hasattr(solver, "debug_tensorization"):
        solver.debug_tensorization = debug_tensorization
    if hasattr(solver, "debug_unification"):
        solver.debug_unification = debug_unification

    return solver


# Backward-compatible wrappers — implementations live in numerics.py


def softplus_protect_kz(kz, min_kz=1e-3, beta=100):
    """Backward-compatible wrapper. See :func:`numerics.softplus_magnitude_floor`."""
    return softplus_magnitude_floor(kz, min_kz, beta=beta)


def softplus_clamp_min(x, min_val=0.0, beta=100):
    """Backward-compatible wrapper. See :func:`numerics.softplus_floor`."""
    return softplus_floor(x, min_val, beta=beta)


class FourierBaseSolver(Cell3D, SolverSubjectMixin, LayerProcessingMixin, FieldCalculationMixin, ExternalRegionsMixin):
    """Base class for Fourier-based electromagnetic solvers.

    This class serves as the foundation for Fourier-based electromagnetic solvers,
    providing common functionality and interface for derived solver implementations
    such as RCWASolver and RDITSolver.

    FourierBaseSolver implements core computational methods for electromagnetic
    simulations of layered structures with periodic boundary conditions. It handles
    the setup of computational grids, material properties, layer stacks, and the
    calculation of optical responses (reflection, transmission, field distributions)
    using Fourier-based methods.

    Attributes:
        tcomplex (torch.dtype): Complex data type used for calculations.
        tfloat (torch.dtype): Float data type used for calculations.
        tint (torch.dtype): Integer data type used for calculations.
        nfloat (np.dtype): NumPy float data type for compatibility.
        device (Union[str, torch.device]): Device used for computation ('cpu' or 'cuda').
        grids (List[int]): Dimensions in real space [height, width].
        harmonics (List[int]): Dimensions in k-space [kx, ky].
        shapes (ShapeGenerator): Generator for creating shape masks.
        layer_manager (LayerManager): Manager for handling material layers.

    Note:
        This is an abstract base class and should not be instantiated directly.
        Use RCWASolver or RDITSolver for concrete implementations, or the
        create_solver() function to create an appropriate solver instance.

    Examples:
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
        grids=[512, 512],
        harmonics=[5, 5]
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
    """

    def __init__(
        self,
        # wavelengths (frequencies) to be solved
        lam0: Union[float, np.ndarray] = np.array([1.0]),
        lengthunit: str = "um",  # length unit used in the solver
        grids: Optional[List[int]] = None,  # dimensions in real space: (H, W)
        harmonics: Optional[List[int]] = None,  # dimensions in k space: (kH, kW)
        materiallist: Optional[List[MaterialClass]] = None,  # list of materials
        t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
        t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
        is_use_FFF: bool = False,  # if use Fast Fourier Factorization
        fff_vector_scheme: str = "POL",
        fff_fourier_weight: float = 1e-2,
        fff_smoothness_weight: float = 1e-3,
        fff_vector_steps: int = 1,
        precision: Precision = Precision.SINGLE,
        device: Union[str, torch.device] = "cpu",
        algorithm: SolverAlgorithm = None,
        debug_batching: bool = False,  # Enable debug output for batched operations
        debug_tensorization: bool = False,  # Enable debug output for tensorization
        debug_unification: bool = False,  # Enable debug output for unification
    ) -> None:
        # Set numerical precision
        if precision == Precision.SINGLE:
            self.tcomplex = torch.complex64
            self.tfloat = torch.float32
            self.tint = torch.int32
            self.nfloat = np.float32
        else:
            self.tcomplex = torch.complex128
            self.tfloat = torch.float64
            self.tint = torch.int64
            self.nfloat = np.float64

        if grids is None:
            grids = [512, 512]
        else:
            grids = list(grids)
        if harmonics is None:
            harmonics = [3, 3]
        else:
            harmonics = list(harmonics)
        if materiallist is None:
            materiallist = []
        else:
            materiallist = list(materiallist)

        Cell3D.__init__(self, lengthunit, grids, harmonics, materiallist, t1, t2, device)
        SolverSubjectMixin.__init__(self)

        # Store debug flags
        self.debug_batching = debug_batching
        self.debug_tensorization = debug_tensorization
        self.debug_unification = debug_unification

        # Initialize tensors that were previously class variables
        self.mesh_fp = torch.empty((3, 3), device=self.device, dtype=self.tfloat)
        self.mesh_fq = torch.empty((3, 3), device=self.device, dtype=self.tfloat)
        self.reci_t1 = torch.empty((2,), device=self.device, dtype=self.tfloat)
        self.reci_t2 = torch.empty((2,), device=self.device, dtype=self.tfloat)
        self.tlam0 = torch.empty((1,), device=self.device, dtype=self.tfloat)
        self.k_0 = torch.empty((1,), device=self.device, dtype=self.tfloat)

        # Set free space wavelength
        if lam0 is None:
            lam0 = np.array([1.0], dtype=self.nfloat)
        if isinstance(lam0, (float, int)):
            self._lam0 = np.array([float(lam0)], dtype=self.nfloat)
        elif isinstance(lam0, np.ndarray):
            self._lam0 = lam0.astype(self.nfloat)
        else:
            try:
                self._lam0 = np.array(lam0, dtype=self.nfloat)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"lam0 must be float or numpy.ndarray, got {type(lam0)}") from exc

        if self._lam0.size == 0:
            raise ValueError("lam0 must be non-empty")

        self.n_freqs = len(self._lam0)
        self.kinc = torch.zeros((1, 3), device=self.device, dtype=self.tfloat)

        self.src = {}
        self.is_use_FFF = is_use_FFF

        self.fff_vector_scheme = fff_vector_scheme
        self.fff_fourier_weight = float(fff_fourier_weight)
        self.fff_smoothness_weight = float(fff_smoothness_weight)
        self.fff_vector_steps = int(fff_vector_steps)
        if self.fff_vector_steps < 1:
            raise ValueError("fff_vector_steps must be a positive integer")

        # Cache for tangent-field generators keyed by lattice/device/hyper-parameters.
        self._vector_generator_cache: Dict[
            Tuple[
                Tuple[int, int],
                Tuple[float, ...],
                Tuple[float, ...],
                str,
                torch.dtype,
                float,
                float,
                int,
            ],
            Any,
        ] = {}

        # Set the solving algorithm strategy
        self._algorithm = algorithm

    @property
    def algorithm(self):
        """Get the current algorithm."""
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        """Set the algorithm to use."""
        if not isinstance(algorithm, SolverAlgorithm):
            raise TypeError("Algorithm must be an instance of SolverAlgorithm")
        self._algorithm = algorithm

    def _setup_reciprocal_space(self):
        """Set up reciprocal lattice vectors and k-space mesh.

        This method is idempotent - it only calculates these values once
        and subsequent calls return immediately. This avoids redundant
        calculations when processing multiple sources.

        Sets up:
        - self.reci_t1, self.reci_t2: Reciprocal lattice vectors
        - self.mesh_fp, self.mesh_fq: k-space mesh grids
        - self.tlam0: Wavelengths as tensor
        - self.k_0: Wave vector magnitudes
        """
        # Check if already initialized
        if hasattr(self, "_reciprocal_space_initialized") and self._reciprocal_space_initialized:
            return

        # Calculate reciprocal lattice vectors
        d_v = self.lattice_t1[0] * self.lattice_t2[1] - self.lattice_t2[0] * self.lattice_t1[1]

        self.reci_t1 = (
            2
            * torch.pi
            * torch.cat(((+self.lattice_t2[1] / d_v).unsqueeze(0), (-self.lattice_t2[0] / d_v).unsqueeze(0)), dim=0)
        )
        self.reci_t2 = (
            2
            * torch.pi
            * torch.cat(((-self.lattice_t1[1] / d_v).unsqueeze(0), (+self.lattice_t1[0] / d_v).unsqueeze(0)), dim=0)
        )

        # Calculate wave vector expansion
        self.tlam0 = torch.tensor(self.lam0, dtype=self.tfloat, device=self.device)
        self.k_0 = 2 * torch.pi / self.tlam0  # k0 with dimensions: n_freqs

        # Set up k-space mesh
        f_p = torch.arange(
            start=-np.floor(self.harmonics[0] / 2), end=np.floor(self.harmonics[0] / 2) + 1, dtype=self.tint, device=self.device
        )
        f_q = torch.arange(
            start=-np.floor(self.harmonics[1] / 2), end=np.floor(self.harmonics[1] / 2) + 1, dtype=self.tint, device=self.device
        )
        [self.mesh_fq, self.mesh_fp] = torch.meshgrid(f_q, f_p, indexing="xy")

        # Mark as initialized
        self._reciprocal_space_initialized = True

    def _solve_nonhomo_layer(self, layer_thickness, p_mat_i, q_mat_i, mat_w0, mat_v0, harmonics, k_0, **kwargs):
        """Delegate to the algorithm strategy."""
        if self._algorithm is None:
            raise ValueError("Solver algorithm not set")
        return self._algorithm.solve_nonhomo_layer(
            layer_thickness, p_mat_i, q_mat_i, mat_w0, mat_v0, harmonics, k_0, **kwargs
        )

    @property
    def lam0(self) -> np.ndarray:
        """Get the free space wavelength array.

        Returns:
            Array of free space wavelengths used in the simulation
        """
        return self._lam0

    @lam0.setter
    def lam0(self, value: Union[float, np.ndarray]) -> None:
        """Set the free space wavelength array.

        Args:
            value: New wavelength value(s) to set

        Raises:
            ValueError: If the input type is not float or numpy.ndarray
        """
        if isinstance(value, float):
            self._lam0 = np.array([value], dtype=self.nfloat)
        elif isinstance(value, np.ndarray):
            self._lam0 = value.astype(self.nfloat)
        else:
            raise TypeError(f"lam0 must be float or numpy.ndarray, got {type(value)}")

    @property
    def device_resolution(self):
        """Get the device resolution metadata.

        Returns the DeviceResolution object containing information about the
        requested device, resolved device, fallback status, and fallback reason.

        Returns:
            DeviceResolution: Device resolution metadata with fields:
                - requested_device: The device string that was requested
                - resolved_device: The actual torch.device that was resolved to
                - fell_back: Boolean indicating if fallback occurred
                - reason: String explaining fallback reason, or None if no fallback
        """
        return self._device_resolution

    def add_source(self, theta: float, phi: float, pte: float, ptm: float, norm_te_dir: str = "y") -> dict:
        """Configure the incident electromagnetic wave source.

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

        Args:
            theta (float): Incident angle (polar angle) in radians.
                   0 corresponds to normal incidence along the z-axis.
                   π/2 corresponds to grazing incidence in the xy-plane.

            phi (float): Azimuthal angle in radians, measured in the xy-plane.
                 0 corresponds to the x-axis.
                 π/2 corresponds to the y-axis.

            pte (float): TE polarization amplitude (can be complex).
                 The TE component has its electric field perpendicular to the plane of incidence.

            ptm (float): TM polarization amplitude (can be complex).
                 The TM component has its magnetic field perpendicular to the plane of incidence.

            norm_te_dir (str, optional): Direction of normal component for TE wave.
                        Options: 'x', 'y', or 'z'.
                        This defines the reference direction for the TE polarization.
                        Default is 'y', meaning the TE component is perpendicular to
                        the y-axis for normal incidence.

        Returns:
            dict: Source configuration dictionary that can be passed to the solve() method,
                containing:
                - 'theta': Polar angle in radians
                - 'phi': Azimuthal angle in radians
            - 'pte': TE polarization amplitude
            - 'ptm': TM polarization amplitude
            - 'norm_te_dir': Direction of normal component for TE wave

        Examples:
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

        Note:
            The convention used in TorchRDIT defines the TE polarization as having
            the electric field perpendicular to the plane of incidence, and the TM
            polarization as having the magnetic field perpendicular to the plane of incidence.

        Keywords:
            source, incident wave, polarization, TE, TM, plane wave, incidence angle,
            azimuthal angle, polar angle, spherical coordinates, circular polarization
        """
        source_dict = {}
        source_dict["theta"] = theta
        source_dict["phi"] = phi
        source_dict["pte"] = pte
        source_dict["ptm"] = ptm
        source_dict["norm_te_dir"] = norm_te_dir

        return source_dict

    def _initialize_k_vectors(self):
        """Initialize k-vectors from self.kinc.

        self.kinc is always (n_sources, n_freqs, 3) from _pre_solve().

        Returns:
            Tuple of 4D tensors (n_sources, n_freqs, harmonics[0], harmonics[1]).
        """
        # kinc is always (n_sources, n_freqs, 3) from _pre_solve
        kinc = self.kinc

        if self.debug_batching:
            print(f"[DEBUG] _initialize_k_vectors: kinc.shape={kinc.shape}")

        # Core tensorized k-vector calculation
        # Calculate wave vector expansion using broadcasting
        # kinc shape: (n_sources, n_freqs, 3)
        # Output shapes: (n_sources, n_freqs, harmonics[0], harmonics[1])
        kx_0 = (
            kinc[:, :, 0, None, None]  # (n_sources, n_freqs, 1, 1)
            - (
                self.mesh_fp[None, None, :, :] * self.reci_t1[0, None, None, None]
                + self.mesh_fq[None, None, :, :] * self.reci_t2[0, None, None, None]
            )
            / self.k_0[None, :, None, None]
        )
        kx_0 = kx_0.to(dtype=self.tcomplex)

        ky_0 = (
            kinc[:, :, 1, None, None]  # (n_sources, n_freqs, 1, 1)
            - (
                self.mesh_fp[None, None, :, :] * self.reci_t1[1, None, None, None]
                + self.mesh_fq[None, None, :, :] * self.reci_t2[1, None, None, None]
            )
            / self.k_0[None, :, None, None]
        )
        ky_0 = ky_0.to(dtype=self.tcomplex)

        # Apply numerical relaxation for stability
        epsilon = 1e-6
        kx_0, ky_0 = self._apply_numerical_relaxation(kx_0, ky_0, epsilon)

        # Calculate kz for reflection and transmission regions
        kz_ref_0 = self._calculate_kz_region(self.ur1, self.er1, kx_0, ky_0, self.layer_manager.is_ref_dispersive)
        kz_trn_0 = self._calculate_kz_region(self.ur2, self.er2, kx_0, ky_0, self.layer_manager.is_trn_dispersive)

        # Always return 4D: (n_sources, n_freqs, harmonics[0], harmonics[1])
        return kx_0, ky_0, kz_ref_0, kz_trn_0

    def _apply_numerical_relaxation(self, kx_0, ky_0, epsilon):
        """Unified numerical relaxation for both single and batched inputs.

        Replace near-zero k-vector components with a small positive epsilon
        to prevent downstream division-by-zero.  Uses ``torch.where`` so
        that the operation is fully differentiable (no in-place mutation).

        Handles both tensor shapes:
        - Single source: (n_freqs, harmonics[0], harmonics[1])
        - Batched sources: (n_sources, n_freqs, harmonics[0], harmonics[1])

        Args:
            kx_0, ky_0: k-vector tensors (any dimensionality)
            epsilon: Small offset value for zero replacement

        Returns:
            Modified kx_0, ky_0 with epsilon replacing near-zero values
        """
        kx_0 = torch.where(torch.abs(kx_0) < epsilon, epsilon, kx_0)
        ky_0 = torch.where(torch.abs(ky_0) < epsilon, epsilon, ky_0)
        return kx_0, ky_0

    def _calculate_kz_region(self, ur, er, kx_0, ky_0, is_dispersive):
        """Calculate kz for a region (reflection or transmission).

        Args:
            ur, er: Material properties (scalars or (n_freqs,) tensors)
            kx_0, ky_0: 4D k-vectors (n_sources, n_freqs, harmonics[0], harmonics[1])
            is_dispersive: Whether the material is dispersive

        Returns:
            kz_0 with same shape as input kx_0, ky_0
        """
        # kx_0, ky_0 are always 4D: (n_sources, n_freqs, H0, H1)
        if not is_dispersive:
            kz_0 = torch.conj(torch.sqrt(torch.conj(ur) * torch.conj(er) - kx_0 * kx_0 - ky_0 * ky_0 + 0 * 1j))
        else:
            # Dispersive: ur, er have shape (n_freqs,) → broadcast to (1, n_freqs, 1, 1)
            ur_er_product = (torch.conj(ur) * torch.conj(er))[None, :, None, None]
            kz_0 = torch.conj(torch.sqrt(ur_er_product - kx_0 * kx_0 - ky_0 * ky_0 + 0 * 1j))
        return kz_0

    def _setup_common_matrices(self, kx_0, ky_0, kz_ref_0, kz_trn_0):
        """Set up common matrices for both single and batched sources.

        This unified function automatically handles both single source (3D tensors)
        and batched sources (4D tensors) by detecting the input dimensions.

        Args:
            kx_0, ky_0: k-vectors with shape (n_sources, n_freqs, harmonics[0], harmonics[1]).
            kz_ref_0, kz_trn_0: kz vectors with same shape as kx_0, ky_0.

        Returns:
            Dictionary of 4D matrices (n_sources, n_freqs, ...).
        """
        # Inputs are always 4D: (n_sources, n_freqs, harmonics[0], harmonics[1])
        n_sources = kx_0.shape[0]
        n_harmonics_squared = self.harmonics[0] * self.harmonics[1]

        # Create identity matrices
        ident_mat_k = torch.eye(n_harmonics_squared, dtype=self.tcomplex, device=self.device)
        ident_mat_k2 = torch.eye(2 * n_harmonics_squared, dtype=self.tcomplex, device=self.device)

        self.ident_mat_k = ident_mat_k
        self.ident_mat_k2 = ident_mat_k2

        # Transform to diagonal matrices - flatten last two dimensions
        # Shape: (n_sources, n_freqs, harmonics[0], harmonics[1]) -> (n_sources, n_freqs, n_harmonics_squared)
        mat_kx = kx_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_ky = ky_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_kz_ref = kz_ref_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_kz_trn = kz_trn_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)

        # Calculate derived matrices
        mat_kx_ky = mat_kx * mat_ky
        mat_ky_kx = mat_ky * mat_kx
        mat_kx_kx = mat_kx * mat_kx
        mat_ky_ky = mat_ky * mat_ky

        # VECTORIZED: Direct call to to_diag_util with batched tensors
        # to_diag_util supports batched inputs through broadcasting
        mat_kx_diag = to_diag_util(mat_kx, self.harmonics)
        mat_ky_diag = to_diag_util(mat_ky, self.harmonics)

        mat_kz = torch.conj(torch.sqrt(1.0 - mat_kx_kx - mat_ky_ky))

        ident_mat_kx_kx = 1.0 - mat_kx_kx
        ident_mat_ky_ky = 1.0 - mat_ky_ky

        # Set up identity matrices with batching
        # Shape: (n_sources, n_freqs, n_harmonics_squared, n_harmonics_squared)
        ident_mat = ident_mat_k[None, None, :, :].expand(n_sources, self.n_freqs, -1, -1)
        zero_mat = torch.zeros(
            size=(n_sources, self.n_freqs, n_harmonics_squared, n_harmonics_squared),
            dtype=self.tcomplex,
            device=self.device,
        )

        # Create block matrices for each source
        mat_w0 = blockmat2x2([[ident_mat, zero_mat], [zero_mat, ident_mat]])

        # Differentiable kz protection (see numerics.py)
        inv_mat_lam = safe_kz_reciprocal(mat_kz, dtype=self.tcomplex)

        # VECTORIZED: Create block matrix using batched operations
        # blockmat2x2 works with batched tensors through torch.cat broadcasting
        mat_v0 = blockmat2x2(
            [
                [
                    to_diag_util(mat_kx_ky * inv_mat_lam, self.harmonics),
                    to_diag_util(ident_mat_kx_kx * inv_mat_lam, self.harmonics),
                ],
                [
                    to_diag_util(-ident_mat_ky_ky * inv_mat_lam, self.harmonics),
                    to_diag_util(-mat_kx_ky * inv_mat_lam, self.harmonics),
                ],
            ]
        )

        # Prepare result dictionary
        result = {
            "mat_kx": mat_kx,
            "mat_ky": mat_ky,
            "mat_kz_ref": mat_kz_ref,
            "mat_kz_trn": mat_kz_trn,
            "mat_kx_ky": mat_kx_ky,
            "mat_ky_kx": mat_ky_kx,
            "mat_kx_kx": mat_kx_kx,
            "mat_ky_ky": mat_ky_ky,
            "mat_kx_diag": mat_kx_diag,
            "mat_ky_diag": mat_ky_diag,
            "mat_kz": mat_kz,
            "mat_w0": mat_w0,
            "mat_v0": mat_v0,
            "ident_mat": ident_mat,
            "zero_mat": zero_mat,
        }

        # Always return 4D: (n_sources, n_freqs, ...)
        return result

    def _print_tensor_shape(self, name: str, tensor: torch.Tensor) -> None:
        """Print tensor shape if debug_batching is enabled.

        Args:
            name: Name of the tensor
            tensor: The tensor to print shape for
        """
        if self.debug_batching:
            print(f"[DEBUG] {name}: {tensor.shape}")

    def _compare_with_sequential(
        self, batched_result: torch.Tensor, sequential_results: List[torch.Tensor], name: str, tolerance: float = 1e-6
    ) -> None:
        """Compare batched results with sequential results for debugging.

        Args:
            batched_result: Result from batched computation
            sequential_results: List of results from sequential computation
            name: Name of the result being compared
            tolerance: Tolerance for comparison
        """
        if self.debug_batching:
            for i, seq_result in enumerate(sequential_results):
                if batched_result.dim() > seq_result.dim():
                    # Extract the i-th source from batched result
                    batch_i = batched_result[i]
                else:
                    batch_i = batched_result

                diff = torch.abs(batch_i - seq_result).max().item()
                print(f"[DEBUG] {name} source {i} max diff: {diff:.2e}")

                if diff > tolerance:
                    print(f"[WARNING] {name} source {i} differs by more than {tolerance}")

    def _solve_structure(self, sources=None, **kwargs):
        """Solve the electromagnetic problem for single or batched sources.

        This unified method handles both single source (2D kinc) and batched sources
        (3D kinc) cases through dimension expansion, eliminating code duplication.

        Args:
            sources: Optional list of source dictionaries for batched solving.
                    If None, uses single source mode from self.src
            **kwargs: Additional parameters that may be required by specific solvers

        Returns:
            SolverResults: Results for single or batched sources (unified interface)
        """
        # sources is always a list (from solve()); kinc is always 3D (B, F, 3)
        n_sources = len(sources) if sources is not None else self.kinc.shape[0]

        if getattr(self, "debug_unification", False):
            print(f"[DEBUG] _solve_structure: n_sources={n_sources}, kinc.shape={self.kinc.shape}")

        self.notify_observers(
            "calculation_starting",
            {"mode": "solving_structure", "n_sources": n_sources, "n_freqs": self.n_freqs, "n_layers": self.layer_manager.nlayer},
        )

        # Initialize k-vectors
        self.notify_observers("initializing_k_vectors")
        kx_0, ky_0, kz_ref_0, kz_trn_0 = self._initialize_k_vectors()

        # Set up matrices for calculation
        self.notify_observers("setting_up_matrices")
        matrices = self._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)

        n_harmonics_squared = self.harmonics[0] * self.harmonics[1]

        # Initialize global scattering matrix — always 4D
        shape = (n_sources, self.n_freqs, 2 * n_harmonics_squared, 2 * n_harmonics_squared)
        smat_global = init_smatrix(shape=shape, dtype=self.tcomplex, device=self.device)

        # Process each layer and update global scattering matrix
        self.notify_observers("processing_layers", {"total": self.layer_manager.nlayer})
        for n_layer in range(self.layer_manager.nlayer):
            self.notify_observers(
                "layer_started",
                {
                    "layer_index": n_layer,
                    "current": n_layer + 1,
                    "total": self.layer_manager.nlayer,
                    "progress": (n_layer / self.layer_manager.nlayer) * 100,
                },
            )

            smat_layer = self._process_layer(n_layer, matrices)
            smat_global = redhstar(smat_global, smat_layer)
            self.notify_observers("layer_completed", {"layer_index": n_layer})

        # Connect to external regions — vectorized over all sources (4D)
        self.notify_observers("connecting_external_regions")
        smat_global, mat_v_ref, mat_v_trn = self._connect_external_regions(smat_global, matrices)

        # Calculate fields — fully vectorized over all sources (no Python loop)
        self.notify_observers("calculating_fields")
        fields = self._calculate_fields_and_efficiencies(
            smat_global, matrices, kx_0, ky_0, mat_v_ref, mat_v_trn,
            kinc=self.kinc, sources=sources,
        )

        # Reshape field Fourier coeffs: (B, F, H², 1) → (B, F, H0, H1)
        H0, H1 = self.harmonics
        field_shape = (n_sources, self.n_freqs, H0, H1)

        def _fmt(key):
            v = fields[key]
            if v is None:
                return None
            return torch.reshape(v, shape=field_shape).transpose(dim0=-2, dim1=-1)

        # Efficiencies are already (B, F, ...) from the vectorized method
        reflection = fields["total_ref_efficiency"]   # (B, F)
        transmission = fields["total_trn_efficiency"]  # (B, F)

        # Pre-compute reshaped field tensors for raw_data (B, F, H0, H1)
        fmt_fields = {}
        for key in ("ref_s_x", "ref_s_y", "ref_s_z", "trn_s_x", "trn_s_y", "trn_s_z",
                     "ref_u_x", "ref_u_y", "ref_u_z", "trn_u_x", "trn_u_y", "trn_u_z"):
            v = fields[key]
            if v is not None:
                fmt_fields[key] = torch.reshape(v, shape=field_shape).transpose(dim0=-2, dim1=-1)
            else:
                fmt_fields[key] = None

        # Detach smat_global once.  The full source-major tensor is stored in an
        # internal cache for correct per-source indexing, while the public root
        # structure_matrix keeps the historical (F, M, M) contract.
        smat_det = ScatteringMatrix(
            S11=smat_global.S11.detach(),
            S12=smat_global.S12.detach(),
            S21=smat_global.S21.detach(),
            S22=smat_global.S22.detach(),
        )

        # Per-source raw_data — full legacy schema for backward compat
        per_source_raw = []
        for i in range(n_sources):
            # Views into the shared detached tensor — no extra memory
            smat_i = SMatrix(
                S11=smat_det.S11[i], S12=smat_det.S12[i],
                S21=smat_det.S21[i], S22=smat_det.S22[i],
            )

            def _get(key):
                v = fmt_fields[key]
                return v[i] if v is not None else None

            per_source_raw.append({
                "smat_structure": smat_i,
                # Field Fourier coefficients
                "ref_s_x": _get("ref_s_x"), "ref_s_y": _get("ref_s_y"), "ref_s_z": _get("ref_s_z"),
                "trn_s_x": _get("trn_s_x"), "trn_s_y": _get("trn_s_y"), "trn_s_z": _get("trn_s_z"),
                "ref_u_x": _get("ref_u_x"), "ref_u_y": _get("ref_u_y"), "ref_u_z": _get("ref_u_z"),
                "trn_u_x": _get("trn_u_x"), "trn_u_y": _get("trn_u_y"), "trn_u_z": _get("trn_u_z"),
                # Backward compat aliases
                "rx": _get("ref_s_x"), "ry": _get("ref_s_y"), "rz": _get("ref_s_z"),
                "tx": _get("trn_s_x"), "ty": _get("trn_s_y"), "tz": _get("trn_s_z"),
                # Efficiencies
                "REF": reflection[i], "TRN": transmission[i],
                "RDE": fields["ref_diff_efficiency"][i], "TDE": fields["trn_diff_efficiency"][i],
                # K-vectors
                "kzref": matrices["mat_kz_ref"][i], "kztrn": matrices["mat_kz_trn"][i],
                "kinc": self.kinc[i], "kx": torch.squeeze(kx_0[i]), "ky": torch.squeeze(ky_0[i]),
                "lattice_t1": self.lattice_t1, "lattice_t2": self.lattice_t2,
                "default_grids": self.grids,
            })

        self.notify_observers("calculation_completed", {"n_sources": n_sources, "n_freqs": self.n_freqs})

        return SolverResults(
            reflection=reflection,
            transmission=transmission,
            reflection_diffraction=fields["ref_diff_efficiency"],
            transmission_diffraction=fields["trn_diff_efficiency"],
            reflection_field=FieldComponents(
                x=_fmt("ref_s_x"), y=_fmt("ref_s_y"), z=_fmt("ref_s_z"),
                mag_x=_fmt("ref_u_x"), mag_y=_fmt("ref_u_y"), mag_z=_fmt("ref_u_z"),
            ),
            transmission_field=FieldComponents(
                x=_fmt("trn_s_x"), y=_fmt("trn_s_y"), z=_fmt("trn_s_z"),
                mag_x=_fmt("trn_u_x"), mag_y=_fmt("trn_u_y"), mag_z=_fmt("trn_u_z"),
            ),
            structure_matrix=ScatteringMatrix(
                S11=smat_det.S11[0], S12=smat_det.S12[0],
                S21=smat_det.S21[0], S22=smat_det.S22[0],
            ),
            _structure_matrix_batched=smat_det,
            wave_vectors=WaveVectors(
                kx=kx_0, ky=ky_0,
                kinc=self.kinc, kzref=matrices["mat_kz_ref"], kztrn=matrices["mat_kz_trn"],
            ),
            raw_data={"_per_source": per_source_raw},
            n_sources=n_sources,
            source_parameters=sources,
            loss=1.0 - reflection - transmission,
            lattice_t1=self.lattice_t1,
            lattice_t2=self.lattice_t2,
            default_grids=self.grids,
        )

    def solve(self, source: Union[dict, List[dict]], **kwargs) -> SolverResults:
        """Solve the electromagnetic problem for the configured structure.

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

        Args:
            source (Union[dict, List[dict]]): Source configuration for the simulation.
                   Can be either:
                   - dict: Single source configuration dictionary containing the incident wave
                     parameters. This should be created using the add_source() method,
                     with the following keys:
                     - 'theta': Polar angle of incidence (in radians)
                     - 'phi': Azimuthal angle of incidence (in radians)
                     - 'pte': Complex amplitude of the TE polarization component
                     - 'ptm': Complex amplitude of the TM polarization component
                     - 'norm_te_dir': Direction of normal component for TE wave (default: 'y')
                   - List[dict]: List of source configuration dictionaries for batched processing.
                     All sources must have the same structure of keys. Batched processing
                     improves performance when simulating multiple incident conditions

            **kwargs: Additional keyword arguments to customize the solution process:
                       Default is False. If True, electric and magnetic field components
                       will be computed and returned.
                     - 'compute_modes' (bool): Whether to compute and return mode information
                       Default is False.
                     - 'store_matrices' (bool): Whether to store system matrices
                       Default is False. Setting to True increases memory usage but
                       can be useful for debugging.
                     - 'return_all' (bool): Whether to return all intermediate results
                       Default is False. Setting to True increases memory usage.

        Returns:
            SolverResults: Unified results container (v0.1.27) supporting both single and batched sources.

                For single source (dict input):
                - reflection: Shape (n_freqs) - Total reflection efficiency
                - transmission: Shape (n_freqs) - Total transmission efficiency
                - reflection_diffraction: Shape (n_freqs, harmonics[0], harmonics[1])
                - transmission_diffraction: Shape (n_freqs, harmonics[0], harmonics[1])
                - reflection_field: FieldComponents with E-field Fourier coefficients (S_x, S_y, S_z)
                                   and H-field Fourier coefficients (U_x, U_y, U_z)
                - transmission_field: FieldComponents with E and H Fourier coefficients
                - structure_matrix: Scattering matrix for the entire structure
                - wave_vectors: Wave vector components (kx, ky, kinc, kzref, kztrn)
                - lattice_t1, lattice_t2: Lattice vectors for field reconstruction
                - Field APIs: get_reflection_interface_fields(), get_transmission_interface_fields()

                For batched sources (list input):
                - reflection: Shape (n_sources, n_freqs)
                - transmission: Shape (n_sources, n_freqs)
                - reflection_diffraction: Shape (n_sources, n_freqs, harmonics[0], harmonics[1])
                - transmission_diffraction: Shape (n_sources, n_freqs, harmonics[0], harmonics[1])
                - Indexing: results[i] returns SolverResults for source i
                - Iteration: for result in results iterates over sources
                - Batched methods: find_optimal_source(), get_parameter_sweep_data()
                - All field APIs work with batched results

                The results also provide helper methods for extracting specific diffraction orders
                and analyzing propagating modes.

        Examples:
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

        Batched source processing (v0.1.27 unified interface):
        ```python
        # Create multiple sources for angle sweep
        deg = np.pi / 180
        sources = [
            solver.add_source(theta=0*deg, phi=0, pte=1.0, ptm=0.0),
            solver.add_source(theta=30*deg, phi=0, pte=1.0, ptm=0.0),
            solver.add_source(theta=45*deg, phi=0, pte=1.0, ptm=0.0),
            solver.add_source(theta=60*deg, phi=0, pte=1.0, ptm=0.0)
        ]

        # Solve with batched sources - returns unified SolverResults
        results = solver.solve(sources)  # Single SolverResults class handles batching

        # Access results for all sources at once
        print(f"Transmission values: {results.transmission[:, 0]}")

        # Access individual results via indexing
        result_30deg = results[1]  # SolverResults for 30° incidence

        # Find optimal angle (batched-only method)
        best_idx = results.find_optimal_source('max_transmission')
        print(f"Best angle: {sources[best_idx]['theta'] * 180/np.pi:.1f}°")

        # Iterate over sources
        for i, result in enumerate(results):
            print(f"Angle {sources[i]['theta']*180/np.pi:.0f}°: T={result.transmission[0]:.3f}")
        ```

        Optimization with multiple sources:
        ```python
        # Optimize for multiple incident angles simultaneously
        mask = solver.get_circle_mask(center=(0, 0), radius=0.25)
        mask = mask.to(torch.float32)
        mask.requires_grad = True

        solver.update_er_with_mask(mask=mask, layer_index=0)

        # Define multiple sources
        sources = [
            solver.add_source(theta=0*deg, phi=0, pte=1.0, ptm=0.0),
            solver.add_source(theta=15*deg, phi=0, pte=1.0, ptm=0.0),
            solver.add_source(theta=30*deg, phi=0, pte=1.0, ptm=0.0)
        ]

        # Optimize for average transmission across all angles
        optimizer = torch.optim.Adam([mask], lr=0.01)
        for i in range(100):
            optimizer.zero_grad()
            results = solver.solve(sources)  # SolverResults with batching
            loss = -results.transmission.mean()  # Maximize average transmission
            loss.backward()
            optimizer.step()
        ```

        Note:
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
        """
        # Normalize input: single dict → [dict], then always process as list.
        is_single = isinstance(source, dict)
        if is_single:
            sources = [source]
        elif isinstance(source, list):
            if not source:
                raise ValueError("At least one source required")
            for src in source:
                if not isinstance(src, dict) or "theta" not in src:
                    raise ValueError(
                        "Invalid source format: each source must be a dict with 'theta', 'phi', 'pte', 'ptm'"
                    )
            sources = source
        else:
            raise TypeError(f"source must be a dict or list of dicts, got {type(source).__name__}")

        self.src = sources[0]  # keep for backward compat
        self._pre_solve(sources)
        result = self._solve_structure(sources, **kwargs)

        # Squeeze source dimension at the API boundary for dict-input callers
        if is_single:
            return result[0]
        return result

    def _pre_solve(self, sources=None) -> None:
        """Set up incident wave vectors for a list of sources.

        Always produces ``self.kinc`` with shape ``(n_sources, n_freqs, 3)``,
        even for a single source (``n_sources = 1``).

        Args:
            sources: List of source dicts, a single dict, or None (uses self.src).
        """
        # Normalize input to list
        if sources is None:
            sources = [self.src]
        elif isinstance(sources, dict):
            sources = [sources]
        # Set up reciprocal space (idempotent)
        self._setup_reciprocal_space()

        n_sources = len(sources)
        self.n_sources = n_sources

        # Debug output
        if self.debug_batching:
            print(f"[DEBUG] _pre_solve: n_sources={n_sources}")

        # Calculate refractive index of external medium
        refractive_1 = torch.sqrt(self.ur1 * self.er1)
        if refractive_1.dim() == 0:  # non-dispersive
            refractive_1 = refractive_1.unsqueeze(0).expand(self.n_freqs)

        # Tensorize theta/phi — always produce shape (n_sources, n_freqs)
        def _to_tensor_preserve(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=self.tfloat)
            return torch.tensor(x, device=self.device, dtype=self.tfloat)

        theta_list = [_to_tensor_preserve(s["theta"]) for s in sources]
        phi_list = [_to_tensor_preserve(s["phi"]) for s in sources]

        # Ensure each is at least 1-D before stacking
        theta_list = [t.unsqueeze(0).expand(self.n_freqs) if t.dim() == 0 else t for t in theta_list]
        phi_list = [p.unsqueeze(0).expand(self.n_freqs) if p.dim() == 0 else p for p in phi_list]

        theta = torch.stack(theta_list)  # (n_sources, n_freqs)
        phi = torch.stack(phi_list)      # (n_sources, n_freqs)

        # kinc components — shape (n_sources, n_freqs) via broadcasting with refractive_1 (n_freqs,)
        kinc_x = refractive_1 * torch.sin(theta) * torch.cos(phi)
        kinc_y = refractive_1 * torch.sin(theta) * torch.sin(phi)
        kinc_z = refractive_1 * torch.cos(theta)

        # Always (n_sources, n_freqs, 3)
        self.kinc = torch.stack([kinc_x, kinc_y, kinc_z], dim=2)

        # Debug output
        if self.debug_batching:
            print(f"[DEBUG] kinc shape: {self.kinc.shape}")

        assert self.kinc.shape == (n_sources, self.n_freqs, 3), (
            f"Expected kinc shape ({n_sources}, {self.n_freqs}, 3), got {self.kinc.shape}"
        )

        # Check if options are set correctly (only needed once, not per source)
        for n_layer in range(self.layer_manager.nlayer):
            if self.layer_manager.layers[n_layer].is_homogeneous is False:
                if self.layer_manager.layers[n_layer].ermat is None:
                    # if not homogenous material, must be with a pattern.
                    # if no pattern assigned before solving, the material will be set as homogeneous
                    self.layer_manager.replace_layer_to_homogeneous(layer_index=n_layer)
                    print(f"Warning: Layer {n_layer} has no pattern assigned, and was changed to homogeneous")

class RCWASolver(FourierBaseSolver):
    """Rigorous Coupled-Wave Analysis (RCWA) Solver.

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

    Example:
        ```python
        # Create an RCWA solver
        solver = RCWASolver(
            lam0=np.array([1.55]),  # Wavelength in micrometers
            grids=[512, 512],        # Real space dimensions
            harmonics=[5, 5],            # Fourier space dimensions
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
    """

    def __init__(
        self,
        lam0: Optional[Union[float, np.ndarray]] = None,
        lengthunit: str = "um",
        grids: list = [512, 512],  # dimensions in real space: (H, W)
        harmonics: list = [3, 3],  # dimensions in k space: (kH, kW)
        materiallist: list = [],  # list of materials
        t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
        t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
        is_use_FFF: bool = False,
        fff_vector_scheme: str = "POL",
        fff_fourier_weight: float = 1e-2,
        fff_smoothness_weight: float = 1e-3,
        fff_vector_steps: int = 1,
        precision: Precision = Precision.SINGLE,
        device: Union[str, torch.device] = "cpu",
        debug_batching: bool = False,
        debug_tensorization: bool = False,
        debug_unification: bool = False,
    ) -> None:
        super().__init__(
            lam0=lam0,
            lengthunit=lengthunit,
            grids=grids,
            harmonics=harmonics,
            materiallist=materiallist,
            t1=t1,
            t2=t2,
            is_use_FFF=is_use_FFF,
            fff_vector_scheme=fff_vector_scheme,
            fff_fourier_weight=fff_fourier_weight,
            fff_smoothness_weight=fff_smoothness_weight,
            fff_vector_steps=fff_vector_steps,
            precision=precision,
            device=device,
            debug_batching=debug_batching,
            debug_tensorization=debug_tensorization,
            debug_unification=debug_unification,
        )

        # Set the algorithm strategy
        self._algorithm = RCWAAlgorithm(self)

    def set_rdit_order(self, rdit_order):
        """Set the order of the R-DIT algorithm.

        This method allows you to configure the order of the Rigorous Diffraction
        Interface Theory algorithm. The R-DIT order affects the numerical stability
        and accuracy of the solution, especially for thick or high-contrast layers.

        Args:
            rdit_order (int): The order of the R-DIT algorithm. Higher orders generally
                       provide better accuracy but may be computationally more expensive.

        Examples:
        ```python
        solver = RCWASolver(
            lam0=np.array([1.55]),
            device='cuda'
        )

        # Set RDIT order to 2 for a balance of accuracy and performance
        solver.set_rdit_order(2)
        ```
        """
        self._algorithm.set_rdit_order(rdit_order)


class RDITSolver(FourierBaseSolver):
    """Eigendecomposition-free Rigorous Diffraction Interface Theory (R-DIT) Solver.

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

    Args:
        lam0: Wavelength(s) to simulate, in the specified length unit. Default is [1.0].
        lengthunit: The unit of length used in the simulation (e.g., 'um', 'nm').
                   Default is 'um' (micrometers).
        grids: Dimensions of the real space grid [height, width]. Default is [512, 512].
        harmonics: Dimensions in Fourier space [kx, ky]. Default is [3, 3].
              This determines the number of Fourier harmonics used.
        materiallist: List of materials used in the simulation. Default is empty list.
        t1: First lattice vector. Default is [[1.0, 0.0]] (unit vector in x-direction).
        t2: Second lattice vector. Default is [[0.0, 1.0]] (unit vector in y-direction).
        is_use_FFF: Whether to use Fast Fourier Factorization. Default is True.
                   This improves convergence for high-contrast material interfaces.
        precision: Numerical precision to use (SINGLE or DOUBLE). Default is SINGLE.
        device: The device to run the solver on ('cpu', 'cuda', etc.). Default is 'cpu'.
               For optimal performance, especially with the R-DIT solver, using 'cuda'
               is highly recommended for GPU acceleration.

    Returns:
        A solver instance (either RCWASolver or RDITSolver) configured according
        to the provided parameters. The returned solver is fully differentiable,
        enabling gradient-based optimization for inverse design tasks.

    Example:
    ```python
    # Create an R-DIT solver
    solver = RDITSolver(
        lam0=np.array([1.55]),  # Wavelength in micrometers
        grids=[512, 512],        # Real space dimensions
        harmonics=[5, 5],            # Fourier space dimensions
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

    References:
        - Huang et al., "Eigendecomposition-free inverse design of meta-optics devices,"
          Opt. Express 32, 13986-13997 (2024)
        - Huang et al., "Inverse Design of Photonic Structures Using Automatic Differentiable
          Rigorous Diffraction Interface Theory," CLEO (2023)
    """

    def __init__(
        self,
        lam0: Optional[Union[float, np.ndarray]] = None,
        lengthunit: str = "um",
        grids: list = [512, 512],  # dimensions in real space: (H, W)
        harmonics: list = [3, 3],  # dimensions in k space: (kH, kW)
        materiallist: list = [],  # list of materials
        t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
        t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
        is_use_FFF: bool = False,
        fff_vector_scheme: str = "POL",
        fff_fourier_weight: float = 1e-2,
        fff_smoothness_weight: float = 1e-3,
        fff_vector_steps: int = 1,
        precision: Precision = Precision.SINGLE,
        device: Union[str, torch.device] = "cpu",
        debug_batching: bool = False,
        debug_tensorization: bool = False,
        debug_unification: bool = False,
    ) -> None:
        super().__init__(
            lam0=lam0,
            lengthunit=lengthunit,
            grids=grids,
            harmonics=harmonics,
            materiallist=materiallist,
            t1=t1,
            t2=t2,
            is_use_FFF=is_use_FFF,
            fff_vector_scheme=fff_vector_scheme,
            fff_fourier_weight=fff_fourier_weight,
            fff_smoothness_weight=fff_smoothness_weight,
            fff_vector_steps=fff_vector_steps,
            precision=precision,
            device=device,
            debug_batching=debug_batching,
            debug_tensorization=debug_tensorization,
            debug_unification=debug_unification,
        )

        # Set the algorithm strategy
        self._algorithm = RDITAlgorithm(self)

    def set_rdit_order(self, rdit_order: int) -> None:
        """Set the order of the R-DIT algorithm.

        This method allows you to configure the order of the Rigorous Diffraction
        Interface Theory algorithm. The R-DIT order affects the numerical stability
        and accuracy of the solution, especially for thick or high-contrast layers.

        The R-DIT order determines the approximation used in the diffraction
        interface theory. Higher orders generally provide better accuracy but
        may be computationally more expensive.

        Args:
            rdit_order: The order of the R-DIT algorithm. Higher orders generally
                       provide better accuracy but may be computationally more
                       expensive. Typical values are 1, 2, or 3.
                       - 1: First-order approximation (fastest, least accurate)
                       - 2: Second-order approximation (good balance)
                       - 3: Third-order approximation (most accurate, slowest)

        Returns:
            None

        Example:
        ```python
        solver = RDITSolver(...)
        solver.set_rdit_order(2)  # Set R-DIT order to 2 for good balance
        ```

        Note:
            The optimal R-DIT order depends on the specific problem. For most
            applications, order 2 provides a good balance between accuracy and
            computational efficiency. For highly demanding applications with
            thick layers or high material contrast, order 3 may be necessary.
        """
        self._algorithm.set_rdit_order(rdit_order)


def get_solver_builder():
    """Get a solver builder for creating a solver with a fluent interface.

    This function returns a new instance of the SolverBuilder class, which provides
    a fluent interface for configuring and creating solver instances. The builder
    pattern allows for a more readable and flexible way to specify solver parameters
    compared to passing all parameters to the constructor at once.

    Returns:
        SolverBuilder: A new solver builder instance.

    Examples:
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

    See Also:
        create_solver_from_builder: For an alternative approach using a configuration function.
    """
    from .builder import SolverBuilder

    return SolverBuilder()


def create_solver_from_builder(
    builder_config: Callable[["SolverBuilder"], "SolverBuilder"],
) -> Union["RCWASolver", "RDITSolver"]:
    """Create a solver from a builder configuration function.

    This function provides an alternative way to create and configure a solver
    using a configuration function that takes a builder and returns a configured
    builder. This approach can be particularly useful when you want to encapsulate
    solver configuration logic in reusable functions.

    Args:
        builder_config (Callable): A function that takes a SolverBuilder instance,
                      configures it, and returns the configured builder.

    Returns:
        Union[RCWASolver, RDITSolver]: A solver instance configured according
        to the builder configuration.

    Examples:
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

    See Also:
        get_solver_builder: For direct access to a builder instance.
    """
    from torchrdit.builder import SolverBuilder

    builder = SolverBuilder()
    builder = builder_config(builder)
    return builder.build()
