"""
TorchRDIT solver module for electromagnetic simulations and inverse design.

This module provides core solver implementations for electromagnetic wave propagation 
through complex layered structures using both Rigorous Coupled-Wave Analysis (RCWA) and 
eigendecomposition-free Rigorous Diffraction Interface Theory (R-DIT) methods.

TorchRDIT is an advanced software package designed for the inverse design of meta-optics 
devices. It provides a GPU-accelerated and fully differentiable framework powered by 
PyTorch, enabling efficient optimization of photonic structures. The R-DIT implementation 
achieves up to 16.2× speedup compared to traditional RCWA-based inverse design methods.

The solvers handle:
- Multiple wavelength simulations
- Arbitrary incident angles
- Heterogeneous material layers
- Computation of reflection/transmission coefficients
- Field calculations
- Efficiency calculations
- Automatic differentiation for gradient-based optimization

Key applications include:
- Parameter-constrained and free-form meta-atoms
- Reconfigurable photonic structures using optical phase-change materials
- High-performance meta-lenses and beam deflectors

Key classes:
- FourierBaseSolver: Base class implementing common Fourier-based calculations
- RCWASolver: Implementation of the RCWA algorithm
- RDITSolver: Eigendecomposition-free implementation of the R-DIT algorithm
- SolverObserver: Interface for tracking solver progress
- SolverSubjectMixin: Enables solvers to notify observers

Helper functions:
- create_solver: Factory function to create solver instances
- create_solver_from_config: Creates solver from configuration dictionary or file
- create_solver_from_builder: Creates solver using the builder pattern

References:
- Huang et al., "Eigendecomposition-free inverse design of meta-optics devices," 
  Opt. Express 32, 13986-13997 (2024)
- Huang et al., "Inverse Design of Photonic Structures Using Automatic Differentiable 
  Rigorous Diffraction Interface Theory," CLEO (2023)
"""
from typing import Union, Optional, Dict, List, Any, Callable
import numpy as np
import torch

from torch.linalg import inv as tinv
from torch.linalg import solve as tsolve
from torch.nn.functional import conv2d as tconv2d
from .cell import Cell3D, CellType
from .utils import blockmat2x2, redhstar, init_smatrix, blur_filter, to_diag_util
from .constants import Algorithm, Precision
from .materials import MaterialClass
from .algorithm import RCWAAlgorithm, RDITAlgorithm, SolverAlgorithm
import time

class SolverObserver:
    """Interface for observers that track solver progress.
    
    This class defines the Observer interface in the Observer design pattern
    for monitoring the progress and state of solvers. Concrete observer
    implementations should inherit from this class and implement the update
    method to receive and process notifications from solvers.
    
    Typical event types include:
    - 'iteration_complete': Triggered after each iteration in iterative methods
    - 'layer_processed': Triggered when a layer has been processed
    - 'solve_started': Triggered when the solve process begins
    - 'solve_completed': Triggered when the solve process completes
    
    Example:
        ```python
        class ProgressObserver(SolverObserver):
            def update(self, event_type: str, data: dict) -> None:
                if event_type == 'iteration_complete':
                    print(f"Completed iteration {data['iteration']} with error {data['error']}")
        
        solver = RCWASolver(...)
        solver.add_observer(ProgressObserver())
        ```
    """
    
    def update(self, event_type: str, data: dict) -> None:
        """Called when the solver notifies of an event.
        
        This method is called by the solver when an event occurs. Concrete
        observer implementations should override this method to process
        notifications from the solver.
        
        Args:
            event_type: The type of event that occurred (e.g., 'iteration_complete',
                        'layer_processed', 'solve_started', 'solve_completed')
            data: Additional data related to the event. The exact contents depend
                  on the event_type, but typically include relevant information
                  about the solver's state at the time of the event.
        """
        pass
        
class SolverSubjectMixin:
    """Mixin class that allows a solver to notify observers of progress.
    
    This class implements the Subject role in the Observer design pattern.
    It maintains a list of observers and provides methods to add, remove,
    and notify observers of events. Classes that inherit from this mixin
    can easily support observer notifications without reimplementing the
    observer management logic.
    
    The mixin handles:
    - Maintaining a list of registered observers
    - Adding and removing observers
    - Notifying all observers when events occur
    
    Example:
        ```python
        class MySolver(BaseSolver, SolverSubjectMixin):
            def __init__(self):
                BaseSolver.__init__(self)
                SolverSubjectMixin.__init__(self)
                
            def solve(self):
                self.notify_observers('solve_started', {'timestamp': time.time()})
                # Solve logic here
                self.notify_observers('solve_completed', {'result': result})
        ```
    """
    
    def __init__(self):
        """Initialize the observer list.
        
        Creates an empty list to store observer objects. This method should be
        called in the constructor of any class that inherits from this mixin.
        """
        self._observers = []
        
    def add_observer(self, observer: SolverObserver) -> None:
        """Add an observer to the notification list.
        
        Registers an observer to receive notifications when events occur.
        If the observer is already registered, it will not be added again.
        
        Args:
            observer: The observer object to add. Must implement the SolverObserver interface.
        """
        if observer not in self._observers:
            self._observers.append(observer)
            
    def remove_observer(self, observer: SolverObserver) -> None:
        """Remove an observer from the notification list.
        
        Unregisters an observer so it no longer receives notifications.
        If the observer is not in the list, no action is taken.
        
        Args:
            observer: The observer object to remove.
        """
        if observer in self._observers:
            self._observers.remove(observer)
            
    def notify_observers(self, event_type: str, data: dict = None) -> None:
        """Notify all observers of an event.
        
        This method calls the update method on all registered observers,
        passing the event type and data to each observer. This is the main
        mechanism for broadcasting events to all observers.
        
        Common event types include:
        - 'solve_started': Triggered when solving begins
        - 'solve_completed': Triggered when solving completes
        - 'iteration_complete': Triggered after each iteration
        - 'layer_processed': Triggered when a layer has been processed
        
        Args:
            event_type: The type of event that occurred. This is a string that
                        identifies the type of event, allowing observers to handle
                        different events differently.
            data: Additional data related to the event (default: None). This is a
                  dictionary that contains event-specific information that might be
                  useful to observers, such as iteration counts, error values, or
                  timing information.
        
        Example:
            ```python
            # In a solver method
            def solve(self):
                self.notify_observers('solve_started', {'time': time.time()})
                # Solving logic...
                self.notify_observers('solve_completed', {'result': result})
            ```
        """
        if data is None:
            data = {}
            
        for observer in self._observers:
            observer.update(event_type, data)

def create_solver_from_config(config: Union[str, Dict[str, Any]], flip: bool = False) -> Union["RCWASolver", "RDITSolver"]:
    """Create a solver from a configuration file or dictionary.
    
    This function creates a solver instance based on a configuration specified
    either as a dictionary or a path to a JSON/YAML configuration file. It uses
    the SolverBuilder pattern internally to construct the solver.
    
    Args:
        config: Either a path to a configuration file (string) or a dictionary
               containing the configuration parameters. The configuration should
               specify all necessary parameters for the solver.
        flip: Whether to flip the coordinate system (default: False). When True,
              the solver will use a flipped coordinate system, which can be useful
              for certain types of simulations or to match other software conventions.
    
    Returns:
        A solver instance (either RCWASolver or RDITSolver) configured according
        to the provided parameters.
    
    Example:
        ```python
        # From dictionary
        config = {
            'algorithm': 'RDIT',
            'wavelengths': [1.55],
            'lengthunit': 'um',
            'rdim': [512, 512],
            'kdim': [5, 5]
        }
        solver = create_solver_from_config(config)
        
        # From file
        solver = create_solver_from_config('my_config.json')
        ```
    """
    # Lazy import to avoid circular dependencies
    from .builder import SolverBuilder
    return SolverBuilder().from_config(config, flip).build()

def create_solver(algorithm: Algorithm = Algorithm.RDIT,
                 precision: Precision = Precision.SINGLE,
                 lam0: np.ndarray = np.array([1.0]),
                 lengthunit: str = 'um',
                 rdim: List[int] = [512, 512],
                 kdim: List[int] = [3,3],
                 materiallist: List[Any] = [],
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),
                 is_use_FFF: bool = True,
                 device: Union[str, torch.device] = 'cpu') -> Union["RCWASolver", "RDITSolver"]:
    """Create a solver with the given parameters.
    
    This is the main factory function for creating solver instances with custom
    parameters. It provides a convenient way to create a solver without having to
    directly instantiate the solver classes or use the builder pattern.
    
    By default, this function creates an R-DIT solver, which is TorchRDIT's
    eigendecomposition-free implementation offering up to 16.2× speedup compared
    to traditional RCWA-based methods. The R-DIT solver is recommended for inverse
    design applications due to its computational efficiency and stable gradients.
    
    Args:
        algorithm: The algorithm to use for solving (RCWA or RDIT). Default is RDIT,
                 which is recommended for most applications, especially inverse design.
        precision: Numerical precision to use (SINGLE or DOUBLE). Default is SINGLE.
        lam0: Wavelength(s) to simulate, in the specified length unit. Default is [1.0].
        lengthunit: The unit of length used in the simulation (e.g., 'um', 'nm').
                   Default is 'um' (micrometers).
        rdim: Dimensions of the real space grid [height, width]. Default is [512, 512].
        kdim: Dimensions in Fourier space [kheight, kwidth]. Default is [3, 3].
              This determines the number of Fourier harmonics used.
        materiallist: List of materials used in the simulation. Default is empty list.
        t1: First lattice vector. Default is [[1.0, 0.0]] (unit vector in x-direction).
        t2: Second lattice vector. Default is [[0.0, 1.0]] (unit vector in y-direction).
        is_use_FFF: Whether to use Fast Fourier Factorization. Default is True.
                   This improves convergence for high-contrast material interfaces.
        device: The device to run the solver on ('cpu', 'cuda', etc.). Default is 'cpu'.
               For optimal performance, especially with the R-DIT solver, using 'cuda'
               is highly recommended for GPU acceleration.
    
    Returns:
        A solver instance (either RCWASolver or RDITSolver) configured according
        to the provided parameters. The returned solver is fully differentiable,
        enabling gradient-based optimization for inverse design tasks.
    
    Example:
        ```python
        # Create a basic R-DIT solver with default parameters
        solver = create_solver()  # Default is R-DIT algorithm
        
        # Create an R-DIT solver with custom settings for meta-optics design
        solver = create_solver(
            algorithm=Algorithm.RDIT,
            lam0=np.array([1.55]),
            rdim=[1024, 1024],
            kdim=[5, 5],
            device='cuda'  # Use GPU acceleration for faster computation
        )
        
        # Create an RCWA solver for comparison or specific use cases
        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.55]),
            rdim=[512, 512],
            kdim=[3, 3]
        )
        ```
    """
    # Lazy import to avoid circular dependencies
    from .builder import SolverBuilder
    
    # Use the builder to create the solver
    return (SolverBuilder()
            .with_algorithm(algorithm)
            .with_precision(precision)
            .with_wavelengths(lam0)
            .with_length_unit(lengthunit)
            .with_real_dimensions(rdim)
            .with_k_dimensions(kdim)
            .with_materials(materiallist)
            .with_lattice_vectors(t1, t2)
            .with_fff(is_use_FFF)
            .with_device(device)
            .build())

class FourierBaseSolver(Cell3D, SolverSubjectMixin):
    """Base Class of Fourier Domain Solver.
    
    This class serves as the foundation for Fourier-based electromagnetic solvers,
    providing common functionality and interface for derived solver implementations
    such as RCWASolver and RDITSolver.
    
    The FourierBaseSolver implements the core computational methods for:
    - Setting up the simulation environment and parameters
    - Managing the layer stack and material properties
    - Computing Fourier transforms and harmonic expansions
    - Constructing matrices for electromagnetic field calculation
    - Calculating S-matrices for multilayer structures
    - Computing reflection and transmission coefficients
    - Determining field distributions
    - Calculating diffraction efficiencies
    
    This class inherits from Cell3D to manage the geometric and material properties
    of the structure, and SolverSubjectMixin to enable observer notifications
    during the solving process.
    
    Note:
        This is an abstract base class and should not be instantiated directly.
        Use RCWASolver or RDITSolver for concrete implementations, or the
        create_solver() function to create an appropriate solver instance.
    """

    def __init__(self,
                 # wavelengths (frequencies) to be solved
                 lam0: Union[float, np.ndarray] = np.array([1.0]),
                 lengthunit: str = 'um',  # length unit used in the solver
                 rdim: Optional[List[int]] = None,  # dimensions in real space: (H, W)
                 kdim: Optional[List[int]] = None,  # dimensions in k space: (kH, kW)
                 materiallist: Optional[List[MaterialClass]] = None,  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 is_use_FFF: bool = True, # if use Fast Fourier Factorization
                 precision: Precision = Precision.SINGLE,
                 algorithm: SolverAlgorithm = None,
                 device: Union[str, torch.device] = 'cpu') -> None:
        Cell3D.__init__(self, lengthunit, rdim, kdim, materiallist, t1, t2, device)
        SolverSubjectMixin.__init__(self)

        # Initialize default values for mutable parameters
        if rdim is None:
            rdim = [512, 512]
        if kdim is None:
            kdim = [3, 3]
        if materiallist is None:
            materiallist = []
        
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

        # Initialize tensors that were previously class variables
        self.mesh_fp = torch.empty((3, 3), device=self.device, dtype=self.tfloat)
        self.mesh_fq = torch.empty((3, 3), device=self.device, dtype=self.tfloat)
        self.reci_t1 = torch.empty((2,), device=self.device, dtype=self.tfloat)
        self.reci_t2 = torch.empty((2,), device=self.device, dtype=self.tfloat)
        self.tlam0 = torch.empty((1,), device=self.device, dtype=self.tfloat)
        self.k_0 = torch.empty((1,), device=self.device, dtype=self.tfloat)

        # Set free space wavelength
        if isinstance(lam0, float):
            self._lam0 = np.array([lam0], dtype=self.nfloat)
        elif isinstance(lam0, np.ndarray):
            self._lam0 = lam0.astype(self.nfloat)
        else:
            raise TypeError(f"lam0 must be float or numpy.ndarray, got {type(lam0)}")

        self.n_freqs = len(self._lam0)
        self.kinc = torch.zeros((1, 3), device=self.device, dtype=self.tfloat)

        self.src = {}
        self.smat_layers = {}
        self.is_use_FFF = is_use_FFF

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
    
    def _solve_nonhomo_layer(self, layer_thickness, p_mat_i, q_mat_i, mat_w0, mat_v0, kdim, k_0, **kwargs):
        """Delegate to the algorithm strategy."""
        if self._algorithm is None:
            raise ValueError("Solver algorithm not set")
        return self._algorithm.solve_nonhomo_layer(
            layer_thickness, p_mat_i, q_mat_i, mat_w0, mat_v0, kdim, k_0, **kwargs)

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

    def add_source(self,
                   theta: float,
                   phi: float,
                   pte: float,
                   ptm: float,
                   norm_te_dir: str = 'y') -> dict:
        """Create and return a source configuration for the incident electromagnetic wave.

        This method configures the incident electromagnetic wave source for simulation.
        The source is defined by its incident angles and polarization components.
        The returned dictionary can be passed directly to the solve() method.
        
        The incident wave direction is specified using spherical coordinates:
        - theta: Polar angle measured from the z-axis (0° = normal incidence)
        - phi: Azimuthal angle measured in the xy-plane from the x-axis
        
        The polarization is specified using TE (transverse electric) and TM 
        (transverse magnetic) components. These can be complex values to represent
        phase differences between the components.

        Args:
            theta: Incident angle (polar angle) in degrees. 0° corresponds to normal
                  incidence along the z-axis, 90° corresponds to grazing incidence
                  in the xy-plane.
            phi: Azimuthal angle in degrees, measured in the xy-plane from the x-axis.
                 0° corresponds to the x-axis, 90° corresponds to the y-axis.
            pte: TE polarization amplitude (complex value). The TE component has its
                 electric field perpendicular to the plane of incidence.
            ptm: TM polarization amplitude (complex value). The TM component has its
                 magnetic field perpendicular to the plane of incidence.
            norm_te_dir: Direction of normal component for TE wave ('x', 'y', or 'z').
                        This defines the reference direction for the TE polarization.
                        Default is 'y', meaning the TE component is perpendicular to
                        the y-axis.

        Returns:
            Dictionary containing source parameters that can be passed to the solve() method:
            - 'theta': Polar angle in degrees
            - 'phi': Azimuthal angle in degrees
            - 'pte': TE polarization amplitude
            - 'ptm': TM polarization amplitude
            - 'norm_te_dir': Direction of normal component for TE wave
        
        Example:
            ```python
            # Normal incidence with pure TE polarization
            source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
            
            # 45-degree incidence with mixed polarization
            source = solver.add_source(theta=45, phi=0, pte=0.7071, ptm=0.7071)
            
            # Use the source in a solve operation
            result = solver.solve(source)
            ```
        """
        source_dict = {}
        source_dict['theta'] = theta
        source_dict['phi'] = phi
        source_dict['pte'] = pte
        source_dict['ptm'] = ptm
        source_dict['norm_te_dir'] = norm_te_dir

        return source_dict
    
    def _initialize_k_vectors(self):
        """Calculate k-vectors common to both solving methods."""
        # Calculate wave vector expansion
        # kx_0, ky_0: (n_freqs, kdim[0], kdim[1])
        kx_0 = self.kinc[:, 0, None, None] - \
            (self.mesh_fp[None, :, :] * self.reci_t1[0, None, None] + \
                self.mesh_fq[None, :, :] * self.reci_t2[0, None, None]) / self.k_0[:, None, None]
        kx_0 = kx_0.to(dtype=self.tcomplex)

        ky_0 = self.kinc[:, 1, None, None] - \
            (self.mesh_fp[None, :, :] * self.reci_t1[1, None, None] + \
                self.mesh_fq[None, :, :] * self.reci_t2[1, None, None]) / self.k_0[:, None, None]
        ky_0 = ky_0.to(dtype=self.tcomplex)

        # Add relaxation for numerical stability
        epsilon = 1e-6
        kx_0, ky_0 = self._apply_numerical_relaxation(kx_0, ky_0, epsilon)

        # Calculate kz for reflection and transmission regions
        kz_ref_0 = self._calculate_kz_region(self.ur1, self.er1, kx_0, ky_0, 
                                            self.layer_manager.is_ref_dispersive)
        kz_trn_0 = self._calculate_kz_region(self.ur2, self.er2, kx_0, ky_0, 
                                            self.layer_manager.is_trn_dispersive)
        
        return kx_0, ky_0, kz_ref_0, kz_trn_0

    def _apply_numerical_relaxation(self, kx_0, ky_0, epsilon):
        """Apply small offset to zero values for numerical stability."""
        zero_indices = torch.nonzero(kx_0 == 0.0, as_tuple=True)
        if len(zero_indices[0]) > 0:
            kx_0[zero_indices] = kx_0[zero_indices] + epsilon
            
        zero_indices = torch.nonzero(ky_0 == 0.0, as_tuple=True)
        if len(zero_indices[0]) > 0:
            ky_0[zero_indices] = ky_0[zero_indices] + epsilon
        
        return kx_0, ky_0

    def _calculate_kz_region(self, ur, er, kx_0, ky_0, is_dispersive):
        """Calculate kz for a region (reflection or transmission)."""
        if not is_dispersive:
            kz_0 = torch.conj(torch.sqrt(
                torch.conj(ur)*torch.conj(er) - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))
        else:
            kz_0 = torch.conj(torch.sqrt(
                (torch.conj(ur)*torch.conj(er))[:, None, None] - kx_0 * kx_0 - ky_0 * ky_0 + 0*1j))
        return kz_0
    
    def _setup_common_matrices(self, kx_0, ky_0, kz_ref_0, kz_trn_0):
        """Set up common matrices for both solving methods.
        
        Args:
            kx_0: kx_0
            ky_0: ky_0
            kz_ref_0: kz_ref_0
            kz_trn_0: kz_trn_0
        """
        n_harmonics = self.kdim[0] * self.kdim[1]

        self.ident_mat_k = torch.eye(n_harmonics, dtype=self.tcomplex, device=self.device)
        self.ident_mat_k2 = torch.eye(2 * n_harmonics, dtype=self.tcomplex, device=self.device)
        
        # Transform to diagonal matrices
        mat_kx = kx_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_ky = ky_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_kz_ref = kz_ref_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        mat_kz_trn = kz_trn_0.transpose(dim0=-2, dim1=-1).flatten(start_dim=-2)
        
        # Calculate derived matrices
        mat_kx_ky = mat_kx * mat_ky
        mat_ky_kx = mat_ky * mat_kx
        mat_kx_kx = mat_kx * mat_kx
        mat_ky_ky = mat_ky * mat_ky
        
        mat_kx_diag = to_diag_util(mat_kx, self.kdim)
        mat_ky_diag = to_diag_util(mat_ky, self.kdim)
        
        mat_kz = torch.conj(torch.sqrt(1.0 - mat_kx_kx - mat_ky_ky))
        
        ident_mat_kx_kx = 1.0 - mat_kx_kx
        ident_mat_ky_ky = 1.0 - mat_ky_ky
        
        # Set up identity matrices appropriately
        ident_mat = self.ident_mat_k[None, :, :].expand(self.n_freqs, -1, -1)
        zero_mat = torch.zeros(size=(self.n_freqs, n_harmonics, n_harmonics), 
                            dtype=self.tcomplex, device=self.device)
        
        # Create block matrices
        mat_w0 = blockmat2x2([[ident_mat, zero_mat], [zero_mat, ident_mat]])
        
        inv_mat_lam = 1 / (1j*mat_kz)
        
        mat_v0 = blockmat2x2([
            [to_diag_util(mat_kx_ky * inv_mat_lam, self.kdim), 
            to_diag_util(ident_mat_kx_kx * inv_mat_lam, self.kdim)],
            [to_diag_util(- ident_mat_ky_ky * inv_mat_lam, self.kdim), 
            to_diag_util(- mat_kx_ky * inv_mat_lam, self.kdim)]
        ])
        
        # Return all matrices that will be needed
        result = {
            'mat_kx': mat_kx, 'mat_ky': mat_ky, 
            'mat_kz_ref': mat_kz_ref, 'mat_kz_trn': mat_kz_trn,
            'mat_kx_ky': mat_kx_ky, 'mat_ky_kx': mat_ky_kx,
            'mat_kx_kx': mat_kx_kx, 'mat_ky_ky': mat_ky_ky,
            'mat_kx_diag': mat_kx_diag, 'mat_ky_diag': mat_ky_diag,
            'mat_kz': mat_kz, 'mat_w0': mat_w0, 'mat_v0': mat_v0,
            'ident_mat': ident_mat, 'zero_mat': zero_mat
        }
        
        return result
    
    def _process_layer(self, n_layer, matrices):
        """Process a single layer and return its scattering matrix.
        
        This method handles both homogeneous and non-homogeneous layers.
        
        Args:
            n_layer: Layer index to process
            matrices: Dictionary of matrices from setup_common_matrices
        
        Returns:
            dict: Scattering matrix for the layer
        """
        layer = self.layer_manager.layers[n_layer]
        smat_layer = {}
        
        # Handle non-homogeneous layers
        if not layer.is_homogeneous:
            # Extract needed matrices
            mat_kx_diag = matrices['mat_kx_diag']
            mat_ky_diag = matrices['mat_ky_diag']
            
            if layer.is_dispersive:
                toeplitz_er = layer.kermat
            else:
                toeplitz_er = self.expand_dims(layer.kermat)
            
            # assuming permeability always non-dispersive
            # Transform dimensions to (n_freqs, n_harmonics, n_harmonics)
            toeplitz_ur = self.expand_dims(layer.kurmat)
            
            # Solve for all frequencies
            solve_ter_mky = tsolve(toeplitz_er, mat_ky_diag)
            solve_ter_mkx = tsolve(toeplitz_er, mat_kx_diag)
            solve_tur_mky = tsolve(toeplitz_ur, mat_ky_diag)
            solve_tur_mkx = tsolve(toeplitz_ur, mat_kx_diag)
            
            # Create P matrix
            p_mat_i = blockmat2x2([[mat_kx_diag @ solve_ter_mky,
                                toeplitz_ur - mat_kx_diag @ solve_ter_mkx],
                                [mat_ky_diag @ solve_ter_mky - toeplitz_ur,
                                - mat_ky_diag @ solve_ter_mkx]])
            
            # Handle Fast Fourier Factorization (FFF)
            if self.is_use_FFF:
                delta_toeplitz_er = toeplitz_er - tinv(self.reciprocal_toeplitz_er)
                q_mat_i = blockmat2x2([[mat_kx_diag @ solve_tur_mky - delta_toeplitz_er @ self.n_xy,
                                    toeplitz_er - mat_kx_diag @ solve_tur_mkx - delta_toeplitz_er @ self.n_yy],
                                    [mat_ky_diag @ solve_tur_mky - toeplitz_er + delta_toeplitz_er @ self.n_xx,
                                    delta_toeplitz_er @ self.n_xy - mat_ky_diag @ solve_tur_mkx]])
            else:
                q_mat_i = blockmat2x2([[mat_kx_diag @ solve_tur_mky,
                                    toeplitz_er - mat_kx_diag @ solve_tur_mkx],
                                    [mat_ky_diag @ solve_tur_mky - toeplitz_er,
                                    - mat_ky_diag @ solve_tur_mkx]])
                
                
            # Call the appropriate algorithm method
            smat_layer = self._solve_nonhomo_layer(layer_thickness=self.layer_manager.layers[n_layer].thickness.to(self.device).to(self.tcomplex),
                                        p_mat_i=p_mat_i,
                                        q_mat_i=q_mat_i,
                                        mat_w0=matrices['mat_w0'],
                                        mat_v0=matrices['mat_v0'],
                                        kdim=self.kdim,
                                        k_0=self.k_0)
        
        # Handle homogeneous layers
        else:
            if layer.is_dispersive:
                # Get material properties for all frequencies
                toeplitz_er = self._matlib[layer.material_name].er.detach().to(self.tcomplex).to(self.device)
                toeplitz_ur = self._matlib[layer.material_name].ur.detach().to(self.tcomplex).to(self.device)
                
                # Calculate common values
                toep_ur_er = toeplitz_ur * toeplitz_er
                conj_toep_ur_er = torch.conj(toeplitz_ur) * torch.conj(toeplitz_er)
                
                # Calculate kz for the layer
                mat_kz_i = torch.conj(torch.sqrt(
                    conj_toep_ur_er[:, None] - matrices['mat_kx'] ** 2 - matrices['mat_ky'] ** 2) + 0*1j).to(self.tcomplex)
                
                # Calculate v matrix
                inv_dmat_lam_i = 1 / (1j*mat_kz_i)
                mat_v_i = 1 / toeplitz_ur * blockmat2x2([[to_diag_util(matrices['mat_kx_ky'] * inv_dmat_lam_i, self.kdim),
                                                            to_diag_util((toep_ur_er[:, None] - matrices['mat_kx_kx']) * inv_dmat_lam_i, self.kdim)],
                                                            [to_diag_util((matrices['mat_ky_ky'] - toep_ur_er[:, None]) * inv_dmat_lam_i, self.kdim),
                                                            - to_diag_util(matrices['mat_ky_kx'] * inv_dmat_lam_i, self.kdim)]])
            else:
                # Get non-dispersive material properties
                toeplitz_er = self._matlib[layer.material_name].er.detach().clone().to(self.device)
                toeplitz_ur = self._matlib[layer.material_name].ur.detach().clone().to(self.device)
                
                # Calculate common values
                toep_ur_er = toeplitz_ur * toeplitz_er
                conj_toep_ur_er = torch.conj(toeplitz_ur) * torch.conj(toeplitz_er)

                mat_kz_i = torch.conj(torch.sqrt(
                    conj_toep_ur_er - matrices['mat_kx'] ** 2 - matrices['mat_ky'] ** 2) + 0*1j).to(self.tcomplex)

                inv_dmat_lam_i = 1 / (1j*mat_kz_i)
                mat_v_i = 1 / toeplitz_ur * blockmat2x2([[to_diag_util(matrices['mat_kx_ky'] * inv_dmat_lam_i, self.kdim),
                                                            to_diag_util((toep_ur_er - matrices['mat_kx_kx']) * inv_dmat_lam_i, self.kdim)],
                                                            [to_diag_util((matrices['mat_ky_ky'] - toep_ur_er) * inv_dmat_lam_i, self.kdim),
                                                            - to_diag_util(matrices['mat_ky_kx'] * inv_dmat_lam_i, self.kdim)]])

            
            # Calculate the layer matrix
            mat_x_i = - 1j*mat_kz_i * self.k_0[:, None] \
                * self.layer_manager.layers[n_layer].thickness.to(self.device).to(self.tcomplex)
            mat_x_i_diag = torch.concat([mat_x_i, mat_x_i], dim=1)
            mat_x_i = to_diag_util(torch.exp(mat_x_i_diag), self.kdim)

            # Calculate Layer Scattering Matrix
            atwi = matrices['mat_w0']
            atvi = tsolve(mat_v_i, matrices['mat_v0'])
            mat_a_i = atwi + atvi
            mat_b_i = atwi - atvi

            solve_ai_xi = tsolve(mat_a_i, mat_x_i)           

            mat_xi_bi = mat_x_i @ mat_b_i

            mat_d_i = mat_a_i - mat_xi_bi @ solve_ai_xi @ mat_b_i

            smat_layer['S11'] = tsolve(
                mat_d_i, mat_xi_bi @ solve_ai_xi @ mat_a_i - mat_b_i)
            smat_layer['S12'] = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tsolve(mat_a_i, mat_b_i))
            smat_layer['S21'] = smat_layer['S12']
            smat_layer['S22'] = smat_layer['S11']
        
        return smat_layer

    def _solve_structure(self, **kwargs) -> dict:
        """Solve the electromagnetic problem for all frequencies in batch.
        
        Implements the simultaneous solution of all wavelengths in the simulation.
        
        Args:
            **kwargs: Additional parameters that may be required by specific solvers
            
        Returns:
            Dictionary containing the solution results
        """
        # Notify that calculation is starting
        self.notify_observers("calculation_starting", {
            "mode": "solving_structure",
            "n_freqs": self.n_freqs,
            "n_layers": self.layer_manager.nlayer
        })
        
        # Initialize k-vectors
        self.notify_observers("initializing_k_vectors")
        kx_0, ky_0, kz_ref_0, kz_trn_0 = self._initialize_k_vectors()
        
        # Set up matrices for calculation
        self.notify_observers("setting_up_matrices")
        matrices = self._setup_common_matrices(kx_0, ky_0, kz_ref_0, kz_trn_0)
        
        n_harmonics = self.kdim[0] * self.kdim[1]

        # Initialize global scattering matrix
        smat_global = init_smatrix(shape=(self.n_freqs,
                          2*n_harmonics, 2*n_harmonics), dtype=self.tcomplex, device=self.device)

        smat_layer = {}

        # Process each layer and update global scattering matrix
        self.notify_observers("processing_layers", {"total": self.layer_manager.nlayer})
        for n_layer in range(self.layer_manager.nlayer):
            self.notify_observers("layer_started", {
                "layer_index": n_layer,
                "current": n_layer + 1,
                "total": self.layer_manager.nlayer,
                "progress": (n_layer / self.layer_manager.nlayer) * 100
            })
            
            # Process layer using the unified method
            smat_layer = self._process_layer(n_layer, matrices)
                
            self.smat_layers[n_layer] = smat_layer
            smat_global = redhstar(smat_global, smat_layer)
            
            self.notify_observers("layer_completed", {
                "layer_index": n_layer
            })
        
        # Connect to external regions (reflection and transmission)
        self.notify_observers("connecting_external_regions")
        smat_global = self._connect_external_regions(smat_global, matrices)
        
        # Store the structure scattering matrix
        smat_structure = {key: value.detach().clone() for key, value in smat_global.items()}
        
        # Calculate fields and efficiencies
        self.notify_observers("calculating_fields")
        fields = self._calculate_fields_and_efficiencies(smat_global, matrices, kx_0, ky_0)
        
        # Format the fields for output
        self.notify_observers("assembling_final_data")
        rx = torch.reshape(fields['ref_field_x'], shape=(self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        ry = torch.reshape(fields['ref_field_y'], shape=(self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        rz = torch.reshape(fields['ref_field_z'], shape=(self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        tx = torch.reshape(fields['trn_field_x'], shape=(self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        ty = torch.reshape(fields['trn_field_y'], shape=(self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        tz = torch.reshape(fields['trn_field_z'], shape=(self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        
        # Assemble the final data dictionary
        data = {
            'smat_structure': smat_structure,
            'smat_layers': self.smat_layers,
            'rx': rx,
            'ry': ry,
            'rz': rz,
            'tx': tx,
            'ty': ty,
            'tz': tz,
            'RDE': fields['ref_diff_efficiency'],
            'TDE': fields['trn_diff_efficiency'],
            'REF': fields['total_ref_efficiency'],
            'TRN': fields['total_trn_efficiency'],
            'kzref': matrices['mat_kz_ref'],
            'kztrn': matrices['mat_kz_trn'],
            'kinc': self.kinc,
            'kx': torch.squeeze(kx_0),
            'ky': torch.squeeze(ky_0)
        }
        
        self.notify_observers("calculation_completed", {"n_freqs": self.n_freqs})
        
        return data

    def solve(self,
              source: dict,
              **kwargs) -> dict:
        """Solve the electromagnetic problem using the configured algorithm.

        This is the main entry point for running simulations. The function prepares 
        the parameters, performs validation, and executes the electromagnetic calculation
        for all specified wavelengths. It calculates reflection/transmission coefficients,
        diffraction efficiencies, and field distributions.
        
        The solver handles all aspects of the calculation, including:
        - Setting up k-vectors based on incident angles
        - Computing layer eigenmodes and eigenvalues
        - Building the S-matrix for the entire structure
        - Calculating fields in each layer
        - Computing diffraction efficiencies
        
        The TorchRDIT implementation is fully differentiable, enabling gradient-based
        optimization for inverse design of photonic structures. This makes it suitable
        for integration with optimization frameworks, neural networks, and topology
        optimization techniques.
        
        Before calling solve(), you should:
        1. Configure the solver with appropriate parameters
        2. Define your material layers using add_layer() and related methods
        3. Set up your source using add_source()

        Args:
            source: Source configuration dictionary containing the incident wave parameters.
                   This should be created using the add_source() method, which returns
                   a properly formatted source dictionary with:
                   - 'theta': Polar angle of incidence (in degrees)
                   - 'phi': Azimuthal angle of incidence (in degrees)
                   - 'pte': Complex amplitude of the TE polarization component
                   - 'ptm': Complex amplitude of the TM polarization component
                  
            **kwargs: Additional parameters that may be required by specific solvers or to
                     customize the solution process:
                     - 'compute_fields': Whether to compute field distributions (default: False)
                     - 'compute_modes': Whether to return mode information (default: False)
                     - 'store_matrices': Whether to store system matrices (default: False)
                     - 'return_all': Whether to return all calculation results (default: False)

        Returns:
            Dictionary containing the solution results with keys:
            - 'R': Reflection coefficients for each diffraction order and wavelength
            - 'T': Transmission coefficients for each diffraction order and wavelength
            - 'R_total': Total reflection efficiency for each wavelength
            - 'T_total': Total transmission efficiency for each wavelength
            - 'wavelengths': List of wavelengths solved
            - 'kx_inc': x-component of incident k-vector
            - 'ky_inc': y-component of incident k-vector
            - 'kz_inc': z-component of incident k-vector
            - 'theta': Incident polar angle (degrees)
            - 'phi': Incident azimuthal angle (degrees)
            - 'fields': Field distributions (if compute_fields=True)
            - 'modes': Mode information (if compute_modes=True)
            - 'matrices': System matrices (if store_matrices=True)
        
        Example:
            ```python
            # Create and configure solver
            solver = create_solver(algorithm=Algorithm.RDIT, 
                                  lam0=np.array([1.55]),
                                  rdim=[512, 512], 
                                  kdim=[3, 3],
                                  device='cuda')
            
            # Add layers to the solver
            solver.add_layer(thickness=0.5, material='Air')
            solver.add_layer(thickness=0.2, material='Si')
            solver.add_layer(thickness=1.0, material='SiO2')
            
            # Set up the source
            source = solver.add_source(theta=45, phi=0, pte=1.0, ptm=0.0)
            
            # Run the solver
            result = solver.solve(source, compute_fields=True)
            
            # Get reflection efficiency
            R_total = result['R_total']
            
            # For inverse design using automatic differentiation
            import torch.optim as optim
            
            # Define parameters to optimize (example: layer thickness)
            thickness = torch.nn.Parameter(torch.tensor(0.2))
            optimizer = optim.Adam([thickness], lr=0.01)
            
            def forward():
                # Update the layer with current parameters
                solver.update_layer_thickness(layer_index=1, thickness=thickness)
                # Solve and compute loss (e.g., maximize transmission at target wavelength)
                result = solver.solve(source)
                loss = -result['T_total'][0]  # Negative because we want to maximize
                return loss
                
            # Optimization loop
            for i in range(100):
                optimizer.zero_grad()
                loss = forward()
                loss.backward()  # Automatic differentiation
                optimizer.step()
                print(f"Iteration {i}, Loss: {loss.item()}")
            ```
        """
        self.src = source
        self._pre_solve()
        return self._solve_structure(**kwargs)

    def _pre_solve(self) -> None:
        """_pre_solve.

        Check parameters before solving.

        Args:

        Returns:
            None:
        """
        # Calculate refrective index of external medium
        refractive_1 = torch.sqrt(self.ur1 * self.er1)
        # refractive_2 = torch.sqrt(self.ur2 * self.er2)

        # kinc: dimensions (n_freqs, 3)

        if isinstance(self.src['theta'], Union[float, int]):
            theta_src = torch.tensor([self.src['theta']], dtype=self.tfloat, device=self.device)[None].repeat(self.n_freqs, 1)
        else:
            theta_src = torch.tensor(self.src['theta'], dtype=self.tfloat, device=self.device)[:, None]
        
        if isinstance(self.src['phi'], Union[float, int]):
            phi_src = torch.tensor([self.src['phi']], dtype=self.tfloat, device=self.device)[None].repeat(self.n_freqs, 1)
        else:
            phi_src = torch.tensor(self.src['phi'], dtype=self.tfloat, device=self.device)[:, None]

        self.kinc = refractive_1[None] * torch.cat((
            torch.sin(theta_src) * torch.cos(phi_src),
            torch.sin(theta_src) * torch.sin(phi_src),
            torch.cos(theta_src)
        ), dim=1)
        assert self.kinc.shape == (self.n_freqs, 3)

        # Calculate reciprocal lattice vectors
        d_v = self.lattice_t1[0] * self.lattice_t2[1] - self.lattice_t2[0] * self.lattice_t1[1]

        self.reci_t1 = 2 * torch.pi * \
            torch.cat(((+self.lattice_t2[1] / d_v).unsqueeze(0),
                       (-self.lattice_t2[0] / d_v).unsqueeze(0)), dim=0)
        self.reci_t2 = 2 * torch.pi * \
            torch.cat(((-self.lattice_t1[1] / d_v).unsqueeze(0),
                       (+self.lattice_t1[0] / d_v).unsqueeze(0)), dim=0)

        # Calculate wave vector expansion
        self.tlam0 = torch.tensor(
            self.lam0, dtype=self.tfloat, device=self.device)
        self.k_0 = 2 * torch.pi / self.tlam0  # k0 with dimensions: n_freqs

        f_p = torch.arange(start=-np.floor(self.kdim[0] / 2), end=np.floor(
            self.kdim[0] / 2) + 1, dtype=self.tint, device=self.device)
        f_q = torch.arange(start=-np.floor(self.kdim[1] / 2), end=np.floor(
            self.kdim[1] / 2) + 1, dtype=self.tint, device=self.device)
        [self.mesh_fq, self.mesh_fp] = torch.meshgrid(f_q, f_p, indexing='xy')

        # check if options are set correctly
        for n_layer in range(self.layer_manager.nlayer):
            if self.layer_manager.layers[n_layer].is_homogeneous is False:
                if self.layer_manager.layers[n_layer].ermat is None:
                    # if not homogenous material, must be with a pattern.
                    # if no pattern assigned before solving, the material will be set as homogeneous
                    self.layer_manager.replace_layer_to_homogeneous(layer_index=n_layer)

                    print(
                        f"Warning: Layer {n_layer} has no pattern assigned, and was changed to homogeneous")

    def update_er_with_mask(self,
                            mask: torch.Tensor,
                            layer_index: int,
                            bg_material: str = 'air',
                            method: str = 'FFT') -> None:
        """Update the permittivity distribution in a layer using a binary mask.
        
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

        Args:
            mask: Binary tensor representing the pattern mask, where:
                 - 1 (or True) represents the foreground material (from the layer's material)
                 - 0 (or False) represents the background material (specified by bg_material)
                 The mask dimensions must match the real-space dimensions (rdim) of the solver.
                 
                 For inverse design applications, this mask can be a differentiable tensor
                 generated from a neural network or optimization process.
            
            layer_index: Index of the layer to update. This should be a valid index in the
                        layer stack (0 to nlayer-1).
            
            bg_material: Name of the background material to use where mask=0. Default is 'air'.
                        This must be a valid material name that exists in the solver's material list.
            
            method: Method for computing the Toeplitz matrix:
                   - 'FFT': Fast Fourier Transform method (works for all cell types)
                   - 'Analytical': Analytical method (only works for Cartesian cells)
                   Default is 'FFT'.

        Returns:
            None
            
        Raises:
            ValueError: If the mask dimensions don't match the solver's real-space dimensions.
            KeyError: If the specified background material doesn't exist in the material list.
            
        Example:
            ```python
            # Basic usage: Create a binary mask for a simple grating pattern
            mask = torch.zeros(512, 512)
            mask[:, :256] = 1.0  # Half the domain has the foreground material
            
            # Update layer 1 with the mask, using air as background
            solver.update_er_with_mask(mask=mask, layer_index=1, bg_material='air')
            
            # Inverse design usage: Using a neural network to generate a mask
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
                result = solver.solve(source)
                loss = -result['T_total'][0]  # Maximize transmission
                
                # Backpropagate and update
                loss.backward()
                optimizer.step()
            ```
            
        Note:
            This method only updates the permittivity (epsilon) distribution. To update
            the permeability (mu) distribution, you would need to modify the layer's
            urmat property directly.
            
        References:
            - Huang et al., "Eigendecomposition-free inverse design of meta-optics devices," 
              Opt. Express 32, 13986-13997 (2024)
            - Huang et al., "Inverse Design of Photonic Structures Using Automatic Differentiable 
              Rigorous Diffraction Interface Theory," CLEO (2023)
        """

        ndim1, ndim2 = mask.size()
        if (ndim1 != self.rdim[0]) or (ndim2 != self.rdim[1]):
            raise ValueError("Mask dims don't match!")

        if self.layer_manager.layers[layer_index].is_homogeneous:
            self.layer_manager.replace_layer_to_grating(layer_index=layer_index)

        er_bg = self._get_bg(layer_index=layer_index, param='er')

        if self.layer_manager.layers[layer_index].is_dispersive is False:
            mask_format = mask
        else:
            mask_format = mask.unsqueeze(-3)

        self.layer_manager.layers[layer_index].mask_format = mask_format.to(self.tfloat)

        if bg_material == 'air':
            self.layer_manager.layers[layer_index].ermat = 1 + (er_bg - 1) * mask_format
        else:
            if self._matlib[bg_material].er.ndim == 0:
                self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er * (1 - mask_format) + \
                    (er_bg - 1) * mask_format
            else:
                self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er[:, None, None] * (1 - mask_format) + \
                (er_bg - 1) * mask_format
        
        if method == 'Analytical':
            if self.cell_type != CellType.Cartesian:
                print(f"method [{method}] does not support the cell type [{self.cell_type}], will use FFT instead.")
                method = 'FFT'
        elif method != 'Analytical' and method != 'FFT':
            method = 'FFT'

        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='er', method=method)

        # permeability always 1.0 for current version
        # currently no support for magnetic materials
        self.layer_manager.layers[layer_index].urmat = self._get_bg(
            layer_index=layer_index, param='ur')
        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='ur', method=method)

        
        if self.is_use_FFF is True: 
            _, _, self.n_xx, self.n_yy, self.n_xy = self._calculate_nv_field(mask=self.layer_manager.layers[layer_index].mask_format.squeeze())
            self.reciprocal_toeplitz_er = self.layer_manager._gen_toeplitz2d(1 / self.layer_manager.layers[layer_index].ermat,
                                                          nharmonic_1=self.kdim[0],
                                                            nharmonic_2=self.kdim[1],
                                                                        method='FFT')

    def update_er_with_mask_extern_NV(self,
                            mask: torch.Tensor,
                            nv_vectors: tuple,
                            layer_index: int,
                            bg_material: str = 'air',
                            method: str = 'FFT') -> None:
        """update_er_with_mask.

        To update the layer permittivity (or permeability) distribution in a specified layer.

        Args:
            mask (torch.Tensor): mask, the new binary pattern mask to be updated.
            layer_index (int): layer_index
            bg_material (str): bg_material, the background material of the pattern (mask == 0).
            method (str): method of computing Toeplitz matrix. ['FFT': for all cell type; 'Analytical': only for Cartesian]

        Returns:
            None:
        """

        ndim1, ndim2 = mask.size()
        if (ndim1 != self.rdim[0]) or (ndim2 != self.rdim[1]):
            raise ValueError("Mask dims don't match!")

        if self.layer_manager.layers[layer_index].is_homogeneous:
            self.layer_manager.replace_layer_to_grating(layer_index=layer_index)

        er_bg = self._get_bg(layer_index=layer_index, param='er')

        if self.layer_manager.layers[layer_index].is_dispersive is False:
            mask_format = mask
        else:
            mask_format = mask.unsqueeze(-3)

        self.layer_manager.layers[layer_index].mask_format = mask_format.to(self.tfloat)

        if bg_material == 'air':
            self.layer_manager.layers[layer_index].ermat = 1 + (er_bg - 1) * mask_format
        else:
            if self._matlib[bg_material].er.ndim == 0:
                self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er * (1 - mask_format) + \
                    (er_bg - 1) * mask_format
            else:
                self.layer_manager.layers[layer_index].ermat = self._matlib[bg_material].er * (1 - mask_format) + \
                (er_bg - 1) * mask_format
        
        if method == 'Analytical':
            if self.cell_type != CellType.Cartesian:
                print(f"method [{method}] does not support the cell type [{self.cell_type}], will use FFT instead.")
                method = 'FFT'
        elif method != 'Analytical' and method != 'FFT':
            method = 'FFT'

        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='er', method=method)

        # permeability always 1.0 for current version
        # currently no support for magnetic materials
        self.layer_manager.layers[layer_index].urmat = self._get_bg(
            layer_index=layer_index, param='ur')
        self.layer_manager.gen_toeplitz_matrix(
            layer_index=layer_index, n_harmonic1=self.kdim[0], n_harmonic2=self.kdim[1], param='ur', method=method)

        
        if self.is_use_FFF is True: 
            norm_vec_x = nv_vectors[0]
            norm_vec_y = nv_vectors[1]
            self.n_xx = self.layer_manager._gen_toeplitz2d(norm_vec_x * norm_vec_x,
                                                   nharmonic_1=self.kdim[0],
                                                   nharmonic_2=self.kdim[1],
                                                     method='FFT')
            self.n_yy = self.layer_manager._gen_toeplitz2d(norm_vec_y * norm_vec_y,
                                                   nharmonic_1=self.kdim[0],
                                                   nharmonic_2=self.kdim[1],
                                                     method='FFT')
            self.n_xy = self.layer_manager._gen_toeplitz2d(norm_vec_x * norm_vec_y,
                                                   nharmonic_1=self.kdim[0],
                                                   nharmonic_2=self.kdim[1],
                                                     method='FFT')
            self.reciprocal_toeplitz_er = self.layer_manager._gen_toeplitz2d(1 / self.layer_manager.layers[layer_index].ermat,
                                                          nharmonic_1=self.kdim[0],
                                                            nharmonic_2=self.kdim[1],
                                                                        method='FFT')

    def _calculate_nv_field(self, mask: torch.Tensor):
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=self.tfloat, device=self.device)[None, None, :, :]
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=self.tfloat, device=self.device)[None, None, :, :]

        mask = (mask > 0.5).to(self.tfloat).to(self.device)

        # use conv2d to compute the gradients
        gradient_x = tconv2d(mask[None, None, :, :], sobel_x, padding=1)
        gradient_y = tconv2d(mask[None, None, :, :], sobel_y, padding=1)
        gradient_mag = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        index_bond_vec = torch.nonzero(gradient_mag.squeeze())
        index_field_vec = torch.nonzero(gradient_mag.squeeze() == 0)


        blurred_mask = blur_filter(mask[None, None, :, :], radius=4, beta=2, num_blur=1, tfloat=self.tfloat, device=self.device)
        gradient_x_br = tconv2d(blurred_mask, sobel_x, padding=1)
        gradient_y_br = tconv2d(blurred_mask, sobel_y, padding=1)

        norm_vec_x = (gradient_mag * gradient_x_br).squeeze()
        norm_vec_y = (gradient_mag * gradient_y_br).squeeze()

        bondary_vec_x = norm_vec_x[index_bond_vec[:, 0], index_bond_vec[:, 1]]
        bondary_vec_y = norm_vec_y[index_bond_vec[:, 0], index_bond_vec[:, 1]]

        field_ind_i = index_field_vec[:, 0]
        field_ind_j = index_field_vec[:, 1]
        bond_ind_i = index_bond_vec[:, 0]
        bond_ind_j = index_bond_vec[:, 1]
        denom = torch.sqrt((field_ind_i[:, None] - bond_ind_i[None, :])**2 + (field_ind_j[:, None] - bond_ind_j[None, :])**2) + 1e-6
        # denom = torch.sqrt((index_field_vec[:, 0][:, None] - index_bond_vec[:, 0][None, :])**2 + (index_field_vec[:,1][:, None] - index_bond_vec[:,1][None, :])**2)
        
        norm_vec_x[index_field_vec[:,0], index_field_vec[:, 1]] = torch.sum(bondary_vec_x[None, :] / denom, dim=1)
        norm_vec_y[index_field_vec[:,0], index_field_vec[:, 1]] = torch.sum(bondary_vec_y[None, :] / denom, dim=1)

        denom_normal = torch.sqrt(norm_vec_x ** 2 + norm_vec_y ** 2) + 1e-6
        norm_vec_x = norm_vec_x / denom_normal
        norm_vec_y = norm_vec_y / denom_normal

        n_xx = self.layer_manager._gen_toeplitz2d(norm_vec_x * norm_vec_x,
                                               nharmonic_1=self.kdim[0],
                                               nharmonic_2=self.kdim[1],
                                                 method='FFT')
        n_yy = self.layer_manager._gen_toeplitz2d(norm_vec_y * norm_vec_y,
                                               nharmonic_1=self.kdim[0],
                                               nharmonic_2=self.kdim[1],
                                                 method='FFT')
        n_xy = self.layer_manager._gen_toeplitz2d(norm_vec_x * norm_vec_y,
                                               nharmonic_1=self.kdim[0],
                                               nharmonic_2=self.kdim[1],
                                                 method='FFT')
        
        return norm_vec_x, norm_vec_y, n_xx, n_yy, n_xy
        
       
    def get_nv_components(self, layer_index: int):
        if self.layer_manager.layers[layer_index].mask_format is not None:
            nx, ny, _, _, _ = self._calculate_nv_field(mask=self.layer_manager.layers[layer_index].mask_format.squeeze())
            return nx, ny
        else:
            return None, None

    def update_layer_thickness(self,
                               layer_index: int,
                               thickness: torch.Tensor):
        """update_layer_thickness.

        Update the thickness of the specified layer.

        Args:
            layer_index (int): layer_index
            thickness (torch.Tensor): thickness
        """

        self.layer_manager.update_layer_thickness(layer_index=layer_index, thickness=thickness)
        # self.layer_manager.layers[layer_index].is_solved = False

    def _get_bg(self, layer_index: int, param='er') -> torch.Tensor:
        """_get_bg.

        Returns the background matrix of the specified layer.

        Args:
            layer_index (int): layer_index
            param:

        Returns:
            torch.Tensor:
        """

        ret_mat = None

        if layer_index < self.layer_manager.nlayer:
            if self.layer_manager.layers[layer_index].is_dispersive is True:
                material_name = self.layer_manager.layers[layer_index].material_name
                if param == 'er':
                    # ret_mat = torch.tensor(self._matlib[material_name].er).unsqueeze(1)\
                    # .unsqueeze(1).repeat(1, self.rdim[0], self.rdim[1]).to(self.device).to(self.tcomplex)
                    ret_mat = self._matlib[material_name].er.detach().clone().unsqueeze(1).unsqueeze(1).repeat(1, self.rdim[0], self.rdim[1]).to(self.device).to(self.tcomplex)
                elif param == 'ur':
                    param_val = self._matlib[material_name].ur.detach().clone()
                    ret_mat = param_val * torch.ones(size=(self.rdim[0], self.rdim[1]),
                                                  dtype=self.tcomplex, device=self.device)
                else:
                    raise ValueError(f"Input parameter [{param}] is illeagal.")

            else:
                material_name = self.layer_manager.layers[layer_index].material_name
                if param == 'er':
                    param_val = self._matlib[material_name].er.detach().clone()
                elif param == 'ur':
                    param_val = self._matlib[material_name].ur.detach().clone()
                else:
                    raise ValueError(f"Input parameter [{param}] is illeagal.")

                ret_mat = param_val * torch.ones(size=(self.rdim[0], self.rdim[1]),
                                              dtype=self.tcomplex, device=self.device)
        else:
            raise ValueError("The index exceeds the max layer number.")

        return ret_mat


    def expand_dims(self, mat: torch.Tensor) -> Optional[torch.Tensor]:
        """Function that expands the input matrix to a standard output dimension without layer information:
            (n_freqs, n_harmonics, n_harmonics)

        Args:
            mat (torch.Tensor): input tensor matrxi
        """

        ret = None

        if mat.ndim == 1 and mat.shape[0] == self.n_freqs:
            # The input tensor with dimension (n_freqs)
            ret = mat[:, None, None]
        elif mat.ndim == 2:
            # The input matrix with dimension (n_harmonics, n_harmonics)
            ret = mat.unsqueeze(0).repeat(self.n_freqs, 1, 1)
        else:
            raise RuntimeError("Not Listed in the Case")

        return ret

    def _connect_external_regions(self, smat_global, matrices):
        """Connect the global scattering matrix to reflection and transmission regions.
        
        Args:
            smat_global: The global scattering matrix to connect to external regions
            matrices: Dictionary of matrices from setup_common_matrices
            
        Returns:
            dict: Updated global scattering matrix with external regions connected
        """
        
        # Connect to reflection region
        inv_mat_lam_ref = 1 / (1j * matrices['mat_kz_ref'])
        
        if self.layer_manager.is_ref_dispersive is False:
            mat_v_ref = (1/self.ur1) * blockmat2x2([
                [to_diag_util(matrices['mat_kx_ky'] * inv_mat_lam_ref, self.kdim),
                    to_diag_util((self.ur1 * self.er1 - matrices['mat_kx_kx']) * inv_mat_lam_ref, self.kdim)],
                [to_diag_util((matrices['mat_ky_ky'] - self.ur1 * self.er1) * inv_mat_lam_ref, self.kdim),
                    - to_diag_util(matrices['mat_ky_kx'] * inv_mat_lam_ref, self.kdim)]
            ])
        else:
            mat_v_ref = (1/self.ur1) * blockmat2x2([
                [to_diag_util(matrices['mat_kx_ky'] * inv_mat_lam_ref, self.kdim),
                    to_diag_util(((self.ur1 * self.er1)[:, None] - matrices['mat_kx_kx']) * inv_mat_lam_ref, self.kdim)],
                [to_diag_util((matrices['mat_ky_ky'] - (self.ur1 * self.er1)[:, None]) * inv_mat_lam_ref, self.kdim),
                    - to_diag_util(matrices['mat_ky_kx'] * inv_mat_lam_ref, self.kdim)]
            ])
        
        mat_w_ref = self.ident_mat_k2[None, :, :].expand(self.n_freqs, -1, -1)
        
        # Calculate reflection region scattering matrix
        smat_ref = {}
        atw1 = mat_w_ref
        atv1 = tsolve(matrices['mat_v0'], mat_v_ref)
        
        # Common code for both cases
        mat_a_1 = atw1 + atv1
        mat_b_1 = atw1 - atv1
        inv_mat_a_1 = tinv(mat_a_1)
        inv_mat_a_1_mat_b_1 = inv_mat_a_1 @ mat_b_1
        smat_ref['S11'] = - inv_mat_a_1_mat_b_1
        smat_ref['S12'] = 2 * inv_mat_a_1
        smat_ref['S21'] = 0.5 * (mat_a_1 - mat_b_1 @ inv_mat_a_1_mat_b_1)
        smat_ref['S22'] = mat_b_1 @ inv_mat_a_1
        
        smat_global = redhstar(smat_ref, smat_global)
        
        # Connect to transmission region
        inv_mat_lam_trn = 1 / (1j * matrices['mat_kz_trn'])
        
        if self.layer_manager.is_trn_dispersive is False:
            mat_v_trn = (1/self.ur2) * blockmat2x2([
                [to_diag_util(matrices['mat_kx_ky'] * inv_mat_lam_trn, self.kdim),
                    to_diag_util((self.ur2 * self.er2 - matrices['mat_kx_kx']) * inv_mat_lam_trn, self.kdim)],
                [to_diag_util((matrices['mat_ky_ky'] - self.ur2 * self.er2) * inv_mat_lam_trn, self.kdim),
                    - to_diag_util(matrices['mat_ky_kx'] * inv_mat_lam_trn, self.kdim)]
            ])
        else:
            mat_v_trn = (1/self.ur2) * blockmat2x2([
                [to_diag_util(matrices['mat_kx_ky'] * inv_mat_lam_trn, self.kdim),
                    to_diag_util(((self.ur2 * self.er2)[:, None] - matrices['mat_kx_kx']) * inv_mat_lam_trn, self.kdim)],
                [to_diag_util((matrices['mat_ky_ky'] - (self.ur2 * self.er2)[:, None]) * inv_mat_lam_trn, self.kdim),
                    - to_diag_util(matrices['mat_ky_kx'] * inv_mat_lam_trn, self.kdim)]
            ])
        
        mat_w_trn = mat_w_ref
        
        # Calculate transmission region scattering matrix
        smat_trn = {}
        atw2 = mat_w_trn
        atv2 = tsolve(matrices['mat_v0'], mat_v_trn)
        
        # Common code for both cases
        mat_a_2 = atw2 + atv2
        mat_b_2 = atw2 - atv2
        inv_mat_a_2 = tinv(mat_a_2)
        inv_mat_a_2_mat_b_2 = inv_mat_a_2 @ mat_b_2
        smat_trn['S11'] = mat_b_2 @ inv_mat_a_2
        smat_trn['S12'] = 0.5 * (mat_a_2 - mat_b_2 @ inv_mat_a_2_mat_b_2)
        smat_trn['S21'] = 2 * inv_mat_a_2
        smat_trn['S22'] = - inv_mat_a_2_mat_b_2
        
        smat_global = redhstar(smat_global, smat_trn)
        
        return smat_global

    def _calculate_fields_and_efficiencies(self, smat_global, matrices, kx_0, ky_0):
        """Calculate fields and diffraction efficiencies based on the scattering matrix.
        
        Args:
            smat_global: The global scattering matrix
            matrices: Dictionary of matrices from setup_common_matrices
            kx_0, ky_0: Wave vectors
            
        Returns:
            dict: Dictionary containing calculated fields and efficiencies
        """
        n_harmonics = self.kdim[0] * self.kdim[1]
        
        # Calculate polarization vector
        norm_vec = torch.tensor([0.0, 0.0, 1.0], dtype=self.tcomplex, device=self.device)
        
        if isinstance(self.src['theta'], Union[float, int]):
            ate = torch.empty_like(norm_vec)
            if np.abs(self.src['theta']) < 1e-3:
                if 'norm_te_dir' not in self.src:
                    ate = torch.tensor([0.0, 1.0, 0.0], dtype=self.tcomplex, device=self.device)[None, :].repeat(self.n_freqs, 1)
                else:
                    if self.src['norm_te_dir'] == 'y':
                        ate = torch.tensor(
                            [0.0, 1.0, 0.0], dtype=self.tcomplex, device=self.device)[None, :].repeat(self.n_freqs, 1)
                    elif self.src['norm_te_dir'] == 'x':
                        ate = torch.tensor(
                            [1.0, 0.0, 0.0], dtype=self.tcomplex, device=self.device)[None, :].repeat(self.n_freqs, 1)
            else:
                ate = torch.cross(self.kinc, norm_vec[None, :].repeat(self.n_freqs, 1), dim=1)
                ate = ate / torch.norm(ate, dim=1).unsqueeze(-1)
        else:
            ate = torch.zeros((self.n_freqs, 3), dtype=self.tcomplex, device=self.device)
            theta_mask = np.abs(self.src['theta']) < 1e-3
            for i in range(self.n_freqs):
                if theta_mask[i]:
                    if 'norm_te_dir' not in self.src:
                        ate[i, :] = torch.tensor([0.0, 1.0, 0.0], dtype=self.tcomplex, device=self.device)
                    else:
                        if self.src['norm_te_dir'] == 'y':
                            ate[i, :] = torch.tensor([0.0, 1.0, 0.0], dtype=self.tcomplex, device=self.device)
                        elif self.src['norm_te_dir'] == 'x':
                            ate[i, :] = torch.tensor([1.0, 0.0, 0.0], dtype=self.tcomplex, device=self.device)
                else:
                    ate[i, :] = torch.cross(self.kinc[i, :], norm_vec, dim=0)
                    ate[i, :] = ate[i, :] / torch.norm(ate[i, :])
        
        atm = torch.cross(ate, self.kinc, dim=1)
        atm = atm / torch.norm(atm)
        
        # Create polarization vector
        if isinstance(self.src['pte'], Union[float, int]):
            pte_vec = self.src['pte'] * ate
        else:
            pte_vec = torch.tensor(self.src['pte'], dtype=self.tcomplex, device=self.device)[:, None] * ate
            
        if isinstance(self.src['ptm'], Union[float, int]):
            ptm_vec = self.src['ptm'] * atm
        else:
            ptm_vec = torch.tensor(self.src['ptm'], dtype=self.tcomplex, device=self.device)[:, None] * atm
        
        pol_vec = pte_vec + ptm_vec
        pol_vec = pol_vec / torch.norm(pol_vec, dim=1).unsqueeze(-1)
        
        # Calculate electric field source vector
        delta = torch.zeros(size=(self.n_freqs, n_harmonics), dtype=self.tcomplex, device=self.device)
        delta[:, (self.kdim[1] // 2) * self.kdim[0] + (self.kdim[0] // 2)] = 1
        esrc = torch.cat((pol_vec[:, 0].unsqueeze(-1) * delta, pol_vec[:, 1].unsqueeze(-1) * delta), dim=1)
        
        # Calculate source vectors
        mat_w_ref = self.ident_mat_k2[None, :, :].expand(self.n_freqs, -1, -1)
        mat_w_trn = mat_w_ref
        
        csrc = tsolve(mat_w_ref, esrc.unsqueeze(-1))
        
        # Calculate reflected fields
        cref = smat_global['S11'] @ csrc
        eref = mat_w_ref @ cref
        
        ref_field_x = eref[:, 0:n_harmonics, :]
        ref_field_y = eref[:, n_harmonics:2*n_harmonics, :]
        
        # Use matrices from the common setup
        ref_field_z = - matrices['mat_kx_diag'] @ tsolve(to_diag_util(matrices['mat_kz_ref'], self.kdim), ref_field_x) \
                        - matrices['mat_ky_diag'] @ tsolve(to_diag_util(matrices['mat_kz_ref'], self.kdim), ref_field_y)
        
        # Calculate transmitted fields
        ctrn = smat_global['S21'] @ csrc
        etrn = mat_w_trn @ ctrn
        trn_field_x = etrn[:, 0:n_harmonics, :]
        trn_field_y = etrn[:, n_harmonics:2*n_harmonics, :]
        trn_field_z = - matrices['mat_kx_diag'] @ tsolve(to_diag_util(matrices['mat_kz_trn'], self.kdim), trn_field_x) \
                        - matrices['mat_ky_diag'] @ tsolve(to_diag_util(matrices['mat_kz_trn'], self.kdim), trn_field_y)
        
        # Calculate diffraction efficiencies
        ref_diff_efficiency = torch.reshape(
            torch.real(self.ur2/self.ur1*to_diag_util(matrices['mat_kz_ref'], self.kdim) / self.kinc[:, 2, None, None]) @
            (torch.real(ref_field_x) ** 2 + torch.imag(ref_field_x) ** 2 +
                torch.real(ref_field_y) ** 2 + torch.imag(ref_field_y) ** 2 +
                torch.real(ref_field_z) ** 2 + torch.imag(ref_field_z) ** 2),
            shape=(self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        
        trn_diff_efficiency = torch.reshape(
            torch.real(self.ur1/self.ur2*to_diag_util(matrices['mat_kz_trn'], self.kdim) / self.kinc[:, 2, None, None]) @
            (torch.real(trn_field_x) ** 2 + torch.imag(trn_field_x) ** 2 +
                torch.real(trn_field_y) ** 2 + torch.imag(trn_field_y) ** 2 +
                torch.real(trn_field_z) ** 2 + torch.imag(trn_field_z) ** 2),
            shape=(self.n_freqs, self.kdim[0], self.kdim[1])).transpose(dim0=-2, dim1=-1)
        
        # Calculate overall reflectance & transmittance
        total_ref_efficiency = torch.sum(ref_diff_efficiency, dim=(-1, -2))
        total_trn_efficiency = torch.sum(trn_diff_efficiency, dim=(-1, -2))
        
        # Create fields dictionary
        fields = {
            'ref_field_x': ref_field_x,
            'ref_field_y': ref_field_y, 
            'ref_field_z': ref_field_z,
            'trn_field_x': trn_field_x,
            'trn_field_y': trn_field_y,
            'trn_field_z': trn_field_z,
            'ref_diff_efficiency': ref_diff_efficiency,
            'trn_diff_efficiency': trn_diff_efficiency,
            'total_ref_efficiency': total_ref_efficiency,
            'total_trn_efficiency': total_trn_efficiency
        }
        
        return fields


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
        result = solver.solve(source)
        ```
    """
    def __init__(self,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3,3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 is_use_FFF: bool = True,
                 precision: Precision = Precision.SINGLE,
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(lam0, lengthunit, rdim, kdim, materiallist, t1, t2, is_use_FFF, precision, device)

        # Set the algorithm strategy
        self._algorithm = RCWAAlgorithm(self)
    
    def set_rdit_order(self, rdit_order):
        """Set R-DIT order."""
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
        rdim: Dimensions of the real space grid [height, width]. Default is [512, 512].
        kdim: Dimensions in Fourier space [kheight, kwidth]. Default is [3, 3].
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
        result = solver.solve(source)
        ```
        
    References:
        - Huang et al., "Eigendecomposition-free inverse design of meta-optics devices," 
          Opt. Express 32, 13986-13997 (2024)
        - Huang et al., "Inverse Design of Photonic Structures Using Automatic Differentiable 
          Rigorous Diffraction Interface Theory," CLEO (2023)
    """

    def __init__(self,
                 lam0: np.ndarray = np.ndarray([]),
                 lengthunit: str = 'um',
                 rdim: list = [512, 512],  # dimensions in real space: (H, W)
                 kdim: list = [3,3],  # dimensions in k space: (kH, kW)
                 materiallist: list = [],  # list of materials
                 t1: torch.Tensor = torch.tensor([[1.0, 0.0]]),  # lattice vector in real space
                 t2: torch.Tensor = torch.tensor([[0.0, 1.0]]),  # lattice vector in real space
                 is_use_FFF: bool = True,
                 precision: Precision = Precision.SINGLE,
                 device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__(lam0, lengthunit, rdim, kdim, materiallist, t1, t2, is_use_FFF, precision, device)

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
    """Get a solver builder for creating a solver with a fluent interface."""
    from .builder import SolverBuilder
    return SolverBuilder()

def create_solver_from_builder(builder_config: Callable[['SolverBuilder'], 'SolverBuilder']) -> Union["RCWASolver", "RDITSolver"]: # type: ignore
    """Create a solver from a builder configuration."""
    from torchrdit.builder import SolverBuilder
    builder = SolverBuilder()
    builder = builder_config(builder)
    return builder.build()