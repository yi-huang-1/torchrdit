from abc import ABC, abstractmethod
import torch
from .solver import tsolve
from .utils import EigComplex, to_diag_util
import math

class SolverAlgorithm(ABC):
    """Abstract base class defining the interface for electromagnetic solver algorithms.
    
    This class defines the common interface that all solver algorithm implementations
    must adhere to. It follows the Strategy pattern, allowing different algorithms
    (RCWA and R-DIT) to be used interchangeably with the same solver interface.
    
    Note:
        Users should not instantiate or use these algorithm classes directly.
        Instead, use the SolverBuilder interface which provides a more user-friendly
        way to configure and create solvers with the appropriate algorithm.
    
    Examples:
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
    """
    
    @abstractmethod
    def solve_nonhomo_layer(self, layer_index, p_mat, q_mat, w0_mat, v0_mat):
        """Solve equations for non-homogeneous layer.
        
        This abstract method defines the interface for solving the electromagnetic
        field equations within a non-homogeneous layer. Different algorithm
        implementations (RCWA, R-DIT) will provide different approaches for
        this calculation.
        
        Note:
            This method is called internally by the solver and should not be
            called directly by users.
        
        Args:
            layer_index (int): Index of the layer to solve.
            p_mat (torch.Tensor): P matrix for the layer.
            q_mat (torch.Tensor): Q matrix for the layer.
            w0_mat (torch.Tensor): W0 matrix.
            v0_mat (torch.Tensor): V0 matrix.
            
        Returns:
            dict: Results of the non-homogeneous layer calculation, typically a
            scattering matrix for the layer.
        
        Keywords:
            layer solving, non-homogeneous, electromagnetic, scattering matrix
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return the name of the algorithm.
        
        This property provides a human-readable identifier for the algorithm.
        
        Returns:
            str: The name of the algorithm (e.g., 'RCWA', 'R-DIT')
        
        Keywords:
            algorithm name, identifier
        """
        pass



class RCWAAlgorithm(SolverAlgorithm):
    """Rigorous Coupled Wave Analysis (RCWA) algorithm implementation.
    
    This class implements the traditional RCWA method, which solves Maxwell's
    equations in the frequency domain by expanding the electromagnetic fields
    and material properties in Fourier series and matching boundary conditions
    at layer interfaces.
    
    While the TorchRDIT package primarily emphasizes the eigendecomposition-free
    R-DIT method for improved performance, this RCWA implementation is provided
    for comparison and for cases where RCWA may be preferred.
    
    Note:
        Users should not instantiate this class directly. Instead, use the
        SolverBuilder interface to create solvers with the appropriate algorithm.
    
    Attributes:
        solver: Reference to the parent solver instance.
        _rdit_order (int): Order parameter for compatibility with R-DIT.
    
    Examples:
    ```python
    # Using SolverBuilder to create a solver with RCWA algorithm
    from torchrdit.builder import SolverBuilder
    from torchrdit.constants import Algorithm
    builder = SolverBuilder()
    solver = builder.with_algorithm(Algorithm.RCWA).build()
    # Additional configuration can be applied through the builder
    solver = builder.with_algorithm(Algorithm.RCWA) \\
                    .with_wavelengths(1.55) \\
                    .with_k_dimensions([5, 5]) \\
                    .build()
    ```
    
    Keywords:
        RCWA, electromagnetic, eigendecomposition, Fourier, Maxwell, scattering
    """
    
    def __init__(self, solver):
        """Initialize the RCWA algorithm instance.
        
        Args:
            solver: Reference to the parent solver that will use this algorithm.
        """
        self.solver = solver
        self._rdit_order = 2  # Default value
    
    @property
    def name(self):
        """Return the name of the algorithm.
        
        Returns:
            str: The name of the algorithm ('RCWA')
        """
        return "RCWA"
    
    def solve_nonhomo_layer(self, layer_thickness, p_mat_i, q_mat_i, mat_w0, mat_v0, kdim, k_0, **kwargs):
        """RCWA implementation for solving non-homogeneous layer.
        
        This method implements the traditional approach for calculating the scattering
        matrix of a non-homogeneous layer using eigenmode decomposition in the
        Rigorous Coupled Wave Analysis (RCWA) algorithm.
        
        Note:
            This method is called internally by the solver and should not be
            called directly by users.
        
        Args:
            layer_thickness (float): Thickness of the layer.
            p_mat_i (torch.Tensor): P matrix for the layer.
            q_mat_i (torch.Tensor): Q matrix for the layer.
            mat_w0 (torch.Tensor): W0 matrix.
            mat_v0 (torch.Tensor): V0 matrix.
            kdim (list): Dimensions in k-space [kheight, kwidth].
            k_0 (torch.Tensor): Wave number.
            **kwargs: Additional parameters.
            
        Returns:
            dict: Dictionary containing the scattering matrix for the layer
                 with keys 'S11', 'S12', 'S21', 'S22'.
        
        Keywords:
            RCWA, scattering matrix, eigenmode, layer calculation, non-homogeneous
        """
        # Compute Eigen Modes
        smat_layer = {}
        mat_lam_i, mat_w_i = EigComplex.apply(p_mat_i @ q_mat_i)

        inv_dmat_lam_i = to_diag_util(1 / torch.sqrt(mat_lam_i), kdim)


        mat_v_i = q_mat_i @ mat_w_i @ inv_dmat_lam_i
        mat_x_i = torch.linalg.matrix_exp(to_diag_util(- torch.sqrt(mat_lam_i) * k_0[:, None] \
            * layer_thickness, kdim))

        # Calculate Layer Scattering Matrix
        mat_a_i = tsolve(mat_w_i, mat_w0) + tsolve(mat_v_i, mat_v0)
        mat_b_i = tsolve(mat_w_i, mat_w0) - tsolve(mat_v_i, mat_v0)

        mat_xb_i = mat_x_i @ mat_b_i
        mat_d_i = mat_a_i - mat_xb_i @ tsolve(mat_a_i, mat_x_i) @ mat_b_i

        smat_layer['S11'] = tsolve(
            mat_d_i, mat_xb_i @ tsolve(mat_a_i, mat_x_i) @ mat_a_i - mat_b_i)
        smat_layer['S12'] = tsolve(mat_d_i, mat_x_i) @ (mat_a_i - mat_b_i @ tsolve(mat_a_i, mat_b_i))
        smat_layer['S21'] = smat_layer['S12']
        smat_layer['S22'] = smat_layer['S11']

        return smat_layer
    
    def set_rdit_order(self, rdit_order):
        """Set R-DIT order for compatibility with RDITAlgorithm.
        
        This method is provided for API compatibility with the R-DIT algorithm,
        allowing for easy switching between algorithms.
        
        Note:
            Users should not call this method directly. RDIT order can be set 
            using SolverBuilder.with_rdit_order() method.
        
        Args:
            rdit_order (int): The order of the R-DIT algorithm. This parameter has
                       minimal effect in the RCWA implementation.
        
        Examples:
        ```python
        # Setting RDIT order through the builder
        from torchrdit.builder import SolverBuilder
        from torchrdit.constants import Algorithm
        solver = SolverBuilder() \\
                .with_algorithm(Algorithm.RCWA) \\
                .with_rdit_order(4) \\
                .build()
        ```
        
        Keywords:
            RCWA, R-DIT, compatibility, order parameter
        """
        self._rdit_order = rdit_order


class RDITAlgorithm(SolverAlgorithm):
    """Rigorous Diffraction Interface Theory (R-DIT) algorithm implementation.
    
    This class implements the eigendecomposition-free R-DIT algorithm, which offers
    improved numerical stability and computational efficiency compared to the
    traditional RCWA approach. The R-DIT method achieves up to 16.2× speedup
    in inverse design applications.
    
    Note:
        Users should not instantiate this class directly. Instead, use the
        SolverBuilder interface to create solvers with the appropriate algorithm.
    
    Attributes:
        solver: Reference to the parent solver instance.
        _rdit_order (int): Order parameter for the R-DIT algorithm approximation.
    
    Examples:
    ```python
    # Using SolverBuilder to create a solver with R-DIT algorithm
    from torchrdit.builder import SolverBuilder
    from torchrdit.constants import Algorithm
    builder = SolverBuilder()
    # Configure with R-DIT algorithm and set order
    solver = builder.with_algorithm(Algorithm.RDIT) \\
                    .with_rdit_order(8) \\
                    .build()
    # Or load configuration from a file
    solver = SolverBuilder().from_config("config.json").build()
    ```
    
    Keywords:
        R-DIT, electromagnetic, eigendecomposition-free, optimization, speedup, scattering
    """
    
    def __init__(self, solver):
        """Initialize the R-DIT algorithm instance.
        
        Args:
            solver: Reference to the parent solver that will use this algorithm.
        """
        self.solver = solver
        self._rdit_order = 10  # Default value
    
    @property
    def name(self):
        """Return the name of the algorithm.
        
        Returns:
            str: The name of the algorithm ('R-DIT')
        
        Keywords:
            algorithm name, R-DIT, identifier
        """
        return "R-DIT"
    
    def solve_nonhomo_layer(self, layer_thickness, p_mat_i, q_mat_i, mat_w0, mat_v0, kdim, k_0, **kwargs):
        """R-DIT implementation for solving non-homogeneous layer.
        
        This method implements the eigendecomposition-free approach for calculating
        the scattering matrix of a non-homogeneous layer using the Rigorous Diffraction
        Interface Theory algorithm.
        
        Note:
            This method is called internally by the solver and should not be
            called directly by users.
        
        Args:
            layer_thickness (float): Thickness of the layer.
            p_mat_i (torch.Tensor): P matrix for the layer.
            q_mat_i (torch.Tensor): Q matrix for the layer.
            mat_w0 (torch.Tensor): W0 matrix.
            mat_v0 (torch.Tensor): V0 matrix.
            kdim (list): Dimensions in k-space [kheight, kwidth].
            k_0 (torch.Tensor): Wave number.
            **kwargs: Additional parameters.
            
        Returns:
            dict: Dictionary containing the scattering matrix for the layer
                 with keys 'S11', 'S12', 'S21', 'S22'.
        
        Keywords:
            R-DIT, scattering matrix, eigendecomposition-free, layer calculation, non-homogeneous
        """
        n_harmonics = kdim[0] * kdim[1]

        smat_layer = {}

        # Construct T matrix
        delta_h = k_0[:, None, None] * layer_thickness / 2.0
        tmat_a_i = torch.eye(2*n_harmonics, 2*n_harmonics).unsqueeze(0).repeat(k_0.shape[0], 1, 1).to(p_mat_i.dtype).to(p_mat_i.device)
        tmat_b_i = torch.zeros(size=(k_0.shape[0], 2*n_harmonics, 2*n_harmonics)).to(p_mat_i.dtype).to(p_mat_i.device)
        tmat_c_i = torch.zeros(size=(k_0.shape[0], 2*n_harmonics, 2*n_harmonics)).to(p_mat_i.dtype).to(p_mat_i.device)
        tmat_d_i = torch.eye(2*n_harmonics, 2*n_harmonics).unsqueeze(0).repeat(k_0.shape[0], 1, 1).to(p_mat_i.dtype).to(p_mat_i.device)

        p_fcoef = torch.eye(2*n_harmonics, 2*n_harmonics).unsqueeze(0).repeat(k_0.shape[0], 1, 1).to(p_mat_i.dtype).to(p_mat_i.device)
        q_fcoef = torch.eye(2*n_harmonics, 2*n_harmonics).unsqueeze(0).repeat(k_0.shape[0], 1, 1).to(p_mat_i.dtype).to(p_mat_i.device)
        
        for irdit_order in range(1, self._rdit_order + 1):
            if (irdit_order % 2) == 0:  # even orders
                p_fcoef = p_fcoef @ q_mat_i
                q_fcoef = q_fcoef @ p_mat_i
                fac = (delta_h**irdit_order / math.factorial(irdit_order)).to(p_mat_i.dtype).to(p_mat_i.device)
                tmat_a_i = tmat_a_i + fac * p_fcoef
                tmat_d_i = tmat_d_i + fac * q_fcoef
            else:  # odd orders
                p_fcoef = p_fcoef @ p_mat_i
                q_fcoef = q_fcoef @ q_mat_i
                fac = (delta_h**irdit_order / math.factorial(irdit_order)).to(p_mat_i.dtype).to(p_mat_i.device)
                tmat_b_i = tmat_b_i + fac * p_fcoef
                tmat_c_i = tmat_c_i + fac * q_fcoef

        # Construct some helper functions
        a_i_w0 = tmat_a_i @ mat_w0
        b_i_v0 = tmat_b_i @ mat_v0
        c_i_w0 = tmat_c_i @ mat_w0
        d_i_v0 = tmat_d_i @ mat_v0

        mat_g1 = a_i_w0 + b_i_v0
        mat_g2 = -c_i_w0 - d_i_v0

        mat_xx1 = tsolve(mat_g2, c_i_w0 - d_i_v0)
        mat_xx2 = tsolve(mat_g1, a_i_w0 - b_i_v0)

        mat_yyi = mat_xx1 - mat_xx2
        mat_zzi = mat_xx1 + mat_xx2

        smat_layer['S11'] = mat_yyi / 2.0
        smat_layer['S12'] = mat_zzi / 2.0
        smat_layer['S21'] = smat_layer['S12']
        smat_layer['S22'] = smat_layer['S11']
        return smat_layer
    
    def set_rdit_order(self, rdit_order):
        """Set the order of the R-DIT algorithm.
        
        The R-DIT order determines the approximation used in the diffraction
        interface theory. Higher orders generally provide better accuracy but
        may be computationally more expensive.
        
        Note:
            Users should not call this method directly. RDIT order can be set 
            using SolverBuilder.with_rdit_order() method.
        
        Args:
            rdit_order (int): The order of the algorithm (typically 1-10). Higher values
                       provide more accurate results at the cost of computational
                       efficiency.
        
        Examples:
        ```python
        # Setting RDIT order through the builder
        from torchrdit.builder import SolverBuilder
        from torchrdit.constants import Algorithm
        solver = SolverBuilder() \\
                        .with_algorithm(Algorithm.RDIT) \\
                        .with_rdit_order(8) \\
                        .build()
        ```
        
        Keywords:
            R-DIT, order, approximation, accuracy, performance, tradeoff
        """
        self._rdit_order = rdit_order