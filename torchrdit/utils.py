""" This file defines some helper function. """
from typing import Callable, Optional, Union, Any, Tuple, List
from functools import partial, wraps

import torch
import skimage.draw as skdraw

from torch.linalg import solve as tsolve
from .materials import MaterialClass

# Function Type
FuncType = Callable[..., Any]

def tensor_params_check(func: Optional[FuncType] = None, check_start_index: int = 1, check_stop_index:int = 1) -> FuncType:
    """Decorator for validating that function parameters are PyTorch tensors.
    
    This decorator checks that the specified positional and keyword arguments of
    the decorated function are PyTorch tensor objects. If any of the checked parameters
    are not tensors, a TypeError is raised.
    
    This is particularly useful for functions that perform tensor operations and 
    require tensor inputs, ensuring type safety at runtime.
    
    Args:
        func: The function to be decorated. If None, returns a partial 
              function that can be used as a decorator with parameters.
        check_start_index: Index of the first parameter to check. Default is 1,
                         which skips the first parameter (typically 'self' in methods).
        check_stop_index: Index of the last parameter to check. Default is 1,
                       meaning only the parameter at check_start_index is checked.
    
    Returns:
        The decorated function if func is provided, otherwise a partial function
        that will decorate the function when applied.
    
    Raises:
        TypeError: If any of the checked parameters are not PyTorch tensors.
        
    Example:
    ```python
    @tensor_params_check(check_start_index=1, check_stop_index=3)
    def add_tensors(self, a, b, c, d=None):
        # This will check that parameters a, b, and c are tensors
        return a + b + c
    ```
    """

    if func is None:
        check_stop_index = max(check_stop_index, check_start_index)
        return partial(tensor_params_check, check_start_index=check_start_index, check_stop_index=check_stop_index)

    @wraps(func)
    def tensor_warpper(*args: Optional[Any], **kwargs: Optional[Any]) -> Optional[Any]:
        n_checked = 0 # ignore the first parameter [self]
        # Type Check
        # Positional parameters
        for index ,val in enumerate(args):
            if check_start_index <= n_checked <= check_stop_index:
                if not isinstance(val, torch.Tensor):
                    errstr = f"Input[{index}]: {val} is not a torch.Tensor type."
                    raise TypeError(errstr)
            n_checked += 1
        # Keywords parameters
        for keys, val in kwargs.items():
            if check_start_index <= n_checked <= check_stop_index:
                if not isinstance(val, torch.Tensor):
                    errstr = f"Input[{keys}]: {keys}={val} is not a torch.Tensor type."
                    raise TypeError(errstr)
            n_checked += 1

        l_args = list(args)

        return func(*tuple(l_args), **kwargs)
    return tensor_warpper

class EigComplex(torch.autograd.Function):
    """Differentiable complex eigendecomposition for PyTorch.
    
    This class implements a custom autograd function that enables differentiable
    eigendecomposition of complex matrices in PyTorch. It provides the forward
    computation and the corresponding backward gradients required for automatic
    differentiation in optimization tasks.
    
    The implementation is based on the analytical gradient formulation described in:
    https://doi.org/10.1038/s42005-021-00568-6
    
    This is particularly important for the R-DIT algorithm which replaces traditional
    eigendecomposition with other numerical approaches for improved efficiency.
    
    Example:
    ```python
    # Apply differentiable eigendecomposition to a complex matrix
    eigen_values, eigen_vectors = EigComplex.apply(complex_matrix)
    
    # Use the results in a differentiable computation
    result = torch.matmul(eigen_vectors, torch.diag(eigen_values))
    
    # Backward pass will compute gradients through the eigendecomposition
    loss = compute_loss(result)
    loss.backward()
    ```
    """

    @staticmethod
    def forward(ctx, input_matrix, eps=1E-6):
        """Forward pass computing eigenvalues and eigenvectors.
        
        Computes the eigendecomposition of the input complex matrix and saves
        the necessary tensors for the backward pass.
        
        Args:
            ctx: Context object for saving tensors for the backward pass
            input_matrix: Complex matrix to decompose
            eps: Small value for numerical stability (default: 1E-6)
            
        Returns:
            tuple: (eigenvalues, eigenvectors) of the input matrix
        """
        # Perform the eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(input_matrix)
        shape = torch.tensor(input_matrix.shape)
        eps = torch.tensor(eps)

        ctx.save_for_backward(eigenvalues, eigenvectors, shape, eps)

        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_matrix_d, grad_matrix_u):
        """Backward pass computing gradients for eigendecomposition.
        
        Computes the gradients of the loss with respect to the input matrix
        based on the gradients of the loss with respect to the eigenvalues
        and eigenvectors.
        
        Args:
            ctx: Context object containing saved tensors from the forward pass
            grad_matrix_d: Gradient of the loss with respect to eigenvalues
            grad_matrix_u: Gradient of the loss with respect to eigenvectors
            
        Returns:
            Gradient of the loss with respect to the input matrix and None for eps
        """
        # matrix_d -> eigenvalues, matrix_u -> eigenvectors
        matrix_d, matrix_u, shape, eps = ctx.saved_tensors
        device = matrix_d.device
        tcomplex = matrix_d.dtype

        # Convert eigenvalues gradient
        grad_matrix_d = grad_matrix_d.unsqueeze(-2) * torch.eye(
            grad_matrix_d.shape[-1], dtype=tcomplex, device=device)

        # Calculate intermediate matrices
        identity_matrix = torch.eye(shape[-2], shape[-1],
                      dtype=tcomplex, device=device)

        matrix_d = matrix_d.unsqueeze(-1)

        matrix_e = matrix_d.adjoint() - matrix_d

        # Lorentzian brodening
        matrix_f = matrix_e / (matrix_e ** 2 + eps)
        matrix_f = matrix_f - identity_matrix * matrix_f

        # Compute the reverse mode gradient of the eigendecomposition of input_matrix
        grad_matrix_a = torch.conj(matrix_f) * (matrix_u.adjoint() @ grad_matrix_u)
        grad_matrix_a = grad_matrix_d + grad_matrix_a
        grad_matrix_a = grad_matrix_a @ matrix_u.adjoint()
        grad_matrix_a = torch.linalg.solve(matrix_u.adjoint(), grad_matrix_a)

        return grad_matrix_a, None


def create_material(
    name: str = 'material1',  # Name of the material
    permittivity: float = 1.0,  # relative permittivity
    permeability: float = 1.0,  # relative permeability
    # true or false if the permittivity of material is dispersive
    dielectric_dispersion: bool = False,
    # path of the user-defined dispersive dielectric data,
    user_dielectric_file: str = None,
    data_format: str = 'freq-eps',  # format of the user-difined data
    data_unit: str = 'thz',  # unit of frequency or wavelength
    max_poly_fit_order: int = 10,  # max polynomial fit order
):
    """Create a material object for use in electromagnetic simulations.
    
    This function creates and returns a MaterialClass instance that can be used
    in TorchRDIT simulations. Materials can be defined with either constant
    properties (non-dispersive) or wavelength-dependent properties (dispersive).
    
    For non-dispersive materials, simple constant values for permittivity and
    permeability are sufficient. For dispersive materials, data can be loaded
    from a file containing wavelength or frequency-dependent properties.
    
    Args:
        name: Unique identifier for the material. This name is used when 
              referencing the material in other functions.
              Default is 'material1'.
        permittivity: Complex relative permittivity (εᵣ) of the material for non-dispersive
                    materials. Default is 1.0 (vacuum). Negative convention is used.
        permeability: Complex relative permeability (μᵣ) of the material for non-dispersive
                    materials. Default is 1.0 (vacuum). Negative convention is used.
        dielectric_dispersion: Whether the material has frequency-dependent
                             permittivity. If True, data must be provided through
                             user_dielectric_file. Default is False.
        user_dielectric_file: Path to a file containing the dispersive material data.
                            Required if dielectric_dispersion is True.
                            Default is None.
        data_format: Format of the data in the dispersive material file.
                   Options are:
                   - 'freq-eps': Frequency and complex permittivity
                   - 'wl-eps': Wavelength and complex permittivity
                   - 'freq-nk': Frequency and complex refractive index
                   - 'wl-nk': Wavelength and complex refractive index
                   Default is 'freq-eps'.
        data_unit: Unit of frequency or wavelength in the dispersive material file.
                 Common values include 'thz', 'ghz', 'mhz', 'um', 'nm'.
                 Default is 'thz'.
        max_poly_fit_order: Maximum order of polynomial fitting for dispersive
                          materials. Higher values can capture more complex
                          dispersive behavior but may lead to overfitting.
                          Default is 10.
    
    Returns:
        MaterialClass: A material object that can be used in TorchRDIT simulations.
    
    Example:
    ```python
    # Create simple materials with constant properties
    air = create_material(name='air', permittivity=1.0)
    silicon = create_material(name='silicon', permittivity=11.7)
    
    # Create a dispersive material from data file
    gold = create_material(
        name='gold',
        dielectric_dispersion=True,
        user_dielectric_file='gold_data.txt',
        data_format='wl-nk',
        data_unit='um'
    )
    ```
    """
    return MaterialClass(name=name,
                     permittivity=permittivity,
                     permeability=permeability,
                     dielectric_dispersion=dielectric_dispersion,
                     user_dielectric_file=user_dielectric_file,
                     data_format=data_format,
                     data_unit=data_unit,
                     max_poly_fit_order=max_poly_fit_order)


def blockmat2x2(mlist: list):
    """Create a block matrix from a 2×2 list of sub-matrices.
    
    This function concatenates four matrices into a single block matrix in a 2×2 pattern:
    
    ```
    [ mlist[0][0]  mlist[0][1] ]
    [ mlist[1][0]  mlist[1][1] ]
    ```
    
    This is commonly used in electromagnetic simulations to construct matrices
    that have a natural 2×2 block structure, such as scattering matrices that
    relate forward and backward propagating waves.
    
    Args:
        mlist: A 2×2 list containing the four sub-matrices to concatenate.
               All matrices must be compatible for concatenation, meaning
               mlist[0][0] and mlist[0][1] must have the same number of rows,
               mlist[1][0] and mlist[1][1] must have the same number of rows,
               mlist[0][0] and mlist[1][0] must have the same number of columns,
               and mlist[0][1] and mlist[1][1] must have the same number of columns.
    
    Returns:
        torch.Tensor: The concatenated block matrix.
        
    Example:
    ```python
    # Create four 2x2 matrices
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    C = torch.tensor([[9, 10], [11, 12]])
    D = torch.tensor([[13, 14], [15, 16]])
    
    # Combine them into a block matrix
    block_matrix = blockmat2x2([[A, B], [C, D]])
    
    # Result is a 4x4 matrix:
    # [[1, 2, 5, 6],
    #  [3, 4, 7, 8],
    #  [9, 10, 13, 14],
    #  [11, 12, 15, 16]]
    ```
    """
    return torch.cat((torch.cat((mlist[0][0], mlist[0][1]), -1),
                      torch.cat((mlist[1][0], mlist[1][1]), -1)),
                     -2)

def init_smatrix(shape: Tuple, dtype: torch.dtype, device: Union[str, torch.device] = 'cpu'):
    """Initialize a scattering matrix with identity transmission and zero reflection.
    
    This function creates a dictionary representing an S-matrix for a layer or structure
    with perfect transmission (identity matrices for S12 and S21) and zero reflection
    (zero matrices for S11 and S22). This corresponds to an initial state with no
    reflections at the interfaces.
    
    In electromagnetic simulations, S-matrices relate incoming and outgoing waves:
    [b1] = [S11 S12] [a1]
    [b2]   [S21 S22] [a2]
    
    where a1, a2 are incoming waves and b1, b2 are outgoing waves.
    
    Args:
        shape: Tuple specifying the dimensions of the S-matrix components.
              For batch processing, this is typically (batch_size, n_harmonics, n_harmonics).
        dtype: PyTorch data type for the matrices (typically torch.complex64 or torch.complex128).
        device: Device to create the tensors on ('cpu' or 'cuda').
               Default is 'cpu'.
    
    Returns:
        dict: S-matrix dictionary with keys 'S11', 'S12', 'S21', 'S22' mapping to the
             corresponding blocks of the scattering matrix.
    
    Note:
        The default initialization represents a non-reflecting layer, where incoming
        waves pass through without reflection (S11=S22=0) and without modification
        (S12=S21=I, where I is the identity matrix).
    """
    smatrix = {}
    if len(shape) == 3:
        smatrix['S11'] = torch.zeros(
            size=shape, dtype=dtype, device=device)
        smatrix['S12'] = torch.eye(shape[-2], shape[-1], dtype=dtype,
                                   device=device).unsqueeze(0).repeat(shape[0], 1, 1)
        smatrix['S21'] = torch.eye(shape[-2], shape[-1], dtype=dtype,
                                   device=device).unsqueeze(0).repeat(shape[0], 1, 1)
        smatrix['S22'] = torch.zeros(
            size=shape, dtype=dtype, device=device)
    else:
        smatrix['S11'] = torch.zeros(
            size=shape, dtype=dtype, device=device)
        smatrix['S12'] = torch.eye(shape[-2], shape[-1], dtype=dtype,
                                   device=device)
        smatrix['S21'] = torch.eye(shape[-2], shape[-1], dtype=dtype,
                                   device=device)
        smatrix['S22'] = torch.zeros(
            size=shape, dtype=dtype, device=device)
        
    return smatrix


def redhstar(smat_a: dict, smat_b: dict, tcomplex: torch.dtype = torch.complex64) -> dict:
    """Compute the Redheffer star product of two scattering matrices.
    
    The Redheffer star product (⋆) combines the scattering matrices of two adjacent
    layers or structures to produce the scattering matrix of the combined system.
    This is a fundamental operation in the transfer matrix method for solving
    multilayer electromagnetic problems.
    
    For two S-matrices A and B, the Redheffer star product C = A ⋆ B is given by:
    - C11 = A11 + A12 B11 (I - A22 B11)^(-1) A21
    - C12 = A12 (I - B11 A22)^(-1) B12
    - C21 = B21 (I - A22 B11)^(-1) A21
    - C22 = B22 + B21 A22 (I - B11 A22)^(-1) B12
    
    Args:
        smat_a: First scattering matrix (dictionary with keys 'S11', 'S12', 'S21', 'S22')
               This represents the layer or structure closer to the reference medium.
        smat_b: Second scattering matrix (dictionary with keys 'S11', 'S12', 'S21', 'S22')
               This represents the layer or structure closer to the transmission medium.
        tcomplex: PyTorch complex data type to use for intermediate calculations.
                Default is torch.complex64.
    
    Returns:
        dict: Combined scattering matrix as a dictionary with keys 'S11', 'S12', 'S21', 'S22'
             representing the four blocks of the scattering matrix of the combined system.
    
    Note:
        The function assumes that the S-matrices are properly formatted and have compatible
        dimensions for matrix operations. For batch processing, both S-matrices should
        have the same batch dimensions.
    """

    # Input dimensions (n_batches, n_freqs, n_harmonics, n_harmonics)
    # Construct identity matrix
    if smat_a['S11'].ndim == 3:
        _, harmonic_m, harmonic_n = smat_a['S11'].shape
    else:
        harmonic_m, harmonic_n = smat_a['S11'].shape
    device = smat_a['S11'].device
    identity_mat = torch.eye(harmonic_m, harmonic_n, dtype=tcomplex, device=device)

    # Compute commom terms
    inv_cycle_1 = identity_mat - smat_b['S11'] @ smat_a['S22']
    cycle_1_smat_b_11  = tsolve(inv_cycle_1, smat_b['S11'])
    cycle_1_smat_b_12  = tsolve(inv_cycle_1, smat_b['S12'])

    # woodbury matrix identity
    cycle_2 = identity_mat + smat_a['S22'] @ cycle_1_smat_b_11
    smat_b_21_cycle_2 = smat_b['S21'] @ cycle_2
    # Compute combined scattering matrix
    smatrix = {}
    smatrix['S11'] = smat_a['S11'] + smat_a['S12'] @ cycle_1_smat_b_11 @ smat_a['S21']
    smatrix['S12'] = smat_a['S12'] @ cycle_1_smat_b_12
    smatrix['S21'] = smat_b_21_cycle_2 @ smat_a['S21']
    smatrix['S22'] = smat_b['S22'] + smat_b_21_cycle_2 @ smat_a['S22'] @ smat_b['S12']

    return smatrix

def _create_blur_kernel(radius: int, device: torch.device = torch.device('cpu'), tfloat: torch.dtype = torch.float32):
    """Create a circular convolution kernel for blurring operations.
    
    Creates a normalized circular (disk-shaped) convolution kernel with the specified
    radius that can be used for blurring operations via convolution. The kernel has
    uniform weights (all pixels within the disk have the same value) and is normalized
    so that the sum of all weights equals 1.
    
    Args:
        radius: Radius of the circular kernel in pixels. The resulting kernel
               will have dimensions (2*radius+1) × (2*radius+1).
        device: PyTorch device to create the kernel on ('cpu' or 'cuda').
               Default is 'cpu'.
        tfloat: PyTorch floating-point data type for the kernel.
               Default is torch.float32.
    
    Returns:
        torch.Tensor: Normalized circular convolution kernel.
        
    Examples:
    ```python
    kernel = _create_blur_kernel(radius=5)
    kernel.shape  # Returns torch.Size([11, 11])
    ```
        
    Keywords:
        convolution, kernel, blur, disk, circle, filter, topology optimization
    """
    row_ind, col_ind = skdraw.disk(center=(radius, radius), radius=radius+1)
    kernel = torch.zeros(size=(2*radius+1, 2*radius+1),
                         dtype=tfloat, device=device)
    kernel[row_ind, col_ind] = 1
    return kernel / torch.sum(kernel)


def operator_blur(rho: torch.Tensor,
                  radius: int = 2,
                  num_blur: int = 1,
                  device: torch.device = torch.device('cpu'),
                  tfloat: torch.dtype = torch.float32) -> torch.Tensor:
    """Apply a blur filter to a tensor using 2D convolution.
    
    This function applies a circular blur filter to the input tensor using
    2D convolution. The blur can be applied multiple times in sequence to
    achieve stronger blurring effects. This is commonly used in topology
    optimization to enforce minimum feature sizes and improve manufacturability.
    
    Args:
        rho: Input tensor to be blurred. Should be a 2D tensor of shape [height, width]
            or a batch of 2D tensors of shape [batch_size, height, width].
        radius: Radius of the circular blur kernel in pixels. Larger values
               create stronger blurring effects. Default is 2.
        num_blur: Number of times to apply the blur operation sequentially.
                Higher values create stronger blurring. Default is 1.
        device: PyTorch device to perform the operation on ('cpu' or 'cuda').
               Default is 'cpu'.
        tfloat: PyTorch floating-point data type for the operation.
               Default is torch.float32.
    
    Returns:
        torch.Tensor: Blurred tensor with the same shape as the input.
        
    Examples:
    ```python
    import torch
    from torchrdit.utils import operator_blur
    input_tensor = torch.rand(1, 1, 32, 32)
    blurred = operator_blur(input_tensor, radius=3, num_blur=2)
    blurred.shape  # Returns torch.Size([1, 1, 32, 32])
    ```
        
    Note:
        This function is often used in conjunction with projection operations
        in topology optimization to enforce minimum feature sizes and improve
        the manufacturability of optimized designs.
        
    Keywords:
        blur, convolution, filter, smoothing, topology optimization, feature size, batch processing
    """

    in_ch = rho.shape[1]
    out_ch = in_ch

    kernel = _create_blur_kernel(radius=radius, device=device, tfloat=tfloat)


    kernel_norm = kernel.view(out_ch, in_ch, kernel.shape[0], kernel.shape[1])

    for _ in range(num_blur):
        rho = torch.nn.functional.conv2d(rho, kernel_norm, padding=radius)

    return rho

def operator_proj(rho: torch.Tensor, eta: float = 0.5, beta: int = 100, num_proj: int = 1) -> torch.Tensor:
    """Apply a projection filter to a tensor for binary optimization.
    
    This function applies a sigmoid-based projection to push values in the input tensor
    toward either 0 or 1, which is useful in topology optimization to enforce binary
    designs. The projection is controlled by two parameters:
    
    - eta: The threshold value (values above eta tend toward 1, below toward 0)
    - beta: The projection strength (higher values create sharper transitions)
    
    The projection can be applied multiple times to strengthen the effect.
    
    The projection function is defined as:
    ρ̂ = (tanh(β·η) + tanh(β·(ρ-η))) / (tanh(β·η) + tanh(β·(1-η)))
    
    Args:
        rho: Input tensor with values ideally between 0 and 1.
        eta: Threshold value for the projection. Values above eta tend toward 1,
            while values below eta tend toward 0. Default is 0.5.
        beta: Projection strength parameter. Higher values create sharper
             transitions between 0 and 1. Default is 100.
        num_proj: Number of times to apply the projection sequentially.
                Higher values create more binary results. Default is 1.
    
    Returns:
        torch.Tensor: Projected tensor with the same shape as the input, with
                   values pushed toward 0 or 1.
    
    Examples:
    ```python
    import torch
    from torchrdit.utils import operator_proj
    input_tensor = torch.rand(10, 10)  # Random values between 0 and 1
    binary_result = operator_proj(input_tensor, beta=200)
    print(binary_result)
    # Values will be pushed closer to 0 or 1
    ```
    
    Note:
        This function is commonly used in topology optimization to enforce binary
        designs (e.g., material/no material) while maintaining differentiability
        for gradient-based optimization.
        
    Keywords:
        projection, binary, topology optimization, thresholding, differentiable, sigmoid
    """
    eta = torch.tensor(eta)
    beta = torch.tensor(beta)

    for _ in range(num_proj):
        rho = torch.div(torch.tanh(beta * eta) + torch.tanh(beta * (rho - eta)),
                        torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta)))

    return rho


def blur_filter(rho: torch.Tensor,
                radius: int = 2,
                num_blur: int = 1,
                beta: int = 100,
                eta: float = 0.5,
                num_proj: int = 1,
                device: torch.device = torch.device('cpu'),
                tfloat: torch.dtype = torch.float32):
    """Apply combined blur and projection filtering for topology optimization.
    
    This function combines blurring and projection operations commonly used in
    topology optimization to enforce minimum feature sizes and binary designs.
    It first applies a blur filter to smooth the input, then applies a projection
    filter to push values toward 0 or 1.
    
    The combined effect helps to:
    1. Enforce minimum feature sizes through blurring
    2. Create binary designs through projection
    3. Maintain differentiability for gradient-based optimization
    
    This is particularly useful in the inverse design of photonic structures
    where both manufacturability constraints and binary material distributions
    are important.
    
    Args:
        rho: Input tensor with values ideally between 0 and 1.
        radius: Radius of the circular blur kernel in pixels.
               Larger values create stronger blurring. Default is 2.
        num_blur: Number of times to apply the blur operation.
                Higher values create stronger blurring. Default is 1.
        beta: Projection strength parameter. Higher values create sharper
             transitions between 0 and 1. Default is 100.
        eta: Threshold value for the projection. Values above eta tend toward 1,
            while values below eta tend toward 0. Default is 0.5.
        num_proj: Number of times to apply the projection.
                Higher values create more binary results. Default is 1.
        device: PyTorch device to perform the operations on ('cpu' or 'cuda').
               Default is 'cpu'.
        tfloat: PyTorch floating-point data type for the operations.
               Default is torch.float32.
    
    Returns:
        torch.Tensor: Filtered tensor with the same shape as the input.
    
    Examples:
    ```python
    import torch
    from torchrdit.utils import blur_filter
    design = torch.rand(1, 1, 64, 64)  # Initial random design
    filtered = blur_filter(design, radius=3, num_blur=2, beta=150)
    print(filtered)
    # Result will have smoother features and more binary values
    ```
    
    Note:
        This function is a convenience wrapper that combines the operator_blur
        and operator_proj functions into a single operation commonly used in
        topology optimization workflows.
        
    Keywords:
        topology optimization, filter, blur, projection, binary, manufacturability, 
        feature size, inverse design, photonics
    """

    rho = operator_blur(rho, radius=radius, num_blur=num_blur, device=device, tfloat=tfloat)
    rho = operator_proj(rho, beta=beta, eta=eta, num_proj=num_proj)

    return rho

def to_diag_util(input_mat: torch.Tensor, kdim: List[int]) -> torch.Tensor:
    """Convert a vector to a diagonal matrix for electromagnetic calculations.
    
    This utility function creates a diagonal matrix from a vector, which is a common
    operation in electromagnetic simulations when converting material properties or
    field components to matrix form for calculations.
    
    The function handles both regular harmonics (n_harmonics) and cases where the
    input represents both polarizations (2*n_harmonics), automatically determining
    the appropriate size based on the input dimensions.
    
    Args:
        input_mat: Input tensor to be converted to a diagonal matrix. This is typically
                 a vector of length n_harmonics or 2*n_harmonics, where n_harmonics
                 is the product of the k-space dimensions.
        kdim: Dimensions in Fourier space as [kheight, kwidth]. These determine
             the number of harmonics used in the calculation.
    
    Returns:
        torch.Tensor: Diagonal matrix with the input values along the diagonal.
                   If input_mat has shape [..., n_harmonics], the output will have
                   shape [..., n_harmonics, n_harmonics]. If input_mat has shape
                   [..., 2*n_harmonics], the output will have shape
                   [..., 2*n_harmonics, 2*n_harmonics].
    
    Examples:
    ```python
    import torch
    from torchrdit.utils import to_diag_util
    input_vector = torch.ones(5)  # A vector with 5 elements
    kdim = [1, 5]  # 1x5 k-space dimension resulting in 5 harmonics
    diagonal_matrix = to_diag_util(input_vector, kdim)
    print(diagonal_matrix.shape)  # Outputs torch.Size([5, 5])
    print(torch.allclose(diagonal_matrix, torch.eye(5)))  # Outputs True
    ```
    
    Note:
        This function is used extensively in the RCWA and R-DIT algorithms when
        constructing matrices for eigenvalue problems and field calculations.
        
    Keywords:
        diagonal matrix, electromagnetic, RCWA, R-DIT, Fourier, harmonics, 
        polarization, matrix construction
    """
    n_harmonics = kdim[0] * kdim[1]
    if input_mat.shape[-1] == n_harmonics:
        return input_mat.unsqueeze(-2) * torch.eye(n_harmonics).to(input_mat.device)
    else:
        return input_mat.unsqueeze(-2) * torch.eye(2 * n_harmonics).to(input_mat.device)