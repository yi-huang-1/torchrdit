""" This file defines some helper function. """
from typing import Callable, Optional, Union, Any, Tuple
from functools import partial, wraps

import torch
import skimage.draw as skdraw

from torch.linalg import solve as tsolve
from .materials import MaterialClass

# Function Type
FuncType = Callable[..., Any]

def tensor_params_check(func: Optional[FuncType] = None, check_start_index: int = 1, check_stop_index:int = 1) -> FuncType:
    """tensor_params_check. This function is used for checking input parameters.

    Args:
        func (Optional[FuncType]): func, fucntions to be checked.
        check_start_index (int): check_start_index, index of the first parameter to be checked.
        check_stop_index (int): check_stop_index, index of the last parameter to be checked.

    Returns:
        FuncType:
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
    """EigComplex.
    This class are used for differentiable eigen-decomposition.
    Reference: https://doi.org/10.1038/s42005-021-00568-6
    """


    @staticmethod
    def forward(ctx, input_matrix, eps=1E-6):
        # Perform the eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(input_matrix)
        shape = torch.tensor(input_matrix.shape)
        eps = torch.tensor(eps)

        ctx.save_for_backward(eigenvalues, eigenvectors, shape, eps)

        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_matrix_d, grad_matrix_u):
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
    """create_material.
    This function returns material objects for solver.

    Args:
        name (str): name of the material.
        permittivity (float): permittivity of the material (if non-dispersive)
        permeability (float): permeability of the material (if non-dispersive)
        dielectric_dispersion (bool): dielectric_dispersion (true if meterial is dispersive)
        user_dielectric_file (str): user_dielectric_file (valid for dispersive material)
        data_format (str): data_format ['freq-eps', 'wl-eps', 'freq-nk', 'wl-nk']
        data_unit (str): data_unit, unit of the frequency of wavelength
        max_poly_fit_order (int): max_poly_fit_order, max polynomial fitting order
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
    """Concatnates block matrices from the given 2 x 2 list.

    Args:
        mlist (list): list with tensors.

        X = [xa, xb;
            xc,  xd]
    """
    return torch.cat((torch.cat((mlist[0][0], mlist[0][1]), -1),
                      torch.cat((mlist[1][0], mlist[1][1]), -1)),
                     -2)

def init_smatrix(shape: Tuple, dtype: torch.dtype , device : Union[str, torch.device] ='cpu'):
    """init_smatrix.
    This function initializes and returns a scattering matrix

    Args:
        shape (Tuple): shape
        dtype (torch.dtype): dtype
        device (Union[str, torch.device]): device
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
    """redhstar.
    This fucntion returns the Redheffer star product of two s-matrices.

    Args:
        smat_a (dict): smat_a
        smat_b (dict): smat_b
        tcomplex (torch.dtype): tcomplex

    Returns:
        dict:
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
    """_create_blur_kernel.
    Helper function used below for creating the conv kernel.
    Args:
        radius (int): radius
        device (torch.device): device
        tfloat (torch.dtype): tfloat
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
    """operator_blur.
    Blur operator implemented via two-dimensional convolution
    Args:
        rho (torch.Tensor): rho
        radius (int): radius
        num_blur (int): num_blur
        device (torch.device): device
        tfloat (torch.dtype): tfloat

    Returns:
        torch.Tensor:
    """

    in_ch = rho.shape[1]
    out_ch = in_ch

    kernel = _create_blur_kernel(radius=radius, device=device, tfloat=tfloat)


    kernel_norm = kernel.view(out_ch, in_ch, kernel.shape[0], kernel.shape[1])

    for _ in range(num_blur):
        rho = torch.nn.functional.conv2d(rho, kernel_norm, padding=radius)

    return rho

def operator_proj(rho: torch.Tensor, eta: float = 0.5, beta: int = 100, num_proj: int = 1) -> torch.Tensor:
    """operator_proj.
    This function makes the density projection.

    Args:
        rho (torch.Tensor): rho
        eta (float): eta, the center of the projection between 0 and 1.
        beta (int): beta, strength of the projection.
        num_proj (int): num_proj, number of projections to be applied.

    Returns:
        torch.Tensor:
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
    """blur_filter.
    This function makes the blur filtering and projections.

    Args:
        rho (torch.Tensor): rho
        radius (int): radius
        num_blur (int): num_blur
        beta (int): beta, strength of the projection.
        eta (float): eta, the center of the projection between 0 and 1.
        num_proj (int): num_proj, number of projections to be applied.
        device (torch.device): device
        tfloat (torch.dtype): tfloat
    """

    rho = operator_blur(rho, radius=radius, num_blur=num_blur, device=device, tfloat=tfloat)
    rho = operator_proj(rho, beta=beta, eta=eta, num_proj=num_proj)

    return rho

