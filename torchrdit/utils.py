import numpy as np
import torch
import skimage.draw as skdraw

from torch.linalg import inv as tinv
from torch.fft import fft2, fftshift
from .materials import materials
from typing import Optional, Tuple, Union

from typing import Callable, Optional, Union, Any
from functools import partial, wraps

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

    assert(isinstance(check_start_index, int))
    if func is None:
        if check_stop_index < check_start_index:
            check_stop_index = check_start_index
        return partial(tensor_params_check, check_start_index=check_start_index, check_stop_index=check_stop_index)

    @wraps(func)
    def tensor_warpper(*args: Optional[Any], **kwargs: Optional[Any]) -> Optional[Any]:
        batch_size = args[0].n_batches
        n_checked = 1 # ignore the first parameter [self]
        # Type Check
        # Positional parameters
        for i ,x in enumerate(args):
            if i > 0:
                if (n_checked >= check_start_index and n_checked <= check_stop_index):
                    if not isinstance(x, torch.Tensor):
                        errstr = f"Input[{i}]: {x} is not a torch.Tensor type."
                        raise TypeError(errstr)
                n_checked += 1
        # Keywords parameters
        for k, v in kwargs.items():       
            if (n_checked >= check_start_index and n_checked <= check_stop_index):
                if not isinstance(v, torch.Tensor):
                    errstr = f"Input[{k}]: {k}={v} is not a torch.Tensor type."
                    raise TypeError(errstr)
            n_checked += 1
                
        # Dimension Check, assert if the parameters match the batch size
        n_checked = 1 # ignore the first parameter [self]
        l_args = list(args)
        for i ,x in enumerate(l_args):
            if i > 0:
                if (n_checked >= check_start_index and n_checked <= check_stop_index):
                    if x.ndim == 0:
                        l_args[i] = x.expand(batch_size)
                    elif x.ndim == 1 and x.shape[0] == 1:
                        l_args[i] = x.expand(batch_size)
                    elif x.ndim == 1 and x.shape[0] != batch_size:
                        errstr = f"The size of the input tensor {list(x.shape)} must match the batch size {[batch_size]} or [1]."
                        raise RuntimeError(errstr)
                    elif x.ndim == 2 and x.shape[0] == 1:
                        l_args[i] = x.expand(batch_size, x.shape[1])
                    elif x.ndim == 2 and x.shape[0] != batch_size:
                        errstr = f"The size of the input tensor {list(x.shape)} must match the batch size {[batch_size,]} or [1,]."
                        raise RuntimeError(errstr)
                    elif x.ndim > 2:
                        errstr = f"The input tensor does not match the batch size."
                        raise RuntimeError(errstr)
                n_checked += 1

        for k, v in kwargs.items():
            if (n_checked >= check_start_index and n_checked <= check_stop_index):
                if v.ndim == 0:
                    kwargs[k] = v.expand(batch_size)
                elif v.ndim == 1 and v.shape[0] == 1:
                    kwargs[k] = v.expand(batch_size)
                elif v.ndim == 1 and v.shape[0] != batch_size:
                    errstr = f"The size of the input tensor {list(v.shape)} must match the batch size {[batch_size]} or [1]."
                    raise RuntimeError(errstr)
                elif v.ndim == 2 and v.shape[0] == 1:
                    kwargs[k] = v.expand(batch_size, v.shape[1])
                elif v.ndim == 2 and v.shape[0] != batch_size:
                    errstr = f"The size of the input tensor {list(v.shape)} must match the batch size {[batch_size, ]} or [1, ]."
                    raise RuntimeError(errstr)
                elif v.ndim > 2:
                    errstr = f"The input tensor does not match the batch size."
                    raise RuntimeError(errstr)
            n_checked += 1
                
        return func(*tuple(l_args), **kwargs)
    return tensor_warpper

class eig_complex(torch.autograd.Function):
    """eig_complex.
    This class are used for differentiable eigen-decomposition.
    Reference: https://doi.org/10.1038/s42005-021-00568-6
    """


    @staticmethod
    def forward(ctx, A, eps=1E-6):
        # Perform the eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        shape = torch.tensor(A.shape)
        eps = torch.tensor(eps)

        ctx.save_for_backward(eigenvalues, eigenvectors, shape, eps)

        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_D, grad_U):
        # D -> eigenvalues, U -> eigenvectors
        D, U, shape, eps = ctx.saved_tensors
        device = D.device
        tcomplex = D.dtype

        # Convert eigenvalues gradient
        grad_D = grad_D.unsqueeze(-2) * torch.eye(
            grad_D.shape[-1], dtype=tcomplex, device=device)

        # Calculate intermediate matrices
        I = torch.eye(shape[-2], shape[-1],
                      dtype=tcomplex, device=device)

        D = D.unsqueeze(-1)

        E = D.adjoint() - D

        # Lorentzian brodening
        F = E / (E ** 2 + eps)
        F = F - I * F

        # Compute the reverse mode gradient of the eigendecomposition of A
        grad_A = torch.conj(F) * (U.adjoint() @ grad_U)
        grad_A = grad_D + grad_A
        grad_A = grad_A @ U.adjoint()
        grad_A = torch.linalg.solve(U.adjoint(), grad_A)

        return grad_A, None


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
    return materials(name=name,
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


def gen_toeplitz2d_dispersive(A: torch.Tensor,
                         P: int = 1,
                         Q: int = 1,
                         tcomplex: torch.dtype = torch.complex64,
                         tint: torch.dtype = torch.int32) -> torch.Tensor:
    """This function constructs Toeplitz matrices from
    a real-space 2D grid.

    Args:
        A (torch.Tensor): Grid in real space: (n_batches, n_freqs, H, W)
        P (int): Number of harmonics for the T1 direction.
        Q (int): Number of harmonics for the T2 direction.

    Returns:
        C (torch.Tensor): Toeplitz matrix: (n_batches, kHW, kHW)
    """

    n_batches, n_freqs, nH, nW = A.size()
    device = A.device

    # Computes indices of spatial harmonics
    P = int(P)
    Q = int(Q)

    kHW = P * Q

    # # construct the convolution matrix
    # C = torch.zeros(size=(n_batches, n_freqs, kHW, kHW),
    #                 dtype=tcomplex, device=device)

    # compute fourier coefficents of A for only the last two dimensions
    fA = fftshift(
        fftshift(fft2(A[:, :, :, :]), dim=-1), dim=-2) / (nH * nW)

    # compute zero-order harmonic
    p0 = torch.floor(torch.tensor(nH / 2)).to(torch.int64)
    q0 = torch.floor(torch.tensor(nW / 2)).to(torch.int64)

    rows = torch.arange(kHW, dtype = tint, device = device)
    cols = torch.arange(kHW, dtype = tint, device = device)

    rows, cols = torch.meshgrid(rows, cols, indexing = "ij")

    ms = torch.div(rows, Q, rounding_mode = 'floor').to(torch.int64)
    prow_t = rows - ms * Q
    js = torch.div(cols, Q, rounding_mode = 'floor').to(torch.int64)
    pcol_t = cols - js * Q
    qf_t = ms - js
    pf_t = prow_t - pcol_t

    C = fA[:, :, p0 - pf_t, q0 - qf_t].to(tcomplex)

    return C

def gen_toeplitz2d_normal(A: torch.Tensor,
                     P: int = 1,
                     Q: int = 1,
                     tcomplex: torch.dtype = torch.complex64,
                     tint: torch.dtype = torch.int32) -> torch.Tensor:
    """This function constructs Toeplitz matrices from
    a real-space 2D grid.

    Args:
        A (torch.Tensor): Grid in real space: (n_batches, H, W)
        P (int): Number of harmonics for the T1 direction.
        Q (int): Number of harmonics for the T2 direction.

    Returns:
        C (torch.Tensor): Toeplitz matrix: (n_batches, kHW, kHW)
    """

    n_batches, nH, nW = A.size()
    device = A.device

    # Computes indices of spatial harmonics
    P = int(P)
    Q = int(Q)

    kHW = P * Q

    # compute fourier coefficents of A for only the last two dimensions
    fA = fftshift(fftshift(fft2(A), dim=-1), dim=-2) / (nH * nW)

    # compute zero-order harmonic
    p0 = torch.floor(torch.tensor(nH / 2)).to(torch.int64)
    q0 = torch.floor(torch.tensor(nW / 2)).to(torch.int64)

    # # construct the convolution matrix
    # C = torch.zeros(size=(n_batches, kHW, kHW),
    #                 dtype=tcomplex, device=device)

    rows = torch.arange(kHW, dtype = tint, device = device)
    cols = torch.arange(kHW, dtype = tint, device = device)

    rows, cols = torch.meshgrid(rows, cols, indexing = "ij")

    ms = torch.div(rows, Q, rounding_mode = 'floor').to(torch.int64)
    prow_t = rows - ms * Q
    js = torch.div(cols, Q, rounding_mode = 'floor').to(torch.int64)
    pcol_t = cols - js * Q
    qf_t = ms - js
    pf_t = prow_t - pcol_t

    C = fA[:, p0 - pf_t, q0 - qf_t].to(tcomplex)


    return C


def init_smatrix(shape: Tuple, dtype: torch.dtype , device : Union[str, torch.device] ='cpu'):
    """init_smatrix.
    This function initializes and returns a scattering matrix

    Args:
        shape (Tuple): shape
        dtype (torch.dtype): dtype
        device (Union[str, torch.device]): device
    """
    smatrix = dict()
    smatrix['S11'] = torch.zeros(
        size=shape, dtype=dtype, device=device)
    smatrix['S12'] = torch.eye(shape[-2], shape[-1], dtype=dtype,
                               device=device).unsqueeze(0).unsqueeze(0).repeat(shape[0], shape[1], 1, 1)
    smatrix['S21'] = torch.eye(shape[-2], shape[-1], dtype=dtype,
                               device=device).unsqueeze(0).unsqueeze(0).repeat(shape[0], shape[1], 1, 1)
    smatrix['S22'] = torch.zeros(
        size=shape, dtype=dtype, device=device)
    return smatrix


def redhstar(SA: dict, SB: dict, tcomplex: torch.dtype = torch.complex64) -> dict:
    """redhstar.
    This fucntion returns the Redheffer star product of two s-matrices.

    Args:
        SA (dict): SA
        SB (dict): SB
        tcomplex (torch.dtype): tcomplex

    Returns:
        dict:
    """

    # Input dimensions (n_batches, n_freqs, kHW, kHW)
    # Construct identity matrix
    _, _, m, n = SA['S11'].shape
    device = SA['S11'].device
    I = torch.eye(m, n, dtype=tcomplex, device=device)

    # Compute commom terms
    D = SA['S12'] @ tinv(I - SB['S11'] @ SA['S22'])
    F = SB['S21'] @ tinv(I - SA['S22'] @ SB['S11'])

    # Compute combined scattering matrix
    SS = dict()
    SS['S11'] = SA['S11'] + D @ SB['S11'] @ SA['S21']
    SS['S12'] = D @ SB['S12']
    SS['S21'] = F @ SA['S21']
    SS['S22'] = SB['S22'] + F @ SA['S22'] @ SB['S12']

    return SS

def _create_blur_kernel(radius: int, device: torch.device = torch.device('cpu'), tfloat: torch.dtype = torch.float32):
    """_create_blur_kernel.
    Helper function used below for creating the conv kernel.
    Args:
        radius (int): radius
        device (torch.device): device
        tfloat (torch.dtype): tfloat
    """
    rr, cc = skdraw.disk(center=(radius, radius), radius=radius+1)
    kernel = torch.zeros(size=(2*radius+1, 2*radius+1),
                         dtype=tfloat, device=device)
    kernel[rr, cc] = 1
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
