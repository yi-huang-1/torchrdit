"""
Torch-native tangent vector field generation.

This module reproduces the tangent-field utilities from the fmmax project
(``https://github.com/facebookresearch/fmmax``). The implementation follows the derivations
of the following papers:
- M. F. Schubert and A. M. Hammond, "Fourier modal method for inverse design of metasurface-enhanced micro-LEDs," Opt. Express 31, 42945 (2023).
- V. Liu and S. Fan, "S4 : A free electromagnetic solver for layered periodic structures," Computer Physics Communications 183, 2233–2244 (2012).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import torch


__all__ = [
    "Expansion",
    "FourierExpansionManager",
    "LatticeVectors",
    "TangentFieldGenerator",
    "compute_tangent_field",
    "generate_expansion",
    "min_array_shape_for_expansion",
]


def _magnitude_floor(dtype: torch.dtype) -> float:
    """Return a stable floor value for squared magnitudes."""
    if dtype == torch.float32:
        return 1e-8
    if dtype == torch.float64:
        return 1e-16
    if dtype == torch.float16:
        return 1e-4
    if dtype == torch.bfloat16:
        return 1e-4
    raise TypeError(f"Unsupported dtype for stability epsilon {dtype}")


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype paired with ``real_dtype``."""
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise TypeError(f"Unsupported real dtype {real_dtype}")


def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
    """Promote ``x`` to a complex dtype if needed."""
    return x if x.is_complex() else x.to(_complex_dtype(x.dtype))


def _cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the 2D cross product."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


@dataclass
class LatticeVectors:
    """Store primitive lattice vectors."""

    u: torch.Tensor
    v: torch.Tensor

    def reciprocal(self) -> LatticeVectors:
        """Return the reciprocal lattice."""
        cross = _cross_product(self.u, self.v)[..., None]
        uprime = torch.stack([self.v[..., 1], -self.v[..., 0]], dim=-1) / cross
        vprime = torch.stack([-self.u[..., 1], self.u[..., 0]], dim=-1) / cross
        return LatticeVectors(u=uprime, v=vprime)

    def to(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> LatticeVectors:
        """Return a copy of the lattice on the requested device and dtype."""
        return LatticeVectors(
            u=self.u.to(device=device or self.u.device, dtype=dtype or self.u.dtype),
            v=self.v.to(device=device or self.v.device, dtype=dtype or self.v.dtype),
        )

    def normalized_basis(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return normalized basis vectors on the requested device/dtype."""
        raw_basis = torch.stack([self.u, self.v], dim=-1)
        area = torch.abs(_cross_product(self.u, self.v))
        normalized = raw_basis / torch.sqrt(area)[..., None, None]
        return normalized.to(device=device or self.u.device, dtype=dtype or self.u.dtype)

    def inverse_metric(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return the inverse lattice metric on the requested device/dtype."""
        basis = self.normalized_basis(device=device, dtype=dtype)
        metric = basis.transpose(-1, -2) @ basis
        return torch.linalg.inv(metric)


@dataclass
class Expansion:
    """Hold Fourier basis coefficients."""

    basis_coefficients: torch.Tensor  # (num_terms, 2)

    @property
    def num_terms(self) -> int:
        """Return the number of retained Fourier terms."""
        return int(self.basis_coefficients.shape[0])


class FourierExpansionManager:
    """Manage Fourier bookkeeping for tangent-field calculations."""

    def __init__(self, lattice_vectors: LatticeVectors, approximate_num_terms: int) -> None:
        self._lattice_vectors = lattice_vectors
        self._expansion = generate_expansion(
            primitive_lattice_vectors=lattice_vectors,
            approximate_num_terms=approximate_num_terms,
        )

    @property
    def primitive_lattice_vectors(self) -> LatticeVectors:
        """Return the primitive lattice vectors used for the expansion."""
        return self._lattice_vectors

    @property
    def expansion(self) -> Expansion:
        """Return the cached Fourier expansion."""
        return self._expansion

    def adapt_to(self, *, device: torch.device, dtype: torch.dtype) -> None:
        """Move internal tensors to the requested device and dtype."""
        self._lattice_vectors = self._lattice_vectors.to(device=device, dtype=dtype)
        coeffs = self._expansion.basis_coefficients
        if coeffs.device != device:
            self._expansion = Expansion(basis_coefficients=coeffs.to(device=device))

    def min_array_shape(self) -> Tuple[int, int]:
        """Return the smallest spatial grid compatible with the expansion."""
        return min_array_shape_for_expansion(self._expansion)

    def project(
        self,
        x: torch.Tensor,
        *,
        axes: Tuple[int, int] = (-2, -1),
        centered_coordinates: bool = False,
    ) -> torch.Tensor:
        """Project ``x`` onto the retained Fourier orders."""
        return fft(
            x,
            expansion=self._expansion,
            axes=axes,
            centered_coordinates=centered_coordinates,
        )

    def reconstruct(
        self,
        y: torch.Tensor,
        *,
        shape: Tuple[int, int],
        axis: int = -1,
        centered_coordinates: bool = False,
    ) -> torch.Tensor:
        """Reconstruct the spatial field from Fourier samples."""
        return ifft(
            y,
            expansion=self._expansion,
            shape=shape,
            axis=axis,
            centered_coordinates=centered_coordinates,
        )

    def fourier_penalty_weights(self) -> torch.Tensor:
        """Return Fourier penalty weights on the current device/dtype."""
        return _fourier_penalty_weights(
            self._lattice_vectors,
            self._expansion,
        )


def _solve_quadratic_for_orders(ratio: float, approximate_num_terms: int) -> int:
    """Solve the quadratic that balances Fourier orders along u and v."""
    a = 4.0 * ratio
    b = 2.0 * (ratio + 1.0)
    c = 1.0 - approximate_num_terms
    discriminant = max(b * b - 4.0 * a * c, 0.0)
    root = (-b + math.sqrt(discriminant)) / (2.0 * a)
    return max(0, int(round(root)))


def _generate_parallelogram_coefficients(
    reciprocal_vectors: LatticeVectors,
    approximate_num_terms: int,
) -> torch.Tensor:
    """Enumerate Fourier coefficients for the expansion."""
    device = reciprocal_vectors.u.device
    dtype = reciprocal_vectors.u.dtype
    ku_spacing = torch.linalg.norm(reciprocal_vectors.u.detach().to(dtype=torch.float64)).item()
    kv_spacing = torch.linalg.norm(reciprocal_vectors.v.detach().to(dtype=torch.float64)).item()

    if ku_spacing == 0 or kv_spacing == 0:
        raise ValueError("Lattice vectors must be non-zero.")

    nu = _solve_quadratic_for_orders(ku_spacing / kv_spacing, approximate_num_terms)
    nv = _solve_quadratic_for_orders(kv_spacing / ku_spacing, approximate_num_terms)

    gu = torch.arange(-nu, nu + 1, device=device, dtype=torch.int64)
    gv = torch.arange(-nv, nv + 1, device=device, dtype=torch.int64)
    if gu.numel() == 0 or gv.numel() == 0:
        gu = torch.zeros(1, dtype=torch.int64, device=device)
        gv = torch.zeros(1, dtype=torch.int64, device=device)

    G1, G2 = torch.meshgrid(gu, gv, indexing="ij")
    G1 = G1.reshape(-1)
    G2 = G2.reshape(-1)
    vectors = (
        G1[:, None].to(dtype=dtype) * reciprocal_vectors.u
        + G2[:, None].to(dtype=dtype) * reciprocal_vectors.v
    )
    magnitude = torch.linalg.norm(vectors, dim=-1)
    order = torch.argsort(magnitude, stable=True)
    coeffs = torch.stack([G1[order], G2[order]], dim=-1)
    return coeffs


def generate_expansion(
    primitive_lattice_vectors: LatticeVectors,
    approximate_num_terms: int,
) -> Expansion:
    """Build a Fourier expansion for the lattice."""
    reciprocal_vectors = primitive_lattice_vectors.reciprocal()
    coeffs = _generate_parallelogram_coefficients(
        reciprocal_vectors, approximate_num_terms
    )
    return Expansion(basis_coefficients=coeffs)


def min_array_shape_for_expansion(expansion: Expansion) -> Tuple[int, int]:
    """Return the minimal grid dimensions for the expansion."""
    coeffs = expansion.basis_coefficients
    if coeffs.numel() == 0:
        return 1, 1
    max_u = int(coeffs[:, 0].abs().max().item())
    max_v = int(coeffs[:, 1].abs().max().item())
    return 2 * max_u + 1, 2 * max_v + 1


def _absolute_axes(axes: Iterable[int], ndim: int) -> Tuple[int, ...]:
    """Convert possibly negative axes to absolute positions."""
    result = []
    for axis in axes:
        if axis not in range(-ndim, ndim):
            raise ValueError(f"Axis {axis} out of bounds for ndim={ndim}")
        absolute = axis % ndim
        if absolute in result:
            raise ValueError(f"Duplicate axis {axis}")
        result.append(absolute)
    return tuple(result)


def _fftshift_phase(length: int, sign: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Generate phase factors for centred FFT coordinates."""
    freq = torch.fft.fftfreq(length, device=device)
    phase = torch.exp(sign * 1j * 2.0 * math.pi * 0.5 * freq)
    return phase.to(dtype=_complex_dtype(dtype))


def _fft2(
    x: torch.Tensor,
    axes: Tuple[int, int],
    norm: str,
    centered_coordinates: bool,
) -> torch.Tensor:
    """Apply a 2D FFT with optional centred coordinates."""
    y = torch.fft.fftn(x, dim=axes, norm=norm)
    if centered_coordinates:
        ax0, ax1 = axes
        phase0 = _fftshift_phase(x.shape[ax0], sign=-1, dtype=x.dtype, device=x.device)
        phase1 = _fftshift_phase(x.shape[ax1], sign=-1, dtype=x.dtype, device=x.device)
        reshape0 = [1] * x.ndim
        reshape1 = [1] * x.ndim
        reshape0[ax0] = phase0.numel()
        reshape1[ax1] = phase1.numel()
        y = y * phase0.view(reshape0) * phase1.view(reshape1)
    return y


def _ifft2(
    x: torch.Tensor,
    axes: Tuple[int, int],
    norm: str,
    centered_coordinates: bool,
) -> torch.Tensor:
    """Apply a 2D inverse FFT with optional centred coordinates."""
    y = x
    if centered_coordinates:
        ax0, ax1 = axes
        phase0 = _fftshift_phase(x.shape[ax0], sign=1, dtype=x.dtype, device=x.device)
        phase1 = _fftshift_phase(x.shape[ax1], sign=1, dtype=x.dtype, device=x.device)
        reshape0 = [1] * x.ndim
        reshape1 = [1] * x.ndim
        reshape0[ax0] = phase0.numel()
        reshape1[ax1] = phase1.numel()
        y = y * phase0.view(reshape0) * phase1.view(reshape1)
    return torch.fft.ifftn(y, dim=axes, norm=norm)


def fft(
    x: torch.Tensor,
    expansion: Expansion,
    axes: Tuple[int, int] = (-2, -1),
    centered_coordinates: bool = True,
) -> torch.Tensor:
    """Project ``x`` onto the retained Fourier orders."""
    axes = _absolute_axes(axes, x.ndim)
    min_shape = min_array_shape_for_expansion(expansion)
    if x.shape[axes[0]] < min_shape[0] or x.shape[axes[1]] < min_shape[1]:
        raise ValueError("Input shape is insufficient for the expansion.")
    transformed = _fft2(x, axes=axes, norm="forward", centered_coordinates=centered_coordinates)
    coeffs = expansion.basis_coefficients.to(device=x.device)
    select_u = coeffs[:, 0]
    select_v = coeffs[:, 1]
    leading = [slice(None)] * axes[0]
    trailing = [slice(None)] * (x.ndim - axes[1] - 1)
    indexed = transformed[tuple(leading + [select_u, select_v] + trailing)]
    return indexed


def ifft(
    y: torch.Tensor,
    expansion: Expansion,
    shape: Tuple[int, int],
    axis: int = -1,
    centered_coordinates: bool = True,
) -> torch.Tensor:
    """Reconstruct ``y`` onto a spatial grid."""
    (axis,) = _absolute_axes((axis,), y.ndim)
    if y.shape[axis] != expansion.num_terms:
        raise ValueError("Fourier axis does not match number of expansion terms.")
    output_shape = y.shape[:axis] + shape + y.shape[axis + 1 :]
    output = torch.zeros(output_shape, dtype=y.dtype, device=y.device)
    coeffs = expansion.basis_coefficients.to(device=output.device)
    select_u = coeffs[:, 0]
    select_v = coeffs[:, 1]
    leading = [slice(None)] * axis
    trailing = [slice(None)] * (y.ndim - axis - 1)
    output[tuple(leading + [select_u, select_v] + trailing)] = y
    axes = (axis, axis + 1)
    return _ifft2(output, axes=axes, norm="forward", centered_coordinates=centered_coordinates)


def _field_magnitude(field: torch.Tensor) -> torch.Tensor:
    """Compute the per-pixel magnitude."""
    magnitude_sq = torch.sum(torch.abs(field) ** 2, dim=-1, keepdim=True)
    base_dtype = field.real.dtype if field.is_complex() else field.dtype
    floor = torch.tensor(
        _magnitude_floor(base_dtype),
        dtype=magnitude_sq.dtype,
        device=field.device,
    )
    magnitude_sq = torch.clamp(magnitude_sq, min=floor)
    return torch.sqrt(magnitude_sq)


def _max_field_magnitude(field: torch.Tensor) -> torch.Tensor:
    """Return the global maximum magnitude."""
    return torch.amax(_field_magnitude(field), dim=(-3, -2), keepdim=True)


def _normalize(field: torch.Tensor) -> torch.Tensor:
    """Normalize a field by its global maximum."""
    max_magnitude = _max_field_magnitude(field)
    dtype = max_magnitude.dtype
    threshold = torch.tensor(
        math.sqrt(_magnitude_floor(dtype)),
        dtype=dtype,
        device=field.device,
    )
    max_magnitude = torch.clamp(max_magnitude, min=threshold)
    return field / max_magnitude


def _normalize_elementwise(field: torch.Tensor) -> torch.Tensor:
    """Normalize each pixel independently."""
    magnitude = _field_magnitude(field)
    threshold = torch.tensor(
        math.sqrt(_magnitude_floor(magnitude.dtype)),
        dtype=magnitude.dtype,
        device=field.device,
    )
    magnitude = torch.clamp(magnitude, min=threshold)
    return field / magnitude


def _angle(x: torch.Tensor) -> torch.Tensor:
    """Compute a stable angle for complex values."""
    abs_x = torch.abs(x)
    zero = torch.zeros(1, dtype=abs_x.dtype, device=x.device)
    is_small = torch.isclose(abs_x, zero, atol=1e-12)
    safe = torch.where(is_small, torch.ones_like(x), x)
    return torch.angle(safe)


def _normalize_jones(field: torch.Tensor) -> torch.Tensor:
    """Convert the field into Jones vectors."""
    field = _normalize(field)
    magnitude = _field_magnitude(field)
    zero = torch.zeros(1, dtype=magnitude.dtype, device=field.device)
    ones = torch.ones_like(magnitude)
    magnitude_near_zero = torch.isclose(magnitude, zero, atol=1e-12)
    magnitude_safe = torch.where(magnitude_near_zero, ones, magnitude)

    fallback = torch.full_like(field[..., :1], 1.0 / math.sqrt(2.0))
    tx_norm = torch.where(magnitude_near_zero, fallback, field[..., :1] / magnitude_safe)
    ty_norm = torch.where(magnitude_near_zero, fallback, field[..., 1:] / magnitude_safe)

    phi = (math.pi / 8.0) * (1.0 + torch.cos(math.pi * magnitude))
    theta = _angle(tx_norm + 1j * ty_norm)
    phase = torch.exp(1j * theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    jx = phase * (tx_norm * cos_phi - ty_norm * 1j * sin_phi)
    jy = phase * (ty_norm * cos_phi + tx_norm * 1j * sin_phi)
    concatenated = torch.cat([jx, jy], dim=-1)
    complex_magnitude = torch.sqrt(torch.abs(concatenated[..., 0]) ** 2 + torch.abs(concatenated[..., 1]) ** 2)
    zero_c = torch.zeros(1, dtype=complex_magnitude.dtype, device=field.device)
    complex_magnitude = torch.where(
        torch.isclose(complex_magnitude, zero_c, atol=1e-12),
        torch.ones_like(complex_magnitude),
        complex_magnitude,
    )
    return concatenated / complex_magnitude.unsqueeze(-1)


def _field_at_max_magnitude(field: torch.Tensor) -> torch.Tensor:
    """Return the sample with maximum magnitude."""
    magnitude = _field_magnitude(field).reshape(-1)
    idx = torch.argmax(magnitude)
    flat = field.reshape(-1, 2)
    return flat[idx]


def _is_1d_field(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detect quasi-1D masks and return the reference angle."""
    ref_field = _field_at_max_magnitude(field)
    ref_angle = _angle(torch.abs(ref_field[0]) + 1j * torch.abs(ref_field[1]))
    angle = _angle(torch.abs(field[..., 0]) + 1j * torch.abs(field[..., 1]))
    magnitude = _field_magnitude(field).squeeze(-1)
    tol = 1e-2

    zero = torch.zeros(1, dtype=magnitude.dtype, device=field.device)
    mask = torch.isclose(magnitude, zero, atol=1e-12)
    for delta in (-2, -1, 0, 1, 2):
        mask = mask | torch.isclose(angle, ref_angle + delta * math.pi, atol=tol)
    is_1d = torch.all(mask)
    is_1d = is_1d | (field.shape[0] == 1) | (field.shape[1] == 1)
    return is_1d, ref_angle


def _periodic_forward_difference(field: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute the periodic forward difference along ``axis``."""
    diff = torch.roll(field, shifts=-1, dims=axis) - field
    return diff * field.shape[axis]


def _transform_gradient(
    partial_grad: torch.Tensor,
    basis_vectors: torch.Tensor,
    *,
    inverse_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the coordinate transform defined by the basis vectors."""
    basis = basis_vectors.to(partial_grad.dtype)
    if inverse_metric is None:
        metric = basis.transpose(-1, -2) @ basis
        inverse_metric = torch.linalg.inv(metric)
    else:
        inverse_metric = inverse_metric.to(partial_grad.dtype)
    return partial_grad @ inverse_metric @ basis.transpose(-1, -2)


def _scalar_field_forward_difference_gradient(
    field: torch.Tensor,
    basis_vectors: torch.Tensor,
    inverse_metric: torch.Tensor,
) -> torch.Tensor:
    """Return the scalar-field gradient using periodic differences."""
    dimension = basis_vectors.shape[-1]
    axes = range(field.ndim - dimension, field.ndim)
    diffs = [_periodic_forward_difference(field, axis=ax) for ax in axes]
    partial_grad = torch.stack(diffs, dim=-1)
    return _transform_gradient(partial_grad, basis_vectors, inverse_metric=inverse_metric)


def _vector_field_forward_difference_gradient(
    field: torch.Tensor,
    primitive_lattice_vectors: LatticeVectors,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for both vector components."""
    basis_vectors = primitive_lattice_vectors.normalized_basis(
        device=field.device,
        dtype=field.dtype,
    )
    inverse_metric = primitive_lattice_vectors.inverse_metric(
        device=field.device,
        dtype=field.dtype,
    )
    return (
        _scalar_field_forward_difference_gradient(field[..., 0], basis_vectors, inverse_metric),
        _scalar_field_forward_difference_gradient(field[..., 1], basis_vectors, inverse_metric),
    )


def _smoothness_loss(
    field: torch.Tensor,
    primitive_lattice_vectors: LatticeVectors,
) -> torch.Tensor:
    """Compute the tangent-field smoothness penalty."""
    grads = _vector_field_forward_difference_gradient(field, primitive_lattice_vectors)
    stacked = torch.stack([torch.abs(grads[0]) ** 2, torch.abs(grads[1]) ** 2], dim=0)
    return stacked.mean()


class ConjugateGradientError(RuntimeError):
    """Raised when the conjugate-gradient solver fails to converge."""


def _solve_sym_pos_linear_system(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    tol: float | None = None,
    max_iter: int | None = None,
    damping: float = 0.0,
) -> torch.Tensor:
    """Solve ``A x = rhs`` using conjugate gradients with matvec callback."""
    if tol is None:
        tol = float(rhs.norm()) * 1e-6
    if max_iter is None:
        max_iter = max(int(rhs.shape[0]), 10)
    damping = float(damping)

    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)
    if rs_old.sqrt() <= tol:
        return x

    eps = torch.finfo(rhs.dtype).eps
    converged = False
    restarts = 0
    max_restarts = 5
    for _ in range(max_iter):
        Ap = matvec(p)
        if not torch.isfinite(Ap).all():
            raise ConjugateGradientError("Conjugate gradient produced non-finite matvec result.")
        if damping:
            Ap = Ap + damping * p
        denom = torch.dot(p, Ap)
        if not torch.isfinite(denom):
            raise ConjugateGradientError("Conjugate gradient produced non-finite curvature estimate.")
        if torch.abs(denom) < eps:
            if restarts < max_restarts:
                p = r.clone()
                restarts += 1
                continue
            if rs_old <= tol * tol:
                converged = True
                break
            raise ConjugateGradientError("Conjugate gradient encountered near-zero curvature.")
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if rs_new.sqrt() <= tol:
            converged = True
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
        if not torch.isfinite(x).all() or not torch.isfinite(r).all() or not torch.isfinite(p).all():
            raise ConjugateGradientError("Conjugate gradient iterate became non-finite.")
    if not converged:
        residual = matvec(x) - rhs
        if damping:
            residual = residual + damping * x
        if residual.norm() > tol:
            raise ConjugateGradientError(
                f"Conjugate gradient failed to converge within {max_iter} iterations."
            )
    return x


def _compute_gradient(
    arr: torch.Tensor,
    primitive_lattice_vectors: LatticeVectors,
) -> torch.Tensor:
    """Return gradients for the indicator mask."""
    basis_vectors = primitive_lattice_vectors.normalized_basis(
        device=arr.device,
        dtype=arr.dtype,
    )
    inverse_metric = primitive_lattice_vectors.inverse_metric(
        device=arr.device,
        dtype=arr.dtype,
    )
    grad_u = (torch.roll(arr, shifts=-1, dims=-2) - torch.roll(arr, shifts=1, dims=-2)) * (arr.shape[-2] / 2.0)
    grad_v = (torch.roll(arr, shifts=-1, dims=-1) - torch.roll(arr, shifts=1, dims=-1)) * (arr.shape[-1] / 2.0)
    partial_grad = torch.stack([grad_u, grad_v], dim=-1)
    return _transform_gradient(partial_grad, basis_vectors, inverse_metric=inverse_metric)


def _alignment_loss(
    field: torch.Tensor,
    target_field: torch.Tensor,
    elementwise_alignment_loss_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute the least-squares alignment loss."""
    elementwise_loss = torch.sum(
        torch.abs(field - target_field) ** 2, dim=-1, keepdim=True
    )
    return (elementwise_alignment_loss_weight * elementwise_loss).mean()


def _transverse_wavevectors(
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
) -> torch.Tensor:
    """Compute transverse wavevectors for the expansion."""
    reciprocal = primitive_lattice_vectors.reciprocal()
    dtype = primitive_lattice_vectors.u.dtype
    coeffs = expansion.basis_coefficients.to(dtype=dtype, device=primitive_lattice_vectors.u.device)
    kx = 2 * math.pi * (
        coeffs[:, 0] * reciprocal.u[0] + coeffs[:, 1] * reciprocal.v[0]
    )
    ky = 2 * math.pi * (
        coeffs[:, 0] * reciprocal.u[1] + coeffs[:, 1] * reciprocal.v[1]
    )
    return torch.stack([kx, ky], dim=-1)


def _fourier_loss(
    fourier_field: torch.Tensor,
    fourier_penalty_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute the Fourier-domain penalty proportional to |k_t|^2."""
    weights = fourier_penalty_weights.to(
        dtype=torch.abs(fourier_field).dtype,
        device=fourier_field.device,
    )
    return torch.sum(torch.abs(fourier_field) ** 2 * weights)


def _fourier_penalty_weights(
    primitive_lattice_vectors: LatticeVectors,
    expansion: Expansion,
) -> torch.Tensor:
    """Precompute transverse |k_t|^2 weights for the Fourier loss."""
    transverse_wavevectors = _transverse_wavevectors(
        primitive_lattice_vectors,
        expansion,
    )
    basis_vectors = torch.stack(
        [primitive_lattice_vectors.u, primitive_lattice_vectors.v],
        dim=-1,
    )
    area = torch.abs(torch.linalg.det(basis_vectors))
    kt_squared = torch.linalg.norm(transverse_wavevectors, dim=-1) ** 2 * area
    return kt_squared.unsqueeze(-1)


def _field_loss(
    fourier_field: torch.Tensor,
    expansion: Expansion,
    primitive_lattice_vectors: LatticeVectors,
    fourier_penalty_weights: torch.Tensor,
    target_field: torch.Tensor,
    elementwise_alignment_loss_weight: torch.Tensor,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> torch.Tensor:
    """Total loss combining alignment, Fourier, and smoothness terms."""
    spatial_shape = target_field.shape[-3:-1]
    field = ifft(
        fourier_field,
        expansion=expansion,
        shape=spatial_shape,
        axis=-2,
        centered_coordinates=False,
    )
    loss = _alignment_loss(field, target_field, elementwise_alignment_loss_weight)
    if fourier_loss_weight > 0:
        loss = loss + fourier_loss_weight * _fourier_loss(
            fourier_field,
            fourier_penalty_weights,
        )
    if smoothness_loss_weight > 0:
        loss = loss + smoothness_loss_weight * _smoothness_loss(field, primitive_lattice_vectors)
    return loss.real


def _complex_to_real(z: torch.Tensor) -> torch.Tensor:
    """Flatten a complex tensor into a real vector."""
    return torch.view_as_real(z).reshape(-1)


def _real_to_complex(vec: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Convert a flattened real vector back to complex form."""
    return torch.view_as_complex(vec.reshape(shape + (2,)))


def _field_loss_value_jac_and_hessian(
    flat_real: torch.Tensor,
    expansion: Expansion,
    primitive_lattice_vectors: LatticeVectors,
    fourier_penalty_weights: torch.Tensor,
    target_field: torch.Tensor,
    elementwise_alignment_loss_weight: torch.Tensor,
    fourier_loss_weight: float,
    smoothness_loss_weight: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Evaluate loss, gradient, and Hessian-vector product callback."""
    field_shape = (expansion.num_terms, 2)

    def loss_from_real(real_vec: torch.Tensor) -> torch.Tensor:
        fourier_field = _real_to_complex(real_vec, field_shape)
        fourier_field = fourier_field.reshape(expansion.num_terms, 2)
        return _field_loss(
            fourier_field,
            expansion=expansion,
            primitive_lattice_vectors=primitive_lattice_vectors,
            fourier_penalty_weights=fourier_penalty_weights,
            target_field=target_field,
            elementwise_alignment_loss_weight=elementwise_alignment_loss_weight,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
        )

    loss_value = loss_from_real(flat_real)
    jac = torch.autograd.grad(loss_value, flat_real, create_graph=True)[0]

    def hessian_matvec(vec: torch.Tensor) -> torch.Tensor:
        product = torch.dot(jac, vec)
        hvp = torch.autograd.grad(product, flat_real, retain_graph=True)[0]
        if not torch.isfinite(hvp).all():
            raise ConjugateGradientError("Hessian matvec produced non-finite result.")
        return hvp

    return loss_value, jac, hessian_matvec


class TangentFieldGenerator:
    """Generate tangent vector fields for patterned masks using Fourier expansions."""

    def __init__(
        self,
        lattice_t1: torch.Tensor,
        lattice_t2: torch.Tensor,
        kdim: Tuple[int, int],
        *,
        fourier_loss_weight: float = 1e-2,
        smoothness_loss_weight: float = 1e-3,
        steps: int = 1,
    ) -> None:
        self._lattice_vectors = LatticeVectors(u=lattice_t1, v=lattice_t2)
        approximate_terms = int(kdim[0] * kdim[1])
        self._expansion_manager = FourierExpansionManager(
            self._lattice_vectors,
            approximate_terms,
        )
        self._default_fourier_loss_weight = float(fourier_loss_weight)
        self._default_smoothness_loss_weight = float(smoothness_loss_weight)
        self._default_steps = int(steps)

    @property
    def lattice_vectors(self) -> LatticeVectors:
        return self._expansion_manager.primitive_lattice_vectors

    @property
    def expansion(self) -> Expansion:
        return self._expansion_manager.expansion

    def compute(
        self,
        mask: torch.Tensor,
        XO: torch.Tensor,
        YO: torch.Tensor,
        *,
        scheme: str = "POL",
        fourier_loss_weight: float | None = None,
        smoothness_loss_weight: float | None = None,
        steps: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute tangent fields for the requested scheme."""
        # `_calculate_vector_field` in `solver.py` squeezes the mask the solver stores
        # (possibly batched as `(N, H, W)` for dispersive stacks or vectorized inverse-
        # design loops) and feeds each slice into this method, which always expects a
        # single 2D mask. Keeping the contract here simple avoids pushing batch logic
        # into the tangent generator while letting the solver reuse the cached instance.
        self._validate_inputs(mask, XO, YO)
        self._ensure_device_dtype(mask)

        resolved_fourier = (
            float(fourier_loss_weight)
            if fourier_loss_weight is not None
            else self._default_fourier_loss_weight
        )
        resolved_smoothness = (
            float(smoothness_loss_weight)
            if smoothness_loss_weight is not None
            else self._default_smoothness_loss_weight
        )
        resolved_steps = int(steps) if steps is not None else self._default_steps

        dispatch = self._scheme_dispatch(
            fourier_loss_weight=resolved_fourier,
            smoothness_loss_weight=resolved_smoothness,
            steps=resolved_steps,
        )
        scheme_upper = scheme.upper()
        if scheme_upper not in dispatch:
            raise ValueError(f"Unsupported scheme '{scheme}'. Expected one of {list(dispatch)}.")
        return dispatch[scheme_upper](mask)

    def _validate_inputs(
        self,
        mask: torch.Tensor,
        XO: torch.Tensor,
        YO: torch.Tensor,
    ) -> None:
        """Validate geometry and coordinate shapes."""
        if mask.ndim != 2:
            raise ValueError(f"`mask` must be 2D, got shape {mask.shape}")
        if mask.shape != XO.shape or mask.shape != YO.shape:
            raise ValueError("`mask`, `XO`, and `YO` must share the same 2D shape.")

    def _scheme_dispatch(
        self,
        *,
        fourier_loss_weight: float,
        smoothness_loss_weight: float,
        steps: int,
    ) -> Dict[str, Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """Return scheme-specific callables."""
        def _pol(arr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            field = self._compute_field(
                arr,
                use_jones_direct=False,
                fourier_loss_weight=fourier_loss_weight,
                smoothness_loss_weight=smoothness_loss_weight,
                steps=steps,
            )
            return field[..., 0], field[..., 1]

        def _normal(arr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            field = self._compute_field(
                arr,
                use_jones_direct=False,
                fourier_loss_weight=fourier_loss_weight,
                smoothness_loss_weight=smoothness_loss_weight,
                steps=steps,
            )
            normalized = _normalize_elementwise(field)
            return normalized[..., 0], normalized[..., 1]

        def _jones(arr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            field = self._compute_field(
                arr,
                use_jones_direct=False,
                fourier_loss_weight=fourier_loss_weight,
                smoothness_loss_weight=smoothness_loss_weight,
                steps=steps,
            )
            jones = _normalize_jones(field)
            return jones[..., 0], jones[..., 1]

        def _jones_direct(arr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            field = self._compute_field(
                arr,
                use_jones_direct=True,
                fourier_loss_weight=fourier_loss_weight,
                smoothness_loss_weight=smoothness_loss_weight,
                steps=steps,
            )
            return field[..., 0], field[..., 1]

        return {
            "POL": _pol,
            "NORMAL": _normal,
            "JONES": _jones,
            "JONES_DIRECT": _jones_direct,
        }

    def _ensure_device_dtype(self, mask: torch.Tensor) -> None:
        """Align internal tensors with the mask device and dtype."""
        if not mask.is_floating_point():
            raise TypeError("`mask` must be a floating-point tensor.")
        self._expansion_manager.adapt_to(device=mask.device, dtype=mask.dtype)

    def _filter_and_adjust_resolution(self, component: torch.Tensor) -> torch.Tensor:
        """Filter and reconstruct a gradient component."""
        transformed = self._expansion_manager.project(
            component,
            axes=(-2, -1),
            centered_coordinates=False,
        )
        min_shape = self._expansion_manager.min_array_shape()
        target_shape = (
            min_shape[0] * (2 if component.shape[-2] > 1 else 1),
            min_shape[1] * (2 if component.shape[-1] > 1 else 1),
        )
        return self._expansion_manager.reconstruct(
            transformed,
            shape=target_shape,
            axis=-1,
            centered_coordinates=False,
        )

    def _compute_field(
        self,
        arr: torch.Tensor,
        *,
        use_jones_direct: bool,
        fourier_loss_weight: float,
        smoothness_loss_weight: float,
        steps: int,
    ) -> torch.Tensor:
        """Execute the Newton solve for a tangent field."""
        expansion = self.expansion
        primitive_lattice_vectors = self.lattice_vectors
        fourier_penalty_weights = self._expansion_manager.fourier_penalty_weights()

        fourier_loss_weight = fourier_loss_weight / expansion.num_terms
        smoothness_loss_weight = smoothness_loss_weight / expansion.num_terms

        grad = _compute_gradient(arr, primitive_lattice_vectors)
        gx = self._filter_and_adjust_resolution(grad[..., 0])
        gy = self._filter_and_adjust_resolution(grad[..., 1])

        grad = torch.stack([gx, gy], dim=-1)
        grad = _normalize(grad)
        elementwise_alignment_weight = _field_magnitude(grad)

        is_1d, grad_angle = _is_1d_field(grad)
        dummy_grad = torch.ones_like(grad)
        grad = torch.where(is_1d, dummy_grad, grad)

        target_field = torch.stack([grad[..., 1], -grad[..., 0]], dim=-1)
        target_field = _normalize_elementwise(target_field)
        if use_jones_direct:
            target_field = _normalize_jones(target_field)
            initial_field = target_field
        else:
            initial_field = _normalize(torch.stack([grad[..., 1], -grad[..., 0]], dim=-1))

        fourier_field = fft(
            initial_field,
            expansion=expansion,
            axes=(-3, -2),
            centered_coordinates=False,
        )
        flat_real = _complex_to_real(fourier_field.reshape(-1))
        flat_real_current = flat_real.clone().detach().requires_grad_(True)

        # Rather than forming the dense Hessian (which scales quadratically in the
        # number of harmonics), we solve each Newton step with a conjugate-gradient
        # loop that only queries Hessian–vector products. Those products are obtained
        # via autograd's `hvp`, keeping both memory usage and runtime manageable for
        # the large grids used when `is_use_fff=True`.
        for iteration in range(max(steps, 1)):
            (
                _,
                jac,
                hessian_matvec,
            ) = _field_loss_value_jac_and_hessian(
                flat_real_current,
                expansion=expansion,
                primitive_lattice_vectors=primitive_lattice_vectors,
                fourier_penalty_weights=fourier_penalty_weights,
                target_field=target_field,
                elementwise_alignment_loss_weight=elementwise_alignment_weight,
                fourier_loss_weight=fourier_loss_weight,
                smoothness_loss_weight=smoothness_loss_weight,
            )
            try:
                delta = _solve_sym_pos_linear_system(
                    hessian_matvec,
                    jac,
                    tol=max(
                        float(jac.norm()) * 1e-8,
                        torch.finfo(jac.dtype).eps,
                    ),
                    max_iter=max(jac.numel() * 2, 10),
                    damping=1e-6,
                )
                if not torch.isfinite(delta).all():
                    raise ConjugateGradientError("Conjugate gradient produced non-finite update.")
            except ConjugateGradientError:
                basis = torch.eye(jac.numel(), dtype=jac.dtype, device=jac.device)
                hessian_cols = []
                for column in basis.split(1, dim=1):
                    hessian_cols.append(hessian_matvec(column.squeeze(1)))
                hessian = torch.stack(hessian_cols, dim=1)
                identity = torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)
                try:
                    delta = torch.linalg.solve(hessian, jac)
                except (RuntimeError, torch.linalg.LinAlgError):
                    regularized = (hessian + 1e-8 * identity).clone().contiguous()
                    try:
                        delta = torch.linalg.solve(regularized, jac)
                    except (RuntimeError, torch.linalg.LinAlgError):
                        pseudo_inverse = torch.linalg.pinv(regularized, rcond=1e-12)
                        delta = pseudo_inverse @ jac

                if not torch.isfinite(delta).all():
                    regularized = (hessian + 1e-6 * identity).clone().contiguous()
                    pseudo_inverse = torch.linalg.pinv(regularized, rcond=1e-12)
                    delta = pseudo_inverse @ jac

                if not torch.isfinite(delta).all():
                    raise ConjugateGradientError("Dense fallback produced non-finite update.")
            next_real = flat_real_current - delta
            if iteration < max(steps, 1) - 1:
                flat_real_current = next_real.detach().requires_grad_(True)
            else:
                flat_real_current = next_real

        fourier_field = _real_to_complex(flat_real_current, (expansion.num_terms, 2))
        field = ifft(
            fourier_field,
            expansion=expansion,
            shape=arr.shape,
            axis=-2,
            centered_coordinates=False,
        )

        field_1d = torch.stack(
            [
                torch.sin(grad_angle),
                torch.cos(grad_angle),
            ],
            dim=-1,
        ).to(field.dtype)
        field_1d = field_1d.view(1, 1, 2)
        field = torch.where(is_1d, field_1d, field)
        return _normalize(field)


def compute_tangent_field(
    mask: torch.Tensor,
    XO: torch.Tensor,
    YO: torch.Tensor,
    lattice_t1: torch.Tensor,
    lattice_t2: torch.Tensor,
    kdim: Tuple[int, int],
    *,
    scheme: str = "POL",
    fourier_loss_weight: float = 1e-2,
    smoothness_loss_weight: float = 1e-3,
    steps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute tangent fields via ``TangentFieldGenerator``."""
    generator = TangentFieldGenerator(
        lattice_t1=lattice_t1,
        lattice_t2=lattice_t2,
        kdim=kdim,
        fourier_loss_weight=fourier_loss_weight,
        smoothness_loss_weight=smoothness_loss_weight,
        steps=steps,
    )
    return generator.compute(
        mask=mask,
        XO=XO,
        YO=YO,
        scheme=scheme,
    )
