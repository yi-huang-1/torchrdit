"""Numerical stability utilities for electromagnetic solvers.

Provides dtype-aware, differentiable protection functions for common
numerical hazards: near-zero division, eigenvalue degeneracy, and
ill-conditioned linear systems.

All functions preserve autograd gradient flow and are safe for use in
inverse-design optimization loops.

Typical usage from solver code::

    from .numerics import safe_kz_reciprocal, safe_sqrt_reciprocal

    inv_kz = safe_kz_reciprocal(mat_kz, dtype=self.tcomplex)
    inv_sqrt_lam = safe_sqrt_reciprocal(eigenvalues, dtype=dtype)
"""

import warnings

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dtype-aware constants
# ---------------------------------------------------------------------------

#: Safe upper bounds on |Re(arg)| fed to matrix_exp, preventing Inf.
#: Derived from ln(finfo.max) with safety margin.
#: float32: exp(88) ≈ 1.6e38 (max ~3.4e38), float64: exp(709) ≈ 8.2e307.
EXP_ARG_MAX_F32: float = 80.0
EXP_ARG_MAX_F64: float = 500.0


def eps_for_dtype(dtype: torch.dtype) -> float:
    """Return a safe magnitude floor appropriate for *dtype*.

    The value is chosen so that ``1/eps`` is well within the dtype's
    representable range and small enough not to disturb physically
    meaningful values (e.g., kz ≈ 1e-4 at grazing incidence).

    Args:
        dtype: A real or complex torch dtype.

    Returns:
        ``1e-6`` for 32-bit types, ``1e-12`` for 64-bit types.
    """
    if dtype in (torch.float32, torch.complex64):
        return 1e-6
    return 1e-12  # float64 / complex128


# ---------------------------------------------------------------------------
# Differentiable primitives
# ---------------------------------------------------------------------------

def softplus_floor(x: torch.Tensor, min_val: float, beta: float = 100.0) -> torch.Tensor:
    """Differentiable lower bound: smooth approximation of ``max(x, min_val)``.

    Uses the identity ``max(x, m) ≈ x + softplus(m - x)`` which is
    exact in the limit ``beta → ∞``.

    Args:
        x: Real-valued tensor.
        min_val: Lower-bound threshold.
        beta: Sharpness of the softplus transition (higher = sharper).

    Returns:
        Tensor with all values ≥ *min_val* (up to softplus smoothing).
    """
    return x + F.softplus(min_val - x, beta=beta)


def softplus_magnitude_floor(
    z: torch.Tensor, min_magnitude: float, beta: float = 100.0
) -> torch.Tensor:
    """Differentiable magnitude floor for complex (or real) tensors.

    Pushes ``|z|`` away from zero while preserving the phase of *z*.
    When ``|z| >> min_magnitude`` the output equals *z*; when
    ``|z| → 0`` the output magnitude is smoothly clamped to
    *min_magnitude*.

    For purely imaginary inputs (e.g., evanescent kz modes) the
    protection is applied along the imaginary axis, not the real axis.

    Args:
        z: Complex- or real-valued tensor.
        min_magnitude: Minimum allowed magnitude.
        beta: Sharpness of the softplus transition.

    Returns:
        Protected tensor with ``|result| >= min_magnitude``.
    """
    z_abs = torch.abs(z)
    protection = F.softplus(min_magnitude - z_abs, beta=beta)
    # Unit-phase direction: z/|z| preserves phase for values well above
    # machine noise; fallback to +1 (real axis) for truly tiny values
    # where phase is numerically meaningless.
    phase_threshold = min_magnitude * 0.01
    direction = torch.where(
        z_abs > phase_threshold,
        z / (z_abs + 1e-30),
        torch.ones_like(z),
    )
    return z + protection * direction


# ---------------------------------------------------------------------------
# Physics-specific safe operations
# ---------------------------------------------------------------------------

def safe_kz_reciprocal(kz: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Compute ``1 / (1j * kz)`` with differentiable magnitude protection.

    This is the standard reciprocal that appears in every V-matrix
    construction (reflection region, transmission region, and layer
    interiors).  At Brewster angle, Wood anomalies, or grazing
    incidence certain kz components approach zero; the softplus floor
    prevents divergence while keeping the operation differentiable.

    Uses a **hybrid relative + absolute floor** per (source, frequency)
    block so that protection adapts to problem scale:

    - ``floor = max(eps_abs, eps_rel * max|Re(kz)|)`` computed over the
      harmonics dimension (last axis).
    - The magnitude protection is applied in normalized coordinates
      ``x = |kz| / floor`` using a fixed-shape softplus, so the transition
      band remains ``O(floor)`` without a dtype-dependent ``beta`` cap.

    This avoids the bias of a fixed floor on near-grazing propagating
    modes while still preventing NaN/Inf at true singularities.

    Args:
        kz: Complex tensor of kz values, shape ``(..., H)`` where the
            last dimension is harmonics.
        dtype: The complex dtype used by the solver (determines epsilon).

    Returns:
        ``1 / (1j * kz_protected)`` where the protection is near-identity
        for ``|kz| >> floor`` and bounded near ``kz = 0``.
    """
    eps_abs = eps_for_dtype(dtype)
    eps_rel = 1e-4

    # Per-block scale: max |Re(kz)| over harmonics.
    # Using Re(kz) excludes evanescent modes (purely imaginary, |kz| can
    # be huge) which would inflate the floor and bias near-grazing
    # propagating modes in the same block.  When all modes are evanescent
    # (Re(kz) ≈ 0), scale → 0 and we fall back to eps_abs.
    scale = kz.real.abs().amax(dim=-1, keepdim=True)

    # Hybrid floor: absolute minimum + relative scaling
    floor = torch.clamp_min(eps_rel * scale, eps_abs)

    # Normalized softplus protection in x = |kz| / floor coordinates.
    # Keeping alpha fixed makes the transition width proportional to floor
    # without needing to clamp beta = alpha / floor in fp64.
    kz_abs = torch.abs(kz)
    alpha = 5.0
    x = kz_abs / floor
    u = alpha * (1.0 - x)
    # Clamp the dimensionless coordinate for stable log1p(exp(...)).
    u_safe = torch.clamp(u, min=-20.0, max=20.0)
    protection = torch.where(
        u > 20.0,
        floor * (1.0 - x),  # deep protection zone: softplus ≈ identity
        floor * (torch.log1p(torch.exp(u_safe)) / alpha),
    )

    # Phase direction: preserve phase of kz for non-tiny values
    phase_threshold = floor * 0.01
    direction = torch.where(
        kz_abs > phase_threshold,
        kz / (kz_abs + 1e-30),
        torch.ones_like(kz),
    )
    kz_safe = kz + protection * direction

    return 1.0 / (1j * kz_safe)


def safe_sqrt_reciprocal(eigenvalues: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Compute ``1 / sqrt(eigenvalues)`` with protection for near-zero values.

    Used in RCWA eigenmode decomposition where the eigenvalues of
    ``P @ Q`` can approach zero for degenerate modes.  Without
    protection this causes NaN/Inf that propagates through the entire
    scattering-matrix calculation.

    Args:
        eigenvalues: Complex tensor of eigenvalues (any shape).
        dtype: The complex dtype used by the solver.

    Returns:
        ``1 / sqrt_protected`` where ``|sqrt(eigenvalues)| >= eps``.
    """
    eps = eps_for_dtype(dtype)
    sqrt_lam = torch.sqrt(eigenvalues)
    sqrt_lam_safe = softplus_magnitude_floor(sqrt_lam, min_magnitude=eps)
    return 1.0 / sqrt_lam_safe


def exp_arg_max_for_dtype(dtype: torch.dtype) -> float:
    """Return a safe exponent clamp for *dtype* (well below overflow)."""
    if dtype in (torch.float32, torch.complex64):
        return EXP_ARG_MAX_F32
    return EXP_ARG_MAX_F64


def clamp_exp_arg(arg: torch.Tensor, max_abs: float | None = None) -> torch.Tensor:
    """Clamp the real part of a complex exponent to prevent overflow.

    ``matrix_exp(diag(arg))`` overflows when ``Re(arg)`` is large.
    This function symmetrically clamps the real part while leaving the
    imaginary part (phase) untouched.

    When *max_abs* is ``None`` the limit is chosen automatically based
    on the tensor's dtype (80 for float32, 500 for float64).

    Args:
        arg: Complex tensor of exponent arguments.
        max_abs: Maximum allowed ``|Re(arg)|``.  If ``None``, inferred
            from *arg.dtype*.

    Returns:
        Tensor with ``Re(result)`` in ``[-max_abs, max_abs]``.
    """
    if max_abs is None:
        max_abs = exp_arg_max_for_dtype(arg.dtype)
    return torch.clamp(arg.real, min=-max_abs, max=max_abs) + 1j * arg.imag


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

def solve_with_retry(
    A: torch.Tensor, b: torch.Tensor, dtype: torch.dtype, max_retries: int = 2
) -> torch.Tensor:
    """Solve ``Ax = b`` with dtype-aware Tikhonov regularization fallback.

    On ``torch.linalg.LinAlgError`` the function adds ``eps * I`` to
    *A* and retries, increasing *eps* by 10x on each attempt.  Raises
    the original error after *max_retries* failures.

    Args:
        A: Coefficient matrix ``(..., N, N)``.
        b: Right-hand side ``(..., N, M)`` or ``(..., N)``.
        dtype: Solver dtype (determines base epsilon).
        max_retries: Maximum number of regularized retries.

    Returns:
        Solution tensor *x* with the same shape as *b*.

    Raises:
        torch.linalg.LinAlgError: If the system remains singular after
            all retries.
    """
    eps_base = eps_for_dtype(dtype)

    for attempt in range(max_retries + 1):
        try:
            return torch.linalg.solve(A, b)
        except torch.linalg.LinAlgError:
            if attempt == max_retries:
                raise
            eps = eps_base * (10 ** attempt)
            warnings.warn(
                f"Singular matrix in solve (attempt {attempt + 1}/{max_retries}), "
                f"adding Tikhonov regularization eps={eps:.1e}",
                stacklevel=2,
            )
            eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
            A = A + eps * eye

    # Unreachable, but keeps the type checker happy.
    return torch.linalg.solve(A, b)
