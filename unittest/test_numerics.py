"""Unit tests for torchrdit.numerics — pure-math, no solver dependency."""

import warnings

import pytest
import torch

from torchrdit.numerics import (
    EXP_ARG_MAX_F32,
    EXP_ARG_MAX_F64,
    clamp_exp_arg,
    eps_for_dtype,
    exp_arg_max_for_dtype,
    safe_kz_reciprocal,
    safe_sqrt_reciprocal,
    softplus_floor,
    softplus_magnitude_floor,
    solve_with_retry,
)


# ---------------------------------------------------------------------------
# eps_for_dtype
# ---------------------------------------------------------------------------

class TestEpsForDtype:
    def test_float32_returns_1e6(self):
        assert eps_for_dtype(torch.float32) == 1e-6

    def test_complex64_returns_1e6(self):
        assert eps_for_dtype(torch.complex64) == 1e-6

    def test_float64_returns_1e12(self):
        assert eps_for_dtype(torch.float64) == 1e-12

    def test_complex128_returns_1e12(self):
        assert eps_for_dtype(torch.complex128) == 1e-12


# ---------------------------------------------------------------------------
# softplus_floor
# ---------------------------------------------------------------------------

class TestSoftplusFloor:
    def test_values_above_min_unchanged(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = softplus_floor(x, min_val=0.5)
        torch.testing.assert_close(result, x, atol=1e-4, rtol=0)

    def test_values_below_min_raised(self):
        x = torch.tensor([0.0, -1.0, -5.0])
        result = softplus_floor(x, min_val=0.5)
        assert (result >= 0.5 - 0.01).all(), f"Got values below floor: {result}"

    def test_gradient_is_finite(self):
        x = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        y = softplus_floor(x, min_val=0.5).sum()
        y.backward()
        assert torch.isfinite(x.grad).all()

    def test_gradient_flows_through(self):
        """Gradient should be non-zero even for values near the floor."""
        x = torch.tensor([0.5], requires_grad=True)  # exactly at min_val
        y = softplus_floor(x, min_val=0.5)
        y.backward()
        assert x.grad is not None and x.grad.abs() > 0


# ---------------------------------------------------------------------------
# softplus_magnitude_floor
# ---------------------------------------------------------------------------

class TestSoftplusMagnitudeFloor:
    def test_large_values_unchanged(self):
        z = torch.tensor([1.0 + 2.0j, -3.0 + 0.5j])
        result = softplus_magnitude_floor(z, min_magnitude=1e-4)
        torch.testing.assert_close(result, z, atol=1e-3, rtol=0)

    def test_near_zero_gets_protected(self):
        z = torch.tensor([1e-10 + 0j, 0.0 + 1e-10j])
        result = softplus_magnitude_floor(z, min_magnitude=1e-4)
        assert (torch.abs(result) >= 1e-4 - 1e-6).all(), (
            f"Magnitude below floor: {torch.abs(result)}"
        )

    def test_exact_zero_gets_protected(self):
        z = torch.tensor([0.0 + 0.0j])
        result = softplus_magnitude_floor(z, min_magnitude=1e-3)
        assert torch.abs(result).item() >= 1e-3 - 1e-5

    def test_preserves_sign_of_real_part(self):
        """Phase preservation: positive real stays positive, negative stays negative.

        Values must be above the phase threshold (min_magnitude * 0.01)
        to trigger phase-preserving direction; below that, fallback is +1.
        """
        # Use values well above phase threshold for min_magnitude=1e-4 → threshold=1e-6
        z_pos = torch.tensor([1e-5 + 0j])
        z_neg = torch.tensor([-1e-5 + 0j])
        r_pos = softplus_magnitude_floor(z_pos, min_magnitude=1e-4)
        r_neg = softplus_magnitude_floor(z_neg, min_magnitude=1e-4)
        assert r_pos.real > 0, f"Expected positive real, got {r_pos}"
        assert r_neg.real < 0, f"Expected negative real, got {r_neg}"

    def test_real_tensor_also_works(self):
        x = torch.tensor([0.0, 1e-8, 1.0])
        result = softplus_magnitude_floor(x, min_magnitude=1e-4)
        assert torch.isfinite(result).all()
        assert (torch.abs(result) >= 1e-4 - 1e-6).all()

    def test_purely_imaginary_preserves_phase(self):
        """Evanescent kz modes are purely imaginary — phase must stay on imaginary axis.

        Values must be above phase_threshold = min_magnitude * 0.01 for
        phase preservation.  Below that, direction falls back to +1 (real axis).
        """
        # min_magnitude=1e-6 → phase_threshold=1e-8
        # Use imaginary values above 1e-8 so phase is preserved
        z = torch.tensor([1e-5j, -1e-5j, 1e-7j])
        result = softplus_magnitude_floor(z, min_magnitude=1e-6)
        for i in range(len(z)):
            r = result[i]
            assert torch.abs(r).item() >= 1e-6 - 1e-8, f"Magnitude too small: {r}"
            # Phase should be preserved: imaginary part dominates
            if z[i].imag > 0:
                assert r.imag > 0, f"Expected positive imag, got {r}"
            else:
                assert r.imag < 0, f"Expected negative imag, got {r}"
            # Real part should be negligible relative to imaginary
            assert abs(r.real) < abs(r.imag) * 0.1, (
                f"Real part too large relative to imag: {r}"
            )

    def test_negative_real_preserves_phase(self):
        """Negative real values above phase threshold stay negative."""
        # min_magnitude=1e-6, phase_threshold=1e-8; use value well above
        z = torch.tensor([-1e-5 + 0j])
        result = softplus_magnitude_floor(z, min_magnitude=1e-6)
        assert result.real < 0, f"Expected negative real, got {result}"

    def test_mixed_phase_preservation(self):
        """Complex values at various phases should keep their direction.

        Uses values above the phase threshold so direction is phase-preserving.
        """
        # min_magnitude=1e-6, phase_threshold=1e-8; use |z|=1e-5 > 1e-8
        angles = [0, torch.pi / 4, torch.pi / 2, torch.pi, -torch.pi / 2]
        for angle in angles:
            z = torch.tensor([1e-5 * torch.exp(torch.tensor(1j * angle))])
            result = softplus_magnitude_floor(z, min_magnitude=1e-6)
            result_angle = torch.angle(result[0]).item()
            input_angle = float(angle)
            angle_diff = abs(result_angle - input_angle)
            angle_diff = min(angle_diff, 2 * torch.pi - angle_diff)
            assert angle_diff < 0.6, (
                f"Phase shifted too much: input={input_angle:.2f}, "
                f"result={result_angle:.2f}, diff={angle_diff:.2f}"
            )


# ---------------------------------------------------------------------------
# safe_kz_reciprocal
# ---------------------------------------------------------------------------

class TestSafeKzReciprocal:
    def test_normal_kz_gives_correct_value(self):
        """For well-behaved kz, result should match 1/(1j*kz)."""
        kz = torch.tensor([0.5 + 0j, 1.0 + 0.1j])
        result = safe_kz_reciprocal(kz, dtype=torch.complex128)
        expected = 1.0 / (1j * kz)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_zero_kz_no_nan(self):
        kz = torch.tensor([0.0 + 0j])
        result = safe_kz_reciprocal(kz, dtype=torch.complex128)
        assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all()

    def test_near_zero_kz_bounded(self):
        kz = torch.tensor([1e-15 + 0j, 0.0 + 1e-15j])
        result = safe_kz_reciprocal(kz, dtype=torch.complex128)
        # Result magnitude should be bounded by ~1/eps
        eps = eps_for_dtype(torch.complex128)
        max_magnitude = 1.0 / eps * 2  # generous bound
        assert (torch.abs(result) <= max_magnitude).all(), (
            f"Result too large: {torch.abs(result)}"
        )

    def test_gradient_finite_at_zero(self):
        kz = torch.tensor([0.0 + 0j], requires_grad=True)
        result = safe_kz_reciprocal(kz, dtype=torch.complex128)
        loss = result.abs().sum()
        loss.backward()
        assert kz.grad is not None and torch.isfinite(kz.grad).all()

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_dtype_awareness(self, dtype):
        """Float32 should use larger epsilon than float64."""
        kz = torch.tensor([1e-5 + 0j], dtype=dtype)
        result = safe_kz_reciprocal(kz, dtype=dtype)
        assert torch.isfinite(result.real).all()

    def test_near_grazing_fresnel_accuracy(self):
        """Protection must not bias kz at 89° — the Fresnel regime.

        At 89° incidence, kz₀ = cos(89°) ≈ 0.01745 is physically valid.
        The protection floor should be far below this value, so the
        reciprocal matches the unprotected analytic result to high precision.
        """
        import math
        theta_deg = 89.0
        cos_theta = math.cos(math.radians(theta_deg))
        # Simulate a (1, 1, H) shaped kz with zeroth order + evanescent modes
        kz = torch.tensor([[[cos_theta + 0j, 0.5 + 0j, 0.0 + 1.5j]]], dtype=torch.complex128)
        result = safe_kz_reciprocal(kz, dtype=torch.complex128)
        expected = 1.0 / (1j * kz)
        # Zeroth-order mode (index 0) must match unprotected value
        torch.testing.assert_close(
            result[0, 0, 0], expected[0, 0, 0], atol=1e-10, rtol=1e-8
        )

    def test_propagating_modes_identity(self):
        """For clearly non-singular propagating modes, protection ≈ identity.

        All kz values here are well above any reasonable floor, so the
        reciprocal must match bare 1/(1j*kz) to machine precision.
        """
        kz = torch.tensor([[0.3 + 0j, 0.7 + 0j, 1.0 + 0j, 0.5 + 0.1j]], dtype=torch.complex128)
        result = safe_kz_reciprocal(kz, dtype=torch.complex128)
        expected = 1.0 / (1j * kz)
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-10)

    def test_large_evanescent_does_not_inflate_floor(self):
        """Evanescent modes with huge |kz| must not bias propagating modes.

        In high-harmonic gratings, evanescent orders can have |kz| ~ 1e3
        (purely imaginary). The scale must use Re(kz) so the floor stays
        proportional to propagating modes, not evanescent ones.
        """
        import math
        cos89 = math.cos(math.radians(89.0))
        # Propagating near-grazing mode + large evanescent modes
        kz = torch.tensor([[cos89 + 0j, 0.5 + 0j, 0.0 + 500j, 0.0 + 1000j]], dtype=torch.complex128)
        result = safe_kz_reciprocal(kz, dtype=torch.complex128)
        expected = 1.0 / (1j * kz)
        # The propagating mode at cos(89°) must match unprotected value
        torch.testing.assert_close(
            result[0, 0], expected[0, 0], atol=1e-10, rtol=1e-8
        )
        # The other propagating mode too
        torch.testing.assert_close(
            result[0, 1], expected[0, 1], atol=1e-12, rtol=1e-10
        )

    @pytest.mark.parametrize("value", [1e-10, 1e-9, 1e-8])
    def test_small_nonzero_fp64_modes_are_near_identity(self, value):
        """Small-but-nonzero fp64 kz values should not be flattened by protection."""
        kz = torch.tensor([value + 0j], dtype=torch.complex128)
        result = safe_kz_reciprocal(kz, dtype=torch.complex128)
        expected = 1.0 / (1j * kz)
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-6)

    def test_wood_anomaly_prevents_nan(self):
        """True kz=0 (Wood anomaly) must not produce NaN/Inf.

        When one or more harmonics have exactly kz=0, the reciprocal
        is protected by the hybrid floor while other modes stay accurate.
        """
        # Mix of zero, near-zero, and normal kz values
        kz = torch.tensor([[0.0 + 0j, 1e-18 + 0j, 0.8 + 0j, 0.0 + 0.5j]], dtype=torch.complex128)
        result = safe_kz_reciprocal(kz, dtype=torch.complex128)
        assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all()
        # The healthy mode (0.8) should still match analytic value
        expected_healthy = 1.0 / (1j * torch.tensor(0.8 + 0j, dtype=torch.complex128))
        torch.testing.assert_close(result[0, 2], expected_healthy, atol=1e-12, rtol=1e-10)


# ---------------------------------------------------------------------------
# safe_sqrt_reciprocal
# ---------------------------------------------------------------------------

class TestSafeSqrtReciprocal:
    def test_normal_eigenvalues(self):
        lam = torch.tensor([1.0 + 0j, 4.0 + 0j])
        result = safe_sqrt_reciprocal(lam, dtype=torch.complex128)
        expected = torch.tensor([1.0 + 0j, 0.5 + 0j])
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_zero_eigenvalue_no_nan(self):
        lam = torch.tensor([0.0 + 0j])
        result = safe_sqrt_reciprocal(lam, dtype=torch.complex128)
        assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all()

    def test_near_zero_eigenvalue_bounded(self):
        lam = torch.tensor([1e-20 + 0j])
        result = safe_sqrt_reciprocal(lam, dtype=torch.complex128)
        eps = eps_for_dtype(torch.complex128)
        max_magnitude = 1.0 / eps * 2
        assert torch.abs(result).item() <= max_magnitude

    def test_negative_eigenvalue_complex_result(self):
        """Negative eigenvalues produce imaginary sqrt — should still be protected."""
        lam = torch.tensor([-1.0 + 0j])
        result = safe_sqrt_reciprocal(lam, dtype=torch.complex128)
        assert torch.isfinite(result.real).all() and torch.isfinite(result.imag).all()

    def test_gradient_finite(self):
        lam = torch.tensor([0.01 + 0j], requires_grad=True)
        result = safe_sqrt_reciprocal(lam, dtype=torch.complex128)
        loss = result.abs().sum()
        loss.backward()
        assert lam.grad is not None and torch.isfinite(lam.grad).all()


# ---------------------------------------------------------------------------
# clamp_exp_arg
# ---------------------------------------------------------------------------

class TestClampExpArg:
    def test_small_values_unchanged(self):
        arg = torch.tensor([1.0 + 2.0j, -3.0 + 0.5j])
        result = clamp_exp_arg(arg)
        torch.testing.assert_close(result, arg)

    def test_large_positive_real_clamped_f64(self):
        arg = torch.tensor([1000.0 + 2.0j], dtype=torch.complex128)
        result = clamp_exp_arg(arg)
        assert result.real.item() == EXP_ARG_MAX_F64
        assert result.imag.item() == 2.0

    def test_large_negative_real_clamped_f64(self):
        arg = torch.tensor([-1000.0 + 0.5j], dtype=torch.complex128)
        result = clamp_exp_arg(arg)
        assert result.real.item() == -EXP_ARG_MAX_F64
        assert result.imag.item() == 0.5

    def test_float32_uses_tighter_clamp(self):
        """float32 overflow starts at exp(88), so clamp must be ≤80."""
        arg = torch.tensor([100.0 + 1.0j], dtype=torch.complex64)
        result = clamp_exp_arg(arg)
        assert result.real.item() == EXP_ARG_MAX_F32  # 80, not 500
        assert result.imag.item() == 1.0
        # Verify the clamped value doesn't overflow float32 exp
        assert torch.isfinite(torch.exp(result.real)).all()

    def test_float32_safe_values_unchanged(self):
        arg = torch.tensor([50.0 + 2.0j], dtype=torch.complex64)
        result = clamp_exp_arg(arg)
        torch.testing.assert_close(result, arg)

    def test_explicit_max_abs_overrides_dtype(self):
        arg = torch.tensor([200.0 + 0j], dtype=torch.complex128)
        result = clamp_exp_arg(arg, max_abs=100.0)
        assert result.real.item() == 100.0

    def test_imaginary_part_never_touched(self):
        arg = torch.tensor([999.0 + 12345.0j, -999.0 - 999.0j])
        result = clamp_exp_arg(arg)
        torch.testing.assert_close(result.imag, arg.imag)

    def test_exp_arg_max_for_dtype(self):
        assert exp_arg_max_for_dtype(torch.float32) == EXP_ARG_MAX_F32
        assert exp_arg_max_for_dtype(torch.complex64) == EXP_ARG_MAX_F32
        assert exp_arg_max_for_dtype(torch.float64) == EXP_ARG_MAX_F64
        assert exp_arg_max_for_dtype(torch.complex128) == EXP_ARG_MAX_F64


# ---------------------------------------------------------------------------
# solve_with_retry
# ---------------------------------------------------------------------------

class TestSolveWithRetry:
    def test_well_conditioned_no_warning(self):
        A = torch.eye(3, dtype=torch.complex128) * 2.0
        b = torch.ones(3, 1, dtype=torch.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fail on any warning
            x = solve_with_retry(A, b, dtype=torch.complex128)
        expected = torch.ones(3, 1, dtype=torch.complex128) * 0.5
        torch.testing.assert_close(x, expected, atol=1e-10, rtol=0)

    def test_singular_matrix_gets_regularized(self):
        # Exactly singular: one zero row
        A = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128)
        b = torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        with pytest.warns(UserWarning, match="Singular matrix"):
            x = solve_with_retry(A, b, dtype=torch.complex128)
        assert torch.isfinite(x).all()

    def test_exhausted_retries_raises(self):
        # With max_retries=0, singular matrix raises immediately (no fallback)
        A = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex128)
        b = torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        with pytest.raises(torch.linalg.LinAlgError):
            solve_with_retry(A, b, dtype=torch.complex128, max_retries=0)

    def test_batched_input(self):
        """Batched (B, N, N) coefficient matrices."""
        A = torch.eye(3, dtype=torch.complex128).unsqueeze(0).expand(4, -1, -1) * 3.0
        b = torch.ones(4, 3, 1, dtype=torch.complex128)
        x = solve_with_retry(A, b, dtype=torch.complex128)
        expected = torch.ones(4, 3, 1, dtype=torch.complex128) / 3.0
        torch.testing.assert_close(x, expected, atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# Backward-compatible wrappers in solver.py
# ---------------------------------------------------------------------------

class TestBackwardCompatWrappers:
    """Ensure old function names + keyword args still work after module split."""

    def test_softplus_protect_kz_keyword_min_kz(self):
        """The old API used min_kz= as keyword — must not raise TypeError."""
        from torchrdit.solver import softplus_protect_kz
        kz = torch.tensor([1e-10 + 0j])
        result = softplus_protect_kz(kz, min_kz=1e-3, beta=100)
        assert torch.abs(result).item() >= 1e-3 - 1e-5

    def test_softplus_clamp_min_keyword_min_val(self):
        """The old API used min_val= as keyword."""
        from torchrdit.solver import softplus_clamp_min
        x = torch.tensor([0.0])
        result = softplus_clamp_min(x, min_val=0.5, beta=100)
        assert result.item() >= 0.5 - 0.01

    def test_softplus_protect_kz_positional_args(self):
        from torchrdit.solver import softplus_protect_kz
        kz = torch.tensor([1e-10 + 0j])
        result = softplus_protect_kz(kz, 1e-3, 100)
        assert torch.abs(result).item() >= 1e-3 - 1e-5
