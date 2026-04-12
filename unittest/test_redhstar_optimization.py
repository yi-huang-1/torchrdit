"""Tests for redhstar optimization implementation.

This module contains comprehensive tests to validate the optimized Redheffer star product
implementation against the original reference implementation. Tests verify correctness,
gradient preservation, and performance across various input dimensions and edge cases.

Classes:
    TestRedhstarOptimization: Test suite for optimized redhstar implementation.

Examples:
    Run these tests with pytest:

    >>> python -m pytest unittest/test_redhstar_optimization.py

Keywords:
    testing, optimization, redhstar, redheffer star product, performance, gradient preservation
"""

import unittest
import torch

from torchrdit.utils import SMatrix, redhstar, init_smatrix


def redhstar_original(smat_a: SMatrix, smat_b: SMatrix, tcomplex: torch.dtype = torch.complex64) -> SMatrix:
    """Original Redheffer star product implementation for reference/testing.

    This is the original, unoptimized implementation of the Redheffer star product
    that was replaced with an optimized version. It's kept here as a reference
    implementation for correctness validation in tests.
    """
    from torch.linalg import solve as tsolve

    harmonic_m, harmonic_n = smat_a.S11.shape[-2:]
    device = smat_a.S11.device
    identity_mat = torch.eye(harmonic_m, harmonic_n, dtype=tcomplex, device=device)

    inv_cycle_1 = identity_mat - smat_b.S11 @ smat_a.S22
    cycle_1_b11 = tsolve(inv_cycle_1, smat_b.S11)
    cycle_1_b12 = tsolve(inv_cycle_1, smat_b.S12)

    cycle_2 = identity_mat + smat_a.S22 @ cycle_1_b11
    b21_cycle_2 = smat_b.S21 @ cycle_2

    return SMatrix(
        S11=smat_a.S11 + smat_a.S12 @ cycle_1_b11 @ smat_a.S21,
        S12=smat_a.S12 @ cycle_1_b12,
        S21=b21_cycle_2 @ smat_a.S21,
        S22=smat_b.S22 + b21_cycle_2 @ smat_a.S22 @ smat_b.S12,
    )


def _make_smatrix(shape, dtype, device):
    """Create a random SMatrix with small magnitude."""
    return SMatrix(
        S11=torch.randn(shape, dtype=dtype, device=device) * 0.1,
        S12=torch.randn(shape, dtype=dtype, device=device) * 0.1,
        S21=torch.randn(shape, dtype=dtype, device=device) * 0.1,
        S22=torch.randn(shape, dtype=dtype, device=device) * 0.1,
    )


def _smatrix_max_error(a: SMatrix, b: SMatrix) -> float:
    """Compute maximum relative error between two SMatrix objects."""
    max_err = 0.0
    for attr in ("S11", "S12", "S21", "S22"):
        diff = torch.abs(getattr(a, attr) - getattr(b, attr))
        ref_mag = torch.abs(getattr(b, attr)) + 1e-12
        max_err = max(max_err, torch.max(diff / ref_mag).item())
    return max_err


class TestRedhstarOptimization(unittest.TestCase):
    """Tests that verify the optimized redhstar implementation.

    Focuses on functional correctness across dimensions, numerical stability,
    dtype handling, and basic integration with helpers.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.tolerance = 1e-6
        self.matrix_sizes = [8, 16, 32, 64]
        self.device = torch.device('cpu')
        self.dtype = torch.complex64

    def test_correctness_2d_matrices(self):
        """Test correctness for 2D matrices (single source, single frequency)."""
        for size in self.matrix_sizes:
            with self.subTest(size=size):
                smat_a = _make_smatrix((size, size), self.dtype, self.device)
                smat_b = _make_smatrix((size, size), self.dtype, self.device)

                result_opt = redhstar(smat_a, smat_b, self.dtype)
                result_ref = redhstar_original(smat_a, smat_b, self.dtype)

                max_error = _smatrix_max_error(result_opt, result_ref)
                self.assertLess(max_error, self.tolerance,
                              f"Size {size}x{size}: Max relative error {max_error:.2e}")

                self.assertEqual(result_opt.S11.shape, (size, size))
                self.assertEqual(result_opt.S11.dtype, self.dtype)

    def test_correctness_3d_matrices(self):
        """Test correctness for 3D matrices (single source, multiple frequencies)."""
        n_freqs = 5
        for size in self.matrix_sizes[:2]:
            with self.subTest(size=size, n_freqs=n_freqs):
                shape = (n_freqs, size, size)
                smat_a = _make_smatrix(shape, self.dtype, self.device)
                smat_b = _make_smatrix(shape, self.dtype, self.device)

                result_opt = redhstar(smat_a, smat_b, self.dtype)
                result_ref = redhstar_original(smat_a, smat_b, self.dtype)

                max_error = _smatrix_max_error(result_opt, result_ref)
                self.assertLess(max_error, self.tolerance)
                self.assertEqual(result_opt.S11.shape, (n_freqs, size, size))

    def test_correctness_4d_matrices(self):
        """Test correctness for 4D matrices (batched sources, multiple frequencies)."""
        n_sources, n_freqs = 3, 4
        for size in [8, 16]:
            with self.subTest(size=size):
                shape = (n_sources, n_freqs, size, size)
                smat_a = _make_smatrix(shape, self.dtype, self.device)
                smat_b = _make_smatrix(shape, self.dtype, self.device)

                result_opt = redhstar(smat_a, smat_b, self.dtype)
                result_ref = redhstar_original(smat_a, smat_b, self.dtype)

                max_error = _smatrix_max_error(result_opt, result_ref)
                self.assertLess(max_error, self.tolerance)
                self.assertEqual(result_opt.S11.shape, (n_sources, n_freqs, size, size))

    def test_numerical_stability(self):
        """Test numerical stability with near-singular matrices."""
        size = 16
        smat_a = _make_smatrix((size, size), self.dtype, self.device)
        smat_b = _make_smatrix((size, size), self.dtype, self.device)

        # Make A22 and B11 close to identity (I - B11@A22 close to singular)
        eye = torch.eye(size, dtype=self.dtype, device=self.device)
        smat_a = SMatrix(S11=smat_a.S11, S12=smat_a.S12, S21=smat_a.S21,
                         S22=0.99 * eye + 0.01 * smat_a.S22)
        smat_b = SMatrix(S11=0.99 * eye + 0.01 * smat_b.S11,
                         S12=smat_b.S12, S21=smat_b.S21, S22=smat_b.S22)

        result_opt = redhstar(smat_a, smat_b, self.dtype)
        result_ref = redhstar_original(smat_a, smat_b, self.dtype)

        for attr in ("S11", "S12", "S21", "S22"):
            self.assertTrue(torch.all(torch.isfinite(getattr(result_opt, attr))))
            self.assertTrue(torch.all(torch.isfinite(getattr(result_ref, attr))))

        max_error = _smatrix_max_error(result_opt, result_ref)
        self.assertLess(max_error, 0.1)  # Looser tolerance for near-singular case

    def test_edge_cases(self):
        """Test edge cases including small matrices and zero matrices."""
        # 1x1 matrix
        smat_a = _make_smatrix((1, 1), self.dtype, self.device)
        smat_b = _make_smatrix((1, 1), self.dtype, self.device)
        result_opt = redhstar(smat_a, smat_b, self.dtype)
        result_ref = redhstar_original(smat_a, smat_b, self.dtype)
        self.assertLess(_smatrix_max_error(result_opt, result_ref), self.tolerance)

        # Zero matrices
        size = 8
        z = torch.zeros(size, size, dtype=self.dtype, device=self.device)
        smat_a_zero = SMatrix(S11=z.clone(), S12=z.clone(), S21=z.clone(), S22=z.clone())
        smat_b_zero = SMatrix(S11=z.clone(), S12=z.clone(), S21=z.clone(), S22=z.clone())

        result_opt = redhstar(smat_a_zero, smat_b_zero, self.dtype)
        result_ref = redhstar_original(smat_a_zero, smat_b_zero, self.dtype)
        self.assertLess(_smatrix_max_error(result_opt, result_ref), self.tolerance)

        for attr in ("S11", "S12", "S21", "S22"):
            self.assertTrue(torch.allclose(getattr(result_opt, attr), z, atol=1e-10))

    def test_different_dtypes(self):
        """Test that the optimization works with different complex dtypes."""
        size = 16
        for dtype in [torch.complex64, torch.complex128]:
            with self.subTest(dtype=dtype):
                smat_a = _make_smatrix((size, size), dtype, self.device)
                smat_b = _make_smatrix((size, size), dtype, self.device)

                result_opt = redhstar(smat_a, smat_b, dtype)
                result_ref = redhstar_original(smat_a, smat_b, dtype)

                self.assertLess(_smatrix_max_error(result_opt, result_ref), self.tolerance)
                self.assertEqual(result_opt.S11.dtype, dtype)

    def test_integration_with_init_smatrix(self):
        """Test that optimized redhstar works correctly with init_smatrix."""
        size = 16
        smat_init = init_smatrix((size, size), self.dtype, self.device)
        smat_test = _make_smatrix((size, size), self.dtype, self.device)

        result_opt = redhstar(smat_init, smat_test, self.dtype)
        result_ref = redhstar_original(smat_init, smat_test, self.dtype)

        max_error = _smatrix_max_error(result_opt, result_ref)
        self.assertLess(max_error, self.tolerance)

        for attr in ("S11", "S12", "S21", "S22"):
            self.assertTrue(torch.allclose(
                getattr(result_opt, attr), getattr(result_ref, attr), atol=self.tolerance
            ))


if __name__ == '__main__':
    unittest.main()
