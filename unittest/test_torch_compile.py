"""Smoke tests for torch.compile readiness.

Verifies that key computational kernels can be traced by torch.compile
without graph breaks. These tests don't assert numeric results (covered
by golden-value tests) -- they only check that compilation succeeds.

Note: PyTorch inductor has limited complex-tensor support. Tests that
exercise the full solve path use ``backend="eager"`` to verify graph
traceability without hitting inductor codegen limitations.
"""

import pytest
import numpy as np
import torch

from torchrdit.solver import create_solver
from torchrdit.constants import Algorithm
from torchrdit.utils import SMatrix, redhstar, to_diag_util, blockmat2x2


# Skip entirely if torch.compile is unavailable (PyTorch < 2.0)
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "compile"),
    reason="torch.compile not available",
)


@pytest.fixture
def simple_solver():
    """Create a minimal solver for compile tests."""
    solver = create_solver(
        algorithm=Algorithm.RCWA,
        lam0=np.array([1.0]),
        harmonics=[3, 3],
        grids=[64, 64],
    )
    solver.add_layer(material_name="air", thickness=torch.tensor(0.5))
    return solver


class TestRedhstarCompile:
    """Verify redhstar can be compiled without graph breaks."""

    def test_redhstar_compile_succeeds(self):
        """redhstar should compile without errors."""
        M = 18  # 2 * 3 * 3
        eye = torch.eye(M, dtype=torch.complex64)
        smat_a = SMatrix(S11=eye.clone(), S12=eye.clone(), S21=eye.clone(), S22=eye.clone())
        smat_b = SMatrix(S11=0.5 * eye, S12=eye.clone(), S21=eye.clone(), S22=0.5 * eye)

        compiled_redhstar = torch.compile(redhstar, fullgraph=False)
        result = compiled_redhstar(smat_a, smat_b)

        assert result.S11.shape == (M, M)
        assert result.S12.shape == (M, M)

    def test_redhstar_compile_batched(self):
        """redhstar should compile with batched (4D) S-matrices."""
        B, F, M = 2, 1, 18
        shape = (B, F, M, M)
        eye = torch.eye(M, dtype=torch.complex64).reshape(1, 1, M, M).expand(shape)
        smat_a = SMatrix(S11=eye.clone(), S12=eye.clone(), S21=eye.clone(), S22=eye.clone())
        smat_b = SMatrix(S11=0.5 * eye, S12=eye.clone(), S21=eye.clone(), S22=0.5 * eye)

        compiled_redhstar = torch.compile(redhstar, fullgraph=False)
        result = compiled_redhstar(smat_a, smat_b)

        assert result.S11.shape == shape


class TestSolveCompile:
    """Verify solve() forward pass is traceable by torch.compile.

    Uses ``backend="eager"`` to test graph tracing without triggering
    PyTorch inductor's limited complex-tensor codegen.
    """

    def test_solve_single_source_eager_compile(self, simple_solver):
        """_solve_structure should be traceable with eager backend."""
        source = simple_solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)

        original = simple_solver._solve_structure
        simple_solver._solve_structure = torch.compile(original, backend="eager", fullgraph=False)

        try:
            result = simple_solver.solve(source)
            assert result.reflection is not None
            assert result.transmission is not None
        finally:
            simple_solver._solve_structure = original

    def test_solve_batched_sources_eager_compile(self, simple_solver):
        """_solve_structure with batched sources should be traceable with eager backend."""
        sources = [
            simple_solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0),
            simple_solver.add_source(theta=10, phi=0, pte=1.0, ptm=0.0),
        ]

        original = simple_solver._solve_structure
        simple_solver._solve_structure = torch.compile(original, backend="eager", fullgraph=False)

        try:
            result = simple_solver.solve(sources)
            assert result.reflection.shape[0] == 2
        finally:
            simple_solver._solve_structure = original


class TestUtilsCompile:
    """Verify utility functions compile cleanly."""

    def test_to_diag_util_compile(self):
        """to_diag_util should compile without graph breaks."""
        x = torch.randn(2, 1, 9, dtype=torch.complex64)
        compiled_fn = torch.compile(to_diag_util, fullgraph=False)
        result = compiled_fn(x, [3, 3])
        assert result.shape == (2, 1, 9, 9)

    def test_blockmat2x2_compile(self):
        """blockmat2x2 should compile without graph breaks."""
        M = 9
        a = torch.randn(M, M, dtype=torch.complex64)
        b = torch.randn(M, M, dtype=torch.complex64)
        compiled_fn = torch.compile(blockmat2x2, fullgraph=False)
        result = compiled_fn([[a, b], [b, a]])
        assert result.shape == (2 * M, 2 * M)
