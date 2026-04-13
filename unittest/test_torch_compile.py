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
from torchrdit.shapes import ShapeGenerator
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

    def test_redhstar_fullgraph_eager(self):
        """redhstar should trace as a single graph with eager backend."""
        M = 18
        eye = torch.eye(M, dtype=torch.complex64)
        smat_a = SMatrix(S11=eye.clone(), S12=eye.clone(), S21=eye.clone(), S22=eye.clone())
        smat_b = SMatrix(S11=0.5 * eye, S12=eye.clone(), S21=eye.clone(), S22=0.5 * eye)

        compiled = torch.compile(redhstar, backend="eager", fullgraph=True)
        result = compiled(smat_a, smat_b)
        assert result.S11.shape == (M, M)

    def test_redhstar_near_singular(self):
        """redhstar should handle near-singular (I - B11 @ A22) via regularization."""
        M = 18
        eye = torch.eye(M, dtype=torch.complex64)
        # A22 ~ inv(B11) makes (I - B11 @ A22) ~ 0
        smat_a = SMatrix(S11=eye.clone(), S12=eye.clone(), S21=eye.clone(), S22=eye.clone())
        smat_b = SMatrix(S11=eye.clone(), S12=eye.clone(), S21=eye.clone(), S22=eye.clone())

        result = redhstar(smat_a, smat_b)
        # Should not raise; result should be finite
        assert torch.all(torch.isfinite(result.S11.real))
        assert torch.all(torch.isfinite(result.S12.real))


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

    def test_solve_single_source_fullgraph_eager(self, simple_solver):
        """_solve_structure should trace as a single graph (eager backend)."""
        source = simple_solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)

        original = simple_solver._solve_structure
        simple_solver._solve_structure = torch.compile(original, backend="eager", fullgraph=True)

        try:
            result = simple_solver.solve(source)
            assert result.reflection is not None
        finally:
            simple_solver._solve_structure = original

    def test_solve_backward_gradient_flow(self):
        """Gradients should flow through the solve path."""
        from torchrdit.utils import create_material

        solver = create_solver(
            algorithm=Algorithm.RCWA,
            lam0=np.array([1.0]),
            harmonics=[3, 3],
            grids=[64, 64],
        )
        si = create_material(name="Si", permittivity=11.7)
        solver.add_materials([si])
        solver.add_layer(material_name="Si", thickness=torch.tensor(0.3), is_homogeneous=False)

        shape_gen = ShapeGenerator.from_solver(solver)
        mask = shape_gen.generate_circle_mask(center=(0, 0), radius=0.25).to(torch.float32)
        mask.requires_grad_(True)
        solver.update_er_with_mask(mask=mask, layer_index=0)

        source = solver.add_source(theta=0, phi=0, pte=1.0, ptm=0.0)
        result = solver.solve(source)
        loss = result.transmission[0]
        loss.backward()

        assert mask.grad is not None
        assert torch.any(mask.grad != 0)

    def test_redhstar_compiled_backward(self):
        """Gradients should flow through compiled redhstar."""
        M = 18
        eye = torch.eye(M, dtype=torch.complex64)
        s11 = (0.5 * eye).requires_grad_(True)
        smat_a = SMatrix(S11=s11, S12=eye.clone(), S21=eye.clone(), S22=eye.clone())
        smat_b = SMatrix(S11=eye.clone(), S12=eye.clone(), S21=eye.clone(), S22=0.5 * eye)

        compiled = torch.compile(redhstar, backend="eager", fullgraph=True)
        result = compiled(smat_a, smat_b)
        loss = result.S11.abs().sum()
        loss.backward()

        assert s11.grad is not None
        assert torch.any(s11.grad != 0)


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
