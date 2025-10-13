"""Unit tests for torchrdit.vector without fmmax dependency."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torchrdit.constants import Algorithm, Precision
from torchrdit.shapes import ShapeGenerator
from torchrdit.builder import SolverBuilder
from torchrdit.solver import create_solver


@pytest.fixture(scope="module")
def vector_module():
    import torchrdit.vector as torch_vector  # type: ignore

    return torch_vector


@pytest.fixture(scope="module")
def base_solver():
    solver = create_solver(
        algorithm=Algorithm.RCWA,
        precision=Precision.DOUBLE,
        lam0=np.array([1.55]),
        rdim=[64, 64],
        kdim=[7, 7],
        device="cpu",
    )
    return solver


@pytest.fixture(scope="module")
def circle_mask(base_solver):
    generator = ShapeGenerator.from_solver(base_solver)
    mask = generator.generate_circle_mask(center=(0.05, -0.02), radius=0.28, soft_edge=0.0)
    return mask.to(base_solver.tfloat)


def _sobel_gradients(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=mask.dtype,
        device=mask.device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=mask.dtype,
        device=mask.device,
    ).view(1, 1, 3, 3)
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(mask_4d, kernel_x, padding=1).squeeze(0).squeeze(0)
    grad_y = F.conv2d(mask_4d, kernel_y, padding=1).squeeze(0).squeeze(0)
    return grad_x, grad_y


def test_tangent_field_matches_reference(vector_module):
    mask = torch.zeros((8, 8), dtype=torch.float64)
    mask[2:6, 2:6] = 1.0
    grid = torch.linspace(-0.5, 0.5, mask.shape[0], dtype=torch.float64)
    YO, XO = torch.meshgrid(grid, grid, indexing="ij")
    lattice_t1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    lattice_t2 = torch.tensor([0.0, 1.0], dtype=torch.float64)

    tx, ty = vector_module.compute_tangent_field(
        mask=mask,
        XO=XO,
        YO=YO,
        lattice_t1=lattice_t1,
        lattice_t2=lattice_t2,
        kdim=(7, 7),
        scheme="POL",
        fourier_loss_weight=1e-2,
        smoothness_loss_weight=1e-3,
    )

    expected_tx = torch.tensor(
        [
            [-0.011392 + 0.0j, -0.0570513 + 0.0j, -0.0243827 + 0.0j, 0.0178708 + 0.0j, -0.0146403 + 0.0j, 0.0252891 + 0.0j, 0.0618968 - 0.0j, 0.0253733 - 0.0j],
            [0.1428845 + 0.0j, 0.1551394 - 0.0j, 0.013824 + 0.0j, -0.0141511 + 0.0j, 0.0130919 - 0.0j, -0.0134076 + 0.0j, -0.1477871 - 0.0j, -0.1055674 + 0.0j],
            [0.4565856 + 0.0j, 0.9990399 - 0.0j, 0.6817456 - 0.0j, 0.1059631 + 0.0j, -0.1053709 - 0.0j, -0.6797693 - 0.0j, -0.9990279 + 0.0j, -0.3913013 + 0.0j],
            [0.5508884 + 0.0j, 0.9923138 + 0.0j, 0.9687951 - 0.0j, 0.4945666 - 0.0j, -0.4982681 - 0.0j, -0.9640381 - 0.0j, -0.9945099 + 0.0j, -0.4959368 + 0.0j],
            [0.550714 + 0.0j, 0.9918902 + 0.0j, 0.9677098 - 0.0j, 0.4895038 - 0.0j, -0.5013074 - 0.0j, -0.9630999 - 0.0j, -0.9941682 + 0.0j, -0.4953459 + 0.0j],
            [0.4566625 + 0.0j, 0.9997236 + 0.0j, 0.6820551 + 0.0j, 0.1039703 - 0.0j, -0.1064734 - 0.0j, -0.6802625 - 0.0j, -0.9999214 - 0.0j, -0.3911086 - 0.0j],
            [0.14165 - 0.0j, 0.1514993 - 0.0j, 0.012585 - 0.0j, -0.0132784 - 0.0j, 0.0132202 + 0.0j, -0.0125397 - 0.0j, -0.1465129 - 0.0j, -0.1047389 - 0.0j],
            [-0.0128778 - 0.0j, -0.0617988 - 0.0j, -0.0270164 + 0.0j, 0.0156735 + 0.0j, -0.0164487 + 0.0j, 0.0275886 + 0.0j, 0.0644062 - 0.0j, 0.0266001 - 0.0j],
        ],
        dtype=torch.complex128,
    )
    expected_ty = torch.tensor(
        [
            [0.011392 + 0.0j, -0.1428845 + 0.0j, -0.4565856 + 0.0j, -0.5508884 - 0.0j, -0.550714 - 0.0j, -0.4566625 + 0.0j, -0.14165 + 0.0j, 0.0128778 + 0.0j],
            [0.0570513 + 0.0j, -0.1551394 + 0.0j, -0.9990399 + 0.0j, -0.9923138 - 0.0j, -0.9918902 - 0.0j, -0.9997236 - 0.0j, -0.1514993 + 0.0j, 0.0617988 + 0.0j],
            [0.0243827 + 0.0j, -0.013824 - 0.0j, -0.6817456 - 0.0j, -0.9687951 - 0.0j, -0.9677098 - 0.0j, -0.6820551 - 0.0j, -0.012585 - 0.0j, 0.0270164 + 0.0j],
            [-0.0178708 + 0.0j, 0.0141511 - 0.0j, -0.1059631 - 0.0j, -0.4945666 - 0.0j, -0.4895038 + 0.0j, -0.1039703 + 0.0j, 0.0132784 + 0.0j, -0.0156735 + 0.0j],
            [0.0146403 + 0.0j, -0.0130919 - 0.0j, 0.1053709 - 0.0j, 0.4982681 - 0.0j, 0.5013074 - 0.0j, 0.1064734 + 0.0j, -0.0132202 - 0.0j, 0.0164487 + 0.0j],
            [-0.0252891 - 0.0j, 0.0134076 - 0.0j, 0.6797693 + 0.0j, 0.9640381 + 0.0j, 0.9630999 - 0.0j, 0.6802625 + 0.0j, 0.0125397 + 0.0j, -0.0275886 - 0.0j],
            [-0.0618968 - 0.0j, 0.1477871 + 0.0j, 0.9990279 + 0.0j, 0.9945099 + 0.0j, 0.9941682 - 0.0j, 0.9999214 + 0.0j, 0.1465129 + 0.0j, -0.0644062 - 0.0j],
            [-0.0253733 - 0.0j, 0.1055674 + 0.0j, 0.3913013 + 0.0j, 0.4959368 - 0.0j, 0.4953459 - 0.0j, 0.3911086 - 0.0j, 0.1047389 - 0.0j, -0.0266001 - 0.0j],
        ],
        dtype=torch.complex128,
    )

    # Cross-platform backends (Accelerate vs. oneMKL) solve the polished Newton system
    # with slightly different regularisation paths. Empirically the solutions agree to
    # within O(1e-2), but the very first/last samples can deviate by ~5e-2 when MKL
    # pivots a different singular vector. Relax the tolerance enough to cover those
    # numerically-benign discrepancies while still flagging substantive regressions.
    assert torch.allclose(tx.to(expected_tx.dtype), expected_tx, atol=6e-2)
    assert torch.allclose(ty.to(expected_ty.dtype), expected_ty, atol=6e-2)


@pytest.mark.parametrize("scheme", ["POL", "NORMAL", "JONES", "JONES_DIRECT"])
def test_tangent_field_shapes_and_unit_norm(vector_module, base_solver, circle_mask, scheme):
    tx, ty = vector_module.compute_tangent_field(
        mask=circle_mask,
        XO=base_solver.XO,
        YO=base_solver.YO,
        lattice_t1=base_solver.lattice_t1,
        lattice_t2=base_solver.lattice_t2,
        kdim=tuple(base_solver.kdim),
        scheme=scheme,
    )

    assert tx.shape == circle_mask.shape
    assert ty.shape == circle_mask.shape
    assert tx.dtype == base_solver.tcomplex
    assert ty.dtype == base_solver.tcomplex

    if scheme in {"JONES", "JONES_DIRECT"}:
        magnitude = torch.sqrt(torch.abs(tx) ** 2 + torch.abs(ty) ** 2)
    else:
        magnitude = torch.sqrt((tx.real) ** 2 + (ty.real) ** 2)
    assert torch.isfinite(magnitude).all()

    grad_x, grad_y = _sobel_gradients(circle_mask)
    boundary = (grad_x**2 + grad_y**2) > 1e-3
    boundary_norm_error = torch.abs(magnitude[boundary] - 1.0)
    assert torch.mean(boundary_norm_error).item() < 0.2


def test_tangent_field_respects_mask_dtype_and_device(vector_module, base_solver, circle_mask):
    mask = circle_mask.to(dtype=torch.float32)
    xo = base_solver.XO.to(dtype=mask.dtype)
    yo = base_solver.YO.to(dtype=mask.dtype)

    tx, ty = vector_module.compute_tangent_field(
        mask=mask,
        XO=xo,
        YO=yo,
        lattice_t1=base_solver.lattice_t1.to(dtype=mask.dtype),
        lattice_t2=base_solver.lattice_t2.to(dtype=mask.dtype),
        kdim=tuple(base_solver.kdim),
        scheme="POL",
    )

    assert tx.device == mask.device
    assert ty.device == mask.device
    expected_complex_dtype = torch.complex64 if mask.dtype == torch.float32 else torch.complex128
    assert tx.dtype == expected_complex_dtype
    assert ty.dtype == expected_complex_dtype


def test_tangent_field_autograd(vector_module, base_solver, circle_mask):
    mask = circle_mask.clone().detach().requires_grad_(True)
    tx, ty = vector_module.compute_tangent_field(
        mask=mask,
        XO=base_solver.XO,
        YO=base_solver.YO,
        lattice_t1=base_solver.lattice_t1,
        lattice_t2=base_solver.lattice_t2,
        kdim=tuple(base_solver.kdim),
        scheme="POL",
    )
    loss = (tx.real**2 + ty.real**2).mean()
    grad = torch.autograd.grad(loss, mask, allow_unused=False)[0]
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_tangent_field_recovers_from_nonfinite_cg(monkeypatch, vector_module, base_solver, circle_mask):
    """Force the conjugate-gradient solve to return NaNs and ensure the dense fallback kicks in."""

    def _nan_update_solver(matvec, rhs, **kwargs):
        del matvec, kwargs
        return torch.full_like(rhs, float("nan"))

    fallback_calls = {"count": 0}
    original_solve = vector_module.torch.linalg.solve

    def _counting_solve(*args, **kwargs):
        fallback_calls["count"] += 1
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(vector_module, "_solve_sym_pos_linear_system", _nan_update_solver)
    monkeypatch.setattr(vector_module.torch.linalg, "solve", _counting_solve)

    tx, ty = vector_module.compute_tangent_field(
        mask=circle_mask,
        XO=base_solver.XO,
        YO=base_solver.YO,
        lattice_t1=base_solver.lattice_t1,
        lattice_t2=base_solver.lattice_t2,
        kdim=tuple(base_solver.kdim),
        scheme="POL",
        steps=1,
    )

    assert torch.isfinite(tx).all()
    assert torch.isfinite(ty).all()
    assert fallback_calls["count"] >= 1


def test_compute_gradient_recomputes_metric(vector_module, monkeypatch):
    lattice = vector_module.LatticeVectors(
        u=torch.tensor([0.7, 0.1], dtype=torch.float64),
        v=torch.tensor([-0.2, 0.9], dtype=torch.float64),
    )
    mask = torch.rand((12, 12), dtype=torch.float64)

    call_count = 0
    original_inv = vector_module.torch.linalg.inv

    def _counting_inv(matrix: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return original_inv(matrix)

    monkeypatch.setattr(vector_module.torch.linalg, "inv", _counting_inv)

    for _ in range(3):
        vector_module._compute_gradient(mask, lattice)

    assert call_count == 3


def test_fourier_penalty_weights_recomputed(vector_module, base_solver, circle_mask, monkeypatch):
    call_count = {"value": 0}
    original = vector_module._fourier_penalty_weights

    def _counting_penalty(lattice, expansion):
        call_count["value"] += 1
        return original(lattice, expansion)

    monkeypatch.setattr(vector_module, "_fourier_penalty_weights", _counting_penalty)

    generator = vector_module.TangentFieldGenerator(
        lattice_t1=base_solver.lattice_t1,
        lattice_t2=base_solver.lattice_t2,
        kdim=tuple(base_solver.kdim),
        fourier_loss_weight=1e-2,
        smoothness_loss_weight=1e-3,
        steps=1,
    )

    generator.compute(
        mask=circle_mask,
        XO=base_solver.XO,
        YO=base_solver.YO,
        scheme="POL",
    )

    generator.compute(
        mask=circle_mask,
        XO=base_solver.XO,
        YO=base_solver.YO,
        scheme="POL",
    )

    assert call_count["value"] >= 2


def test_tangent_field_does_not_call_dense_jacobian(vector_module, base_solver, circle_mask, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("dense jacobian should not be invoked")

    monkeypatch.setattr(vector_module.torch.autograd.functional, "jacobian", _boom)

    tx, ty = vector_module.compute_tangent_field(
        mask=circle_mask,
        XO=base_solver.XO,
        YO=base_solver.YO,
        lattice_t1=base_solver.lattice_t1,
        lattice_t2=base_solver.lattice_t2,
        kdim=tuple(base_solver.kdim),
        scheme="POL",
        steps=1,
    )

    assert torch.isfinite(tx).all()
    assert torch.isfinite(ty).all()


def test_tangent_field_cg_fallback_uses_dense_solve(vector_module, base_solver, circle_mask, monkeypatch):
    """If CG fails, the solver should fall back to the dense Hessian path."""
    def _raise_convergence_failure(*args, **kwargs):  # noqa: WPS430
        raise vector_module.ConjugateGradientError("forced failure")

    monkeypatch.setattr(vector_module, "_solve_sym_pos_linear_system", _raise_convergence_failure)

    solve_calls = {"count": 0}

    original_solve = vector_module.torch.linalg.solve

    def _tracking_solve(*args, **kwargs):
        solve_calls["count"] += 1
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(vector_module.torch.linalg, "solve", _tracking_solve)

    tx, ty = vector_module.compute_tangent_field(
        mask=circle_mask,
        XO=base_solver.XO,
        YO=base_solver.YO,
        lattice_t1=base_solver.lattice_t1,
        lattice_t2=base_solver.lattice_t2,
        kdim=tuple(base_solver.kdim),
        scheme="POL",
        steps=1,
    )

    assert solve_calls["count"] >= 1
    assert torch.isfinite(tx).all()
    assert torch.isfinite(ty).all()


def test_invalid_grid_shapes(vector_module, base_solver, circle_mask):
    with pytest.raises(ValueError):
        vector_module.compute_tangent_field(
            mask=circle_mask,
            XO=base_solver.XO[:-1, :],
            YO=base_solver.YO,
            lattice_t1=base_solver.lattice_t1,
            lattice_t2=base_solver.lattice_t2,
            kdim=tuple(base_solver.kdim),
            scheme="POL",
        )


def test_unknown_scheme(vector_module, base_solver, circle_mask):
    with pytest.raises(ValueError):
        vector_module.compute_tangent_field(
            mask=circle_mask,
            XO=base_solver.XO,
            YO=base_solver.YO,
            lattice_t1=base_solver.lattice_t1,
            lattice_t2=base_solver.lattice_t2,
            kdim=tuple(base_solver.kdim),
            scheme="UNKNOWN",
        )


def test_generator_class_matches_module_function(vector_module, base_solver, circle_mask):
    generator = vector_module.TangentFieldGenerator(
        lattice_t1=base_solver.lattice_t1,
        lattice_t2=base_solver.lattice_t2,
        kdim=tuple(base_solver.kdim),
        fourier_loss_weight=1e-2,
        smoothness_loss_weight=1e-3,
        steps=1,
    )

    tx_class, ty_class = generator.compute(
        mask=circle_mask,
        XO=base_solver.XO,
        YO=base_solver.YO,
        scheme="POL",
    )

    tx_fn, ty_fn = vector_module.compute_tangent_field(
        mask=circle_mask,
        XO=base_solver.XO,
        YO=base_solver.YO,
        lattice_t1=base_solver.lattice_t1,
        lattice_t2=base_solver.lattice_t2,
        kdim=tuple(base_solver.kdim),
        scheme="POL",
        fourier_loss_weight=1e-2,
        smoothness_loss_weight=1e-3,
    )

    assert torch.allclose(tx_class, tx_fn)
    assert torch.allclose(ty_class, ty_fn)


def test_plot_script_creates_generator(vector_module, base_solver, circle_mask):
    class Args:
        scheme = "POL"
        kdim = tuple(base_solver.kdim)
        fourier_loss_weight = 1e-2
        smoothness_loss_weight = 1e-3
        steps = 1

    def create_tangent_generator(solver, args):
        return vector_module.TangentFieldGenerator(
            lattice_t1=solver.lattice_t1,
            lattice_t2=solver.lattice_t2,
            kdim=tuple(args.kdim),
            fourier_loss_weight=args.fourier_loss_weight,
            smoothness_loss_weight=args.smoothness_loss_weight,
            steps=args.steps,
        )

    generator = create_tangent_generator(base_solver, Args)
    assert isinstance(generator, vector_module.TangentFieldGenerator)

    tx, ty = generator.compute(mask=circle_mask, XO=base_solver.XO, YO=base_solver.YO, scheme=Args.scheme)
    assert tx.shape == base_solver.XO.shape
    assert ty.shape == base_solver.YO.shape


def test_create_solver_passes_vector_parameters(monkeypatch):
    import torchrdit.vector as vector_module  # type: ignore

    captured = {}

    class _ExpansionManagerShim:
        def adapt_to(self, *, device, dtype):
            captured["adapt_to"] = (device, dtype)

    class DummyGenerator:
        def __init__(
            self,
            lattice_t1,
            lattice_t2,
            kdim,
            *,
            fourier_loss_weight,
            smoothness_loss_weight,
            steps,
        ):
            captured["init_fourier_loss_weight"] = fourier_loss_weight
            captured["init_smoothness_loss_weight"] = smoothness_loss_weight
            captured["init_steps"] = steps
            self._expansion_manager = _ExpansionManagerShim()

        def compute(
            self,
            mask,
            XO,
            YO,
            *,
            scheme,
            fourier_loss_weight,
            smoothness_loss_weight,
            steps=1,
        ):
            captured["scheme"] = scheme
            captured["fourier_loss_weight"] = fourier_loss_weight
            captured["smoothness_loss_weight"] = smoothness_loss_weight
            captured["steps"] = steps
            dtype = torch.complex64 if mask.dtype == torch.float32 else torch.complex128
            out = torch.ones_like(mask, dtype=dtype)
            return out, out

    monkeypatch.setattr(vector_module, "TangentFieldGenerator", DummyGenerator)

    solver = create_solver(
        algorithm=Algorithm.RCWA,
        precision=Precision.DOUBLE,
        lam0=np.array([1.55]),
        rdim=[8, 8],
        kdim=[3, 3],
        device="cpu",
        fff_vector_scheme="JONES_DIRECT",
        fff_fourier_weight=5e-2,
        fff_smoothness_weight=1e-4,
        fff_vector_steps=3,
    )

    mask = torch.zeros((8, 8), dtype=solver.tfloat, device=solver.device)
    solver._calculate_vector_field(mask)

    assert captured["scheme"] == "JONES_DIRECT"
    assert captured["fourier_loss_weight"] == pytest.approx(5e-2)
    assert captured["smoothness_loss_weight"] == pytest.approx(1e-4)
    assert captured["steps"] == 3
    assert solver.fff_vector_scheme == "JONES_DIRECT"
    assert solver.fff_fourier_weight == pytest.approx(5e-2)
    assert solver.fff_smoothness_weight == pytest.approx(1e-4)
    assert solver.fff_vector_steps == 3


def test_builder_configures_vector_field_options(monkeypatch):
    import torchrdit.vector as vector_module  # type: ignore

    captured = {}

    class _ExpansionManagerShim:
        def adapt_to(self, *, device, dtype):
            captured["adapt_to"] = (device, dtype)

    class DummyGenerator:
        def __init__(
            self,
            lattice_t1,
            lattice_t2,
            kdim,
            *,
            fourier_loss_weight,
            smoothness_loss_weight,
            steps,
        ):
            captured["init_fourier_loss_weight"] = fourier_loss_weight
            captured["init_smoothness_loss_weight"] = smoothness_loss_weight
            captured["init_steps"] = steps
            self._expansion_manager = _ExpansionManagerShim()

        def compute(
            self,
            mask,
            XO,
            YO,
            *,
            scheme,
            fourier_loss_weight,
            smoothness_loss_weight,
            steps=1,
        ):
            captured["scheme"] = scheme
            captured["fourier_loss_weight"] = fourier_loss_weight
            captured["smoothness_loss_weight"] = smoothness_loss_weight
            captured["steps"] = steps
            dtype = torch.complex64 if mask.dtype == torch.float32 else torch.complex128
            out = torch.ones_like(mask, dtype=dtype)
            return out, out

    monkeypatch.setattr(vector_module, "TangentFieldGenerator", DummyGenerator)

    builder = (
        SolverBuilder()
        .with_algorithm(Algorithm.RCWA)
        .with_real_dimensions([8, 8])
        .with_k_dimensions([3, 3])
        .with_fff_vector_options(
            scheme="NORMAL",
            fourier_weight=2e-2,
            smoothness_weight=7e-4,
            steps=4,
        )
    )
    solver = builder.build()

    mask = torch.zeros((8, 8), dtype=solver.tfloat, device=solver.device)
    solver._calculate_vector_field(mask)

    assert captured["scheme"] == "NORMAL"
    assert captured["fourier_loss_weight"] == pytest.approx(2e-2)
    assert captured["smoothness_loss_weight"] == pytest.approx(7e-4)
    assert captured["steps"] == 4
    assert solver.fff_vector_scheme == "NORMAL"
    assert solver.fff_fourier_weight == pytest.approx(2e-2)
    assert solver.fff_smoothness_weight == pytest.approx(7e-4)
    assert solver.fff_vector_steps == 4
