import numpy as np
import os
import sys
from pathlib import Path
import pytest
import torch

from torchrdit.constants import Algorithm, Precision
from torchrdit.solver import create_solver, get_solver_builder
from torchrdit.shapes import ShapeGenerator
from torchrdit.utils import create_material, operator_proj

FM_MAX_SRC = Path(__file__).resolve().parents[2] / "docs" / "fmmax" / "src"
if FM_MAX_SRC.exists() and str(FM_MAX_SRC) not in sys.path:
    sys.path.insert(0, str(FM_MAX_SRC))


# Common units and geometry helpers
UM = 1.0
NM = 1e-3 * UM
DEG = np.pi / 180.0


def _hex_unit_mask(a, r, solver: "FourierBaseSolver"):
    """Construct hexagonal unit-cell mask from 4 circles using solver's lattice."""
    b = a * np.sqrt(3)
    sg = ShapeGenerator.from_solver(solver)
    c1 = sg.generate_circle_mask(center=[0, b / 2], radius=r)
    c2 = sg.generate_circle_mask(center=[0, -b / 2], radius=r)
    c3 = sg.generate_circle_mask(center=[a / 2, 0], radius=r)
    c4 = sg.generate_circle_mask(center=[-a / 2, 0], radius=r)
    mask = sg.combine_masks(mask1=c1, mask2=c2, operation='union')
    mask = sg.combine_masks(mask1=mask, mask2=c3, operation='union')
    mask = sg.combine_masks(mask1=mask, mask2=c4, operation='union')
    return 1 - mask


@pytest.mark.parametrize(
    "algorithm",
    [Algorithm.RCWA, Algorithm.RDIT],
)
def test_solver_transmission_consistency_with_fff_toggle(algorithm):
    """Ensure baseline solve remains stable with and without full Fourier factorization."""
    a = 850 * NM
    r = 220 * NM
    h1 = torch.tensor(180 * NM, dtype=torch.float32)
    h2 = torch.tensor(250 * NM, dtype=torch.float32)
    t1 = torch.tensor([[a, 0]], dtype=torch.float32)
    t2 = torch.tensor([[0, a]], dtype=torch.float32)

    mat_air = create_material(name='Air', permittivity=1.0)
    mat_sin = create_material(name='SiN', permittivity=(1.9) ** 2)
    mat_fs = create_material(name='FusedSilica', permittivity=(1.45) ** 2)

    dev = create_solver(
        algorithm=algorithm,
        precision=Precision.DOUBLE,
        rdim=[192, 192],
        kdim=[7, 7],
        lam0=np.array([1550 * NM]),
        lengthunit='um',
        t1=t1,
        t2=t2,
    )
    if algorithm == Algorithm.RDIT:
        dev.set_rdit_order(6)

    dev.update_ref_material(ref_material=mat_air)
    dev.update_trn_material(trn_material=mat_fs)
    dev.add_layer(material_name=mat_sin, thickness=h1, is_homogeneous=False, is_optimize=False)
    dev.add_layer(material_name=mat_air, thickness=h2, is_homogeneous=True, is_optimize=False)

    src = dev.add_source(theta=0 * DEG, phi=0 * DEG, pte=1, ptm=0)

    mask = _hex_unit_mask(a=a, r=r, solver=dev)
    dev.update_er_with_mask(mask=mask, layer_index=0, method='FFT')

    data_fff = dev.solve(src, is_use_FFF=True)
    data_no_fff = dev.solve(src, is_use_FFF=False)

    t_fff = data_fff.transmission[0].item()
    r_fff = data_fff.reflection[0].item()
    t_no = data_no_fff.transmission[0].item()
    r_no = data_no_fff.reflection[0].item()

    assert np.isfinite(t_fff) and np.isfinite(r_fff)
    assert np.isfinite(t_no) and np.isfinite(r_no)
    assert np.isclose(t_fff, t_no, atol=5e-2)
    assert np.isclose(r_fff, r_no, atol=5e-2)


def test_solver_fff_vector_field_matches_generator():
    """FFF vector field Toeplitz matrices should agree with tangent generator."""

    mat_air = create_material(name="Air", permittivity=1.0)
    mat_sin = create_material(name="SiN", permittivity=(2.0) ** 2)

    dev = create_solver(
        algorithm=Algorithm.RCWA,
        precision=Precision.DOUBLE,
        rdim=[64, 64],
        kdim=[5, 5],
        lam0=np.array([1.55]),
        is_use_FFF=True,
    )
    dev.update_ref_material(ref_material=mat_air)
    dev.update_trn_material(trn_material=mat_air)
    dev.add_layer(material_name=mat_sin, thickness=torch.tensor(0.35, dtype=torch.float64), is_homogeneous=False)

    mask = _hex_unit_mask(a=0.8, r=0.25, solver=dev).to(dev.tfloat)
    dev.update_er_with_mask(mask=mask, layer_index=0, method="FFT")

    from torchrdit import vector as vector_module  # noqa: WPS433

    tx, ty = vector_module.compute_tangent_field(
        mask=mask,
        XO=dev.XO,
        YO=dev.YO,
        lattice_t1=dev.lattice_t1,
        lattice_t2=dev.lattice_t2,
        kdim=tuple(dev.kdim),
        scheme="POL",
        fourier_loss_weight=1e-2,
        smoothness_loss_weight=1e-3,
    )

    denom = (tx.abs() ** 2 + ty.abs() ** 2).clamp_min(1e-12)
    p_xx = (tx.abs() ** 2) / denom
    p_yy = (ty.abs() ** 2) / denom
    p_xy = tx * ty.conj() / denom

    expected_n_xx = dev.layer_manager._gen_toeplitz2d(
        p_xx.to(dev.tcomplex),
        nharmonic_1=dev.kdim[0],
        nharmonic_2=dev.kdim[1],
        method="FFT",
    )
    expected_n_yy = dev.layer_manager._gen_toeplitz2d(
        p_yy.to(dev.tcomplex),
        nharmonic_1=dev.kdim[0],
        nharmonic_2=dev.kdim[1],
        method="FFT",
    )
    expected_n_xy = dev.layer_manager._gen_toeplitz2d(
        p_xy.to(dev.tcomplex),
        nharmonic_1=dev.kdim[0],
        nharmonic_2=dev.kdim[1],
        method="FFT",
    )
    expected_n_yx = dev.layer_manager._gen_toeplitz2d(
        (ty * tx.conj() / denom).to(dev.tcomplex),
        nharmonic_1=dev.kdim[0],
        nharmonic_2=dev.kdim[1],
        method="FFT",
    )

    p_xx, p_yy, p_xy, p_yx = dev._calculate_vector_field(mask=mask)

    assert torch.allclose(p_xx, expected_n_xx, atol=5e-4, rtol=5e-4)
    assert torch.allclose(p_yy, expected_n_yy, atol=5e-4, rtol=5e-4)
    assert torch.allclose(p_xy, expected_n_xy, atol=5e-4, rtol=5e-4)
    assert torch.allclose(p_yx, expected_n_yx, atol=5e-4, rtol=5e-4)


def test_calculate_vector_field_reuses_generator(monkeypatch):
    """Ensure `_calculate_vector_field` reuses a cached tangent-field generator."""
    mat_air = create_material(name="Air", permittivity=1.0)
    mat_sin = create_material(name="SiN", permittivity=(2.0) ** 2)

    dev = create_solver(
        algorithm=Algorithm.RCWA,
        precision=Precision.DOUBLE,
        rdim=[32, 32],
        kdim=[5, 5],
        lam0=np.array([1.55]),
        is_use_FFF=True,
    )
    dev.update_ref_material(ref_material=mat_air)
    dev.update_trn_material(trn_material=mat_air)
    dev.add_layer(material_name=mat_sin, thickness=torch.tensor(0.35, dtype=torch.float64), is_homogeneous=False)

    mask = torch.rand((32, 32), dtype=dev.tfloat, device=dev.device)

    from torchrdit import vector as vector_module  # noqa: WPS433

    instantiations = 0
    real_generator_cls = vector_module.TangentFieldGenerator

    class CountingGenerator(real_generator_cls):
        def __init__(self, *args, **kwargs):
            nonlocal instantiations
            instantiations += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(vector_module, "TangentFieldGenerator", CountingGenerator)

    dev._calculate_vector_field(mask=mask)
    dev._calculate_vector_field(mask=mask)

    assert instantiations == 1


def test_calculate_vector_field_reuses_toeplitz_inputs(monkeypatch):
    """`_calculate_vector_field` should avoid redundant Toeplitz builds."""
    mat_air = create_material(name="Air", permittivity=1.0)
    mat_sin = create_material(name="SiN", permittivity=(2.0) ** 2)

    dev = create_solver(
        algorithm=Algorithm.RCWA,
        precision=Precision.DOUBLE,
        rdim=[32, 32],
        kdim=[5, 5],
        lam0=np.array([1.55]),
        is_use_FFF=True,
    )
    dev.update_ref_material(ref_material=mat_air)
    dev.update_trn_material(trn_material=mat_air)
    dev.add_layer(material_name=mat_sin, thickness=torch.tensor(0.35, dtype=torch.float64), is_homogeneous=False)

    mask = torch.rand((32, 32), dtype=dev.tfloat, device=dev.device)

    call_sequence = []
    original_toeplitz = dev.layer_manager._gen_toeplitz2d

    def _counting_toeplitz(field, *args, **kwargs):
        call_sequence.append(field)
        return original_toeplitz(field, *args, **kwargs)

    monkeypatch.setattr(dev.layer_manager, "_gen_toeplitz2d", _counting_toeplitz)

    dev._calculate_vector_field(mask=mask)

    assert len(call_sequence) == 4
    assert torch.allclose(call_sequence[3], torch.conj(call_sequence[2]))


def test_calculate_vector_field_reuses_toeplitz_inputs_unit_batch(monkeypatch):
    """Toeplitz simplification should also hold for singleton-batch masks."""
    mat_air = create_material(name="Air", permittivity=1.0)
    mat_sin = create_material(name="SiN", permittivity=(2.0) ** 2)

    dev = create_solver(
        algorithm=Algorithm.RCWA,
        precision=Precision.DOUBLE,
        rdim=[32, 32],
        kdim=[5, 5],
        lam0=np.array([1.55]),
        is_use_FFF=True,
    )
    dev.update_ref_material(ref_material=mat_air)
    dev.update_trn_material(trn_material=mat_air)
    dev.add_layer(material_name=mat_sin, thickness=torch.tensor(0.35, dtype=torch.float64), is_homogeneous=False)

    mask = torch.rand((32, 32), dtype=dev.tfloat, device=dev.device).unsqueeze(0)

    call_sequence = []
    original_toeplitz = dev.layer_manager._gen_toeplitz2d

    def _counting_toeplitz(field, *args, **kwargs):
        call_sequence.append(field)
        return original_toeplitz(field, *args, **kwargs)

    monkeypatch.setattr(dev.layer_manager, "_gen_toeplitz2d", _counting_toeplitz)

    dev._calculate_vector_field(mask=mask)

    assert len(call_sequence) == 4
    assert torch.allclose(call_sequence[3], torch.conj(call_sequence[2]))


def test_calculate_vector_field_reuses_toeplitz_inputs_multi_batch(monkeypatch):
    """Toeplitz simplification should extend to batched masks."""
    mat_air = create_material(name="Air", permittivity=1.0)
    mat_sin = create_material(name="SiN", permittivity=(2.0) ** 2)

    dev = create_solver(
        algorithm=Algorithm.RCWA,
        precision=Precision.DOUBLE,
        rdim=[32, 32],
        kdim=[5, 5],
        lam0=np.array([1.55]),
        is_use_FFF=True,
    )
    dev.update_ref_material(ref_material=mat_air)
    dev.update_trn_material(trn_material=mat_air)
    dev.add_layer(material_name=mat_sin, thickness=torch.tensor(0.35, dtype=torch.float64), is_homogeneous=False)

    batch_size = 3
    mask = torch.rand((batch_size, 32, 32), dtype=dev.tfloat, device=dev.device)

    call_sequence = []
    original_toeplitz = dev.layer_manager._gen_toeplitz2d

    def _counting_toeplitz(field, *args, **kwargs):
        call_sequence.append(field)
        return original_toeplitz(field, *args, **kwargs)

    monkeypatch.setattr(dev.layer_manager, "_gen_toeplitz2d", _counting_toeplitz)

    dev._calculate_vector_field(mask=mask)

    assert len(call_sequence) == 4
    assert call_sequence[0].shape[0] == batch_size
    assert torch.allclose(call_sequence[3], torch.conj(call_sequence[2]))


def test_solver_get_vector_components_matches_generator():
    """`get_vector_components` should reproduce tangent generator results."""
    mat_air = create_material(name="Air", permittivity=1.0)
    mat_sin = create_material(name="SiN", permittivity=(2.0) ** 2)

    dev = create_solver(
        algorithm=Algorithm.RCWA,
        precision=Precision.DOUBLE,
        rdim=[64, 64],
        kdim=[5, 5],
        lam0=np.array([1.55]),
        is_use_FFF=True,
    )
    dev.update_ref_material(ref_material=mat_air)
    dev.update_trn_material(trn_material=mat_air)
    dev.add_layer(material_name=mat_sin, thickness=torch.tensor(0.35, dtype=torch.float64), is_homogeneous=False)

    mask = _hex_unit_mask(a=0.8, r=0.25, solver=dev).to(dev.tfloat)
    dev.update_er_with_mask(mask=mask, layer_index=0, method="FFT")

    solver_tx, solver_ty = dev.get_vector_components(layer_index=0)
    assert solver_tx is not None and solver_ty is not None

    from torchrdit import vector as vector_module  # noqa: WPS433

    expected_tx, expected_ty = vector_module.compute_tangent_field(
        mask=mask,
        XO=dev.XO,
        YO=dev.YO,
        lattice_t1=dev.lattice_t1,
        lattice_t2=dev.lattice_t2,
        kdim=tuple(dev.kdim),
        scheme="POL",
        fourier_loss_weight=1e-2,
        smoothness_loss_weight=1e-3,
    )

    assert torch.allclose(solver_tx, expected_tx.to(solver_tx.dtype), atol=1e-6, rtol=1e-6)
    assert torch.allclose(solver_ty, expected_ty.to(solver_ty.dtype), atol=1e-6, rtol=1e-6)


def _build_basic_solver(rdim=32, kdim=3):
    dev = create_solver(
        algorithm=Algorithm.RCWA,
        precision=Precision.DOUBLE,
        rdim=[rdim, rdim],
        kdim=[kdim, kdim],
        lam0=np.array([1.55]),
        lengthunit="um",
        is_use_FFF=False,
    )
    mat_air = create_material(name="Air", permittivity=1.0)
    mat_sin = create_material(name="SiN", permittivity=(2.0) ** 2)
    dev.update_ref_material(ref_material=mat_air)
    dev.update_trn_material(trn_material=mat_air)
    return dev, mat_air, mat_sin


def test_layer_slicing_matches_manual_stack():
    """Sliced layers should match manually duplicated stacks."""
    slice_count = 3
    total_thickness = torch.tensor(0.36, dtype=torch.float64)

    dev_sliced, mat_air, mat_sin = _build_basic_solver()
    dev_manual, _, _ = _build_basic_solver()

    mask = torch.zeros(dev_sliced.rdim, dtype=dev_sliced.tfloat)
    mask[8:24, 8:24] = 1.0

    dev_sliced.add_layer(
        material_name=mat_sin,
        thickness=total_thickness.clone(),
        is_homogeneous=False,
        slice_count=slice_count,
    )
    dev_sliced.add_layer(material_name=mat_air, thickness=torch.tensor(0.1, dtype=torch.float64), is_homogeneous=True)
    dev_sliced.update_er_with_mask(mask=mask, layer_index=0, method="FFT")
    assert getattr(dev_sliced.layer_manager.layers[0], "slice_count") == slice_count

    per_slice_thickness = total_thickness / slice_count
    for idx in range(slice_count):
        dev_manual.add_layer(
            material_name=mat_sin,
            thickness=per_slice_thickness.clone(),
            is_homogeneous=False,
        )
        dev_manual.update_er_with_mask(mask=mask, layer_index=idx, method="FFT")
    dev_manual.add_layer(material_name=mat_air, thickness=torch.tensor(0.1, dtype=torch.float64), is_homogeneous=True)

    src_sliced = dev_sliced.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
    src_manual = dev_manual.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)

    result_sliced = dev_sliced.solve(src_sliced, is_use_FFF=False)
    result_manual = dev_manual.solve(src_manual, is_use_FFF=False)

    torch.testing.assert_close(result_sliced.transmission, result_manual.transmission, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(result_sliced.reflection, result_manual.reflection, atol=1e-6, rtol=1e-4)


def test_layer_slicing_preserves_gradients():
    """Repeated sub-slice multiplication must preserve autograd connectivity."""
    slice_count = 2
    total_thickness = torch.tensor(0.24, dtype=torch.float64)

    dev, mat_air, mat_sin = _build_basic_solver()

    mask = torch.rand(dev.rdim, dtype=dev.tfloat)
    mask.requires_grad_(True)

    dev.add_layer(
        material_name=mat_sin,
        thickness=total_thickness,
        is_homogeneous=False,
        slice_count=slice_count,
    )
    dev.update_er_with_mask(mask=mask, layer_index=0, method="FFT")

    src = dev.add_source(theta=0.0, phi=0.0, pte=1.0, ptm=0.0)
    result = dev.solve(src, is_use_FFF=False)
    loss = result.transmission.sum()
    loss.backward()

    assert mask.grad is not None
    assert torch.all(torch.isfinite(mask.grad))


def test_layer_slice_count_survives_replace_to_grating():
    """update_er_with_mask should keep the configured slice_count even after conversion to grating."""
    slice_count = 5
    dev, mat_air, mat_sin = _build_basic_solver()
    dev.add_layer(
        material_name=mat_sin,
        thickness=torch.tensor(0.18, dtype=torch.float64),
        is_homogeneous=True,
        slice_count=slice_count,
    )

    mask = torch.ones(dev.rdim, dtype=dev.tfloat)
    dev.update_er_with_mask(mask=mask, layer_index=0, method="FFT")

    assert dev.layer_manager.layers[0].slice_count == slice_count


def test_replace_layer_preserves_slice_count():
    """LayerManager replacements should retain slice_count metadata."""
    slice_count = 3
    dev, _, mat_sin = _build_basic_solver()
    dev.add_layer(
        material_name=mat_sin,
        thickness=torch.tensor(0.12, dtype=torch.float64),
        is_homogeneous=False,
        slice_count=slice_count,
    )

    dev.layer_manager.replace_layer_to_homogeneous(layer_index=0)
    assert dev.layer_manager.layers[0].slice_count == slice_count

    dev.layer_manager.replace_layer_to_grating(layer_index=0)
    assert dev.layer_manager.layers[0].slice_count == slice_count


@pytest.mark.parametrize(
    "algorithm, precision, expected_dtype",
    [
        (Algorithm.RCWA, Precision.DOUBLE, np.float64),
        (Algorithm.RDIT, Precision.DOUBLE, np.float64),
        (Algorithm.RCWA, Precision.SINGLE, np.float32),
        (Algorithm.RDIT, Precision.SINGLE, np.float32),
    ],
)
def test_hex_cell_transmission_reflection_golden(algorithm, precision, expected_dtype):
    """Golden numerical validation on a hex-cell GMRF structure for both algorithms/precisions.

    Validates transmission/reflection against reference values and checks dtype matches precision.
    """
    a = 1150 * NM
    r = 400 * NM
    h1 = torch.tensor(230 * NM, dtype=torch.float32)
    h2 = torch.tensor(345 * NM, dtype=torch.float32)
    t1 = torch.tensor([[a / 2, -a * np.sqrt(3) / 2]], dtype=torch.float32)
    t2 = torch.tensor([[a / 2, a * np.sqrt(3) / 2]], dtype=torch.float32)

    # Materials
    n_SiO, n_SiN, n_fs = 1.4496, 1.9360, 1.5100
    mat_sio = create_material(name='SiO', permittivity=n_SiO**2)
    mat_sin = create_material(name='SiN', permittivity=n_SiN**2)
    mat_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

    dev = create_solver(
        algorithm=algorithm,
        precision=precision,
        rdim=[512, 512],
        kdim=[9, 9],
        lam0=np.array([1540 * NM]),
        lengthunit='um',
        t1=t1,
        t2=t2,
    )
    if algorithm == Algorithm.RDIT:
        # Keep order consistent with previous golden runs
        dev.set_rdit_order(10)

    dev.update_trn_material(trn_material=mat_fs)
    dev.add_layer(material_name=mat_sio, thickness=h1, is_homogeneous=False, is_optimize=True)
    dev.add_layer(material_name=mat_sin, thickness=h2, is_homogeneous=True, is_optimize=False)

    src = dev.add_source(theta=0 * DEG, phi=0 * DEG, pte=1, ptm=0)

    mask = _hex_unit_mask(a=a, r=r, solver=dev)
    dev.update_er_with_mask(mask=mask, layer_index=0)
    data = dev.solve(src)

    T = data.transmission[0].detach().numpy()
    R = data.reflection[0].detach().numpy()

    # Golden values with tolerance consistent with prior tests
    assert np.isclose(T, 0.92, atol=2e-2)
    assert np.isclose(R, 0.07, atol=2e-2)
    # Precision/dtype contract
    assert T.dtype == expected_dtype
    assert R.dtype == expected_dtype


def _make_analytical_mask(radius_var, dev):
    rsq = (dev.XO) ** 2 + (dev.YO) ** 2
    mask = rsq - (radius_var) ** 2 + 0.5
    return operator_proj(mask, eta=0.5, beta=100)


@pytest.mark.parametrize("algorithm", [Algorithm.RCWA, Algorithm.RDIT])
def test_analytical_vs_fft_parity(algorithm):
    """Parity check between Analytical and FFT permittivity expansions.

    Uses a circular inclusion; requires close agreement between methods.
    """
    a = 1150 * NM
    r = 400 * NM
    h1 = torch.tensor(230 * NM, dtype=torch.float32)
    h2 = torch.tensor(345 * NM, dtype=torch.float32)
    t1 = torch.tensor([[a, 0]], dtype=torch.float32)
    t2 = torch.tensor([[0, a]], dtype=torch.float32)

    # Materials
    n_SiO, n_SiN, n_fs = 1.4496, 1.9360, 1.5100
    mat_sio = create_material(name='SiO', permittivity=n_SiO**2)
    mat_sin = create_material(name='SiN', permittivity=n_SiN**2)
    mat_fs = create_material(name='FusedSilica', permittivity=n_fs**2)

    dev = create_solver(
        algorithm=algorithm,
        precision=Precision.DOUBLE,
        rdim=[512, 512],
        kdim=[9, 9],
        lam0=np.array([1540 * NM]),
        lengthunit='um',
        t1=t1,
        t2=t2,
    )
    if algorithm == Algorithm.RDIT:
        dev.set_rdit_order(10)

    dev.update_trn_material(trn_material=mat_fs)
    dev.add_layer(material_name=mat_sio, thickness=h1, is_homogeneous=False, is_optimize=True)
    dev.add_layer(material_name=mat_sin, thickness=h2, is_homogeneous=True, is_optimize=False)
    src = dev.add_source(theta=0 * DEG, phi=0 * DEG, pte=1, ptm=0)

    mask = _make_analytical_mask(r, dev)

    dev.update_er_with_mask(mask=mask, layer_index=0, method='Analytical')
    data_ana = dev.solve(src)
    dev.update_er_with_mask(mask=mask, layer_index=0, method='FFT')
    data_fft = dev.solve(src, is_use_FFF=False)

    # Compare with explicit tolerance rather than integer rounding
    assert np.isclose(
        data_fft.transmission[0].item(), data_ana.transmission[0].item(), atol=1e-3
    )
    assert np.isclose(
        data_fft.reflection[0].item(), data_ana.reflection[0].item(), atol=1e-3
    )
    # Sanity: both are double
    assert data_ana.transmission[0].detach().numpy().dtype == np.float64
    assert data_fft.transmission[0].detach().numpy().dtype == np.float64


def test_dispersive_nonhomogeneous_layer_energy_and_values():
    """Dispersive materials in a non-homogeneous top layer; energy sanity + expected band.

    Uses real dispersive data files Si_C-e.txt and SiO2-e.txt. This keeps a single,
    representative dispersive test with a realistic geometry and tolerances that
    account for reduced k-space truncation in tests.
    """
    a = 1150 * NM
    b = a * np.sqrt(3)
    r = 400 * NM
    h1 = torch.tensor(230 * NM, dtype=torch.float64)
    h2 = torch.tensor(345 * NM, dtype=torch.float64)
    t1 = torch.tensor([[a / 2, -a * np.sqrt(3) / 2]], dtype=torch.float64)
    t2 = torch.tensor([[a / 2, a * np.sqrt(3) / 2]], dtype=torch.float64)

    # Builder-based solver with manageable sizes for CI
    dev = (
        get_solver_builder()
        .with_algorithm(Algorithm.RDIT)
        .with_precision(Precision.DOUBLE)
        .with_real_dimensions([256, 256])
        .with_k_dimensions([9, 9])
        .with_wavelengths(np.array([1540 * NM]))
        .with_length_unit('um')
        .with_lattice_vectors(t1, t2)
        .with_fff(True)
        .build()
    )

    # Dispersive and non-dispersive materials
    base = os.path.dirname(__file__)
    mat_sic = create_material(
        name='SiC', dielectric_dispersion=True,
        user_dielectric_file=os.path.join(base, 'Si_C-e.txt'),
        data_format='freq-eps', data_unit='thz'
    )
    mat_sio2 = create_material(
        name='SiO2', dielectric_dispersion=True,
        user_dielectric_file=os.path.join(base, 'SiO2-e.txt'),
        data_format='freq-eps', data_unit='thz'
    )
    mat_sin = create_material(name='SiN', permittivity=(1.9360) ** 2)
    mat_fs = create_material(name='FusedSilica', permittivity=(1.5100) ** 2)

    dev.add_materials([mat_sic, mat_sio2, mat_sin, mat_fs])
    dev.update_trn_material(trn_material=mat_fs)

    # Non-homogeneous top (dispersive SiC), homogeneous bottom (SiN)
    dev.add_layer(material_name=mat_sic, thickness=h1, is_homogeneous=False)
    dev.add_layer(material_name=mat_sin, thickness=h2, is_homogeneous=True)

    src = dev.add_source(theta=0 * DEG, phi=0 * DEG, pte=1, ptm=0)
    sg = ShapeGenerator.from_solver(dev)
    c1 = sg.generate_circle_mask(center=[0, b / 2], radius=r)
    c2 = sg.generate_circle_mask(center=[0, -b / 2], radius=r)
    c3 = sg.generate_circle_mask(center=[a / 2, 0], radius=r)
    c4 = sg.generate_circle_mask(center=[-a / 2, 0], radius=r)
    mask = sg.combine_masks(mask1=c1, mask2=c2, operation='union')
    mask = sg.combine_masks(mask1=mask, mask2=c3, operation='union')
    mask = sg.combine_masks(mask1=mask, mask2=c4, operation='union')
    mask = (1 - mask).to(torch.float64)
    mask.requires_grad = True

    dev.update_er_with_mask(mask=mask, layer_index=0)
    data = dev.solve(src)

    # Energy sanity
    T = data.transmission[0].item()
    R = data.reflection[0].item()
    assert 0 <= T <= 1
    assert 0 <= R <= 1
    assert T + R <= 1 + 1e-4

    # Expected band based on validated results; loosened due to reduced k-dim
    expected_T, expected_R = 0.43316791878804345, 0.5431967324784357
    assert expected_T - 0.1 <= T <= expected_T + 0.1
    assert expected_R - 0.1 <= R <= expected_R + 0.1
