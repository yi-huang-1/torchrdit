import os

import numpy as np
import pytest
import torch

from torchrdit.material_proxy import MaterialDataProxy
from torchrdit.utils import create_material


def _load_freq_eps_file_as_wl_eps_table(file_path: str) -> np.ndarray:
    proxy = MaterialDataProxy()
    return proxy.load_data(file_path, data_format="freq-eps", data_unit="thz", target_unit="um")


def _table_to_eps_complex(table: np.ndarray) -> np.ndarray:
    eps1 = table[:, 1]
    eps2 = table[:, 2]
    return (eps1 - 1j * eps2).astype(np.complex128)


def _table_to_nk(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eps1 = table[:, 1].astype(float)
    eps2 = table[:, 2].astype(float)

    # For epsilon = eps1 - 1j*eps2 (eps2 >= 0), compute n,k such that:
    # (n - 1j*k)^2 = epsilon  <=>  epsilon = (n^2 - k^2) - 1j*(2*n*k)
    abs_eps = np.sqrt(eps1**2 + eps2**2)
    n = np.sqrt(np.clip((abs_eps + eps1) / 2.0, 0.0, None))
    k = np.sqrt(np.clip((abs_eps - eps1) / 2.0, 0.0, None))
    return n, k


@pytest.mark.parametrize(
    "filename,mat_name",
    [
        ("Si_C-e.txt", "SiC"),
        ("SiO2-e.txt", "SiO2"),
    ],
)
def test_inmemory_dispersive_matches_file_based_for_reference_files(filename: str, mat_name: str):
    base = os.path.dirname(__file__)
    file_path = os.path.join(base, filename)

    table = _load_freq_eps_file_as_wl_eps_table(file_path)
    wavelengths_um = table[:, 0].astype(float)
    eps_complex = _table_to_eps_complex(table)
    n, k = _table_to_nk(table)

    mat_file = create_material(
        name=f"{mat_name}_file",
        dielectric_dispersion=True,
        user_dielectric_file=file_path,
        data_format="freq-eps",
        data_unit="thz",
    )
    mat_mem_eps = create_material(
        name=f"{mat_name}_mem_eps",
        dielectric_dispersion=True,
        user_dielectric_wavelengths_um=wavelengths_um.tolist(),
        user_dielectric_eps=eps_complex.tolist(),
    )
    mat_mem_nk = create_material(
        name=f"{mat_name}_mem_nk",
        dielectric_dispersion=True,
        user_dielectric_wavelengths_um=wavelengths_um.tolist(),
        user_dielectric_n=n.tolist(),
        user_dielectric_k=k.tolist(),
    )

    # Compare at several in-range wavelengths (use table points to avoid interpolation artifacts)
    idx = np.unique(np.linspace(0, len(wavelengths_um) - 1, 7, dtype=int))
    test_wls = wavelengths_um[idx]

    eps_file = mat_file.get_permittivity(test_wls, "um")
    eps_mem_eps = mat_mem_eps.get_permittivity(test_wls, "um")
    eps_mem_nk = mat_mem_nk.get_permittivity(test_wls, "um")

    torch.testing.assert_close(eps_mem_eps, eps_file, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(eps_mem_nk, eps_file, rtol=1e-6, atol=1e-6)


def test_inmemory_eps_normalizes_positive_imag_convention():
    wavelengths_um = [1.0, 1.5, 2.0]
    eps_convention = [2.1 - 0.10j, 2.05 - 0.08j, 2.0 - 0.05j]
    eps_opposite = [complex(e.real, -e.imag) for e in eps_convention]  # flip sign to +Im

    mat_ok = create_material(
        name="eps_ok",
        dielectric_dispersion=True,
        user_dielectric_wavelengths_um=wavelengths_um,
        user_dielectric_eps=eps_convention,
    )
    mat_flip = create_material(
        name="eps_flip",
        dielectric_dispersion=True,
        user_dielectric_wavelengths_um=wavelengths_um,
        user_dielectric_eps=eps_opposite,
    )

    test_wls = np.array([1.0, 1.5, 2.0], dtype=float)
    eps_ok = mat_ok.get_permittivity(test_wls, "um")
    eps_flip = mat_flip.get_permittivity(test_wls, "um")
    torch.testing.assert_close(eps_ok, eps_flip, rtol=0, atol=0)


def test_inmemory_inputs_are_mutually_exclusive_with_file():
    base = os.path.dirname(__file__)
    file_path = os.path.join(base, "SiO2-e.txt")

    with pytest.raises(ValueError, match="cannot be used together with user_dielectric_file"):
        create_material(
            name="bad_mix",
            dielectric_dispersion=True,
            user_dielectric_file=file_path,
            data_format="freq-eps",
            data_unit="thz",
            user_dielectric_wavelengths_um=[1.0, 2.0],
            user_dielectric_eps=[2.0 - 0.0j, 2.1 - 0.0j],
        )


def test_inmemory_eps_and_nk_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either .* not both"):
        create_material(
            name="bad_mix2",
            dielectric_dispersion=True,
            user_dielectric_wavelengths_um=[1.0, 2.0],
            user_dielectric_eps=[2.0 - 0.0j, 2.1 - 0.0j],
            user_dielectric_n=[1.4, 1.45],
            user_dielectric_k=[0.0, 0.0],
        )


def test_inmemory_requires_dielectric_dispersion_true():
    with pytest.raises(ValueError, match="dielectric_dispersion must be True"):
        create_material(
            name="bad_flag",
            dielectric_dispersion=False,
            user_dielectric_wavelengths_um=[1.0, 2.0],
            user_dielectric_eps=[2.0 - 0.0j, 2.1 - 0.0j],
        )

