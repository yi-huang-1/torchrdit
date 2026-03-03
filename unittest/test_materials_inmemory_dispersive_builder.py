import os

import numpy as np
import torch

from torchrdit.builder import _create_materials
from torchrdit.material_proxy import MaterialDataProxy


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


def test_builder_inmemory_dispersion_matches_file_based():
    base = os.path.dirname(__file__)
    file_path = os.path.join(base, "SiO2-e.txt")

    table = _load_freq_eps_file_as_wl_eps_table(file_path)
    wavelengths_um = table[:, 0].astype(float)
    eps_complex = _table_to_eps_complex(table)
    n, k = _table_to_nk(table)

    mats = _create_materials(
        {
            "file": {
                "dielectric_dispersion": True,
                "dielectric_file": file_path,
                "data_format": "freq-eps",
                "data_unit": "thz",
            },
            "mem_eps": {
                "dielectric_dispersion": True,
                "dispersion": {"wavelengths_um": wavelengths_um.tolist(), "eps": eps_complex.tolist()},
            },
            "mem_nk": {
                "dielectric_dispersion": True,
                "dispersion": {"wavelengths_um": wavelengths_um.tolist(), "n": n.tolist(), "k": k.tolist()},
            },
        },
        config_dir=base,
    )

    idx = np.unique(np.linspace(0, len(wavelengths_um) - 1, 7, dtype=int))
    test_wls = wavelengths_um[idx]

    eps_file = mats["file"].get_permittivity(test_wls, "um")
    eps_mem_eps = mats["mem_eps"].get_permittivity(test_wls, "um")
    eps_mem_nk = mats["mem_nk"].get_permittivity(test_wls, "um")

    torch.testing.assert_close(eps_mem_eps, eps_file, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(eps_mem_nk, eps_file, rtol=1e-6, atol=1e-6)


def test_builder_dispersion_n_only_defaults_k_to_zero():
    mats = _create_materials(
        {
            "n_only": {
                "dielectric_dispersion": True,
                "dispersion": {"wavelengths_um": [1.0, 2.0], "n": [1.5, 1.6]},
            },
            "nk0": {
                "dielectric_dispersion": True,
                "dispersion": {"wavelengths_um": [1.0, 2.0], "n": [1.5, 1.6], "k": [0.0, 0.0]},
            },
        },
        config_dir=None,
    )
    test_wls = np.array([1.0, 1.5, 2.0], dtype=float)
    eps_n_only = mats["n_only"].get_permittivity(test_wls, "um")
    eps_nk0 = mats["nk0"].get_permittivity(test_wls, "um")
    torch.testing.assert_close(eps_n_only, eps_nk0, rtol=0, atol=0)


def test_builder_single_point_nk_dispersion_is_supported():
    n = 1.5
    k = 0.1
    mats = _create_materials(
        {
            "single_nk": {
                "dielectric_dispersion": True,
                "dispersion": {"wavelengths_um": [1.55], "n": [n], "k": [k]},
            }
        },
        config_dir=None,
    )

    test_wls = np.array([1.0, 1.55, 2.0], dtype=float)
    eps = mats["single_nk"].get_permittivity(test_wls, "um")
    expected = torch.full_like(eps, (n**2 - k**2) - 1j * (2 * n * k))
    torch.testing.assert_close(eps, expected, rtol=1e-6, atol=1e-6)
