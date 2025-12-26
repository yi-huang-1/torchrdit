import os

import numpy as np
import pytest
import torch

from torchrdit.builder import _create_materials
from torchrdit.interface import SpecError, _parse_materials
from torchrdit.material_proxy import MaterialDataProxy


def _load_freq_eps_file_as_wl_eps_table(file_path: str) -> np.ndarray:
    proxy = MaterialDataProxy()
    return proxy.load_data(file_path, data_format="freq-eps", data_unit="thz", target_unit="um")


def _table_to_eps_complex(table: np.ndarray) -> np.ndarray:
    eps1 = table[:, 1]
    eps2 = table[:, 2]
    return (eps1 - 1j * eps2).astype(np.complex128)


def _table_to_eps_opposite_convention(table: np.ndarray) -> np.ndarray:
    eps1 = table[:, 1]
    eps2 = table[:, 2]
    return (eps1 + 1j * eps2).astype(np.complex128)


def _table_to_nk(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eps1 = table[:, 1].astype(float)
    eps2 = table[:, 2].astype(float)

    # For epsilon = eps1 - 1j*eps2 (eps2 >= 0), compute n,k such that:
    # (n - 1j*k)^2 = epsilon  <=>  epsilon = (n^2 - k^2) - 1j*(2*n*k)
    abs_eps = np.sqrt(eps1**2 + eps2**2)
    n = np.sqrt(np.clip((abs_eps + eps1) / 2.0, 0.0, None))
    k = np.sqrt(np.clip((abs_eps - eps1) / 2.0, 0.0, None))
    return n, k


@pytest.mark.parametrize("filename", ["Si_C-e.txt", "SiO2-e.txt"])
def test_interface_inmemory_dispersion_matches_file_based(filename: str):
    base = os.path.dirname(__file__)
    file_path = os.path.join(base, filename)

    table = _load_freq_eps_file_as_wl_eps_table(file_path)
    wavelengths_um = table[:, 0].astype(float)
    eps_complex = _table_to_eps_complex(table)
    n, k = _table_to_nk(table)

    mats = _parse_materials(
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
        caller_dir=base,
    )

    idx = np.unique(np.linspace(0, len(wavelengths_um) - 1, 7, dtype=int))
    test_wls = wavelengths_um[idx]

    eps_file = mats["file"].get_permittivity(test_wls, "um")
    eps_mem_eps = mats["mem_eps"].get_permittivity(test_wls, "um")
    eps_mem_nk = mats["mem_nk"].get_permittivity(test_wls, "um")

    torch.testing.assert_close(eps_mem_eps, eps_file, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(eps_mem_nk, eps_file, rtol=1e-6, atol=1e-6)


def test_interface_dispersion_n_only_defaults_k_to_zero():
    mats = _parse_materials(
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
        caller_dir=None,
    )
    test_wls = np.array([1.0, 1.5, 2.0], dtype=float)
    eps_n_only = mats["n_only"].get_permittivity(test_wls, "um")
    eps_nk0 = mats["nk0"].get_permittivity(test_wls, "um")
    torch.testing.assert_close(eps_n_only, eps_nk0, rtol=0, atol=0)


def test_interface_inmemory_eps_normalizes_opposite_imag_convention():
    base = os.path.dirname(__file__)
    file_path = os.path.join(base, "SiO2-e.txt")

    table = _load_freq_eps_file_as_wl_eps_table(file_path)
    wavelengths_um = table[:, 0].astype(float)
    eps_ref = _table_to_eps_complex(table)
    eps_opposite = _table_to_eps_opposite_convention(table)

    mats = _parse_materials(
        {
            "ref": {
                "dielectric_dispersion": True,
                "dispersion": {"wavelengths_um": wavelengths_um.tolist(), "eps": eps_ref.tolist()},
            },
            "flip": {
                "dielectric_dispersion": True,
                "dispersion": {"wavelengths_um": wavelengths_um.tolist(), "eps": eps_opposite.tolist()},
            },
        },
        config_dir=base,
        caller_dir=base,
    )

    idx = np.unique(np.linspace(0, len(wavelengths_um) - 1, 7, dtype=int))
    test_wls = wavelengths_um[idx]
    eps_ref_out = mats["ref"].get_permittivity(test_wls, "um")
    eps_flip_out = mats["flip"].get_permittivity(test_wls, "um")
    torch.testing.assert_close(eps_ref_out, eps_flip_out, rtol=0, atol=0)


def test_interface_rejects_dispersion_without_dielectric_dispersion_true():
    with pytest.raises(SpecError, match=r"spec\.materials\.SiO2: dispersion is only allowed when dielectric_dispersion is True"):
        _parse_materials(
            {
                "SiO2": {
                    "dispersion": {
                        "wavelengths_um": [1.0, 2.0],
                        "eps": [2.0 - 0.0j, 2.1 - 0.0j],
                    }
                }
            },
            config_dir=None,
            caller_dir=None,
        )


def test_interface_rejects_dispersion_with_dielectric_file():
    with pytest.raises(SpecError, match=r"spec\.materials\.SiO2 cannot specify both dielectric_file and dispersion"):
        _parse_materials(
            {
                "SiO2": {
                    "dielectric_dispersion": True,
                    "dielectric_file": "x.txt",
                    "dispersion": {
                        "wavelengths_um": [1.0, 2.0],
                        "eps": [2.0 - 0.0j, 2.1 - 0.0j],
                    },
                }
            },
            config_dir=None,
            caller_dir=None,
        )


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


def test_inmemory_dispersion_accepts_numpy_and_torch_inputs():
    wavelengths_um = np.array([1.0, 1.5, 2.0], dtype=float)
    eps = np.array([2.10 - 0.0j, 2.08 - 0.0j, 2.05 - 0.0j], dtype=np.complex128)
    n = np.array([1.45, 1.46, 1.47], dtype=float)
    k = np.array([0.01, 0.02, 0.03], dtype=float)

    ref = _parse_materials(
        {
            "eps": {"dielectric_dispersion": True, "dispersion": {"wavelengths_um": wavelengths_um.tolist(), "eps": eps.tolist()}},
            "nk": {"dielectric_dispersion": True, "dispersion": {"wavelengths_um": wavelengths_um.tolist(), "n": n.tolist(), "k": k.tolist()}},
        },
        config_dir=None,
        caller_dir=None,
    )

    mats_np = _parse_materials(
        {
            "eps": {"dielectric_dispersion": True, "dispersion": {"wavelengths_um": wavelengths_um, "eps": eps}},
            "nk": {"dielectric_dispersion": True, "dispersion": {"wavelengths_um": wavelengths_um, "n": n, "k": k}},
        },
        config_dir=None,
        caller_dir=None,
    )

    mats_torch = _parse_materials(
        {
            "eps": {
                "dielectric_dispersion": True,
                "dispersion": {
                    "wavelengths_um": torch.tensor(wavelengths_um, dtype=torch.float32, requires_grad=True),
                    "eps": torch.tensor(eps, dtype=torch.complex64, requires_grad=True),
                },
            },
            "nk": {
                "dielectric_dispersion": True,
                "dispersion": {
                    "wavelengths_um": torch.tensor(wavelengths_um, dtype=torch.float32),
                    "n": torch.tensor(n, dtype=torch.float32),
                    "k": torch.tensor(k, dtype=torch.float32),
                },
            },
        },
        config_dir=None,
        caller_dir=None,
    )

    test_wls = np.array([1.0, 1.25, 1.8, 2.0], dtype=float)
    for key in ("eps", "nk"):
        torch.testing.assert_close(mats_np[key].get_permittivity(test_wls, "um"), ref[key].get_permittivity(test_wls, "um"))
        torch.testing.assert_close(mats_torch[key].get_permittivity(test_wls, "um"), ref[key].get_permittivity(test_wls, "um"))

    if torch.cuda.is_available():
        mats_cuda = _parse_materials(
            {
                "eps": {
                    "dielectric_dispersion": True,
                    "dispersion": {
                        "wavelengths_um": torch.tensor(wavelengths_um, device="cuda"),
                        "eps": torch.tensor(eps, dtype=torch.complex64, device="cuda"),
                    },
                }
            },
            config_dir=None,
            caller_dir=None,
        )
        torch.testing.assert_close(mats_cuda["eps"].get_permittivity(test_wls, "um"), ref["eps"].get_permittivity(test_wls, "um"))
