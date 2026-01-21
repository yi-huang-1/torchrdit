import pytest


def test_builder_rejects_unknown_material_keys():
    from torchrdit.solver import create_solver_from_config

    config = {
        "algorithm": "RCWA",
        "wavelengths": [1.55],
        "grids": [16, 16],
        "harmonics": [3, 3],
        "materials": {
            "Si": {"permittivity": 12.0, "unknown": 1},
        },
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True}],
    }

    with pytest.raises(ValueError, match=r"materials.*Si.*unknown"):
        create_solver_from_config(config)


def test_builder_rejects_unknown_layer_keys():
    from torchrdit.solver import create_solver_from_config

    config = {
        "algorithm": "RCWA",
        "wavelengths": [1.55],
        "grids": [16, 16],
        "harmonics": [3, 3],
        "materials": {
            "Si": {"permittivity": 12.0},
        },
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True, "unknown": 1}],
    }

    with pytest.raises(ValueError, match=r"layers\[0\].*unknown"):
        create_solver_from_config(config)


def test_builder_rejects_base_path_key():
    from torchrdit.solver import create_solver_from_config

    config = {
        "algorithm": "RCWA",
        "wavelengths": [1.55],
        "grids": [16, 16],
        "harmonics": [3, 3],
        "base_path": ".",
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True}],
    }

    with pytest.raises(ValueError, match=r"base_path"):
        create_solver_from_config(config)
