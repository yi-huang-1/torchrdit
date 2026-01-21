import pytest


def _interface_spec():
    return {
        "solver": {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
        },
        "materials": {},
        "layers": [],
    }


def _flat_solver_config():
    return {
        "algorithm": "RCWA",
        "wavelengths": [1.55],
        "grids": [16, 16],
        "harmonics": [3, 3],
    }


def test_detect_config_shape_interface():
    from torchrdit.config_io import detect_config_shape

    assert detect_config_shape(_interface_spec()) == "interface"


def test_detect_config_shape_flat_solver():
    from torchrdit.config_io import detect_config_shape

    assert detect_config_shape(_flat_solver_config()) == "flat_solver"


def test_detect_config_shape_rejects_mixed():
    from torchrdit.config_io import detect_config_shape

    config = _interface_spec()
    config["wavelengths"] = [1.55]

    with pytest.raises(ValueError, match="mixed"):
        detect_config_shape(config)


def test_builder_rejects_auto_harmonics():
    from torchrdit.builder import SolverBuilder

    config = _flat_solver_config()
    config["harmonics"] = "auto"
    config["maxG"] = 25

    with pytest.raises(ValueError, match="auto"):
        SolverBuilder().from_config(config)
