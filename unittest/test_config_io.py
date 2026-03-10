import pytest




def _flat_solver_config():
    return {
        "algorithm": "RCWA",
        "wavelengths": [1.55],
        "grids": [16, 16],
        "harmonics": [3, 3],
    }




def test_detect_config_shape_flat_solver():
    from torchrdit.config_io import detect_config_shape

    assert detect_config_shape(_flat_solver_config()) == "flat_solver"




def test_builder_rejects_auto_harmonics():
    from torchrdit.builder import SolverBuilder

    config = _flat_solver_config()
    config["harmonics"] = "auto"
    config["maxG"] = 25

    with pytest.raises(ValueError, match="auto"):
        SolverBuilder().from_config(config)
