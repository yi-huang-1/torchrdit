import pytest


def _base_config():
    return {
        "algorithm": "RCWA",
        "wavelengths": [1.55],
        "grids": [16, 16],
        "harmonics": [3, 3],
    }


def test_builder_accepts_grids_harmonics_keys():
    from torchrdit.builder import SolverBuilder

    builder = SolverBuilder().from_config(_base_config())
    solver = builder.build()
    assert solver.grids == [16, 16]
    assert solver.harmonics == [3, 3]


@pytest.mark.parametrize(
    "bad_key, hint",
    [
        ("r" + "dim", "grids"),
        ("k" + "dim", "harmonics"),
    ],
)
def test_builder_rejects_legacy_dimension_keys(bad_key, hint):
    from torchrdit.builder import SolverBuilder

    config = {
        "algorithm": "RCWA",
        "wavelengths": [1.55],
        bad_key: [16, 16] if bad_key == "grids" else [3, 3],
    }

    with pytest.raises(ValueError, match=rf"{bad_key}.*{hint}"):
        SolverBuilder().from_config(config)


def test_builder_warns_on_interface_only_keys():
    from torchrdit.builder import SolverBuilder

    config = _base_config()
    config["sources"] = {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0}
    config["vars"] = {"$t": 0.1}
    config["output"] = {"type": "torch"}

    with pytest.warns(UserWarning, match=r"interface\.py"):
        SolverBuilder().from_config(config)
