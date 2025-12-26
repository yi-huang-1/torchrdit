import numpy as np
import pytest


def _minimal_spec(*, output_type: str):
    return {
        "solver": {
            "algorithm": "RCWA",
            "precision": "SINGLE",
            "wavelengths": [1.55],
            "length_unit": "um",
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "cpu",
        },
        "materials": {
            "Si": {"permittivity": 12.0},
        },
        "layers": [
            {"material": "Si", "thickness": 0.2, "is_homogeneous": True},
        ],
        "sources": {"name": "s0", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": output_type},
    }


def test_solve_returns_structured_dict_torch():
    from torchrdit.interface import solve
    import torch

    out = solve(_minimal_spec(output_type="torch"))
    assert isinstance(out, dict)
    assert "efficiency" in out
    assert "TRN" not in out  # interface is isolated from legacy keys
    assert isinstance(out["efficiency"]["transmission"], torch.Tensor)


def test_solve_returns_structured_dict_numpy():
    from torchrdit.interface import solve

    out = solve(_minimal_spec(output_type="numpy"))
    assert isinstance(out, dict)
    assert "efficiency" in out
    assert "TRN" not in out  # interface is isolated from legacy keys
    assert isinstance(out["efficiency"]["transmission"], np.ndarray)


def test_solve_multi_source_returns_dict_by_name():
    from torchrdit.interface import solve

    spec = _minimal_spec(output_type="torch")
    spec["sources"] = [
        {"name": "TE", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        {"name": "TM", "theta": 0.0, "phi": 0.0, "pte": 0.0, "ptm": 1.0},
    ]

    out = solve(spec)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"TE", "TM"}
    assert "efficiency" in out["TE"]
    assert "efficiency" in out["TM"]


@pytest.mark.parametrize("shape_type", ["circle", "rectangle", "polygon"])
def test_solve_patterned_layer_shapes(shape_type):
    from torchrdit.interface import solve

    spec = {
        "solver": {
            "algorithm": "RCWA",
            "precision": "SINGLE",
            "wavelengths": [1.55],
            "length_unit": "um",
            "grids": [32, 32],
            "harmonics": [3, 3],
            "device": "cpu",
        },
        "materials": {"Si": {"permittivity": 12.0}, "SiO2": {"permittivity": 2.25}},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.2,
                "is_homogeneous": False,
                "pattern": {
                    "bg_material": "SiO2",
                    "shapes": [],
                    "layer_shape": "s1",
                },
            }
        ],
        "sources": {"name": "s0", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    if shape_type == "circle":
        spec["layers"][0]["pattern"]["shapes"] = [{"name": "s1", "type": "circle", "center": [0.0, 0.0], "radius": 0.2}]
    elif shape_type == "rectangle":
        spec["layers"][0]["pattern"]["shapes"] = [
            {"name": "s1", "type": "rectangle", "center": [0.0, 0.0], "x_size": 0.4, "y_size": 0.2, "angle": 0.0}
        ]
    elif shape_type == "polygon":
        spec["layers"][0]["pattern"]["shapes"] = [
            {
                "name": "s1",
                "type": "polygon",
                "points": [[-0.2, -0.2], [0.2, -0.2], [0.0, 0.2]],
                "soft_edge": 0.0,
            }
        ]

    out = solve(spec)
    assert isinstance(out, dict)
    assert "efficiency" in out


def test_solve_vars_support_expression_reference():
    from torchrdit.interface import solve

    spec = {
        "solver": {
            "algorithm": "RCWA",
            "precision": "SINGLE",
            "wavelengths": [1.55],
            "length_unit": "um",
            "grids": [16, 16],
            "harmonics": [3, 3],
            "device": "cpu",
        },
        "materials": {"Si": {"permittivity": 12.0}},
        "vars": {"$s": 0.1, "$t": "$s + 0.1"},
        "layers": [{"material": "Si", "thickness": "$t", "is_homogeneous": True}],
        "sources": {"name": "s0", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    out = solve(spec)
    assert isinstance(out, dict)
    assert "efficiency" in out
