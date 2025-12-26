import pytest


def test_optimize_updates_trainable_var_and_supports_multi_source_objective():
    from torchrdit.interface import optimize
    import torch

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
        "vars": {"$t": torch.tensor(0.2, requires_grad=True)},
        "layers": [{"material": "Si", "thickness": "$t", "is_homogeneous": True}],
        "sources": [
            {"name": "TE", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"name": "TM", "theta": 0.0, "phi": 0.0, "pte": 0.0, "ptm": 1.0},
        ],
        "output": {"type": "torch"},
    }

    objective = "-(Results['TE']['efficiency']['transmission'][0] + Results['TM']['efficiency']['transmission'][0])"
    out = optimize(
        spec,
        objective=objective,
        options={"steps": 2, "optimizer": {"name": "adam", "lr": 1e-2}, "return_best": True},
    )

    assert isinstance(out, dict)
    assert "loss_history" in out
    assert len(out["loss_history"]) == 2
    assert abs(out["loss_history"][1] - out["loss_history"][0]) > 1e-12
    assert "vars" in out and "$t" in out["vars"]

    t_val = out["vars"]["$t"]
    assert isinstance(t_val, torch.Tensor)
    assert t_val.requires_grad


def test_optimize_supports_pattern_depending_on_var():
    from torchrdit.interface import optimize
    import torch

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
        "vars": {"$r": torch.tensor(0.15, requires_grad=True)},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.2,
                "is_homogeneous": False,
                "pattern": {
                    "bg_material": "SiO2",
                    "shapes": [{"name": "c1", "type": "circle", "center": [0.0, 0.0], "radius": "$r"}],
                    "layer_shape": "c1",
                },
            }
        ],
        "sources": {"name": "s0", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    objective = "-Results['s0']['efficiency']['transmission'][0]"
    out = optimize(spec, objective=objective, options={"steps": 1, "optimizer": {"name": "adam", "lr": 1e-2}})

    assert isinstance(out["vars"]["$r"], torch.Tensor)
    assert out["vars"]["$r"].requires_grad
