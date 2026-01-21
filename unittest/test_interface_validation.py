import pytest


def test_interface_unknown_top_level_key_raises():
    from torchrdit.interface import solve

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
        "unknown_key": 123,
    }

    with pytest.raises(ValueError, match=r"spec\.unknown_key"):
        solve(spec)


def test_interface_unknown_nested_key_raises():
    from torchrdit.interface import solve

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True, "bad": 1}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(ValueError, match=r"spec\.layers\[0\]\.bad"):
        solve(spec)


def test_interface_pattern_on_homogeneous_layer_raises():
    from torchrdit.interface import solve

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}, "SiO2": {"permittivity": 2.25}},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.1,
                "is_homogeneous": True,
                "pattern": {
                    "bg_material": "SiO2",
                    "shapes": [{"type": "circle", "center": [0.0, 0.0], "radius": 0.2}],
                },
            }
        ],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(ValueError, match=r"spec\.layers\[0\]\.pattern"):
        solve(spec)


def test_interface_accepts_homogeneous_layer_without_is_homogeneous():
    from torchrdit.interface import solve

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [8, 8], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    out = solve(spec)
    assert isinstance(out, dict)
    assert "efficiency" in out


def test_interface_pattern_requires_is_homogeneous_flag():
    from torchrdit.interface import solve

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [8, 8], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1, "pattern": {"layer_shape": "m", "shapes": []}}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(ValueError, match=r"spec\.layers\[0\]\.is_homogeneous"):
        solve(spec)


def test_interface_legacy_dict_var_node_rejected():
    from torchrdit.interface import solve

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}},
        "vars": {"$t": 0.2},
        "layers": [{"material": "Si", "thickness": {"$var": "t"}, "is_homogeneous": True}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(ValueError, match=r"\$var"):
        solve(spec)


def test_interface_missing_var_reference_raises():
    from torchrdit.interface import solve

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}},
        "vars": {"$s": 0.1},
        "layers": [{"material": "Si", "thickness": "$t", "is_homogeneous": True}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(ValueError, match=r"Unknown variable"):
        solve(spec)


def test_interface_var_definitions_are_order_insensitive():
    from torchrdit.interface import _compile_vars
    import torch

    compiled = _compile_vars(
        {
            "$r": "$s + 1",
            "$t": "$r + $s",
            "$s": torch.tensor(0.5),
        },
        device="cpu",
    )

    out = compiled.evaluate()
    assert set(out.keys()) == {"$s", "$r", "$t"}
    assert float(out["$r"]) == pytest.approx(1.5)
    assert float(out["$t"]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "bad_key, hint",
    [
        ("r" + "dim", "grids"),
        ("k" + "dim", "harmonics"),
        ("lam0", "wavelengths"),
    ],
)
def test_interface_solver_legacy_keys_raise(bad_key, hint):
    from torchrdit.interface import solve

    solver = {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]}
    solver[bad_key] = solver.pop(hint) if hint in solver else [1]
    spec = {
        "solver": solver,
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(ValueError, match=rf"spec\.solver\.{bad_key}"):
        solve(spec)


def test_interface_solver_base_path_rejected():
    from torchrdit.interface import solve

    spec = {
        "solver": {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [16, 16],
            "harmonics": [3, 3],
            "base_path": ".",
        },
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(ValueError, match=r"spec\.solver\.base_path"):
        solve(spec)


def test_interface_dict_spec_resolves_relative_dielectric_file_from_caller_dir(monkeypatch, tmp_path):
    from torchrdit.interface import solve

    monkeypatch.chdir(tmp_path)
    spec = {
        "solver": {
            "algorithm": "RCWA",
            "wavelengths": [1.55],
            "grids": [8, 8],
            "harmonics": [3, 3],
            "device": "cpu",
        },
        "materials": {
            "SiO2": {
                "dielectric_dispersion": True,
                "dielectric_file": "SiO2-e.txt",
                "data_format": "freq-eps",
                "data_unit": "thz",
            }
        },
        "layers": [{"material": "SiO2", "thickness": 0.1, "is_homogeneous": True}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    out = solve(spec)
    assert isinstance(out, dict)
    assert "efficiency" in out


def test_interface_solve_accepts_spec_path_and_resolves_material_file(monkeypatch, tmp_path):
    import json
    from torchrdit.interface import solve

    # Create a minimal dispersive material file in the same directory as the spec.
    mat_file = tmp_path / "mat.txt"
    mat_file.write_text(
        "\n".join(
            [
                "180.0\t2.25\t0.0",
                "190.0\t2.25\t0.0",
                "200.0\t2.25\t0.0",
                "210.0\t2.25\t0.0",
                "220.0\t2.25\t0.0",
                "230.0\t2.25\t0.0",
            ]
        )
        + "\n"
    )

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "solver": {
                    "algorithm": "RCWA",
                    "wavelengths": [1.55],
                    "grids": [8, 8],
                    "harmonics": [3, 3],
                    "device": "cpu",
                },
                "materials": {
                    "SiO2": {
                        "dielectric_dispersion": True,
                        "dielectric_file": "mat.txt",
                        "data_format": "freq-eps",
                        "data_unit": "thz",
                    }
                },
                "layers": [{"material": "SiO2", "thickness": 0.1, "is_homogeneous": True}],
                "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
                "output": {"type": "torch"},
            }
        )
    )

    # Change CWD away from the spec directory; resolution should still work.
    other = tmp_path / "other"
    other.mkdir(exist_ok=True)
    monkeypatch.chdir(other)
    out = solve(str(spec_path))
    assert isinstance(out, dict)
    assert "efficiency" in out


def test_interface_duplicate_source_names_raise():
    from torchrdit.interface import solve, SpecError

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True}],
        "sources": [
            {"name": "dup", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
            {"name": "dup", "theta": 0.1, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        ],
        "output": {"type": "torch"},
    }

    with pytest.raises(SpecError, match=r"Duplicate source name.*spec\.sources\[1\]\.name"):
        solve(spec)


@pytest.mark.parametrize(
    "bad_solver",
    [
        {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16], "harmonics": [3, 3]},
        {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16, 16], "harmonics": [3, 3]},
        {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3]},
        {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3, 3]},
        {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [-1, 16], "harmonics": [3, 3]},
        {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [0, 3]},
        {"algorithm": "RCWA", "wavelengths": [], "grids": [16, 16], "harmonics": [3, 3]},
        {"algorithm": "RCWA", "wavelengths": 1.55, "grids": [16, 16], "harmonics": [3, 3]},
    ],
)
def test_interface_solver_dim_validation_raises(bad_solver):
    from torchrdit.interface import solve, SpecError

    spec = {
        "solver": bad_solver,
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(SpecError, match=r"spec\.solver\.(wavelengths|grids|harmonics)"):
        solve(spec)


@pytest.mark.parametrize("harmonics", [[4, 3], [3, 4], [4, 4]])
def test_interface_solver_harmonics_even_raises(harmonics):
    from torchrdit.interface import _normalize_solver_spec, SpecError

    solver = {
        "algorithm": "RCWA",
        "wavelengths": [1.55],
        "grids": [16, 16],
        "harmonics": harmonics,
    }

    with pytest.raises(SpecError, match=r"spec\.solver\.harmonics.*odd"):
        _normalize_solver_spec(solver)


@pytest.mark.parametrize("bad_maxG", [0, -1, 1.5, True, "10"])
def test_interface_solver_auto_maxG_invalid_raises(bad_maxG):
    from torchrdit.interface import _normalize_solver_spec, SpecError

    solver = {
        "algorithm": "RCWA",
        "wavelengths": [1.55],
        "grids": [16, 16],
        "harmonics": "auto",
        "maxG": bad_maxG,
    }

    with pytest.raises(SpecError, match=r"spec\.solver\.maxG"):
        _normalize_solver_spec(solver)


def test_interface_invalid_output_type_raises():
    from torchrdit.interface import solve, SpecError

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.1, "is_homogeneous": True}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "jax"},
    }

    with pytest.raises(SpecError, match=r"spec\.output\.type"):
        solve(spec)


def test_interface_var_key_without_dollar_raises():
    from torchrdit.interface import solve, SpecError

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}},
        "vars": {"t": 0.2},
        "layers": [{"material": "Si", "thickness": 0.2, "is_homogeneous": True}],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(SpecError, match=r"spec\.vars keys must be"):
        solve(spec)


def test_interface_var_cycles_raise():
    from torchrdit.interface import _compile_vars, SpecError

    compiled = _compile_vars({"$a": "$b + 1", "$b": "$a + 1"}, device="cpu")
    with pytest.raises(SpecError, match=r"Cyclic variable definition"):
        compiled.evaluate()


def test_interface_var_disallows_function_calls():
    from torchrdit.interface import _compile_vars, SpecError

    compiled = _compile_vars({"$s": 1.0, "$t": "sin($s)"}, device="cpu")
    with pytest.raises(SpecError, match=r"Unsupported syntax in var expression"):
        compiled.evaluate()


def test_interface_pattern_op_references_unknown_name_raises():
    from torchrdit.interface import solve, SpecError

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
        "materials": {"Si": {"permittivity": 12.0}, "SiO2": {"permittivity": 2.25}},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.2,
                "is_homogeneous": False,
                "pattern": {
                    "bg_material": "SiO2",
                    "shapes": [{"name": "m1", "type": "op", "expr": "union(c1, r1)"}],
                    "layer_shape": "m1",
                },
            }
        ],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(SpecError, match=r"Unknown shape name.*spec\.layers\[0\]\.pattern\.shapes\[0\]"):
        solve(spec)


def test_interface_pattern_layer_shape_unknown_raises():
    from torchrdit.interface import solve, SpecError

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}, "SiO2": {"permittivity": 2.25}},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.2,
                "is_homogeneous": False,
                "pattern": {
                    "bg_material": "SiO2",
                    "shapes": [{"name": "c1", "type": "circle", "center": [0.0, 0.0], "radius": 0.2}],
                    "layer_shape": "missing",
                },
            }
        ],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(SpecError, match=r"spec\.layers\[0\]\.pattern\.layer_shape references unknown name"):
        solve(spec)


def test_interface_pattern_mask_shape_mismatch_raises():
    from torchrdit.interface import solve, SpecError
    import numpy as np

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3]},
        "materials": {"Si": {"permittivity": 12.0}, "SiO2": {"permittivity": 2.25}},
        "vars": {"$m": np.zeros((15, 16), dtype=np.float32)},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.2,
                "is_homogeneous": False,
                "pattern": {"bg_material": "SiO2", "shapes": [{"name": "m1", "type": "mask", "value": "$m"}], "layer_shape": "m1"},
            }
        ],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    with pytest.raises(SpecError, match=r"spec\.layers\[0\]\.pattern\.shapes\[0\]\.value must have shape"):
        solve(spec)


def test_interface_pattern_mask_allows_out_of_range_values():
    from torchrdit.interface import solve
    import torch

    spec = {
        "solver": {"algorithm": "RCWA", "wavelengths": [1.55], "grids": [16, 16], "harmonics": [3, 3], "device": "cpu"},
        "materials": {"Si": {"permittivity": 12.0}, "SiO2": {"permittivity": 2.25}},
        "vars": {"$m": torch.full((16, 16), 1.2)},
        "layers": [
            {
                "material": "Si",
                "thickness": 0.2,
                "is_homogeneous": False,
                "pattern": {"bg_material": "SiO2", "shapes": [{"name": "m1", "type": "mask", "value": "$m"}], "layer_shape": "m1"},
            }
        ],
        "sources": {"theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    result = solve(spec)
    assert "efficiency" in result
