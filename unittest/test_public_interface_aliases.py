def test_package_level_simulate_alias_runs():
    import torchrdit as tr

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
        "layers": [{"material": "Si", "thickness": 0.2, "is_homogeneous": True}],
        "sources": {"name": "s0", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }

    out = tr.simulate(spec)
    assert isinstance(out, dict)
    assert "efficiency" in out
