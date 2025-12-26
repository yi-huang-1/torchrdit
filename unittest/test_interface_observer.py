import pytest


def _minimal_spec():
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
        "materials": {"Si": {"permittivity": 12.0}},
        "layers": [{"material": "Si", "thickness": 0.2, "is_homogeneous": True}],
        "sources": {"name": "s0", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
        "output": {"type": "torch"},
    }


def test_interface_solve_uses_solver_observer_by_default(monkeypatch):
    import torchrdit.interface as iface

    events: list[tuple[str, dict]] = []

    class RecordingObserver:
        def update(self, event_type: str, data: dict) -> None:
            events.append((event_type, dict(data or {})))

    monkeypatch.setattr(iface, "_DEFAULT_SOLVER_OBSERVER_FACTORY", lambda: RecordingObserver())

    out = iface.solve(_minimal_spec())
    assert isinstance(out, dict)
    assert "efficiency" in out
    assert any(e == "calculation_starting" for e, _ in events)
    assert any(e == "calculation_completed" for e, _ in events)


def test_interface_optimize_throttles_solver_observer_to_every_10_steps(monkeypatch):
    import torchrdit.interface as iface
    import torch

    events: list[tuple[str, dict]] = []

    class RecordingObserver:
        def update(self, event_type: str, data: dict) -> None:
            events.append((event_type, dict(data or {})))

    monkeypatch.setattr(iface, "_DEFAULT_SOLVER_OBSERVER_FACTORY", lambda: RecordingObserver())

    spec = _minimal_spec()
    spec["vars"] = {"$t": torch.tensor(0.2, requires_grad=True)}
    spec["layers"][0]["thickness"] = "$t"

    objective = "-Results['s0']['efficiency']['transmission'][0]"
    iface.optimize(spec, objective=objective, options={"steps": 21, "optimizer": {"name": "adam", "lr": 1e-2}})

    starts = [d for e, d in events if e == "calculation_starting"]
    completes = [d for e, d in events if e == "calculation_completed"]

    # Hardcoded interface policy: emit solver observer output on step 1, every 10 steps, and the final step.
    assert len(starts) == 4
    assert len(completes) == 4

    assert [d.get("opt_step") for d in starts] == [1, 10, 20, 21]
    assert all(d.get("opt_total_steps") == 21 for d in starts)
