from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_SOLVER_SETUP_KEYS = {
    "algorithm",
    "precision",
    "wavelengths",
    "length_unit",
    "grids",
    "harmonics",
    "maxg",
    "lattice_vectors",
    "use_fff",
    "is_use_fff",
    "fff_vector_scheme",
    "fff_fourier_weight",
    "fff_smoothness_weight",
    "fff_vector_steps",
    "device",
    "rdit_order",
    "trn_material",
    "ref_material",
}


def load_config(config: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load a JSON config from disk or accept a dict directly."""
    if isinstance(config, (str, os.PathLike)):
        path = Path(config).expanduser()
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
        except FileNotFoundError as e:
            raise ValueError(f"Config file not found: {config}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON config at {config}: {e}") from e
        if not isinstance(loaded, dict):
            raise ValueError(f"Config JSON must be an object, got {type(loaded)}")
        return loaded, str(path.resolve().parent)

    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dict or JSON file path, got {type(config)}")

    return config, None


def detect_config_shape(obj: Dict[str, Any]) -> str:
    """Detect whether config is interface-shaped or a flat solver config."""
    if not isinstance(obj, dict):
        raise TypeError(f"Config must be a dict, got {type(obj)}")

    lowered_keys = set()
    for key in obj.keys():
        if not isinstance(key, str):
            raise TypeError(f"Config keys must be strings, got {type(key)}")
        lowered_keys.add(key.lower())

    has_solver = "solver" in lowered_keys
    has_setup_keys = bool(lowered_keys & _SOLVER_SETUP_KEYS)

    if has_solver and has_setup_keys:
        raise ValueError("mixed config: 'solver' with top-level solver keys")
    if has_solver:
        return "interface"
    if has_setup_keys:
        return "flat_solver"
    if lowered_keys & {"materials", "layers"}:
        return "flat_solver"

    raise ValueError("Unable to detect config shape")

