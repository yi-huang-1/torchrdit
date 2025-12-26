"""
TorchRDIT regulated interface (spec-driven simulation + optimization).

Recommended usage:

```python
import torchrdit as tr
results = tr.simulate(spec)  # forward simulation (alias of tr.solve)
results2 = tr.simulate("spec.json")  # load a JSON spec file
opt = tr.optimize(spec, objective="...", options={...})  # inverse design (minimize objective)
```

This module provides a strict, user-facing API:
- Unknown keys raise :class:`SpecError` with a full dotted path.
- Spec values accept Python / NumPy / torch; torch values preserve autograd where the underlying implementation is torch-based (e.g. vars/layers), but in-memory dispersive material samples are converted to CPU NumPy (no grads).
- Variables are defined only in ``spec["vars"]`` (keys start with ``$``) and referenced elsewhere as strings (e.g., ``"$t"``).
- For file-based dispersive materials, relative ``materials[*].dielectric_file`` paths resolve from the spec file directory (if loaded from JSON), then caller script directory, then CWD.

---

Entry points:
- ``torchrdit.simulate(spec)``: forward simulation (recommended, namespaced)
- ``torchrdit.optimize(spec, objective=..., options=None)``: inverse design
- ``torchrdit.interface.solve(spec)`` / ``torchrdit.interface.optimize(...)``: module-level access

---

**Structured results schema** (returned by ``solve(spec)`` and used in ``optimize`` as ``Results[...]``)

``solve(spec)`` returns:
- Single source: ``result_dict``
- Multi source: ``{source_name: result_dict}``

Each ``result_dict`` contains:
- ``efficiency``: ``{"reflection": (n_wavelengths,), "transmission": (n_wavelengths,)}``
- ``diffraction_efficiency``: ``{"reflection": (n_wavelengths,kx,ky), "transmission": (n_wavelengths,kx,ky)}``
- ``field_fourier_coefficients`` (k-space coefficients, not real-space fields):
  - ``reflection``/``transmission`` → ``{"E": {"x","y","z"}, "H": {"x","y","z"}}`` (each tensor shape ``(n_wavelengths,kx,ky)``; H entries may be ``None``)
- ``scattering_matrix``: ``{"structure": {"S11","S12","S21","S22"}}`` (per-wavelength complex matrices)
- ``wavevectors``: ``{"kx": (kx,ky), "ky": (kx,ky), "incident": (n_wavelengths,3), "kz": {"reflection","transmission"}}``
- ``lattice``: ``{"t1","t2"}``
- ``grids``: ``{"real_space_shape": (H,W)}``

Output conversion (applies to all returned dicts, including optimize outputs):
- ``spec["output"]["type"] == "torch"`` → torch tensors
- ``spec["output"]["type"] == "numpy"`` → NumPy arrays (torch tensors detached + moved to CPU)

---

**Input spec schema** (strict; unknown keys raise)

Top-level ``spec``:
``{"solver": {...}, "materials": {...}?, "vars": {...}?, "layers": [...]?, "sources": {...}|[...], "output": {...}?}``

``spec["solver"]`` (required keys: ``algorithm``, ``wavelengths``, ``grids``, ``harmonics``):
- ``algorithm``: ``"RCWA"|"RDIT"`` (case-insensitive)
- ``precision``: ``"SINGLE"|"DOUBLE"`` (optional)
- ``wavelengths``: non-empty ``list|tuple|numpy.ndarray|torch.Tensor``
- ``length_unit``: ``"m"|"dm"|"cm"|"mm"|"um"|"nm"|"pm"|"angstrom"`` (optional)
- ``grids``: ``[Ny,Nx]`` (2 positive ints; real-space grid size)
- ``harmonics``: ``[Ky,Kx]`` (2 positive ints; Fourier harmonics)
- ``lattice_vectors``: ``{"t1":[x,y], "t2":[x,y]}`` (optional)
- ``use_fff``: ``bool`` (optional)
- ``device``: torch device string (optional; e.g. ``"cpu"``, ``"cuda:0"``)
- ``rdit_order``: ``int`` (optional; meaningful for ``algorithm="RDIT"``)
- ``trn_material`` / ``ref_material``: ``str`` (optional; names of external media materials)

Interface isolation: internal keys ``lam0/rdim/kdim`` are rejected (use ``wavelengths/grids/harmonics``).

``spec["vars"]`` (optional; the only variable entry point):
- Keys must start with ``$``. Values can be primitives/NumPy/torch, or expression strings referencing other vars.
- References elsewhere in spec use string form (e.g., ``"thickness": "$t"``).
- Expressions: arithmetic only; no attribute access and no function calls; order-insensitive evaluation with cycle/missing-var errors.

``spec["materials"]`` (optional):
- ``{name: {"permittivity": ...}}`` for non-dispersive materials, or:
- file-based dispersive:
  ``{name: {"dielectric_dispersion": True, "dielectric_file": "...", "data_format": "freq-eps|wl-eps|freq-nk|wl-nk", "data_unit": "thz|ghz|mhz|um|nm"}}``
- in-memory dispersive (wavelength samples are always in micrometers):
  ``{name: {"dielectric_dispersion": True, "dispersion": {"wavelengths_um": [...], "eps": [...]}}}``
  or ``{name: {"dielectric_dispersion": True, "dispersion": {"wavelengths_um": [...], "n": [...], "k": [...]?}}}``

Rules:
- If ``dielectric_dispersion`` is omitted/False, only ``permittivity`` is allowed.
- If ``dielectric_dispersion`` is True, exactly one of ``dielectric_file`` or ``dispersion`` must be provided.
- ``dispersion`` is a structured object with keys ``wavelengths_um`` and either:
  - ``eps`` (complex permittivity samples), or
  - ``n`` (+ optional ``k``) (refractive index samples; ``k`` defaults to 0).
- ``data_format``/``data_unit`` apply only to file-based dispersion and are rejected when using ``dispersion``.

Complex ε convention:
- TorchRDIT uses a time-harmonic convention where lossy permittivity has ``Im(ε) < 0``.
- For ``dispersion.eps``, both ``eps1 - 1j*eps2`` and ``eps1 + 1j*eps2`` inputs are accepted; the implementation normalizes to the internal convention.

``spec["layers"]`` (optional; list):
- Each layer: ``{"material": str, "thickness": scalar|"$var", "is_homogeneous": bool, "is_optimize": bool?, "slice_count": int?, "pattern": {...}?}``
- ``pattern`` is required iff ``is_homogeneous == False`` and forbidden iff ``is_homogeneous == True``.

Patterned layers: ``pattern = {"bg_material": str?, "method": "FFT|Analytical"?, "soft_edge": float?, "shapes": [...], "layer_shape": str}``
- Node types in ``pattern["shapes"]`` (executed in order; each requires ``{"name": str, "type": ...}``):
  - ``type=="circle"``: ``{"center":[x,y], "radius": r, "soft_edge": float?}``
  - ``type=="rectangle"``: ``{"center":[x,y], "x_size": a, "y_size": b, "angle": rad?, "soft_edge": float?}``
  - ``type=="polygon"``: ``{"points":[[x,y],...], "center":[x,y]?, "angle": rad?, "invert": bool?, "soft_edge": float?}``
  - ``type=="mask"``: ``{"value": 2D mask|"$var"}`` with shape == ``grids`` and values in ``[0,1]`` (strict)
  - ``type=="op"``: ``{"expr": "union(a,b)|intersection(a,b)|difference(a,b)|subtract(a,b)"}`` (nested calls allowed; args must reference prior node names)
  - ``type=="gds"``: not supported (raises)

``spec["sources"]`` (required; dict or list of dicts):
- Source keys: ``{"name": str?, "theta": rad, "phi": rad, "pte": amp, "ptm": amp, "norm_te_dir": "x|y|z"?}``
  - ``theta``, ``phi``: float-like radians
  - ``pte``, ``ptm``: float/complex-like (Python/NumPy/torch scalar)

``spec["output"]`` (optional): ``{"type": "torch"|"numpy"}``

---

**Optimization API** (``optimize(spec, objective=..., options=...)``; objective is always minimized)

``objective``:
- ``str`` expression; supports arithmetic, indexing/slicing, and function calls to:
  ``abs, real, imag, mean, sum, min, max``.
- Environment:
  - ``Results``/``S``: per-source structured results dicts
  - ``Vars``/``V``: variables (also injected as plain names without the ``$`` prefix)
- Disallows attribute access (``x.y``) and imports.

``options`` (optional):
- ``{"steps": int?, "optimizer": {"name": "adam", "lr": float}? , "grad_clip": {"max_norm": float}? , "return_best": bool?}``

Return value:
- Always: ``{"loss_history": [...], "vars": {...}}``
- Plus either: ``{"best": {"loss": float, "vars": {...}, "results": {...}}}`` (if ``return_best``) or ``{"last_results": {...}}``.

---

Full spec example (commented):

```python
import torch
import torchrdit as tr

spec = {
  # --- solver config (interface-level keys only; strict) ---
  "solver": {
    "algorithm": "RCWA",              # "RCWA" or "RDIT" (case-insensitive input accepted, normalized internally)
    "precision": "SINGLE",            # optional
    "wavelengths": [1.55],            # required: list/array of wavelengths
    "length_unit": "um",              # optional
    "grids": [64, 64],                # required: (Ny, Nx) real-space grid size
    "harmonics": [3, 3],              # required: (Ky, Kx) Fourier harmonics
    "device": "cpu",                  # optional: "cpu" / "cuda" / ...
  },

  # --- user variables (sole entry point) ---
  "vars": {
    "$s": 0.10,                       # scalar constant
    "$t": "$s + 0.10",                # expression var (derived; re-evaluated during optimization)
    "$r": torch.tensor(0.15, requires_grad=True),  # trainable leaf tensor (optimization will update it)
    # Optional: user-provided mask (must match grids exactly and values must be within [0,1])
    # If you want it trainable, parameterize it to stay in [0,1] (e.g., torch.sigmoid(raw)) before passing here.
    "$user_mask": torch.rand(64, 64), # custom mask input (not trainable in this example)
  },

  # --- materials ---
  "materials": {
    "Si": {"permittivity": 12.0},
    "SiO2": {"permittivity": 2.25},
    # Dispersive (in-memory) example:
    "Au": {
      "dielectric_dispersion": True,
      "dispersion": {
        "wavelengths_um": [1.0, 1.5, 2.0],
        "n": [0.45, 0.35, 0.30],
        "k": [6.5, 7.0, 7.5],
      },
    },
    # Dispersive (file-based) example:
    # "Au": {
    #   "dielectric_dispersion": True,
    #   "dielectric_file": "materials/Au.txt",
    #   "data_format": "freq-eps",
    #   "data_unit": "thz",
    # },
  },

  # --- geometry: layers (patterned layers supported) ---
  "layers": [
    {
      "material": "Si",
      "thickness": "$t",              # variable reference (string form)
      "is_homogeneous": False,        # patterned layer
      "pattern": {
        "bg_material": "SiO2",        # background material for mask fill
        "method": "FFT",              # forwarded to update_er_with_mask
        "soft_edge": 0.001,           # default shape soft_edge

        # Ordered operation program executed top-to-bottom:
        # - nodes define named shapes/masks/ops
        # - op expr supports nested calls (union/intersection/difference/subtract)
        # - final applied mask is selected by layer_shape
        "shapes": [
          {"name": "c1", "type": "circle",    "center": [0.0, 0.0], "radius": "$r"},
          {"name": "r1", "type": "rectangle", "center": [0.0, 0.0], "x_size": 0.40, "y_size": 0.20, "angle": 0.0},
          {"name": "p1", "type": "polygon",   "points": [[-0.2,-0.2],[0.2,-0.2],[0.0,0.2]], "soft_edge": 0.0},

          # user-provided mask node (must have shape == grids and values within [0,1])
          {"name": "m_user", "type": "mask", "value": "$user_mask"},

          # op nodes can be nested:
          {"name": "m1", "type": "op", "expr": "subtract(union(c1, r1), p1)"},
          {"name": "m2", "type": "op", "expr": "union(m1, m_user)"},
        ],
        "layer_shape": "m2",          # which named node to apply as the layer mask
      },
    },
  ],

  # --- sources: dict or list ---
  "sources": [
    {"name": "TE", "theta": 0.0, "phi": 0.0, "pte": 1.0, "ptm": 0.0},
    {"name": "TM", "theta": 0.0, "phi": 0.0, "pte": 0.0, "ptm": 1.0},
  ],

  # --- output conversion for returned results dict(s) ---
  "output": {"type": "torch"},        # "torch" or "numpy"
}

# Forward simulation (recommended entry point)
results = tr.simulate(spec)

# Inverse design: objective is always minimized
objective = "-0.5*(Results['TE']['efficiency']['transmission'][0] + Results['TM']['efficiency']['transmission'][0])"
opt_out = tr.optimize(spec, objective=objective, options={"steps": 10, "optimizer": {"name": "adam", "lr": 1e-2}})
```
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .builder import SolverBuilder
from .observers import ConsoleProgressObserver
from .path_utils import infer_caller_dir, resolve_data_path
from .shapes import ShapeGenerator
from .utils import create_material


class SpecError(ValueError):
    """Raised when an interface spec or options dict is invalid."""

    pass


_DEFAULT_SOLVER_OBSERVER_FACTORY = lambda: ConsoleProgressObserver(verbose=False)


class _OptimizeStepThrottledObserver:
    def __init__(self, delegate: Any, *, every: int, total_steps: int) -> None:
        self._delegate = delegate
        self._every = int(every)
        self._total_steps = int(total_steps)
        self._step_index = 0

    def set_step(self, step_index: int) -> None:
        self._step_index = int(step_index)

    def _is_active(self) -> bool:
        # 1-based step display policy:
        # - Always show step 1 and final step
        # - Show every N steps (10, 20, ...)
        step = self._step_index + 1
        if step <= 1:
            return True
        if step >= self._total_steps:
            return True
        return (step % self._every) == 0

    def update(self, event_type: str, data: dict) -> None:
        if self._is_active():
            payload = dict(data or {})
            payload["opt_step"] = self._step_index + 1
            payload["opt_total_steps"] = self._total_steps
            self._delegate.update(event_type, payload)


def _path_join(base: str, part: str) -> str:
    if not base:
        return part
    if part.startswith("["):
        return f"{base}{part}"
    return f"{base}.{part}"


def _raise_unknown_keys(path: str, unknown: Sequence[str], allowed: Sequence[str]) -> None:
    unknown_sorted = ", ".join(sorted(_path_join(path, k) for k in unknown))
    allowed_sorted = ", ".join(sorted(allowed))
    raise SpecError(f"Unknown key(s): {unknown_sorted}. Allowed keys at {path}: {allowed_sorted}")


def _check_keys(path: str, obj: Mapping[str, Any], *, allowed: Iterable[str]) -> None:
    allowed_set = set(allowed)
    unknown = [k for k in obj.keys() if k not in allowed_set]
    if unknown:
        _raise_unknown_keys(path, unknown, sorted(allowed_set))


def _require_type(path: str, value: Any, expected: Union[type, Tuple[type, ...]]) -> None:
    if not isinstance(value, expected):
        raise SpecError(f"Expected {path} to be {expected}, got {type(value)}")


def _as_tensor_preserve(value: Any, *, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None):
    if isinstance(value, torch.Tensor):
        if device is None and dtype is None:
            return value
        return value.to(device=device or value.device, dtype=dtype or value.dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


_VAR_NAME_RE = re.compile(r"^\$[A-Za-z_]\w*$")
_VAR_TOKEN_RE = re.compile(r"\$[A-Za-z_]\w*")


def _is_var_name(name: str) -> bool:
    return bool(_VAR_NAME_RE.match(name))


def _strip_var_prefix(name: str) -> str:
    return name[1:] if name.startswith("$") else name


class _SafeVarExprEvaluator:
    def __init__(self, expr: str, *, path: str) -> None:
        self._expr = expr
        self._path = path
        self._tree = ast.parse(expr, mode="eval")

    def eval(self, env: Mapping[str, Any]) -> Any:
        return self._eval_node(self._tree.body, env)

    def _eval_node(self, node: ast.AST, env: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise SpecError(f"Unknown name in var expression at {self._path}: {node.id!r}")
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, env)
            right = self._eval_node(node.right, env)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise SpecError(f"Unsupported binary operator in var expression at {self._path}: {type(node.op)}")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, env)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise SpecError(f"Unsupported unary operator in var expression at {self._path}: {type(node.op)}")
        if isinstance(node, (ast.Tuple, ast.List, ast.Dict, ast.Subscript, ast.Call, ast.Attribute, ast.Lambda)):
            raise SpecError(f"Unsupported syntax in var expression at {self._path}: {type(node).__name__}")
        raise SpecError(f"Unsupported syntax in var expression at {self._path}: {type(node).__name__}")


def _transform_var_expr(expr: str) -> Tuple[str, Dict[str, str]]:
    """Replace `$name` tokens with valid Python identifiers for AST parsing."""
    mapping: Dict[str, str] = {}

    def repl(m: re.Match[str]) -> str:
        tok = m.group(0)  # like "$s"
        ident = f"__var_{tok[1:]}"
        mapping[ident] = tok
        return ident

    return _VAR_TOKEN_RE.sub(repl, expr), mapping


@dataclass(frozen=True)
class _CompiledVars:
    raw: Dict[str, Any]
    exprs: Dict[str, Tuple[_SafeVarExprEvaluator, Dict[str, str]]]
    device: torch.device

    def evaluate(self) -> Dict[str, Any]:
        memo: Dict[str, Any] = {}
        visiting: set[str] = set()

        def resolve(name: str) -> Any:
            if name in memo:
                return memo[name]
            if name in visiting:
                raise SpecError(f"Cyclic variable definition detected at spec.vars.{name}")
            if name not in self.raw:
                raise SpecError(f"Unknown variable {name!r} referenced in spec.vars")

            visiting.add(name)
            raw_val = self.raw[name]
            if name in self.exprs:
                evaluator, ident_to_var = self.exprs[name]
                env = {ident: resolve(var_name) for ident, var_name in ident_to_var.items()}
                val = evaluator.eval(env)
            else:
                val = raw_val

            visiting.remove(name)
            memo[name] = val
            return val

        for k in self.raw.keys():
            resolve(k)
        return memo

    def trainable_params(self) -> List[torch.Tensor]:
        params: List[torch.Tensor] = []
        for name, raw_val in self.raw.items():
            if name in self.exprs:
                continue
            if isinstance(raw_val, torch.nn.Parameter):
                params.append(raw_val)
            elif isinstance(raw_val, torch.Tensor) and raw_val.requires_grad:
                if not raw_val.is_leaf:
                    raise SpecError(
                        f"spec.vars.{name} must be a leaf Tensor/Parameter when requires_grad=True (got non-leaf)"
                    )
                params.append(raw_val)
        return params


def _compile_vars(spec_vars: Any, *, device: Union[str, torch.device]) -> _CompiledVars:
    if spec_vars is None:
        return _CompiledVars(raw={}, exprs={}, device=torch.device(device))
    _require_type("spec.vars", spec_vars, dict)

    raw: Dict[str, Any] = {}
    exprs: Dict[str, Tuple[_SafeVarExprEvaluator, Dict[str, str]]] = {}
    dev = torch.device(device)

    for name, val in spec_vars.items():
        if not isinstance(name, str) or not _is_var_name(name):
            raise SpecError(f"spec.vars keys must be '$<identifier>' strings, got {name!r}")
        raw[name] = val

    for name, val in raw.items():
        path = f"spec.vars.{name}"
        if isinstance(val, str):
            expr_py, ident_to_var = _transform_var_expr(val)
            evaluator = _SafeVarExprEvaluator(expr_py, path=path)
            # Validate referenced vars exist
            for dep in ident_to_var.values():
                if dep not in raw:
                    raise SpecError(f"Unknown variable {dep!r} referenced at {path}")
            exprs[name] = (evaluator, ident_to_var)
        elif isinstance(val, torch.Tensor):
            if val.device != dev:
                if val.requires_grad:
                    raise SpecError(f"{path} is trainable but on {val.device}; expected {dev}")
                raw[name] = val.to(device=dev)

    return _CompiledVars(raw=raw, exprs=exprs, device=dev)


def _validate_var_refs_in_spec(spec: Any, *, vars_keys: Iterable[str]) -> None:
    """Pre-scan the spec (excluding spec.vars) for `$var` string references."""
    keys = set(vars_keys)

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                # Skip the vars section; it is validated/compiled separately.
                if path == "spec" and k == "vars":
                    continue
                walk(v, _path_join(path, str(k)))
            return
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
            return
        if isinstance(obj, tuple):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
            return
        if isinstance(obj, str) and obj.startswith("$"):
            if not _is_var_name(obj):
                raise SpecError(f"Invalid variable reference at {path}: {obj!r} (expected '$<identifier>')")
            if obj not in keys:
                raise SpecError(f"Unknown variable {obj!r} referenced at {path}")

    walk(spec, "spec")


def _resolve_vars_node(value: Any, vars_map: Mapping[str, Any], path: str) -> Any:
    if isinstance(value, dict) and "$var" in value:
        raise SpecError(
            f"Dict-style variable nodes are not supported at {path} (found '$var'); use '$name' string references"
        )
    if isinstance(value, str) and value.startswith("$"):
        if not _is_var_name(value):
            raise SpecError(f"Invalid variable reference at {path}: {value!r} (expected '$<identifier>')")
        if value not in vars_map:
            raise SpecError(f"Unknown variable {value!r} referenced at {path}")
        return vars_map[value]
    if isinstance(value, list):
        return [_resolve_vars_node(v, vars_map, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_resolve_vars_node(v, vars_map, f"{path}[{i}]") for i, v in enumerate(value))
    if isinstance(value, dict):
        return {k: _resolve_vars_node(v, vars_map, _path_join(path, k)) for k, v in value.items()}
    return value


def _convert_tree_to_numpy(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _convert_tree_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_tree_to_numpy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_tree_to_numpy(v) for v in obj)
    return obj


def _postprocess_output(obj: Any, *, output_type: str) -> Any:
    if output_type == "torch":
        return obj
    if output_type == "numpy":
        return _convert_tree_to_numpy(obj)
    raise SpecError(f"Invalid output.type: {output_type!r} (expected 'torch' or 'numpy')")


def _parse_output(spec_output: Any) -> str:
    if spec_output is None:
        return "torch"
    _require_type("spec.output", spec_output, dict)
    _check_keys("spec.output", spec_output, allowed={"type"})
    out_type = spec_output.get("type", "torch")
    if out_type not in {"torch", "numpy"}:
        raise SpecError("spec.output.type must be 'torch' or 'numpy'")
    return out_type


def _parse_sources(spec_sources: Any) -> List[Dict[str, Any]]:
    if spec_sources is None:
        raise SpecError("spec.sources is required")
    if isinstance(spec_sources, dict):
        sources = [spec_sources]
    elif isinstance(spec_sources, list):
        sources = spec_sources
    else:
        raise SpecError(f"Expected spec.sources to be dict or list, got {type(spec_sources)}")

    parsed: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for i, src in enumerate(sources):
        path = f"spec.sources[{i}]"
        _require_type(path, src, dict)
        _check_keys(path, src, allowed={"name", "theta", "phi", "pte", "ptm", "norm_te_dir"})
        if "theta" not in src or "phi" not in src or "pte" not in src or "ptm" not in src:
            raise SpecError(f"{path} requires keys: theta, phi, pte, ptm")
        name = src.get("name", f"s{i}")
        if not isinstance(name, str):
            raise SpecError(f"Expected {path}.name to be str")
        if name in seen_names:
            raise SpecError(f"Duplicate source name {name!r} at {path}.name")
        seen_names.add(name)
        norm_te_dir = src.get("norm_te_dir", "y")
        if norm_te_dir not in {"x", "y", "z"}:
            raise SpecError(f"{path}.norm_te_dir must be one of 'x', 'y', 'z'")
        parsed.append(
            {
                "name": name,
                "theta": float(src["theta"]),
                "phi": float(src["phi"]),
                "pte": src["pte"],
                "ptm": src["ptm"],
                "norm_te_dir": norm_te_dir,
            }
        )
    return parsed


def _parse_materials(
    spec_materials: Any,
    *,
    config_dir: Optional[Union[str, os.PathLike[str]]] = None,
    caller_dir: Optional[Union[str, os.PathLike[str]]] = None,
) -> Dict[str, Any]:
    if spec_materials is None:
        return {}
    _require_type("spec.materials", spec_materials, dict)
    mats: Dict[str, Any] = {}
    for name, props in spec_materials.items():
        if not isinstance(name, str):
            raise SpecError("spec.materials keys must be str")
        if not isinstance(props, dict):
            raise SpecError(f"spec.materials.{name} must be a dict")
        allowed = {"permittivity", "dielectric_dispersion", "dielectric_file", "dispersion", "data_format", "data_unit"}
        _check_keys(f"spec.materials.{name}", props, allowed=allowed)
        if "dielectric_dispersion" in props and not isinstance(props["dielectric_dispersion"], bool):
            raise SpecError(f"spec.materials.{name}.dielectric_dispersion must be bool")

        is_dispersive = bool(props.get("dielectric_dispersion", False))
        if not is_dispersive:
            forbidden = {"dielectric_file", "dispersion", "data_format", "data_unit"} & set(props.keys())
            if forbidden:
                forbidden_str = ", ".join(sorted(forbidden))
                raise SpecError(
                    f"spec.materials.{name}: {forbidden_str} is only allowed when dielectric_dispersion is True"
                )
            if "permittivity" not in props:
                raise SpecError(f"spec.materials.{name} requires permittivity when dielectric_dispersion is False")
            mats[name] = create_material(name=name, permittivity=props["permittivity"])
            continue

        # Dispersive material:
        if "permittivity" in props:
            raise SpecError(f"spec.materials.{name}.permittivity is not allowed when dielectric_dispersion is True")

        has_file = "dielectric_file" in props
        has_disp = "dispersion" in props
        if has_file and has_disp:
            raise SpecError(f"spec.materials.{name} cannot specify both dielectric_file and dispersion")
        if not has_file and not has_disp:
            raise SpecError(f"spec.materials.{name} requires dielectric_file or dispersion when dielectric_dispersion is True")

        if has_file:
            try:
                dielectric_file = resolve_data_path(
                    props["dielectric_file"],
                    config_dir=config_dir,
                    caller_dir=caller_dir,
                )
            except ValueError as e:
                raise SpecError(f"spec.materials.{name}.dielectric_file: {e}") from e
            mats[name] = create_material(
                name=name,
                dielectric_dispersion=True,
                user_dielectric_file=dielectric_file,
                data_format=props.get("data_format", "freq-eps"),
                data_unit=props.get("data_unit", "thz"),
            )
            continue

        # In-memory dispersion:
        if "data_format" in props or "data_unit" in props:
            raise SpecError(
                f"spec.materials.{name}: data_format/data_unit are not used with in-memory dispersion; "
                f"remove them and provide wavelengths_um + (eps or n,k)"
            )

        disp_path = f"spec.materials.{name}.dispersion"
        disp = props["dispersion"]
        _require_type(disp_path, disp, dict)
        _check_keys(disp_path, disp, allowed={"wavelengths_um", "eps", "n", "k"})
        if "wavelengths_um" not in disp:
            raise SpecError(f"{disp_path}.wavelengths_um is required")

        has_eps = "eps" in disp
        has_n = "n" in disp
        if has_eps and has_n:
            raise SpecError(f"{disp_path} cannot specify both eps and n/k")
        if not has_eps and not has_n:
            raise SpecError(f"{disp_path} requires eps or n (+ optional k)")

        if has_eps and ("k" in disp or "n" in disp):
            raise SpecError(f"{disp_path}: when eps is provided, n/k are not allowed")

        mats[name] = create_material(
            name=name,
            dielectric_dispersion=True,
            user_dielectric_wavelengths_um=disp["wavelengths_um"],
            user_dielectric_eps=disp.get("eps"),
            user_dielectric_n=disp.get("n"),
            user_dielectric_k=disp.get("k"),
        )
    return mats


def _parse_layers(spec_layers: Any) -> List[Dict[str, Any]]:
    if spec_layers is None:
        return []
    _require_type("spec.layers", spec_layers, list)
    parsed: List[Dict[str, Any]] = []
    for i, layer in enumerate(spec_layers):
        path = f"spec.layers[{i}]"
        _require_type(path, layer, dict)
        _check_keys(path, layer, allowed={"material", "thickness", "is_homogeneous", "is_optimize", "slice_count", "pattern"})
        if "material" not in layer or "thickness" not in layer or "is_homogeneous" not in layer:
            raise SpecError(f"{path} requires keys: material, thickness, is_homogeneous")
        if not isinstance(layer["material"], str):
            raise SpecError(f"{path}.material must be str")
        if not isinstance(layer["is_homogeneous"], bool):
            raise SpecError(f"{path}.is_homogeneous must be bool")
        if "is_optimize" in layer and not isinstance(layer["is_optimize"], bool):
            raise SpecError(f"{path}.is_optimize must be bool")
        if "slice_count" in layer:
            sc = layer["slice_count"]
            if isinstance(sc, bool) or not isinstance(sc, (int, np.integer)):
                raise SpecError(f"{path}.slice_count must be int")
            if int(sc) < 1:
                raise SpecError(f"{path}.slice_count must be >= 1")

        if layer["is_homogeneous"] and "pattern" in layer:
            raise SpecError(f"{path}.pattern is not allowed when is_homogeneous is True")
        if not layer["is_homogeneous"] and "pattern" not in layer:
            raise SpecError(f"{path} requires pattern when is_homogeneous is False")

        parsed.append(layer)
    return parsed


def _validate_spec(spec: Any) -> None:
    _require_type("spec", spec, dict)
    _check_keys("spec", spec, allowed={"solver", "materials", "layers", "sources", "vars", "output"})
    if "solver" not in spec:
        raise SpecError("spec.solver is required")


def _normalize_solver_spec(spec_solver: Any, *, path: str = "spec.solver") -> Dict[str, Any]:
    """Normalize the interface solver spec into a SolverBuilder-compatible config.

    Interface-level naming:
    - ``wavelengths``: list/array of wavelengths.
    - ``grids``: real-space grid size (mapped to builder key ``rdim``).
    - ``harmonics``: Fourier harmonics (mapped to builder key ``kdim``).

    Legacy/low-level keys like ``rdim``/``kdim``/``lam0`` are rejected to keep the
    interface spec isolated from internal implementation details.
    """
    _require_type(path, spec_solver, dict)

    legacy_map = {"lam0": "wavelengths", "rdim": "grids", "kdim": "harmonics"}
    legacy_present = [k for k in spec_solver.keys() if k in legacy_map]
    if legacy_present:
        details = ", ".join(f"{_path_join(path, k)} (use '{legacy_map[k]}')" for k in sorted(legacy_present))
        raise SpecError(f"Unsupported solver key(s): {details}")

    allowed_keys = {
        "algorithm",
        "precision",
        "wavelengths",
        "length_unit",
        "grids",
        "harmonics",
        "lattice_vectors",
        "use_fff",
        "device",
        "rdit_order",
        "trn_material",
        "ref_material",
    }
    _check_keys(path, spec_solver, allowed=allowed_keys)

    for required in ("algorithm", "wavelengths", "grids", "harmonics"):
        if required not in spec_solver:
            raise SpecError(f"{_path_join(path, required)} is required")

    cfg = dict(spec_solver)

    algo = cfg.get("algorithm")
    if not isinstance(algo, str) or not algo.strip():
        raise SpecError(f"Expected {_path_join(path, 'algorithm')} to be a non-empty string")
    cfg["algorithm"] = algo.upper()
    if cfg["algorithm"] not in {"RCWA", "RDIT"}:
        raise SpecError(f"{_path_join(path, 'algorithm')} must be 'RCWA' or 'RDIT' (case-insensitive input accepted)")

    if "precision" in cfg:
        prec = cfg["precision"]
        if not isinstance(prec, str) or not prec.strip():
            raise SpecError(f"Expected {_path_join(path, 'precision')} to be a non-empty string")
        cfg["precision"] = prec.upper()
        if cfg["precision"] not in {"SINGLE", "DOUBLE"}:
            raise SpecError(f"{_path_join(path, 'precision')} must be 'SINGLE' or 'DOUBLE' (case-insensitive input accepted)")

    wavelengths = cfg.get("wavelengths")
    if isinstance(wavelengths, (str, bytes)) or wavelengths is None:
        raise SpecError(
            f"Expected {_path_join(path, 'wavelengths')} to be a non-empty sequence/array, got {type(wavelengths)}"
        )
    if isinstance(wavelengths, (list, tuple)):
        if len(wavelengths) < 1:
            raise SpecError(f"{_path_join(path, 'wavelengths')} must not be empty")
    elif isinstance(wavelengths, (np.ndarray, torch.Tensor)):
        if int(wavelengths.size if isinstance(wavelengths, np.ndarray) else wavelengths.numel()) < 1:
            raise SpecError(f"{_path_join(path, 'wavelengths')} must not be empty")
    else:
        raise SpecError(
            f"Expected {_path_join(path, 'wavelengths')} to be a list/tuple/numpy array/torch tensor, got {type(wavelengths)}"
        )

    def _require_int_pair(dim_path: str, value: Any) -> Tuple[int, int]:
        if isinstance(value, (list, tuple)):
            seq = list(value)
        elif isinstance(value, np.ndarray):
            seq = value.flatten().tolist()
        elif isinstance(value, torch.Tensor):
            seq = value.detach().flatten().cpu().tolist()
        else:
            raise SpecError(f"Expected {dim_path} to be a 2-item sequence, got {type(value)}")
        if len(seq) != 2:
            raise SpecError(f"Expected {dim_path} to have length 2, got {len(seq)}")
        out: List[int] = []
        for j, item in enumerate(seq):
            if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                raise SpecError(f"Expected {dim_path}[{j}] to be int, got {type(item)}")
            out.append(int(item))
        if out[0] <= 0 or out[1] <= 0:
            raise SpecError(f"{dim_path} values must be > 0, got {tuple(out)}")
        return out[0], out[1]

    cfg["grids"] = list(_require_int_pair(_path_join(path, "grids"), cfg.get("grids")))
    cfg["harmonics"] = list(_require_int_pair(_path_join(path, "harmonics"), cfg.get("harmonics")))

    cfg["rdim"] = cfg.pop("grids")
    cfg["kdim"] = cfg.pop("harmonics")
    return cfg


def _build_solver(spec_solver: Mapping[str, Any]):
    builder = SolverBuilder().from_config(dict(spec_solver))
    return builder.build()


def _apply_materials_to_solver(solver, materials: Mapping[str, Any]) -> None:
    if materials:
        solver.add_materials(list(materials.values()))

def _apply_external_media_to_solver(solver, *, solver_spec: Mapping[str, Any], materials: Mapping[str, Any]) -> None:
    trn = solver_spec.get("trn_material")
    if isinstance(trn, str) and trn.lower() != "air":
        solver.update_trn_material(materials.get(trn, trn))
    ref = solver_spec.get("ref_material")
    if isinstance(ref, str) and ref.lower() != "air":
        solver.update_ref_material(materials.get(ref, ref))


def _build_mask_from_pattern(
    *,
    shape_gen: ShapeGenerator,
    pattern: Mapping[str, Any],
    vars_map: Mapping[str, Any],
    path: str,
) -> torch.Tensor:
    _require_type(path, pattern, dict)
    _check_keys(path, pattern, allowed={"bg_material", "method", "soft_edge", "shapes", "layer_shape"})
    if "shapes" not in pattern:
        raise SpecError(f"{path}.shapes is required")
    if "layer_shape" not in pattern:
        raise SpecError(f"{path}.layer_shape is required")
    layer_shape = pattern["layer_shape"]
    if not isinstance(layer_shape, str) or not layer_shape:
        raise SpecError(f"{path}.layer_shape must be a non-empty string")

    nodes = pattern["shapes"]
    _require_type(f"{path}.shapes", nodes, list)
    if not nodes:
        raise SpecError(f"{path}.shapes must not be empty")

    soft_edge_default = float(pattern.get("soft_edge", 0.001))
    expected_dim = tuple(int(x) for x in getattr(shape_gen, "rdim", ()))
    mask_device = shape_gen.XO.device
    mask_dtype = getattr(shape_gen, "tfloat", torch.float32)

    def _mask_range_check(mask: torch.Tensor, *, node_path: str) -> None:
        if torch.is_complex(mask):
            raise SpecError(f"{node_path} mask must be real-valued")
        with torch.no_grad():
            if not torch.isfinite(mask).all().item():
                raise SpecError(f"{node_path} mask contains non-finite values")
            m_min = float(mask.amin().detach().cpu().item())
            m_max = float(mask.amax().detach().cpu().item())
        if m_min < 0.0 or m_max > 1.0:
            raise SpecError(f"{node_path} mask values must be within [0,1], got min={m_min:.6g}, max={m_max:.6g}")

    class _SafeMaskExprEvaluator:
        def __init__(self, expr: str, *, node_path: str) -> None:
            self._expr = expr
            self._path = node_path
            self._tree = ast.parse(expr, mode="eval")

        def eval(self, masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return self._eval_node(self._tree.body, masks)

        def _eval_node(self, node: ast.AST, masks: Mapping[str, torch.Tensor]) -> torch.Tensor:
            if isinstance(node, ast.Name):
                if node.id not in masks:
                    raise SpecError(f"Unknown shape name {node.id!r} referenced at {self._path}")
                return masks[node.id]
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise SpecError(f"Only simple op(a,b) calls are allowed at {self._path}")
                op = node.func.id
                if op not in {"union", "intersection", "difference", "subtract"}:
                    raise SpecError(f"Unsupported op {op!r} at {self._path}")
                if len(node.args) != 2 or node.keywords:
                    raise SpecError(f"op calls must be op(a, b) with 2 positional args at {self._path}")
                a = self._eval_node(node.args[0], masks)
                b = self._eval_node(node.args[1], masks)
                return shape_gen.combine_masks(a, b, operation=op)
            if isinstance(node, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Subscript, ast.Attribute, ast.Dict, ast.List, ast.Tuple)):
                raise SpecError(f"Unsupported syntax in op expr at {self._path}: {type(node).__name__}")
            raise SpecError(f"Unsupported syntax in op expr at {self._path}: {type(node).__name__}")

    masks: Dict[str, torch.Tensor] = {}
    for i, node in enumerate(nodes):
        npath = f"{path}.shapes[{i}]"
        _require_type(npath, node, dict)
        if "name" not in node:
            raise SpecError(f"{npath}.name is required")
        if "type" not in node:
            raise SpecError(f"{npath}.type is required")
        name = node["name"]
        ntype = node["type"]
        if not isinstance(name, str) or not name:
            raise SpecError(f"{npath}.name must be a non-empty string")
        if not isinstance(ntype, str) or not ntype:
            raise SpecError(f"{npath}.type must be a non-empty string")
        if not name.isidentifier():
            raise SpecError(f"{npath}.name must be a valid identifier, got {name!r}")
        if name in masks:
            raise SpecError(f"Duplicate shape name {name!r} at {npath}.name")

        if ntype == "circle":
            _check_keys(npath, node, allowed={"name", "type", "center", "radius", "soft_edge"})
            center = _resolve_vars_node(node.get("center", [0.0, 0.0]), vars_map, f"{npath}.center")
            radius = _resolve_vars_node(node.get("radius", 0.1), vars_map, f"{npath}.radius")
            soft_edge = float(node.get("soft_edge", soft_edge_default))
            masks[name] = shape_gen.generate_circle_mask(center=tuple(center), radius=radius, soft_edge=soft_edge)
        elif ntype == "rectangle":
            _check_keys(npath, node, allowed={"name", "type", "center", "x_size", "y_size", "angle", "soft_edge"})
            center = _resolve_vars_node(node.get("center", [0.0, 0.0]), vars_map, f"{npath}.center")
            x_size = _resolve_vars_node(node.get("x_size", 0.2), vars_map, f"{npath}.x_size")
            y_size = _resolve_vars_node(node.get("y_size", 0.2), vars_map, f"{npath}.y_size")
            angle = _resolve_vars_node(node.get("angle", 0.0), vars_map, f"{npath}.angle")
            soft_edge = float(node.get("soft_edge", soft_edge_default))
            masks[name] = shape_gen.generate_rectangle_mask(
                center=tuple(center), x_size=x_size, y_size=y_size, angle=angle, soft_edge=soft_edge
            )
        elif ntype == "polygon":
            _check_keys(npath, node, allowed={"name", "type", "points", "center", "angle", "invert", "soft_edge"})
            if "points" not in node:
                raise SpecError(f"{npath}.points is required")
            points = _resolve_vars_node(node["points"], vars_map, f"{npath}.points")
            center = _resolve_vars_node(node.get("center"), vars_map, f"{npath}.center") if "center" in node else None
            angle = _resolve_vars_node(node.get("angle"), vars_map, f"{npath}.angle") if "angle" in node else None
            invert = bool(node.get("invert", False))
            soft_edge = float(node.get("soft_edge", soft_edge_default))
            masks[name] = shape_gen.generate_polygon_mask(
                polygon_points=points,
                center=tuple(center) if center is not None else None,
                angle=angle,
                invert=invert,
                soft_edge=soft_edge,
            )
        elif ntype == "mask":
            _check_keys(npath, node, allowed={"name", "type", "value"})
            if "value" not in node:
                raise SpecError(f"{npath}.value is required")
            raw_val = _resolve_vars_node(node["value"], vars_map, f"{npath}.value")
            if isinstance(raw_val, torch.Tensor):
                mask_t = raw_val
                if mask_t.device != mask_device:
                    if mask_t.requires_grad:
                        raise SpecError(f"{npath}.value is trainable but on {mask_t.device}; expected {mask_device}")
                    mask_t = mask_t.to(device=mask_device)
            else:
                mask_t = torch.as_tensor(raw_val, device=mask_device)
            if mask_t.ndim != 2 or tuple(mask_t.shape) != expected_dim:
                raise SpecError(f"{npath}.value must have shape {expected_dim}, got {tuple(mask_t.shape)}")
            mask_t = mask_t.to(dtype=mask_dtype)
            _mask_range_check(mask_t, node_path=npath)
            masks[name] = mask_t
        elif ntype == "op":
            _check_keys(npath, node, allowed={"name", "type", "expr"})
            if "expr" not in node or not isinstance(node["expr"], str) or not node["expr"].strip():
                raise SpecError(f"{npath}.expr must be a non-empty string")
            expr = node["expr"]
            if "$" in expr:
                raise SpecError(f"{npath}.expr must not contain '$' variable references")
            evaluator = _SafeMaskExprEvaluator(expr, node_path=npath)
            masks[name] = evaluator.eval(masks)
        elif ntype == "gds":
            raise SpecError(f"Unsupported shape type at {npath}.type: 'gds' (not supported in interface)")
        else:
            raise SpecError(f"Unsupported shape type at {npath}.type: {ntype!r}")

    if layer_shape not in masks:
        raise SpecError(f"{path}.layer_shape references unknown name {layer_shape!r}")
    return masks[layer_shape]


def _apply_layers_to_solver(
    solver,
    layers: Sequence[Mapping[str, Any]],
    *,
    vars_map: Mapping[str, Any],
) -> None:
    shape_gen = ShapeGenerator.from_solver(solver)
    solver_device = solver.device if isinstance(solver.device, torch.device) else torch.device(solver.device)

    for i, layer in enumerate(layers):
        path = f"spec.layers[{i}]"
        thickness = _resolve_vars_node(layer["thickness"], vars_map, f"{path}.thickness")
        is_hom = layer["is_homogeneous"]
        is_opt = bool(layer.get("is_optimize", False))
        slice_count = int(layer.get("slice_count", 1))
        if slice_count < 1:
            raise SpecError(f"{path}.slice_count must be >= 1")

        if isinstance(thickness, torch.Tensor):
            thickness_t = thickness
            if torch.is_complex(thickness_t):
                raise SpecError(f"{path}.thickness must be real-valued")
            if thickness_t.device != solver_device:
                if thickness_t.requires_grad:
                    raise SpecError(
                        f"{path}.thickness is trainable but on {thickness_t.device}; expected {solver_device}"
                    )
                thickness_t = thickness_t.to(device=solver_device)
            if thickness_t.dtype != solver.tfloat:
                thickness_t = thickness_t.to(dtype=solver.tfloat)
        else:
            thickness_t = torch.as_tensor(thickness, device=solver_device, dtype=solver.tfloat)

        solver.add_layer(
            material_name=layer["material"],
            thickness=thickness_t,
            is_homogeneous=is_hom,
            is_optimize=is_opt,
            slice_count=slice_count,
        )

        if not is_hom:
            pattern = layer["pattern"]
            mask = _build_mask_from_pattern(shape_gen=shape_gen, pattern=pattern, vars_map=vars_map, path=f"{path}.pattern")
            bg_material = pattern.get("bg_material", "air")
            method = pattern.get("method", "FFT")
            solver.update_er_with_mask(mask=mask, layer_index=i, bg_material=bg_material, method=method)


def _load_spec(spec: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load a spec from an in-memory dict or a JSON file path.

    Returns:
        (spec_dict, config_dir): config_dir is the directory containing the JSON file
        when loaded from disk; otherwise None.
    """
    if isinstance(spec, (str, os.PathLike)):
        path = Path(spec).expanduser()
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
        except FileNotFoundError as e:
            raise SpecError(f"Spec file not found: {spec}") from e
        except json.JSONDecodeError as e:
            raise SpecError(f"Invalid JSON spec at {spec}: {e}") from e
        _require_type("spec", loaded, dict)
        return loaded, str(path.resolve().parent)

    _require_type("spec", spec, dict)
    return spec, None


def solve(spec: Any) -> Any:
    """Run a forward simulation from a regulated spec dict.

    Args:
        spec: Either a spec dictionary describing solver/materials/layers/sources, or a
            path to a JSON spec file.

    Returns:
        A structured results dict from :meth:`torchrdit.results.SolverResults.to_structured_dict`.
        For a single source, returns a single dict. For multiple sources, returns
        a dict mapping ``source_name -> result_dict``.

        If ``spec["output"]["type"] == "numpy"``, any torch tensors inside the
        returned dict(s) are converted to NumPy arrays (detached, moved to CPU).

    Raises:
        SpecError: If the spec contains unknown keys, missing required fields,
            incompatible settings (e.g., ``pattern`` on homogeneous layers), or
            invalid types.
    """
    spec, config_dir = _load_spec(spec)
    _validate_spec(spec)

    output_type = _parse_output(spec.get("output"))

    solver_spec = _normalize_solver_spec(spec["solver"])

    compiled_vars = _compile_vars(spec.get("vars"), device=solver_spec.get("device", "cpu"))
    _validate_var_refs_in_spec(spec, vars_keys=compiled_vars.raw.keys())
    vars_map = compiled_vars.evaluate()

    build_spec = dict(solver_spec)
    # Builder validates ref/trn materials at build-time, but interface materials are added post-build.
    # Build with default air and update external media after materials are applied.
    if "trn_material" in build_spec:
        build_spec["trn_material"] = "air"
    if "ref_material" in build_spec:
        build_spec["ref_material"] = "air"
    solver = _build_solver(build_spec)
    solver.add_observer(_DEFAULT_SOLVER_OBSERVER_FACTORY())

    caller_dir = infer_caller_dir(skip_files=("interface.py", "__init__.py", "path_utils.py"))
    materials = _parse_materials(
        spec.get("materials"),
        config_dir=config_dir,
        caller_dir=str(caller_dir) if caller_dir is not None else None,
    )
    _apply_materials_to_solver(solver, materials)
    _apply_external_media_to_solver(solver, solver_spec=solver_spec, materials=materials)

    layers = _parse_layers(spec.get("layers"))
    _apply_layers_to_solver(solver, layers, vars_map=vars_map)

    sources = _parse_sources(spec.get("sources"))

    if len(sources) == 1:
        s0 = sources[0]
        src = solver.add_source(theta=s0["theta"], phi=s0["phi"], pte=s0["pte"], ptm=s0["ptm"], norm_te_dir=s0["norm_te_dir"])
        res = solver.solve(src).to_structured_dict()
        return _postprocess_output(res, output_type=output_type)

    source_objs = [
        solver.add_source(theta=s["theta"], phi=s["phi"], pte=s["pte"], ptm=s["ptm"], norm_te_dir=s["norm_te_dir"])
        for s in sources
    ]
    batched = solver.solve(source_objs)
    results_by_name: Dict[str, Any] = {}
    for i, s in enumerate(sources):
        res = batched[i].to_structured_dict()
        results_by_name[s["name"]] = _postprocess_output(res, output_type=output_type)
    return results_by_name


class _SafeObjectiveEvaluator:
    def __init__(self, expr: str, *, functions: Mapping[str, Callable[..., Any]]) -> None:
        self._expr = expr
        self._functions = dict(functions)
        self._tree = ast.parse(expr, mode="eval")

    def eval(self, env: Mapping[str, Any]) -> Any:
        return self._eval_node(self._tree.body, env)

    def _eval_node(self, node: ast.AST, env: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in self._functions:
                return self._functions[node.id]
            raise SpecError(f"Unknown name in objective: {node.id!r}")
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, env)
            right = self._eval_node(node.right, env)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise SpecError(f"Unsupported binary operator in objective: {type(node.op)}")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, env)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise SpecError(f"Unsupported unary operator in objective: {type(node.op)}")
        if isinstance(node, ast.Subscript):
            value = self._eval_node(node.value, env)
            slc = node.slice
            if isinstance(slc, ast.Constant):
                key = slc.value
            elif isinstance(slc, ast.Slice):
                lower = self._eval_node(slc.lower, env) if slc.lower is not None else None
                upper = self._eval_node(slc.upper, env) if slc.upper is not None else None
                step = self._eval_node(slc.step, env) if slc.step is not None else None
                key = slice(lower, upper, step)
            else:
                key = self._eval_node(slc, env)
            return value[key]
        if isinstance(node, ast.Call):
            func = self._eval_node(node.func, env)
            if not callable(func):
                raise SpecError("Objective call target is not callable")
            if isinstance(node.func, ast.Attribute):
                raise SpecError("Attribute calls are not allowed in objective")
            args = [self._eval_node(a, env) for a in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value, env) for kw in node.keywords}
            return func(*args, **kwargs)
        if isinstance(node, (ast.Tuple, ast.List)):
            elts = [self._eval_node(e, env) for e in node.elts]
            return elts if isinstance(node, ast.List) else tuple(elts)
        if isinstance(node, ast.Dict):
            keys = [self._eval_node(k, env) for k in node.keys]
            vals = [self._eval_node(v, env) for v in node.values]
            return dict(zip(keys, vals))
        if isinstance(node, ast.Attribute):
            raise SpecError("Attribute access is not allowed in objective")
        raise SpecError(f"Unsupported syntax in objective: {type(node).__name__}")


def _objective_functions():
    def _to_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    def abs_(x: Any):
        return torch.abs(_to_tensor(x))

    def real(x: Any):
        return torch.real(_to_tensor(x))

    def imag(x: Any):
        return torch.imag(_to_tensor(x))

    def mean(x: Any):
        if isinstance(x, (list, tuple)):
            return torch.stack([_to_tensor(v) for v in x]).mean()
        return _to_tensor(x).mean()

    def sum_(x: Any):
        if isinstance(x, (list, tuple)):
            return torch.stack([_to_tensor(v) for v in x]).sum()
        return _to_tensor(x).sum()

    def min_(x: Any):
        if isinstance(x, (list, tuple)):
            return torch.stack([_to_tensor(v) for v in x]).min()
        return _to_tensor(x).min()

    def max_(x: Any):
        if isinstance(x, (list, tuple)):
            return torch.stack([_to_tensor(v) for v in x]).max()
        return _to_tensor(x).max()

    return {
        "abs": abs_,
        "real": real,
        "imag": imag,
        "mean": mean,
        "sum": sum_,
        "min": min_,
        "max": max_,
    }


def optimize(spec: Any, *, objective: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Minimize a string objective by optimizing variables defined in ``spec["vars"]``.

    The objective is parsed and evaluated via a restricted AST allowlist (no
    attribute access, no imports). The evaluation environment contains:

    - ``Results``: dict mapping ``source_name -> structured_result_dict`` (torch tensors).
    - ``Vars``: dict mapping ``var_name -> torch.Tensor/Parameter``.
    - Aliases: ``S`` for ``Results`` and ``V`` for ``Vars``.
    - Each variable name is also injected as a top-level name.

    Notes:
        ``optimize`` is isolated from ``solve`` and builds its own solver instance
        which is reused across optimization steps.

    Args:
        spec: Either a spec dictionary describing solver/materials/layers/sources, or a
            path to a JSON spec file.
            The interface expects user-facing solver keys (``wavelengths/grids/harmonics``),
            and rejects internal keys like ``rdim/kdim/lam0``.
        objective: String expression to minimize. Must evaluate to a scalar.
        options: Optimization options (strictly validated). Supported keys:
            ``steps``, ``optimizer`` (currently ``adam``), ``grad_clip``,
            and ``return_best``.

    Returns:
        A dict with:
        - ``loss_history``: list of float losses.
        - ``vars``: final variable values (torch tensors or NumPy arrays per output type).
        - ``best`` (optional): best loss/vars/results if ``return_best=True``.
        - ``last_results`` (otherwise): last-step results by source.

    Raises:
        SpecError: If the spec/options are invalid or the objective expression
            cannot be evaluated safely to a scalar torch tensor.
    """
    spec, config_dir = _load_spec(spec)
    _validate_spec(spec)
    if not isinstance(objective, str) or not objective.strip():
        raise SpecError("objective must be a non-empty string")
    options = dict(options or {})

    _check_keys(
        "options",
        options,
        allowed={"steps", "optimizer", "return_best", "grad_clip"},
    )
    steps = int(options.get("steps", 100))
    if steps < 1:
        raise SpecError("options.steps must be >= 1")

    opt_cfg = options.get("optimizer", {"name": "adam", "lr": 1e-2})
    _require_type("options.optimizer", opt_cfg, dict)
    _check_keys("options.optimizer", opt_cfg, allowed={"name", "lr"})
    opt_name = opt_cfg.get("name", "adam")
    lr = float(opt_cfg.get("lr", 1e-2))

    output_type = _parse_output(spec.get("output"))

    solver_spec = _normalize_solver_spec(spec["solver"])
    device = solver_spec.get("device", "cpu")

    compiled_vars = _compile_vars(spec.get("vars"), device=device)
    _validate_var_refs_in_spec(spec, vars_keys=compiled_vars.raw.keys())
    vars_map = compiled_vars.evaluate()
    trainable_params = compiled_vars.trainable_params()
    if not trainable_params:
        raise SpecError("No trainable variables found in spec.vars")

    if opt_name.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
    else:
        raise SpecError(f"Unsupported optimizer: {opt_name!r}")

    build_spec = dict(solver_spec)
    if "trn_material" in build_spec:
        build_spec["trn_material"] = "air"
    if "ref_material" in build_spec:
        build_spec["ref_material"] = "air"
    solver = _build_solver(build_spec)
    throttled_observer = _OptimizeStepThrottledObserver(
        _DEFAULT_SOLVER_OBSERVER_FACTORY(),
        every=10,
        total_steps=steps,
    )
    solver.add_observer(throttled_observer)
    caller_dir = infer_caller_dir(skip_files=("interface.py", "__init__.py", "path_utils.py"))
    materials = _parse_materials(
        spec.get("materials"),
        config_dir=config_dir,
        caller_dir=str(caller_dir) if caller_dir is not None else None,
    )
    _apply_materials_to_solver(solver, materials)
    _apply_external_media_to_solver(solver, solver_spec=solver_spec, materials=materials)
    layers = _parse_layers(spec.get("layers"))
    _apply_layers_to_solver(solver, layers, vars_map=vars_map)
    sources = _parse_sources(spec.get("sources"))
    source_objs = [
        (
            s["name"],
            solver.add_source(theta=s["theta"], phi=s["phi"], pte=s["pte"], ptm=s["ptm"], norm_te_dir=s["norm_te_dir"]),
        )
        for s in sources
    ]

    evaluator = _SafeObjectiveEvaluator(objective, functions=_objective_functions())

    best = {"loss": None, "vars": None, "results": None}
    loss_history: List[float] = []

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        vars_map = compiled_vars.evaluate()

        throttled_observer.set_step(step_index)
        solver_device = solver.device if isinstance(solver.device, torch.device) else torch.device(solver.device)

        # Update layer thicknesses (covers derived var expressions).
        for i, layer in enumerate(layers):
            thickness = _resolve_vars_node(layer["thickness"], vars_map, f"spec.layers[{i}].thickness")
            if isinstance(thickness, torch.Tensor):
                thickness_t = thickness
                if torch.is_complex(thickness_t):
                    raise SpecError(f"spec.layers[{i}].thickness must be real-valued")
                if thickness_t.device != solver_device:
                    if thickness_t.requires_grad:
                        raise SpecError(
                            f"spec.layers[{i}].thickness is trainable but on {thickness_t.device}; expected {solver_device}"
                        )
                    thickness_t = thickness_t.to(device=solver_device)
                if thickness_t.dtype != solver.tfloat:
                    thickness_t = thickness_t.to(dtype=solver.tfloat)
            else:
                thickness_t = torch.as_tensor(thickness, device=solver_device, dtype=solver.tfloat)
            solver.update_layer_thickness(layer_index=i, thickness=thickness_t)

        # Update patterned layers: rebuild masks each step.
        shape_gen = ShapeGenerator.from_solver(solver)
        for i, layer in enumerate(layers):
            if layer.get("is_homogeneous") is False:
                mask = _build_mask_from_pattern(
                    shape_gen=shape_gen,
                    pattern=layer["pattern"],
                    vars_map=vars_map,
                    path=f"spec.layers[{i}].pattern",
                )
                solver.update_er_with_mask(
                    mask=mask,
                    layer_index=i,
                    bg_material=layer["pattern"].get("bg_material", "air"),
                    method=layer["pattern"].get("method", "FFT"),
                )

        results_by_name: Dict[str, Any] = {}
        if len(source_objs) == 1:
            name, src = source_objs[0]
            results_by_name[name] = solver.solve(src).to_structured_dict()
        else:
            batched = solver.solve([src for _, src in source_objs])
            for i, (name, _) in enumerate(source_objs):
                results_by_name[name] = batched[i].to_structured_dict()

        vars_env: Dict[str, Any] = {}
        for k, v in vars_map.items():
            vars_env[k] = v
            stripped = _strip_var_prefix(k)
            vars_env[stripped] = v

        env = {
            "Results": results_by_name,
            "Vars": vars_env,
            "S": results_by_name,
            "V": vars_env,
            **{k: v for k, v in vars_env.items() if not k.startswith("$")},
        }
        loss = evaluator.eval(env)
        if not isinstance(loss, torch.Tensor):
            loss = torch.as_tensor(loss, device=solver.device, dtype=torch.float32)
        if loss.numel() != 1:
            raise SpecError("Objective must evaluate to a scalar tensor")
        loss.backward()

        grad_clip = options.get("grad_clip")
        if grad_clip is not None:
            _require_type("options.grad_clip", grad_clip, dict)
            _check_keys("options.grad_clip", grad_clip, allowed={"max_norm"})
            max_norm = float(grad_clip.get("max_norm"))
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_norm)

        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        loss_history.append(loss_val)

        if options.get("return_best", False):
            if best["loss"] is None or loss_val < best["loss"]:
                best["loss"] = loss_val
                best["vars"] = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in vars_map.items()}
                best["results"] = {k: v for k, v in results_by_name.items()}

    final_vars = compiled_vars.evaluate()

    out = {
        "loss_history": loss_history,
        "vars": final_vars,
    }
    if best["loss"] is not None:
        out["best"] = {
            "loss": best["loss"],
            "vars": best["vars"],
            "results": _postprocess_output(best["results"], output_type=output_type),
        }
    else:
        out["last_results"] = _postprocess_output(results_by_name, output_type=output_type)

    # Always post-process vars for output type (numpy means detach)
    if output_type == "numpy":
        out["vars"] = _postprocess_output(out["vars"], output_type=output_type)
        if "best" in out and out["best"].get("vars") is not None:
            out["best"]["vars"] = _postprocess_output(out["best"]["vars"], output_type=output_type)

    return out


class Interface:
    """Facade class exposing the regulated public interface (solve + optimize)."""

    @staticmethod
    def solve(spec: Any) -> Any:
        return solve(spec)

    @staticmethod
    def optimize(spec: Any, *, objective: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return optimize(spec, objective=objective, options=options)
