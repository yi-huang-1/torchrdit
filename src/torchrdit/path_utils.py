from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Optional, Sequence


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return None


def infer_caller_dir(*, skip_files: Optional[Sequence[str]] = None) -> Optional[Path]:
    """Infer the directory of the external caller (best-effort).

    This is used to resolve relative file references when a user passes an in-memory
    dict spec (as opposed to loading a spec from a file path).
    """
    skip_files_set = set(skip_files or ())
    for frame_info in inspect.stack()[2:]:
        filename = frame_info.filename
        if not filename or filename.startswith("<"):
            continue
        base = os.path.basename(filename)
        if base in skip_files_set:
            continue
        if base.startswith("interface"):
            continue
        try:
            return Path(filename).resolve().parent
        except OSError:
            continue
    return None


def resolve_data_path(
    raw_path: str,
    *,
    config_dir: Optional[os.PathLike[str] | str] = None,
    caller_dir: Optional[os.PathLike[str] | str] = None,
    cwd: Optional[os.PathLike[str] | str] = None,
) -> str:
    """Resolve a user-provided path to an existing filesystem path.

    Resolution order for relative paths:
    1) config_dir (directory containing the loaded JSON config/spec)
    2) caller_dir (directory of the caller script)
    3) cwd (current working directory)
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Expected file path to be a non-empty string, got {type(raw_path)}")

    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    path = Path(expanded)

    if path.is_absolute():
        if path.exists():
            return str(path)
        raise ValueError(f"Material data file not found: {raw_path} (resolved: {path})")

    attempted: list[Path] = []
    if config_dir is not None:
        attempted.append(Path(config_dir) / path)
    if caller_dir is not None:
        attempted.append(Path(caller_dir) / path)
    attempted.append(Path(cwd) / path if cwd is not None else Path.cwd() / path)

    found = _first_existing([p.resolve() for p in attempted])
    if found is not None:
        return str(found)

    attempted_str = ", ".join(str(p.resolve()) for p in attempted)
    raise ValueError(
        f"Material data file not found: {raw_path} (attempted: {attempted_str}). "
        f"Use an absolute path or load the spec/config from a file path."
    )

