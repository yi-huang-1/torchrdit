"""Device resolution helpers for TorchRDIT solver construction.

This module centralizes device normalization and fallback decisions before Torch
allocations occur in builder, solver, and utility entry points. TorchRDIT v0.1
supports `cpu` and `cuda` backends only. Unsupported backends, unavailable CUDA
devices, and MPS requests fall back to CPU with an explicit `UserWarning` and a
`DeviceResolution` record describing the outcome.

Classes:
    DeviceResolution: Immutable record describing requested and resolved device state.

Functions:
    resolve_device: Normalize and validate a requested device before allocation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch

_SUPPORTED_BACKENDS = frozenset({"cpu", "cuda"})


@dataclass(frozen=True)
class DeviceResolution:
    """Describe how a requested device maps to an actual runtime device.

    Attributes:
        requested_device: Original device string requested by the caller.
        resolved_device: Actual `torch.device` used for allocations.
        fell_back: Whether TorchRDIT had to degrade to a different backend.
        reason: Explanation for the fallback, or `None` when no fallback occurred.
    """

    requested_device: str
    resolved_device: torch.device
    fell_back: bool
    reason: Optional[str]


def _fallback(requested: str, reason: str) -> DeviceResolution:
    warnings.warn(reason, UserWarning)
    return DeviceResolution(
        requested_device=requested,
        resolved_device=torch.device("cpu"),
        fell_back=True,
        reason=reason,
    )


def resolve_device(device: Union[str, torch.device]) -> DeviceResolution:
    """Resolve a user-provided device request to a supported runtime device.

    TorchRDIT currently supports `cpu` and `cuda` backends only. MPS is rejected
    unconditionally because the solver stack relies on complex dtypes that are not
    supported on that backend. Any unsupported or unavailable device request is
    converted to CPU with an accompanying warning.

    Args:
        device: Requested device as a string or `torch.device` instance.

    Returns:
        A `DeviceResolution` record describing the requested device, the resolved
        runtime device, whether fallback occurred, and the fallback reason.
    """

    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device

    backend = device_str.split(":")[0]

    if backend == "mps":
        return _fallback(
            device_str,
            "mps backend rejected: complex-dtype operations are unsupported on mps.",
        )

    if backend not in _SUPPORTED_BACKENDS:
        return _fallback(
            device_str,
            f"Unsupported device '{device_str}'. Falling back to cpu.",
        )

    if backend == "cpu":
        return DeviceResolution(
            requested_device=device_str,
            resolved_device=torch.device("cpu"),
            fell_back=False,
            reason=None,
        )

    if backend == "cuda":
        if not torch.cuda.is_available():
            return _fallback(
                device_str,
                "CUDA not available. Falling back to cpu.",
            )

        index = None
        if ":" in device_str:
            index = int(device_str.split(":")[1])
            count = torch.cuda.device_count()
            if index >= count:
                return _fallback(
                    device_str,
                    f"CUDA device index {index} exceeds available devices "
                    f"({count}). Falling back to cpu.",
                )

        return DeviceResolution(
            requested_device=device_str,
            resolved_device=torch.device(device_str),
            fell_back=False,
            reason=None,
        )

    return _fallback(device_str, f"Unhandled device '{device_str}'. Falling back to cpu.")
