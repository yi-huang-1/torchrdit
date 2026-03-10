from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch

_SUPPORTED_BACKENDS = frozenset({"cpu", "cuda"})


@dataclass(frozen=True)
class DeviceResolution:
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
