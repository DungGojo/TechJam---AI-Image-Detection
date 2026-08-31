"""Single source of truth for the compute device.

Every script in this repo must call `get_device()` here. Nothing hardcodes
"cuda", "mps" or "cpu" anywhere else.
"""

from __future__ import annotations

import os

import torch

_PRIORITY = ("cuda", "mps", "cpu")


def _available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "mps":
        return torch.backends.mps.is_available()
    return True


def get_device(override: str | None = None) -> torch.device:
    """Resolve an available CUDA, MPS, or CPU device."""
    requested = override or os.environ.get("AIGC_DEVICE")
    if requested:
        requested = requested.strip().lower()
        base = requested.split(":", 1)[0]
        if base not in _PRIORITY:
            raise ValueError(f"unknown device {requested!r}; expected one of {_PRIORITY}")
        if not _available(base):
            raise RuntimeError(f"device {requested!r} was requested but is not available")
        return torch.device(requested)

    for name in _PRIORITY:
        if _available(name):
            return torch.device(name)
    return torch.device("cpu")


def device_report() -> dict[str, object]:
    """Facts about the machine, for check_env.py and for run provenance."""
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "selected": str(get_device()),
    }
