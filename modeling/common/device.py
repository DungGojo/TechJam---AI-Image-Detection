"""Single source of truth for the compute device.

Every script in this repo must call `get_device()` here. Nothing hardcodes
"cuda", "mps" or "cpu" anywhere else.
"""

from __future__ import annotations

import os

import torch

_DEVICE_PRIORITY = ("cuda", "mps", "cpu")



def _is_device_available(device_name: str) -> bool:
    """Check whether a given hardware accelerator backend is operational."""
    if device_name == "cuda":
        return torch.cuda.is_available()
    if device_name == "mps":
        return torch.backends.mps.is_available()
    return True


def get_device(override: str | None = None) -> torch.device:
    """Resolve an available CUDA, MPS, or CPU device following standard priority.

    Args:
        override: Optional explicit device string (e.g. 'cuda:0', 'mps', 'cpu').
                  If omitted, checked against AIGC_DEVICE env var or auto-detected.

    Returns:
        torch.device instance.

    Raises:
        ValueError: If the requested device type is not supported.
        RuntimeError: If the requested device is not available on the host machine.
    """
    device_string = override or os.environ.get("AIGC_DEVICE")
    if device_string:
        device_string = device_string.strip().lower()
        device_type = device_string.split(":", 1)[0]
        if device_type not in _DEVICE_PRIORITY:
            raise ValueError(f"unknown device {device_string!r}; expected one of {_DEVICE_PRIORITY}")
        if not _is_device_available(device_type):
            raise RuntimeError(f"device {device_string!r} was requested but is not available")
        return torch.device(device_string)

    for device_name in _DEVICE_PRIORITY:
        if _is_device_available(device_name):
            return torch.device(device_name)
    return torch.device("cpu")


def device_report() -> dict[str, object]:
    """Gather hardware and PyTorch runtime facts for run provenance and diagnostics."""
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "selected": str(get_device()),
    }

