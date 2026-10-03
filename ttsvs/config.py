"""Paths and device selection."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_dir

ENV_HOME = "TTSVS_HOME"


def data_dir() -> Path:
    """Root data folder: $TTSVS_HOME if set, else the OS user data dir."""
    override = os.environ.get(ENV_HOME)
    path = Path(override).expanduser() if override else Path(user_data_dir("ttsvs"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def voices_dir() -> Path:
    path = data_dir() / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir() -> Path:
    path = data_dir() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(device: str = "auto") -> str:
    """Map "auto" to the best available device (cuda > mps > cpu)."""
    device = (device or "auto").lower()
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device {device!r}; use auto, cpu, cuda or mps.")
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def gpu_backend() -> str:
    """Diagnostic label: "rocm", "cuda", "mps" or "cpu".

    ROCm builds of PyTorch expose the AMD GPU as a "cuda" device, so
    resolve_device() returns "cuda" there; this tells the two apart.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
