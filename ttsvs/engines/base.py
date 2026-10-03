"""Abstract TTS engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EngineError(RuntimeError):
    """Raised when an engine cannot load or run."""


class TTSEngine(ABC):
    name: str
    sample_rate: int
    max_chars: int = 300  # longest text chunk to send to synthesize() at once

    def __init__(self, device: str = "auto"):
        self.device = device

    @abstractmethod
    def load(self) -> None:
        """Load the model lazily; must be idempotent."""

    @abstractmethod
    def synthesize(self, text: str, reference_wav: Path | None, **params) -> np.ndarray:
        """Return float32 mono audio in [-1, 1] at self.sample_rate."""

    def supported_params(self) -> dict[str, dict]:
        """name -> {default, min, max, help}"""
        return {}

    def resolve_params(self, params: dict) -> dict:
        """Drop unknown/None params, fill defaults and clamp to ranges."""
        out: dict = {}
        for key, spec in self.supported_params().items():
            value = params.get(key)
            if value is None:
                value = spec["default"]
            out[key] = min(max(float(value), spec["min"]), spec["max"])
        return out
