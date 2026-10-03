"""Dependency-free engine for tests and UI work without a model."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from .base import TTSEngine


class DummyEngine(TTSEngine):
    name = "dummy"
    sample_rate = 24000
    SECONDS_PER_CHAR = 0.05

    def load(self) -> None:
        return None

    def synthesize(self, text: str, reference_wav: Path | None, **params) -> np.ndarray:
        n = max(int(len(text) * self.SECONDS_PER_CHAR * self.sample_rate), self.sample_rate // 10)
        # Pitch derived from the text so output is deterministic but varies per input.
        freq = 180.0 + (zlib.crc32(text.encode("utf-8")) % 200)
        t = np.arange(n) / self.sample_rate
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        fade = min(n // 2, int(0.01 * self.sample_rate))
        if fade:
            ramp = np.linspace(0.0, 1.0, fade)
            tone[:fade] *= ramp
            tone[-fade:] *= ramp[::-1]
        return tone.astype(np.float32)

    def supported_params(self) -> dict[str, dict]:
        return {"speed": {"default": 1.0, "min": 0.5, "max": 2.0, "help": "Ignored by the dummy engine."}}
