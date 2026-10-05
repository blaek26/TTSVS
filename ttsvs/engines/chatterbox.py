"""Chatterbox (Resemble AI) engine. torch/chatterbox are imported lazily."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config
from ..audio import shorten_silences, trim_silence
from .base import EngineError, TTSEngine

INSTALL_HINT = 'The Chatterbox engine is not installed. Run: pip install "ttsvs[chatterbox]"'


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    sample_rate = 24000
    # Chatterbox tends to garble or babble at the end of long inputs; a sentence
    # or two per call keeps it clean.
    max_chars = 200

    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self._model = None

    def supported_params(self) -> dict[str, dict]:
        return {
            "exaggeration": {
                "default": 0.5, "min": 0.25, "max": 2.0,
                "help": "Emotion/expressiveness. Higher is more dramatic.",
            },
            "cfg_weight": {
                "default": 0.5, "min": 0.0, "max": 1.0,
                "help": "Pacing and adherence to the reference voice. Lower is slower/more deliberate.",
            },
            "temperature": {
                "default": 0.8, "min": 0.05, "max": 5.0,
                "help": "Randomness. Lower is more stable.",
            },
        }

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            raise EngineError(INSTALL_HINT) from exc

        device = config.resolve_device(self.device)
        try:
            self._model = ChatterboxTTS.from_pretrained(device=device)
        except (RuntimeError, AttributeError) as exc:
            # Checkpoints saved on CUDA fail to deserialize on cpu/mps without map_location.
            if device == "cuda" or "map_location" not in str(exc) and "CUDA" not in str(exc):
                raise EngineError(f"Could not load the Chatterbox model: {exc}") from exc
            self._model = self._load_with_map_location(torch, ChatterboxTTS, device)
        self.sample_rate = int(self._model.sr)

    @staticmethod
    def _load_with_map_location(torch, cls, device: str):
        original = torch.load
        map_location = torch.device(device)

        def patched(*args, **kwargs):
            kwargs.setdefault("map_location", map_location)
            return original(*args, **kwargs)

        torch.load = patched
        try:
            return cls.from_pretrained(device=device)
        except Exception as exc:
            raise EngineError(f"Could not load the Chatterbox model on {device}: {exc}") from exc
        finally:
            torch.load = original

    def synthesize(self, text: str, reference_wav: Path | None, **params) -> np.ndarray:
        self.load()
        p = self.resolve_params(params)
        kwargs = dict(p)
        if reference_wav is not None:
            kwargs["audio_prompt_path"] = str(reference_wav)
        wav = self._model.generate(text, **kwargs)
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        out = np.clip(np.asarray(wav, dtype=np.float32).squeeze(), -1.0, 1.0)
        # Drop the quiet tail the model often leaves after the last word.
        out = trim_silence(out, self.sample_rate, threshold_db=-45.0, pad_s=0.08)
        return shorten_silences(out, self.sample_rate)
