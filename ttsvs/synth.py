"""Text + voice -> audio, with chunking and stitching."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import audio
from .engines import get_engine
from .text import split_text
from .voices import VoiceLibrary

PAUSE_S = 0.2


class Synthesizer:
    def __init__(self, engine: str = "chatterbox", device: str = "auto",
                 library: VoiceLibrary | None = None):
        self.engine = get_engine(engine, device)
        self._library = library
        self._loaded = False

    @property
    def library(self) -> VoiceLibrary:
        if self._library is None:
            self._library = VoiceLibrary()
        return self._library

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.engine.load()
            self._loaded = True

    def speak(self, text: str, voice_id: str | None = None,
              progress: Callable[[int, int], None] | None = None,
              **params) -> tuple[int, np.ndarray]:
        chunks = split_text(text, self.engine.max_chars)
        if not chunks:
            raise ValueError("There is no text to speak.")
        reference = self.library.get(voice_id).reference_path if voice_id else None
        self._ensure_loaded()

        pieces: list[np.ndarray] = []
        total = len(chunks)
        if progress:
            progress(0, total)
        for i, chunk in enumerate(chunks, start=1):
            pieces.append(self.engine.synthesize(chunk, reference, **params))
            if progress:
                progress(i, total)
        sr = self.engine.sample_rate  # Chatterbox sets the real rate on load
        return sr, audio.concat_with_pauses(pieces, sr, PAUSE_S)

    def speak_to_file(self, text: str, out_path: Path, voice_id: str | None = None,
                      **params) -> Path:
        sr, data = self.speak(text, voice_id, **params)
        return audio.save_audio(out_path, data, sr)
