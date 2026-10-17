"""Text + voice -> audio, with chunking and stitching."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio
from .engines import get_engine
from .history import History
from .lexicon import Lexicon
from .text import split_text
from .voices import Voice, VoiceLibrary

PAUSE_S = 0.2
SPEED_MIN = 0.5
SPEED_MAX = 2.0


@dataclass
class _Plan:
    """Everything a run needs, resolved up front so bad input fails before any model loads."""

    chunks: list[str]
    voice: Voice | None
    engine_params: dict  # resolved engine settings only (never "speed")
    speed: float


def _clamp_speed(value) -> float:
    speed = float(value)
    if math.isnan(speed):
        raise ValueError("Speed must be a number.")
    return min(max(speed, SPEED_MIN), SPEED_MAX)


class Synthesizer:
    def __init__(self, engine: str = "chatterbox", device: str = "auto",
                 library: VoiceLibrary | None = None,
                 lexicon: Lexicon | None = None, history: History | None = None,
                 save_history: bool = False):
        self.engine = get_engine(engine, device)
        self._library = library
        self._lexicon = lexicon
        self._history = history
        self.save_history = save_history
        self._loaded = False

    @property
    def library(self) -> VoiceLibrary:
        if self._library is None:
            self._library = VoiceLibrary()
        return self._library

    @property
    def lexicon(self) -> Lexicon:
        if self._lexicon is None:
            self._lexicon = Lexicon()
        return self._lexicon

    @property
    def history(self) -> History:
        if self._history is None:
            self._history = History()
        return self._history

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.engine.load()
            self._loaded = True

    def _merged(self, voice: Voice | None, params: dict) -> tuple[dict, float]:
        """(resolved engine params, clamped speed): explicit > voice defaults > engine defaults."""
        merged: dict = dict(voice.defaults) if voice is not None else {}
        merged.update({k: v for k, v in params.items() if v is not None})
        raw_speed = merged.pop("speed", None)
        speed = _clamp_speed(1.0 if raw_speed is None else raw_speed)
        engine_params = self.engine.resolve_params(merged)
        engine_params.pop("speed", None)  # speed is ours; engines never see it
        return engine_params, speed

    def effective_params(self, voice_id: str | None, params: dict) -> dict:
        """The settings a call would really use (for the UI and history), including speed."""
        voice = self.library.get(voice_id) if voice_id else None
        engine_params, speed = self._merged(voice, dict(params or {}))
        return {**engine_params, "speed": speed}

    def _prepare(self, text: str, voice_id: str | None, params: dict) -> _Plan:
        chunks = split_text(self.lexicon.apply(text), self.engine.max_chars)
        if not chunks:
            raise ValueError("There is no text to speak.")
        voice = self.library.get(voice_id) if voice_id else None
        engine_params, speed = self._merged(voice, params)
        return _Plan(chunks, voice, engine_params, speed)

    def _run(self, plan: _Plan) -> Iterator[tuple[int, np.ndarray, int, int]]:
        reference = plan.voice.reference_path if plan.voice else None
        self._ensure_loaded()
        sr = self.engine.sample_rate  # Chatterbox sets the real rate on load
        gap = np.zeros(int(PAUSE_S * sr), dtype=np.float32)
        total = len(plan.chunks)
        for i, chunk in enumerate(plan.chunks, start=1):
            piece = self.engine.synthesize(chunk, reference, **plan.engine_params)
            if plan.speed != 1.0:
                piece = audio.time_stretch(piece, sr, plan.speed)
            piece = audio.fade(piece, sr)
            if i < total:
                piece = np.concatenate([piece, gap])
            yield sr, piece, i, total

    def speak_stream(self, text: str, voice_id: str | None = None,
                     **params) -> Iterator[tuple[int, np.ndarray, int, int]]:
        """Yield (sample_rate, chunk_audio, index, total) as each chunk finishes.

        Chunks already carry their fades and a trailing pause (except the last),
        so concatenating them gives exactly what speak() returns. Bad input
        (no text, unknown voice) raises here, before the first chunk.
        """
        return self._run(self._prepare(text, voice_id, params))

    def speak(self, text: str, voice_id: str | None = None,
              progress: Callable[[int, int], None] | None = None,
              **params) -> tuple[int, np.ndarray]:
        plan = self._prepare(text, voice_id, params)
        total = len(plan.chunks)
        if progress:
            progress(0, total)
        pieces: list[np.ndarray] = []
        for sr, piece, i, total in self._run(plan):
            pieces.append(piece)
            if progress:
                progress(i, total)
        result = np.concatenate(pieces)
        if self.save_history:
            self._save_to_history(sr, result, text, plan)
        return sr, result

    def _save_to_history(self, sr: int, data: np.ndarray, text: str, plan: _Plan) -> None:
        voice = plan.voice
        try:
            self.history.add(
                sr, data, text,
                voice.id if voice else None,
                voice.name if voice else None,
                self.engine.name,
                {**plan.engine_params, "speed": plan.speed},
            )
        except Exception as exc:  # the audio is already made; don't lose it over a save problem
            warnings.warn(f"Could not save to history: {exc}", RuntimeWarning, stacklevel=3)

    def speak_to_file(self, text: str, out_path: Path, voice_id: str | None = None,
                      **params) -> Path:
        sr, data = self.speak(text, voice_id, **params)
        return audio.save_audio(out_path, data, sr)
