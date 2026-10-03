"""Voice library: cleaned reference clips plus metadata on disk."""

from __future__ import annotations

import json
import math
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import audio, config

REFERENCE_SR = 24000
MIN_DURATION_S = 5.0
MAX_DURATION_S = 60.0


class VoiceError(Exception):
    """Base class for voice library errors."""


class VoiceNotFoundError(VoiceError, KeyError):
    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else ""


class VoiceExistsError(VoiceError):
    pass


@dataclass
class Voice:
    id: str  # slug, folder name
    name: str  # display name
    reference_path: Path  # cleaned 24 kHz mono WAV
    created_at: str
    notes: str = ""
    duration_s: float = 0.0
    consent: bool = False  # user confirmed they may clone this voice
    defaults: dict = field(default_factory=dict)  # engine param -> value; may include "speed"


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "voice"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clean_defaults(raw) -> dict[str, float]:
    """Keep only name -> finite number entries (tolerates hand-edited files)."""
    if not isinstance(raw, dict):
        return {}
    return {k: float(v) for k, v in raw.items() if isinstance(k, str) and _is_number(v)}


def check_duration(seconds: float) -> str | None:
    """Return a warning if a clip is outside the recommended 5-60 s range."""
    if seconds < MIN_DURATION_S:
        return (
            f"Clip is only {seconds:.1f}s long; 10-30 seconds of clean speech "
            "gives much better cloning."
        )
    if seconds > MAX_DURATION_S:
        return f"Clip is {seconds:.0f}s long; only about the first 10-30 seconds are useful, consider trimming it."
    return None


class VoiceLibrary:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else config.voices_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def _read(self, folder: Path) -> Voice | None:
        meta = folder / "voice.json"
        ref = folder / "reference.wav"
        if not meta.is_file() or not ref.is_file():
            return None
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            return Voice(
                id=folder.name,
                name=data["name"],
                reference_path=ref,
                created_at=data.get("created_at", ""),
                notes=data.get("notes", ""),
                duration_s=float(data.get("duration_s", 0.0)),
                consent=bool(data.get("consent", False)),
                defaults=_clean_defaults(data.get("defaults")),
            )
        except (ValueError, KeyError, OSError, TypeError):
            return None

    def _write(self, voice: Voice) -> None:
        data = asdict(voice)
        data.pop("id")
        data.pop("reference_path")
        (voice.reference_path.parent / "voice.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def list(self) -> list[Voice]:
        voices = [v for f in sorted(self.root.iterdir()) if f.is_dir() and (v := self._read(f))]
        return sorted(voices, key=lambda v: v.name.lower())

    def _find(self, key: str) -> Voice | None:
        voices = self.list()
        for v in voices:
            if v.id == key:
                return v
        low = key.strip().lower()
        for v in voices:
            if v.name.lower() == low or v.id == slugify(key):
                return v
        return None

    def get(self, voice_id: str) -> Voice:
        """Look up by id or display name (case-insensitive)."""
        voice = self._find(voice_id)
        if voice is None:
            known = ", ".join(v.name for v in self.list()) or "none yet"
            raise VoiceNotFoundError(f"Voice {voice_id!r} not found (available: {known}).")
        return voice

    def add(self, name: str, audio_path: Path, notes: str = "", consent: bool = False) -> Voice:
        name = name.strip()
        if not name:
            raise ValueError("Voice name must not be empty.")
        if any(v.name.lower() == name.lower() for v in self.list()):
            raise VoiceExistsError(f"A voice named {name!r} already exists.")

        data, sr = audio.load_audio(audio_path)
        data = audio.resample(data, sr, REFERENCE_SR)
        data = audio.trim_silence(data, REFERENCE_SR)
        if data.size == 0 or float(abs(data).max()) == 0.0:
            raise audio.AudioError("The clip contains no audible speech.")
        data = audio.normalize(data)

        base = slugify(name)
        voice_id, n = base, 2
        while (self.root / voice_id).exists():
            voice_id, n = f"{base}-{n}", n + 1
        folder = self.root / voice_id
        folder.mkdir(parents=True)
        try:
            ref = folder / "reference.wav"
            audio.save_audio(ref, data, REFERENCE_SR)
            voice = Voice(
                id=voice_id,
                name=name,
                reference_path=ref,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                notes=notes,
                duration_s=round(len(data) / REFERENCE_SR, 3),
                consent=bool(consent),
            )
            self._write(voice)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return voice

    def delete(self, voice_id: str) -> None:
        voice = self.get(voice_id)
        shutil.rmtree(voice.reference_path.parent)

    def rename(self, voice_id: str, new_name: str) -> Voice:
        """Change the display name; the id (folder) stays stable."""
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("Voice name must not be empty.")
        voice = self.get(voice_id)
        for other in self.list():
            if other.id != voice.id and other.name.lower() == new_name.lower():
                raise VoiceExistsError(f"A voice named {new_name!r} already exists.")
        voice.name = new_name
        self._write(voice)
        return voice

    def set_defaults(self, voice_id: str, params: dict) -> Voice:
        """Save engine settings (and "speed") that load whenever this voice is picked.

        An empty dict clears them. Values must be finite numbers.
        """
        if not isinstance(params, dict):
            raise ValueError("Defaults must be a dict of setting name to number.")
        for key, value in params.items():
            if not isinstance(key, str):
                raise ValueError(f"Setting names must be strings, got {key!r}.")
            if not _is_number(value):
                raise ValueError(f"Setting {key!r} must be a number, got {value!r}.")
        voice = self.get(voice_id)
        voice.defaults = {k: float(v) for k, v in params.items()}
        self._write(voice)
        return voice
