"""Audio loading, cleanup and saving helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

try:  # soxr gives high quality resampling; fall back to linear interpolation
    import soxr
except ImportError:  # pragma: no cover
    soxr = None

_SF_EXTS = {".wav", ".flac", ".ogg", ".oga"}


class AudioError(RuntimeError):
    """Raised when audio cannot be read or written."""


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def _load_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise AudioError(
            f"Cannot read {path.suffix} files without ffmpeg. Install ffmpeg "
            "or convert the file to WAV/FLAC/OGG first."
        )
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "44100", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), 44100


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any supported file as (float32 mono, sample_rate)."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"Audio file not found: {path}")
    if path.suffix.lower() in _SF_EXTS:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            return _to_mono(data), int(sr)
        except Exception as exc:
            if ffmpeg_path() is None:
                raise AudioError(f"Could not read {path}: {exc}") from exc
    data, sr = _load_ffmpeg(path)
    return data, sr


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    if soxr is not None:
        return soxr.resample(audio, sr, target_sr).astype(np.float32)
    n = int(round(len(audio) * target_sr / sr))  # pragma: no cover
    x_new = np.linspace(0, len(audio) - 1, n)
    return np.interp(x_new, np.arange(len(audio)), audio).astype(np.float32)  # pragma: no cover


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0, pad_s: float = 0.1) -> np.ndarray:
    """Strip leading/trailing samples quieter than threshold_db below peak."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > peak * 10 ** (threshold_db / 20))
    pad = int(pad_s * sr)
    start = max(int(loud[0]) - pad, 0)
    end = min(int(loud[-1]) + pad + 1, len(audio))
    return audio[start:end]


def normalize(audio: np.ndarray, target_peak: float = 0.95, target_rms_db: float | None = None) -> np.ndarray:
    """Peak-normalize; optionally scale to an RMS target first (peak still capped)."""
    audio = audio.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        return audio
    if target_rms_db is not None:
        rms = float(np.sqrt(np.mean(audio**2)))
        gain = 10 ** (target_rms_db / 20) / max(rms, 1e-9)
        gain = min(gain, target_peak / peak)
    else:
        gain = target_peak / peak
    return (audio * gain).astype(np.float32)


def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    """Write WAV (or MP3 when ffmpeg is installed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        ffmpeg = ffmpeg_path()
        if ffmpeg is None:
            raise AudioError("MP3 output needs ffmpeg. Install it or use a .wav filename.")
        proc = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", str(path)],
            input=audio.tobytes(),
            capture_output=True,
        )
        if proc.returncode != 0:
            raise AudioError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")
        return path
    if suffix in ("", ".wav"):
        sf.write(str(path if suffix else path.with_suffix(".wav")), audio, sr, subtype="PCM_16")
        return path if suffix else path.with_suffix(".wav")
    if suffix in (".flac", ".ogg"):
        sf.write(str(path), audio, sr)
        return path
    raise AudioError(f"Unsupported output format {suffix!r}; use .wav (or .mp3 with ffmpeg).")


def fade(audio: np.ndarray, sr: int, fade_s: float = 0.01) -> np.ndarray:
    """Short linear fade-in and fade-out so chunk edges don't click."""
    audio = np.array(audio, dtype=np.float32)
    n = min(int(fade_s * sr), len(audio) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def concat_with_pauses(chunks: list[np.ndarray], sr: int, pause_s: float = 0.2) -> np.ndarray:
    """Join chunks with pause_s seconds of silence between them, fading each edge."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(pause_s * sr), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        if i:
            parts.append(gap)
        parts.append(fade(chunk, sr))
    return np.concatenate(parts)


def _atempo_chain(rate: float) -> str:
    """ffmpeg filter string for a tempo change; each atempo factor stays in 0.5-2.0."""
    factors: list[float] = []
    while rate > 2.0:
        factors.append(2.0)
        rate /= 2.0
    while rate < 0.5:
        factors.append(0.5)
        rate /= 0.5
    factors.append(rate)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def _time_stretch_librosa(audio: np.ndarray, rate: float) -> np.ndarray | None:
    """Phase-vocoder stretch via librosa, or None when librosa is unusable."""
    try:
        import librosa
    except Exception:  # not installed, or a broken numba/scipy stack
        return None
    try:
        out = librosa.effects.time_stretch(audio, rate=rate)
    except Exception:
        return None
    return np.ascontiguousarray(out, dtype=np.float32)


def _time_stretch_ffmpeg(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise AudioError(
            "Changing the speed needs librosa (installed with the Chatterbox "
            "engine) or ffmpeg. Install ffmpeg, or set speed to 1.0."
        )
    proc = subprocess.run(
        [
            ffmpeg, "-v", "error",
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
            "-filter:a", _atempo_chain(rate),
            "-f", "f32le", "-ar", str(sr), "-ac", "1", "-",
        ],
        input=np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not change the speed: {proc.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Change speed without changing pitch; rate > 1 is faster.

    Uses librosa when importable, else ffmpeg's atempo filter (chained so each
    step stays within its 0.5-2.0 range), else raises AudioError. A rate of 1.0
    returns the input unchanged.
    """
    rate = float(rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("Speed must be a positive number.")
    audio = np.asarray(audio, dtype=np.float32)
    if rate == 1.0 or audio.size == 0:
        return audio
    stretched = _time_stretch_librosa(audio, rate)
    if stretched is None:
        stretched = _time_stretch_ffmpeg(audio, sr, rate)
    return np.clip(stretched, -1.0, 1.0)
