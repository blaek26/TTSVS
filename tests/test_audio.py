import shutil
import sys
import types

import numpy as np
import pytest

from ttsvs import audio


def test_roundtrip_wav(tmp_path):
    data = (0.5 * np.sin(np.linspace(0, 100, 24000))).astype(np.float32)
    out = audio.save_audio(tmp_path / "a.wav", data, 24000)
    loaded, sr = audio.load_audio(out)
    assert sr == 24000 and loaded.dtype == np.float32 and loaded.ndim == 1
    assert np.allclose(loaded, data, atol=1e-3)


def test_load_stereo_to_mono(sample_wav):
    data, sr = audio.load_audio(sample_wav)
    assert sr == 44100 and data.ndim == 1


def test_load_missing_file(tmp_path):
    with pytest.raises(audio.AudioError):
        audio.load_audio(tmp_path / "nope.wav")


def test_resample_length():
    data = np.zeros(44100, dtype=np.float32)
    assert abs(len(audio.resample(data, 44100, 24000)) - 24000) <= 2
    assert audio.resample(data, 44100, 44100) is not None


def test_trim_silence():
    sr = 1000
    data = np.concatenate([np.zeros(1000), np.full(500, 0.5), np.zeros(1000)]).astype(np.float32)
    trimmed = audio.trim_silence(data, sr, pad_s=0.0)
    assert len(trimmed) == 500
    assert len(audio.trim_silence(np.zeros(10, dtype=np.float32), sr)) == 10


def test_normalize_peak():
    out = audio.normalize(np.array([0.1, -0.2], dtype=np.float32), target_peak=0.9)
    assert pytest.approx(float(np.abs(out).max()), rel=1e-5) == 0.9
    zeros = np.zeros(4, dtype=np.float32)
    assert np.array_equal(audio.normalize(zeros), zeros)


def test_concat_with_pauses():
    a = np.ones(100, dtype=np.float32)
    out = audio.concat_with_pauses([a, a, a], 1000, 0.05)
    assert len(out) == 300 + 2 * 50
    assert len(audio.concat_with_pauses([], 1000)) == 0


def test_unsupported_format(tmp_path):
    with pytest.raises(audio.AudioError):
        audio.save_audio(tmp_path / "a.xyz", np.zeros(10, dtype=np.float32), 1000)


def test_fade_ramps_edges_and_keeps_length():
    sr = 1000
    x = np.ones(200, dtype=np.float32)
    y = audio.fade(x, sr, fade_s=0.01)
    assert len(y) == len(x)
    assert y[0] == 0.0 and y[-1] == 0.0
    assert y[100] == 1.0
    assert x[0] == 1.0  # input untouched


# -- time_stretch ----------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _tone(seconds=2.0, sr=24000, freq=440.0):
    t = np.arange(int(seconds * sr)) / sr
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_freq(x, sr):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spectrum)) * sr / len(x)


def test_time_stretch_rate_one_returns_input():
    x = _tone(0.1)
    assert audio.time_stretch(x, 24000, 1.0) is x
    assert len(audio.time_stretch(np.zeros(0, dtype=np.float32), 24000, 2.0)) == 0


def test_time_stretch_rejects_bad_rate():
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            audio.time_stretch(_tone(0.1), 24000, bad)


@pytest.mark.parametrize("rate", [0.3, 0.5, 0.75, 1.5, 2.0, 3.5, 5.0])
def test_atempo_chain_stays_in_range_and_multiplies_to_rate(rate):
    factors = [float(part.split("=")[1]) for part in audio._atempo_chain(rate).split(",")]
    assert all(0.5 <= f <= 2.0 for f in factors)
    assert np.prod(factors) == pytest.approx(rate, rel=1e-4)


@needs_ffmpeg
@pytest.mark.parametrize("rate", [2.0, 0.5, 1.25, 3.0, 0.4])
def test_time_stretch_ffmpeg_scales_length_and_keeps_pitch(rate, monkeypatch):
    monkeypatch.setattr(audio, "_time_stretch_librosa", lambda a, r: None)  # force the ffmpeg path
    x = _tone(2.0)
    y = audio.time_stretch(x, 24000, rate)
    assert y.dtype == np.float32 and y.ndim == 1
    assert len(y) == pytest.approx(len(x) / rate, rel=0.1)
    assert _dominant_freq(y, 24000) == pytest.approx(440.0, abs=15.0)
    assert np.abs(y).max() <= 1.0


def test_time_stretch_uses_librosa_when_importable(monkeypatch):
    calls = []

    def fake_stretch(y, rate):
        calls.append(rate)
        return y[: int(len(y) / rate)] * 2  # pretend; also checks the final clip

    fake = types.SimpleNamespace(effects=types.SimpleNamespace(time_stretch=fake_stretch))
    monkeypatch.setitem(sys.modules, "librosa", fake)
    monkeypatch.setattr(audio, "ffmpeg_path", lambda: None)  # would fail if ffmpeg were needed
    x = np.full(1000, 0.8, dtype=np.float32)
    y = audio.time_stretch(x, 24000, 2.0)
    assert calls == [2.0] and len(y) == 500
    assert y.max() == pytest.approx(1.0)  # 1.6 clipped


def test_time_stretch_falls_back_when_librosa_is_broken(monkeypatch):
    def boom(y, rate):
        raise RuntimeError("numba exploded")

    fake = types.SimpleNamespace(effects=types.SimpleNamespace(time_stretch=boom))
    monkeypatch.setitem(sys.modules, "librosa", fake)
    monkeypatch.setattr(audio, "ffmpeg_path", lambda: None)
    with pytest.raises(audio.AudioError, match="ffmpeg"):
        audio.time_stretch(_tone(0.1), 24000, 1.5)


def test_time_stretch_without_librosa_or_ffmpeg_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "librosa", None)  # makes `import librosa` fail
    monkeypatch.setattr(audio, "ffmpeg_path", lambda: None)
    with pytest.raises(audio.AudioError, match="librosa"):
        audio.time_stretch(_tone(0.1), 24000, 1.5)
