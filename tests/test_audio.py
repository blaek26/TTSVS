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
