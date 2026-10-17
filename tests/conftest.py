import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(autouse=True)
def ttsvs_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("TTSVS_HOME", str(home))
    return home


@pytest.fixture
def sample_wav(tmp_path):
    """8 s of 44.1 kHz stereo tone with 1 s of silence on both ends."""
    sr = 44100
    t = np.arange(int(6 * sr)) / sr
    tone = 0.4 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    pad = np.zeros(sr, dtype=np.float32)
    mono = np.concatenate([pad, tone, pad])
    path = tmp_path / "sample.wav"
    sf.write(str(path), np.stack([mono, mono], axis=1), sr)
    return path
