import numpy as np
import pytest
import soundfile as sf

from ttsvs.engines import EngineError, available_engines, get_engine
from ttsvs.synth import Synthesizer
from ttsvs.voices import VoiceLibrary, VoiceNotFoundError


def test_registry():
    assert {"dummy", "chatterbox"} <= set(available_engines())
    eng = get_engine("dummy")
    assert eng.name == "dummy" and eng.sample_rate == 24000
    with pytest.raises(EngineError):
        get_engine("nope")


def test_dummy_engine_deterministic():
    eng = get_engine("dummy")
    a = eng.synthesize("hello", None)
    assert a.dtype == np.float32 and np.abs(a).max() <= 1.0
    assert np.array_equal(a, eng.synthesize("hello", None))
    assert len(eng.synthesize("hello world, longer", None)) > len(a)


def test_resolve_params_clamps():
    from ttsvs.engines.chatterbox import ChatterboxEngine

    eng = ChatterboxEngine()
    p = eng.resolve_params({"exaggeration": 99, "temperature": None, "bogus": 1})
    assert p == {"exaggeration": 2.0, "cfg_weight": 0.5, "temperature": 0.8}


def test_speak_end_to_end(sample_wav, tmp_path):
    lib = VoiceLibrary()
    lib.add("Anna", sample_wav)
    synth = Synthesizer("dummy", library=lib)
    calls = []
    sr, audio = synth.speak("First sentence. Second sentence.\n\nThird.", "anna",
                            progress=lambda i, n: calls.append((i, n)))
    assert sr == 24000 and audio.ndim == 1 and len(audio) > 0
    assert calls[-1] == (2, 2) and calls[0] == (0, 2)

    out = synth.speak_to_file("Hello there.", tmp_path / "out" / "x.wav", "Anna")
    assert out.exists() and sf.info(str(out)).samplerate == 24000


def test_speak_without_voice_and_errors():
    synth = Synthesizer("dummy")
    assert len(synth.speak("Hi.")[1]) > 0
    with pytest.raises(ValueError):
        synth.speak("   ")
    with pytest.raises(VoiceNotFoundError):
        synth.speak("Hi.", "ghost")


def test_chatterbox_missing_gives_hint(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "chatterbox", None)
    with pytest.raises(EngineError, match=r"ttsvs\[chatterbox\]"):
        get_engine("chatterbox").load()
