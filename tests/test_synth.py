import shutil

import numpy as np
import pytest
import soundfile as sf

from ttsvs import audio as audio_mod
from ttsvs.engines import EngineError, available_engines, get_engine
from ttsvs.history import History
from ttsvs.lexicon import Lexicon
from ttsvs.synth import PAUSE_S, Synthesizer
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


# -- phase 2: speed, defaults, lexicon, streaming, history ---------------------------

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
LONG_TEXT = "One sentence here. Another sentence there. A third one follows. And a fourth."


def spy_on(synth, extra_params=None):
    """Record what the engine is asked to do; optionally advertise engine params."""
    calls = []
    real = synth.engine.synthesize

    def spy(text, reference, **params):
        calls.append((text, reference, params))
        return real(text, reference, **params)

    synth.engine.synthesize = spy
    if extra_params is not None:
        synth.engine.supported_params = lambda: extra_params
    return calls


FAKE_PARAMS = {
    "exaggeration": {"default": 0.5, "min": 0.25, "max": 2.0, "help": ""},
    "cfg_weight": {"default": 0.5, "min": 0.0, "max": 1.0, "help": ""},
}


@needs_ffmpeg
def test_speed_changes_length_and_is_not_passed_to_the_engine(monkeypatch):
    monkeypatch.setattr(audio_mod, "_time_stretch_librosa", lambda a, r: None)
    synth = Synthesizer("dummy")
    calls = spy_on(synth)
    _, normal = synth.speak("Hello there, my friend.")
    _, fast = synth.speak("Hello there, my friend.", speed=2.0)
    _, slow = synth.speak("Hello there, my friend.", speed=0.5)
    assert len(fast) == pytest.approx(len(normal) / 2, rel=0.12)
    assert len(slow) == pytest.approx(len(normal) * 2, rel=0.12)
    assert calls and all("speed" not in params for _, _, params in calls)


@needs_ffmpeg
def test_speed_is_applied_per_chunk_before_fades(monkeypatch):
    monkeypatch.setattr(audio_mod, "_time_stretch_librosa", lambda a, r: None)
    synth = Synthesizer("dummy")
    synth.engine.max_chars = 25
    chunks = list(synth.speak_stream("First part here. Second part here.", speed=2.0))
    assert len(chunks) == 2
    sr, first, i, n = chunks[0]
    assert (i, n) == (1, 2)
    raw = synth.engine.synthesize("First part here.", None)
    stretched = audio_mod.time_stretch(raw, sr, 2.0)
    expected = np.concatenate([audio_mod.fade(stretched, sr), np.zeros(int(PAUSE_S * sr), dtype=np.float32)])
    assert np.allclose(first, expected, atol=1e-6)


def test_speed_1_does_not_touch_time_stretch(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("time_stretch must not be called at speed 1.0")

    monkeypatch.setattr(audio_mod, "time_stretch", boom)
    sr, out = Synthesizer("dummy").speak("Hello there.", speed=1.0)
    assert len(out) > 0


def test_speed_is_clamped(monkeypatch):
    seen = []
    monkeypatch.setattr(audio_mod, "time_stretch", lambda a, sr, rate: (seen.append(rate), a)[1])
    synth = Synthesizer("dummy")
    synth.speak("Hi there.", speed=10)
    synth.speak("Hi there.", speed=0.01)
    assert seen == [2.0, 0.5]
    assert synth.effective_params(None, {"speed": 99})["speed"] == 2.0
    assert synth.effective_params(None, {"speed": 0.1})["speed"] == 0.5
    with pytest.raises(ValueError):
        synth.effective_params(None, {"speed": float("nan")})


def test_effective_params_precedence(sample_wav):
    lib = VoiceLibrary()
    lib.add("Anna", sample_wav)
    lib.set_defaults("anna", {"exaggeration": 0.9, "speed": 1.4})
    synth = Synthesizer("chatterbox", library=lib)  # never loaded; only its param table is used
    # voice defaults beat engine defaults
    assert synth.effective_params("anna", {}) == {
        "exaggeration": 0.9, "cfg_weight": 0.5, "temperature": 0.8, "speed": 1.4,
    }
    # explicit values beat voice defaults; None means "not given"
    got = synth.effective_params("anna", {"exaggeration": 0.3, "speed": None, "cfg_weight": None})
    assert got["exaggeration"] == 0.3 and got["speed"] == 1.4 and got["cfg_weight"] == 0.5
    # engine clamping applies to everything, unknown names are dropped
    got = synth.effective_params("anna", {"exaggeration": 99, "bogus": 1})
    assert got["exaggeration"] == 2.0 and "bogus" not in got
    # no voice: engine defaults and speed 1.0
    assert synth.effective_params(None, {}) == {
        "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "speed": 1.0,
    }
    with pytest.raises(VoiceNotFoundError):
        synth.effective_params("ghost", {})


def test_voice_defaults_reach_the_engine_but_speed_does_not(sample_wav):
    lib = VoiceLibrary()
    lib.add("Anna", sample_wav)
    lib.set_defaults("anna", {"exaggeration": 0.9, "cfg_weight": 0.2, "speed": 1.0})
    synth = Synthesizer("dummy", library=lib)
    calls = spy_on(synth, FAKE_PARAMS)
    synth.speak("Hello there.", "anna", cfg_weight=0.7)
    (_, reference, params), = calls
    assert params == {"exaggeration": 0.9, "cfg_weight": 0.7}
    assert reference == lib.get("anna").reference_path


def test_lexicon_is_applied_before_splitting(tmp_path):
    lex = Lexicon(tmp_path / "lex.json")
    lex.set("TTSVS", "T T S V S")
    synth = Synthesizer("dummy", lexicon=lex)
    calls = spy_on(synth)
    synth.speak("Welcome to ttsvs. ttsvs is local.")
    assert " ".join(text for text, _, _ in calls) == "Welcome to T T S V S. T T S V S is local."


def test_default_lexicon_comes_from_ttsvs_home():
    Lexicon().set("Nguyen", "Win")
    synth = Synthesizer("dummy")
    calls = spy_on(synth)
    synth.speak("Hello Nguyen.")
    assert calls[0][0] == "Hello Win."


def test_lexicon_can_leave_nothing_to_speak(tmp_path):
    lex = Lexicon(tmp_path / "lex.json")
    lex.set("um", "")
    with pytest.raises(ValueError):
        Synthesizer("dummy", lexicon=lex).speak("um")


def test_engine_max_chars_drives_splitting():
    synth = Synthesizer("dummy")
    synth.engine.max_chars = 30
    calls = spy_on(synth)
    synth.speak(LONG_TEXT)
    assert len(calls) > 1 and all(len(text) <= 30 for text, _, _ in calls)


def test_speak_stream_yields_chunks_that_add_up_to_speak():
    synth = Synthesizer("dummy")
    synth.engine.max_chars = 40
    chunks = list(synth.speak_stream(LONG_TEXT))
    total = len(chunks)
    assert total > 1
    assert [(i, n) for _, _, i, n in chunks] == [(i, total) for i in range(1, total + 1)]
    assert all(sr == 24000 and a.dtype == np.float32 and a.ndim == 1 for sr, a, _, _ in chunks)
    sr, whole = synth.speak(LONG_TEXT)
    assert np.array_equal(np.concatenate([a for _, a, _, _ in chunks]), whole)
    # every chunk but the last ends with the pause; the last ends on its fade
    gap = int(PAUSE_S * sr)
    assert all(np.all(a[-gap:] == 0) for _, a, _, _ in chunks[:-1])
    assert abs(chunks[-1][1][-1]) < 1e-3


def test_speak_stream_is_lazy_and_validates_up_front():
    synth = Synthesizer("dummy")
    synth.engine.max_chars = 40
    calls = spy_on(synth)
    stream = synth.speak_stream(LONG_TEXT)
    assert calls == []  # nothing generated until iterated
    next(stream)
    assert len(calls) == 1
    with pytest.raises(ValueError):
        synth.speak_stream("   ")
    with pytest.raises(VoiceNotFoundError):
        synth.speak_stream("Hi.", "ghost")


def test_speak_stream_passes_params_like_speak(sample_wav):
    lib = VoiceLibrary()
    lib.add("Anna", sample_wav)
    synth = Synthesizer("dummy", library=lib)
    calls = spy_on(synth, FAKE_PARAMS)
    list(synth.speak_stream("Hello.", "anna", exaggeration=1.5, speed=1.0))
    assert calls[0][2] == {"exaggeration": 1.5, "cfg_weight": 0.5}


def test_history_off_by_default_and_property_is_lazy(ttsvs_home):
    synth = Synthesizer("dummy")
    synth.speak("Hello.")
    assert not (ttsvs_home / "history").exists()  # nothing created yet
    assert synth.history.list() == []
    assert synth.history is synth.history and (ttsvs_home / "history").is_dir()


def test_save_history_records_speak(sample_wav, tmp_path):
    lib = VoiceLibrary()
    lib.add("Anna", sample_wav)
    lib.set_defaults("anna", {"speed": 1.0})
    hist = History(tmp_path / "h")
    lex = Lexicon(tmp_path / "lex.json")
    lex.set("Hello", "Hi")
    synth = Synthesizer("dummy", library=lib, lexicon=lex, history=hist, save_history=True)
    sr, audio = synth.speak("Hello world.", "ANNA")
    (item,) = hist.list()
    assert item.text == "Hello world."  # what the user typed, before the lexicon
    assert (item.voice_id, item.voice_name, item.engine) == ("anna", "Anna", "dummy")
    assert item.params == synth.effective_params("anna", {})
    assert item.duration_s == pytest.approx(len(audio) / sr, abs=0.001)
    loaded, file_sr = audio_mod.load_audio(item.audio_path)
    assert file_sr == sr and len(loaded) == len(audio)


def test_save_history_without_voice(tmp_path):
    hist = History(tmp_path)
    Synthesizer("dummy", history=hist, save_history=True).speak("Just text.")
    (item,) = hist.list()
    assert item.voice_id is None and item.voice_name is None


def test_streaming_alone_does_not_touch_history(tmp_path):
    hist = History(tmp_path)
    synth = Synthesizer("dummy", history=hist, save_history=True)
    list(synth.speak_stream("Streamed text."))
    assert hist.list() == []


def test_speak_to_file_saves_one_history_item(tmp_path):
    hist = History(tmp_path / "h")
    synth = Synthesizer("dummy", history=hist, save_history=True)
    synth.speak_to_file("Hello.", tmp_path / "out.wav")
    assert len(hist.list()) == 1


def test_failed_generation_is_not_saved(tmp_path):
    hist = History(tmp_path)
    synth = Synthesizer("dummy", history=hist, save_history=True)
    with pytest.raises(ValueError):
        synth.speak("   ")
    with pytest.raises(VoiceNotFoundError):
        synth.speak("Hi.", "ghost")
    assert hist.list() == []


def test_history_save_error_does_not_lose_the_audio(tmp_path, monkeypatch):
    hist = History(tmp_path)

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(hist, "add", broken)
    synth = Synthesizer("dummy", history=hist, save_history=True)
    with pytest.warns(RuntimeWarning, match="disk full"):
        sr, audio = synth.speak("Hello.")
    assert len(audio) > 0
