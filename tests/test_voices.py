import json

import pytest
import soundfile as sf

from ttsvs import config
from ttsvs.voices import (
    VoiceExistsError, VoiceLibrary, VoiceNotFoundError, check_duration, slugify,
)


def test_config_honors_env(ttsvs_home):
    assert config.data_dir() == ttsvs_home
    assert config.voices_dir() == ttsvs_home / "voices"
    assert config.resolve_device("cpu") == "cpu"
    assert config.resolve_device("auto") in {"cpu", "cuda", "mps"}
    with pytest.raises(ValueError):
        config.resolve_device("tpu")


def test_slugify():
    assert slugify("  Ünï Voice #1! ") == "uni-voice-1"
    assert slugify("!!!") == "voice"


def test_add_list_get(sample_wav):
    lib = VoiceLibrary()
    voice = lib.add("My Voice", sample_wav, notes="hi")
    info = sf.info(str(voice.reference_path))
    assert (info.samplerate, info.channels) == (24000, 1)
    assert 5.9 < voice.duration_s < 6.5  # silence trimmed (+ small padding)
    assert voice.id == "my-voice" and voice.notes == "hi"
    assert [v.id for v in lib.list()] == ["my-voice"]
    assert lib.get("my-voice").name == "My Voice"
    assert lib.get("MY VOICE").id == "my-voice"


def test_duplicate_name_rejected(sample_wav):
    lib = VoiceLibrary()
    lib.add("Anna", sample_wav)
    with pytest.raises(VoiceExistsError):
        lib.add("anna", sample_wav)


def test_slug_collision_deduplicated(sample_wav):
    lib = VoiceLibrary()
    a = lib.add("A B", sample_wav)
    b = lib.add("A-B", sample_wav)
    assert a.id != b.id and b.id == "a-b-2"


def test_rename_and_delete(sample_wav):
    lib = VoiceLibrary()
    lib.add("One", sample_wav)
    lib.add("Two", sample_wav)
    renamed = lib.rename("one", "Uno")
    assert renamed.id == "one" and lib.get("Uno").id == "one"
    assert lib.get("One").name == "Uno"  # old id still resolves
    with pytest.raises(VoiceExistsError):
        lib.rename("uno", "two")
    lib.delete("Uno")
    assert [v.id for v in lib.list()] == ["two"]
    with pytest.raises(VoiceNotFoundError):
        lib.delete("uno")


def test_persists_across_instances(sample_wav, tmp_path):
    root = tmp_path / "lib"
    VoiceLibrary(root).add("Zed", sample_wav)
    assert VoiceLibrary(root).get("zed").name == "Zed"


def test_check_duration():
    assert check_duration(2.0) and check_duration(120.0)
    assert check_duration(15.0) is None


def test_gpu_backend():
    assert config.gpu_backend() in {"rocm", "cuda", "mps", "cpu"}


# -- phase 2: consent and per-voice defaults ----------------------------------------


def test_config_phase2_dirs(ttsvs_home):
    assert config.history_dir() == ttsvs_home / "history" and config.history_dir().is_dir()
    assert config.books_dir() == ttsvs_home / "books" and config.books_dir().is_dir()
    assert config.lexicon_path() == ttsvs_home / "lexicon.json"


def test_add_stores_consent(sample_wav):
    lib = VoiceLibrary()
    assert lib.add("No", sample_wav).consent is False
    assert lib.add("Yes", sample_wav, consent=True).consent is True
    fresh = VoiceLibrary()
    assert fresh.get("yes").consent is True and fresh.get("no").consent is False


def test_old_voice_json_still_loads(sample_wav):
    lib = VoiceLibrary()
    voice = lib.add("Old", sample_wav)
    meta = voice.reference_path.parent / "voice.json"
    legacy = json.loads(meta.read_text(encoding="utf-8"))
    legacy.pop("consent")
    legacy.pop("defaults")
    meta.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = VoiceLibrary().get("old")
    assert loaded.consent is False and loaded.defaults == {}


def test_malformed_defaults_are_ignored_on_load(sample_wav):
    lib = VoiceLibrary()
    voice = lib.add("Odd", sample_wav)
    meta = voice.reference_path.parent / "voice.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["defaults"] = {"speed": 1.2, "bad": "x", "flag": True}
    meta.write_text(json.dumps(data), encoding="utf-8")
    assert lib.get("odd").defaults == {"speed": 1.2}
    data["defaults"] = ["not", "a", "dict"]
    meta.write_text(json.dumps(data), encoding="utf-8")
    assert lib.get("odd").defaults == {}


def test_set_defaults_persists(sample_wav, tmp_path):
    root = tmp_path / "lib"
    lib = VoiceLibrary(root)
    lib.add("Anna", sample_wav)
    voice = lib.set_defaults("anna", {"exaggeration": 0.8, "speed": 1, "cfg_weight": 0.3})
    assert voice.defaults == {"exaggeration": 0.8, "speed": 1.0, "cfg_weight": 0.3}
    assert VoiceLibrary(root).get("Anna").defaults == voice.defaults
    # Other fields survive the rewrite.
    assert VoiceLibrary(root).get("Anna").duration_s > 5
    assert lib.set_defaults("anna", {}).defaults == {}
    assert VoiceLibrary(root).get("anna").defaults == {}


@pytest.mark.parametrize("bad", [
    "nope", None, [("speed", 1.0)], {"speed": "fast"}, {"speed": None}, {"speed": True},
    {"speed": float("nan")}, {1: 0.5},
])
def test_set_defaults_validates(sample_wav, bad):
    lib = VoiceLibrary()
    lib.add("Anna", sample_wav)
    with pytest.raises(ValueError):
        lib.set_defaults("anna", bad)
    assert lib.get("anna").defaults == {}


def test_set_defaults_unknown_voice():
    with pytest.raises(VoiceNotFoundError):
        VoiceLibrary().set_defaults("ghost", {"speed": 1.0})


def test_rename_keeps_consent_and_defaults(sample_wav):
    lib = VoiceLibrary()
    lib.add("One", sample_wav, consent=True)
    lib.set_defaults("one", {"speed": 1.3})
    renamed = lib.rename("one", "Uno")
    assert renamed.consent is True and renamed.defaults == {"speed": 1.3}
    assert VoiceLibrary().get("uno").defaults == {"speed": 1.3}
