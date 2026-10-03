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
