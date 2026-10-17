import json
import re
import shutil
from datetime import datetime

import numpy as np
import pytest
import soundfile as sf

from ttsvs import config, history as history_mod
from ttsvs.history import History, HistoryItem

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def clip(seconds=0.5, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def add(hist, text="Hello there.", **kw):
    args = dict(voice_id="anna", voice_name="Anna", engine="dummy", params={"speed": 1.0})
    args.update(kw)
    return hist.add(24000, clip(), text, **args)


def test_default_root(ttsvs_home):
    assert History().root == config.history_dir()


def test_add_writes_audio_and_metadata(tmp_path):
    hist = History(tmp_path / "h")
    item = add(hist, params={"exaggeration": np.float32(0.5), "speed": 1.25})
    assert isinstance(item, HistoryItem)
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", item.id)
    assert item.audio_path == tmp_path / "h" / item.id / "audio.wav"
    info = sf.info(str(item.audio_path))
    assert (info.samplerate, info.channels) == (24000, 1)
    assert item.duration_s == pytest.approx(0.5, abs=0.001)
    datetime.fromisoformat(item.created_at)  # valid ISO 8601
    meta = json.loads((tmp_path / "h" / item.id / "item.json").read_text(encoding="utf-8"))
    assert meta["text"] == "Hello there." and meta["voice_name"] == "Anna" and meta["engine"] == "dummy"
    assert meta["params"] == {"exaggeration": 0.5, "speed": 1.25}  # numpy scalars became plain numbers


def test_roundtrip_through_a_new_instance(tmp_path):
    item = add(History(tmp_path), text="Ünïcode, “quotes”", voice_id=None, voice_name=None)
    again = History(tmp_path).get(item.id)
    assert again == item
    assert again.voice_id is None and again.voice_name is None
    assert again.text == "Ünïcode, “quotes”"


def test_ids_unique_and_sorted_within_one_second(tmp_path):
    hist = History(tmp_path)
    ids = [add(hist, text=f"clip {i}").id for i in range(25)]
    assert len(set(ids)) == 25 and ids == sorted(ids)
    assert [i.id for i in hist.list()] == ids[::-1]  # newest first


def test_ids_stay_unique_with_a_frozen_clock(tmp_path, monkeypatch):
    frozen = datetime(2026, 10, 3, 7, 25, 0, 123456, tzinfo=history_mod.timezone.utc)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(history_mod, "datetime", Frozen)
    hist = History(tmp_path)
    ids = [add(hist, text=f"clip {i}").id for i in range(5)]
    assert len(set(ids)) == 5 and ids == sorted(ids)
    assert ids[0].startswith("20261003-072500-")
    assert [i.text for i in hist.list()] == [f"clip {i}" for i in range(4, -1, -1)]


def test_list_limit_and_empty(tmp_path):
    hist = History(tmp_path)
    assert hist.list() == []
    for i in range(4):
        add(hist, text=str(i))
    assert [i.text for i in hist.list(limit=2)] == ["3", "2"]
    assert len(hist.list(limit=None)) == 4 and hist.list(limit=0) == []


def test_get_missing_raises_keyerror(tmp_path):
    hist = History(tmp_path)
    for bad in ("nope", "../etc", "", ".hidden", "a/b"):
        with pytest.raises(KeyError):
            hist.get(bad)
    with pytest.raises(KeyError):
        hist.delete("nope")


def test_delete_removes_folder(tmp_path):
    hist = History(tmp_path)
    keep, drop = add(hist, text="keep"), add(hist, text="drop")
    hist.delete(drop.id)
    assert not drop.audio_path.parent.exists()
    assert [i.id for i in hist.list()] == [keep.id]
    with pytest.raises(KeyError):
        hist.get(drop.id)


def test_clear_returns_count(tmp_path):
    hist = History(tmp_path)
    assert hist.clear() == 0
    for _ in range(3):
        add(hist)
    (tmp_path / "unrelated").mkdir()  # not ours; left alone
    assert hist.clear() == 3
    assert hist.list() == [] and (tmp_path / "unrelated").is_dir()


def test_incomplete_or_corrupt_items_are_skipped_and_cleared(tmp_path):
    hist = History(tmp_path)
    good = add(hist)
    half = tmp_path / "20200101-000000-0000"  # crash before item.json was written
    half.mkdir()
    sf.write(str(half / "audio.wav"), clip(), 24000)
    bad = tmp_path / "20200101-000000-0001"
    bad.mkdir()
    (bad / "item.json").write_text("{nope", encoding="utf-8")
    (bad / "audio.wav").write_bytes(b"")
    assert [i.id for i in hist.list()] == [good.id]
    assert hist.clear() == 3 and not half.exists() and not bad.exists()


def test_add_failure_leaves_no_folder(tmp_path):
    hist = History(tmp_path)
    with pytest.raises(Exception):
        hist.add(0, clip(), "x", None, None, "dummy", {})  # sample rate 0: soundfile refuses
    assert list(tmp_path.iterdir()) == []


def test_odd_param_values_do_not_break_saving(tmp_path):
    hist = History(tmp_path)
    item = add(hist, params={"path": tmp_path, "arr": np.array([1, 2])})
    assert item.params == {"path": str(tmp_path), "arr": [1, 2]}


def test_export_wav_is_a_copy(tmp_path):
    hist = History(tmp_path / "h")
    item = add(hist)
    out = hist.export(item.id, tmp_path / "out" / "clip.wav")
    assert out == tmp_path / "out" / "clip.wav"
    assert out.read_bytes() == item.audio_path.read_bytes()
    assert hist.export(item.id, tmp_path / "noext") == tmp_path / "noext.wav"
    with pytest.raises(KeyError):
        hist.export("missing", tmp_path / "x.wav")


@needs_ffmpeg
def test_export_mp3(tmp_path):
    hist = History(tmp_path / "h")
    item = add(hist)
    out = hist.export(item.id, tmp_path / "clip.mp3")
    assert out.suffix == ".mp3" and out.stat().st_size > 500
    assert sf.info(str(out)).duration == pytest.approx(0.5, abs=0.15)


def test_export_mp3_without_ffmpeg_gives_clear_error(tmp_path, monkeypatch):
    from ttsvs import audio

    hist = History(tmp_path / "h")
    item = add(hist)
    monkeypatch.setattr(audio, "ffmpeg_path", lambda: None)
    with pytest.raises(audio.AudioError, match="ffmpeg"):
        hist.export(item.id, tmp_path / "clip.mp3")
