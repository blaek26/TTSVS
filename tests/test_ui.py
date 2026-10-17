"""UI tests: the Backend handlers are called directly (no browser needed)."""
import io
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

gr = pytest.importorskip("gradio")

from ttsvs import ui  # noqa: E402
from ttsvs.ui import DEFAULT_VOICE, Backend, build_ui  # noqa: E402

LONG_TEXT = " ".join(
    f"This is sentence number {i}, and it is long enough to need a chunk of its own." for i in range(12))


@pytest.fixture
def be():
    return Backend("dummy")


@pytest.fixture
def clip(tmp_path):
    sr = 24000
    t = np.arange(sr * 6) / sr
    path = tmp_path / "clip.wav"
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return path


@pytest.fixture
def book_file(tmp_path):
    path = tmp_path / "Tiny Tale.txt"
    path.write_text(
        "Chapter 1\n\nIt was a dark night. The rain fell.\n\n"
        "Chapter 2\n\nThe morning came with sun. Birds sang.\n", encoding="utf-8")
    return path


def slider_defaults(be, voice=None):
    return be.slider_values(voice)


def run(gen):
    """Drain a handler generator, returning every yielded tuple."""
    return list(gen)


def is_update(value):
    return isinstance(value, dict) and value.get("__type__") == "update"


# ------------------------------------------------------------------ layout

def test_build_ui_constructs():
    demo = build_ui(engine="dummy")
    assert isinstance(demo, gr.Blocks)
    labels = {getattr(b, "label", None) for b in demo.blocks.values()}
    assert {"Speak", "History", "Audiobook", "Pronunciation", "Voices"} <= labels
    assert "Stream while generating (experimental)" in labels


def test_build_ui_chatterbox_sliders_are_engine_params_plus_speed():
    demo = build_ui(engine="chatterbox")  # constructing never loads the model
    sliders = [b.label for b in demo.blocks.values() if isinstance(b, gr.Slider)]
    # Speak tab and Audiobook tab each get: exaggeration, cfg_weight, temperature, speed
    assert sliders.count("speed") == 2 and sliders.count("exaggeration") == 2


def test_dummy_engine_speed_not_duplicated(be):
    # The dummy engine advertises "speed" itself; the UI must not draw a second slider for it.
    assert "speed" not in be.params_spec
    assert len(be.slider_values()) == len(be.params_spec) + 1


def test_launch_is_local_only(monkeypatch, tmp_path):
    captured = {}

    def fake_launch(self, *args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(gr.Blocks, "launch", fake_launch)
    ui.launch(port=7999, engine="dummy")
    assert captured["server_name"] == "127.0.0.1"
    assert captured["share"] is False
    assert captured["server_port"] == 7999
    assert captured["inbrowser"] is False
    from ttsvs import config
    assert str(config.data_dir()) in captured["allowed_paths"]


# ------------------------------------------------------------------ helpers

def test_count_text():
    assert ui.count_text("") == "0 characters"
    assert ui.count_text(None) == "0 characters"
    out = ui.count_text("x" * 300)
    assert "300 characters" in out and "20 s" in out
    assert "min" in ui.count_text("x" * 9000)


def test_wav_bytes_roundtrip():
    audio = np.sin(np.linspace(0, 100, 2400)).astype(np.float32) * 0.5
    data = ui._wav_bytes(24000, audio)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getnframes() == 2400


def test_friendly_messages():
    from ttsvs.voices import VoiceExistsError, VoiceNotFoundError
    assert "already exists" in ui._friendly(VoiceExistsError("x"))
    assert "not found" in ui._friendly(VoiceNotFoundError("Voice 'a' not found"))
    assert ui._friendly(KeyError("No history item 'x'.")) == "No history item 'x'."
    assert "missing" in ui._friendly(ImportError("pypdf"))
    assert ui._friendly(ValueError("bad value")) == "bad value"


# ------------------------------------------------------------------ generate

def test_generate_non_streaming_saves_history(be):
    outs = run(be.generate("Hello there. This is a test.", DEFAULT_VOICE, False,
                           *slider_defaults(be)))
    final = outs[-1]
    assert is_update(final[0])  # live-playback output untouched
    sr, audio = final[1]
    assert sr == 24000 and audio.dtype == np.int16 and len(audio) > 0
    assert "Done:" in final[2] and "History" in final[2]
    rows = final[3]
    assert len(rows) == 1 and rows[0][2] == "Hello there. This is a test."
    items = be.history.list()
    assert len(items) == 1 and items[0].engine == "dummy" and items[0].voice_id is None
    assert items[0].params["speed"] == 1.0


def test_generate_streaming_yields_wav_chunks_then_full_clip(be):
    be.streaming_supported = True
    outs = run(be.generate(LONG_TEXT, DEFAULT_VOICE, True, *slider_defaults(be)))
    chunks = [o[0] for o in outs if isinstance(o[0], bytes)]
    assert len(chunks) >= 3
    frames = 0
    for blob in chunks:
        with wave.open(io.BytesIO(blob)) as w:
            assert w.getframerate() == 24000
            frames += w.getnframes()
    sr, audio = outs[-1][1]
    # the chunks add up to exactly the clip that was saved (up to int16 rounding of the join)
    assert abs(frames - len(audio)) <= 1
    assert "Done:" in outs[-1][2]
    # saved once, with the whole clip
    items = be.history.list()
    assert len(items) == 1
    assert abs(items[0].duration_s - len(audio) / sr) < 0.01
    # earlier yields never claimed the final clip
    assert all(o[1] is None or is_update(o[1]) for o in outs[:-1])


def test_generate_streaming_falls_back_without_ffmpeg(be):
    be.streaming_supported = False
    outs = run(be.generate(LONG_TEXT, DEFAULT_VOICE, True, *slider_defaults(be)))
    assert not any(isinstance(o[0], bytes) for o in outs)  # no live chunks
    assert len(outs[-1][1][1]) > 0 and len(be.history.list()) == 1


def test_generate_reports_progress(be):
    seen = []
    run(be.generate(LONG_TEXT, DEFAULT_VOICE, False, *slider_defaults(be),
                    progress=lambda frac, desc="": seen.append((frac, desc))))
    assert seen[0][0] == 0 and seen[-1][0] == 1.0
    assert any("Synthesizing" in d for _, d in seen)


def test_generate_with_voice_uses_its_defaults_and_records_it(be, clip):
    be.library.add("Alice", clip, consent=True)
    be.library.set_defaults("Alice", {"speed": 1.0})
    voice = be.find_voice("Alice")
    outs = run(be.generate("Hello Alice.", "Alice", False, *slider_defaults(be, voice)))
    assert "Done:" in outs[-1][2]
    item = be.history.list()[0]
    assert item.voice_name == "Alice" and item.voice_id == voice.id


def test_generate_errors_are_friendly(be):
    with pytest.raises(gr.Error, match="type some text"):
        run(be.generate("   ", DEFAULT_VOICE, False, *slider_defaults(be)))
    with pytest.raises(gr.Error, match="not found"):
        run(be.generate("hi", "Nobody", False, *slider_defaults(be)))
    with pytest.raises(gr.Error):
        run(be.generate("hi", DEFAULT_VOICE, False, 1.0, 2.0, 3.0))  # wrong number of settings
    assert be.history.list() == []


def test_generate_speed_without_backend_is_friendly(be, monkeypatch):
    monkeypatch.setattr("ttsvs.audio._time_stretch_librosa", lambda audio, rate: None)
    monkeypatch.setattr("ttsvs.audio.ffmpeg_path", lambda: None)
    values = slider_defaults(be)[:-1] + [1.5]
    with pytest.raises(gr.Error, match="ffmpeg"):
        run(be.generate("Some words to say.", DEFAULT_VOICE, False, *values))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="speed needs librosa or ffmpeg")
def test_generate_speed_changes_clip_length(be):
    normal = run(be.generate(LONG_TEXT, DEFAULT_VOICE, False, *slider_defaults(be)))[-1][1]
    values = slider_defaults(be)[:-1] + [2.0]
    fast = run(be.generate(LONG_TEXT, DEFAULT_VOICE, False, *values))[-1][1]
    assert len(fast[1]) < len(normal[1]) * 0.65
    assert be.history.list()[0].params["speed"] == 2.0


# ------------------------------------------------------------------ speak extras

def test_load_text_file(be, tmp_path):
    f = tmp_path / "note.txt"
    f.write_bytes("Café au lait.\r\nSecond line.".encode("utf-8"))
    assert be.load_text_file(str(f)) == "Café au lait.\nSecond line."
    latin = tmp_path / "latin.txt"
    latin.write_bytes("naïve".encode("cp1252"))
    assert be.load_text_file(str(latin)) == "naïve"
    assert is_update(be.load_text_file(None))


def test_load_text_file_too_big(be, tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "MAX_TEXT_FILE_CHARS", 100)
    f = tmp_path / "big.txt"
    f.write_text("word " * 100)
    with pytest.raises(gr.Error, match="Audiobook"):
        be.load_text_file(str(f))


def test_save_defaults_and_voice_change_loads_them(be, clip):
    be.library.add("Alice", clip, consent=True)
    values = slider_defaults(be)[:-1] + [1.4]
    msg, rows = be.save_voice_defaults("Alice", *values)
    assert "Alice" in msg and "speed=1.4" in msg
    assert be.library.get("Alice").defaults["speed"] == 1.4
    assert any(r[0] == "Alice" and "speed=1.4" in r[3] for r in rows)
    # picking the voice loads them; the default voice goes back to engine defaults
    assert be.on_voice_change("Alice")[-1] == 1.4
    assert be.on_voice_change(DEFAULT_VOICE) == be.slider_values()
    assert be.on_voice_change(DEFAULT_VOICE)[-1] == 1.0


def test_save_defaults_needs_a_cloned_voice(be):
    with pytest.raises(gr.Error, match="cloned voices"):
        be.save_voice_defaults(DEFAULT_VOICE, *slider_defaults(be))


def test_slider_values_clip_stale_defaults(be, clip):
    be.library.add("Alice", clip, consent=True)
    be.library.set_defaults("Alice", {"speed": 9.0, "bogus": 1.0})
    assert be.on_voice_change("Alice")[-1] == 2.0  # clamped into the slider range


# ------------------------------------------------------------------ history

def make_items(be, n=3):
    for i in range(n):
        run(be.generate(f"Clip number {i} is here.", DEFAULT_VOICE, False, *slider_defaults(be)))


def test_history_rows_newest_first(be):
    make_items(be)
    rows = be.history_rows()
    assert [r[2] for r in rows] == ["Clip number 2 is here.", "Clip number 1 is here.",
                                    "Clip number 0 is here."]
    assert all(len(r) == len(ui.HISTORY_HEADERS) for r in rows)
    assert rows[0][4] == be.history.list()[0].id


def test_history_rows_shorten_long_text(be):
    run(be.generate("word " * 60, DEFAULT_VOICE, False, *slider_defaults(be)))
    text = be.history_rows()[0][2]
    assert len(text) <= 60 and text.endswith("…")


def test_select_history(be):
    make_items(be, 1)
    item = be.history.list()[0]
    audio, details, selected, export, status = be.select_history(item.id)
    assert Path(audio) == item.audio_path and selected == item.id
    assert "Clip number 0" in details and "dummy" in details
    assert export["visible"] is False
    # nothing selected clears everything
    cleared = be.select_history(None)
    assert cleared[0] is None and cleared[2] is None
    with pytest.raises(gr.Error, match="No history item"):
        be.select_history("20000101-000000-0000")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="mp3 needs ffmpeg")
def test_export_history_mp3(be):
    make_items(be, 1)
    item = be.history.list()[0]
    update, msg = be.export_history(item.id)
    path = Path(update["value"])
    assert path.suffix == ".mp3" and path.stat().st_size > 0 and update["visible"] is True
    assert str(path) in msg


def test_export_history_without_ffmpeg_is_friendly(be, monkeypatch):
    make_items(be, 1)
    monkeypatch.setattr(ui, "_ffmpeg_available", lambda: False)
    with pytest.raises(gr.Error, match="ffmpeg"):
        be.export_history(be.history.list()[0].id)
    with pytest.raises(gr.Error, match="Select a history item"):
        be.export_history(None)


def test_use_history_settings(be, clip):
    be.library.add("Alice", clip, consent=True)
    values = slider_defaults(be)[:-1] + [1.25]
    run(be.generate("Reuse this exact text.", "Alice", False, *values))
    item = be.history.list()[0]
    out = be.use_history_settings(item.id)
    text, voice_update, *rest = out
    *sliders, speed, tabs, msg = rest
    assert text == "Reuse this exact text."
    assert voice_update["value"] == "Alice"
    assert speed == 1.25 and len(sliders) == len(be.params_spec)
    assert isinstance(tabs, gr.Tabs) and tabs.selected == "speak"
    assert item.id in msg


def test_use_history_settings_voice_gone(be, clip):
    be.library.add("Alice", clip, consent=True)
    run(be.generate("Hi there.", "Alice", False, *slider_defaults(be, be.find_voice("Alice"))))
    be.library.delete("Alice")
    out = be.use_history_settings(be.history.list()[0].id)
    assert out[1]["value"] == DEFAULT_VOICE
    assert "no longer exists" in out[-1]


def test_delete_history_item(be):
    make_items(be, 2)
    first, second = [i.id for i in be.history.list()]
    table, audio, details, selected, export, msg = be.delete_history(first)
    assert [r[4] for r in table] == [second] and audio is None and selected is None
    with pytest.raises(gr.Error, match="Select"):
        be.delete_history(None)
    with pytest.raises(gr.Error, match="No history item"):
        be.delete_history(first)  # already gone: friendly, not a traceback


def test_clear_history_needs_confirmation(be):
    make_items(be, 2)
    with pytest.raises(gr.Error, match="confirmation"):
        be.clear_history(False)
    assert len(be.history.list()) == 2
    table, audio, details, selected, export, msg, confirm = be.clear_history(True)
    assert table == [] and "2 items" in msg and confirm is False
    assert be.history.list() == []


# ------------------------------------------------------------------ pronunciation

def test_lexicon_add_update_remove(be):
    assert be.lexicon_rows() == []
    rows, msg = be.lexicon_set("TTSVS", "T T S V S")
    assert rows == [["TTSVS", "T T S V S"]] and "TTSVS" in msg
    rows, _ = be.lexicon_set("ttsvs", "tee tee ess vee ess")  # update, case-insensitive
    assert rows == [["ttsvs", "tee tee ess vee ess"]]
    be.lexicon_set("Nguyen", "Win")
    assert [r[0] for r in be.lexicon_rows()] == ["Nguyen", "ttsvs"]
    rows, msg, word, say = be.lexicon_remove("nguyen")
    assert [r[0] for r in rows] == ["ttsvs"] and word == "" and say == ""


def test_lexicon_errors(be):
    with pytest.raises(gr.Error, match="word"):
        be.lexicon_set("  ", "x")
    with pytest.raises(gr.Error, match="how it should be said"):
        be.lexicon_set("word", " ")
    with pytest.raises(gr.Error, match="not in the dictionary"):
        be.lexicon_remove("ghost")
    with pytest.raises(gr.Error, match="remove"):
        be.lexicon_remove("")


def test_lexicon_is_applied_when_generating(be):
    run(be.generate("foo", DEFAULT_VOICE, False, *slider_defaults(be)))
    short = be.history.list()[0].duration_s
    be.lexicon_set("foo", "a much longer replacement phrase for the word")
    run(be.generate("foo", DEFAULT_VOICE, False, *slider_defaults(be)))
    assert be.history.list()[0].duration_s > short * 3
    assert be.history.list()[0].text == "foo"  # history keeps what was typed


def test_preview_say_as(be):
    sr, audio = be.preview_say_as("T T S V S", DEFAULT_VOICE)
    assert sr == 24000 and audio.dtype == np.int16 and len(audio) > 0
    assert be.history.list() == []  # previews are not history
    with pytest.raises(gr.Error, match="say as"):
        be.preview_say_as("  ", DEFAULT_VOICE)


# ------------------------------------------------------------------ voices

def test_add_voice_records_consent_and_shows_it(be, clip):
    out = be.add_voice("Alice", str(clip), "my notes", True)
    msg, name, audio, notes, consent, main, manage, book, table = out
    assert "Added **Alice**" in msg and name == "" and audio is None and consent is False
    assert main["value"] == "Alice" and manage["value"] == "Alice" and book["value"] == "Alice"
    assert "Alice" in main["choices"] or ("Alice", "Alice") in main["choices"]
    voice = be.library.get("Alice")
    assert voice.consent is True
    row = table[0]
    assert row[0] == "Alice" and row[2] == "yes" and row[4] == "my notes"


def test_voice_table_shows_missing_consent(be, clip):
    be.library.add("Old", clip)  # added before consent was recorded
    assert be.voice_rows()[0][2] == "not recorded"


def test_add_voice_validation(be, clip):
    with pytest.raises(gr.Error, match="permission"):
        be.add_voice("Alice", str(clip), "", False)
    with pytest.raises(gr.Error, match="name"):
        be.add_voice("  ", str(clip), "", True)
    with pytest.raises(gr.Error, match="upload or record"):
        be.add_voice("Alice", None, "", True)
    be.add_voice("Alice", str(clip), "", True)
    with pytest.raises(gr.Error, match="already exists"):
        be.add_voice("alice", str(clip), "", True)


def test_add_voice_bad_audio_is_friendly(be, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(gr.Error):
        be.add_voice("Broken", str(bad), "", True)
    assert be.library.list() == []


def test_rename_and_delete_voice(be, clip):
    be.add_voice("Alice", str(clip), "", True)
    main, manage, book, table, cleared = be.rename_voice("Alice", "Alicia", "Alice", "Alice")
    assert main["value"] == "Alicia" and book["value"] == "Alicia"  # pickers follow the rename
    assert table[0][0] == "Alicia" and cleared == ""
    main, manage, book, table, preview = be.delete_voice("Alicia", "Alicia", "Alicia")
    assert table == [] and main["value"] == DEFAULT_VOICE and manage["value"] is None
    with pytest.raises(gr.Error, match="Select a voice"):
        be.delete_voice(None)
    with pytest.raises(gr.Error, match="not found"):
        be.rename_voice("Ghost", "x")


def test_rename_and_delete_leave_other_selections_alone(be, clip):
    be.library.add("Alice", clip, consent=True)
    be.library.add("Bob", clip, consent=True)
    main, manage, book, table, _ = be.rename_voice("Alice", "Alicia", "Bob", DEFAULT_VOICE)
    assert (main["value"], manage["value"], book["value"]) == ("Bob", "Alicia", DEFAULT_VOICE)
    main, manage, book, table, _ = be.delete_voice("Alicia", "Bob", "Bob")
    assert (main["value"], book["value"]) == ("Bob", "Bob") and manage["value"] == "Bob"
    assert [r[0] for r in table] == ["Bob"]


def test_voice_updates_keep_current_selection(be, clip):
    be.library.add("Alice", clip, consent=True)
    be.library.add("Bob", clip, consent=True)
    main, manage, book = be.voice_updates("Bob", "Alice", "Bob")
    assert (main["value"], manage["value"], book["value"]) == ("Bob", "Alice", "Bob")
    main, manage, book = be.voice_updates("Gone", "Gone", "Gone")  # deleted meanwhile
    assert (main["value"], manage["value"], book["value"]) == (DEFAULT_VOICE, "Alice", DEFAULT_VOICE)


def test_guided_script_is_about_25_seconds():
    words = len(ui.GUIDED_SCRIPT.split())
    assert 55 <= words <= 80  # roughly 150 words per minute


# ------------------------------------------------------------------ audiobook

def test_load_book_file(be, book_file):
    title, rows, chapters, out_dir, status = be.load_book_file(str(book_file))
    assert title == "Tiny Tale"
    assert [r[0] for r in rows] == [1, 2] and [r[1] for r in rows] == ["Chapter 1", "Chapter 2"]
    assert all(r[2] > 0 for r in rows) and len(chapters) == 2
    assert Path(out_dir).name == "Tiny Tale" and Path(out_dir).parent.name == "books"
    assert "2" in status
    assert be.load_book_file(None) == ("", [], None, "", "")


def test_load_book_file_errors_are_friendly(be, tmp_path):
    bad = tmp_path / "x.docx"
    bad.write_text("x")
    with pytest.raises(gr.Error, match="Unsupported"):
        be.load_book_file(str(bad))
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n")
    with pytest.raises(gr.Error, match="No readable text"):
        be.load_book_file(str(empty))


def test_render_book_ui_and_resume(be, book_file, tmp_path):
    _, _, chapters, _, _ = be.load_book_file(str(book_file))
    out = tmp_path / "out"
    defaults = slider_defaults(be)
    seen = []
    progress = lambda frac, desc="": seen.append((frac, desc))  # noqa: E731
    allowed = []
    be.allow_path = allowed.append

    status, files = be.render_book_ui(chapters, "Tiny Tale", "", DEFAULT_VOICE, "wav", str(out),
                                      True, *defaults, progress=progress)
    names = sorted(Path(f).name for f in files)
    assert names == ["01 - Chapter 1.wav", "02 - Chapter 2.wav"]
    assert "2" in status and "same folder" in status
    assert seen[0][1].startswith("Chapter 1/2") and seen[-1][0] == 1.0
    assert sorted(Path(p).name for p in allowed) == names  # download links are allowed
    assert be.history.list() == []  # chapters never land in the history
    stamps = {Path(f).name: Path(f).stat().st_mtime_ns for f in files}

    be.render_book_ui(chapters, "Tiny Tale", "", DEFAULT_VOICE, "wav", str(out), True, *defaults)
    assert {Path(f).name: Path(f).stat().st_mtime_ns for f in files} == stamps  # resumed, not redone


def test_render_book_ui_chapter_range_and_default_folder(be, book_file):
    from ttsvs import config
    _, _, chapters, _, _ = be.load_book_file(str(book_file))
    status, files = be.render_book_ui(chapters, "Tiny Tale", "2", DEFAULT_VOICE, "wav", "", True,
                                      *slider_defaults(be))
    assert [Path(f).name for f in files] == ["02 - Chapter 2.wav"]
    assert Path(files[0]).parent == config.books_dir() / "Tiny Tale"


def test_render_book_ui_uses_override_sliders_when_asked(be, book_file, tmp_path, monkeypatch):
    _, _, chapters, _, _ = be.load_book_file(str(book_file))
    seen = []
    import ttsvs.book as book

    real = book.render_book

    def spy(synth, chapters, out_dir, voice_id=None, **kwargs):
        seen.append(kwargs)
        return real(synth, chapters, out_dir, voice_id, **kwargs)

    monkeypatch.setattr(book, "render_book", spy)
    values = slider_defaults(be)[:-1] + [1.0]
    be.render_book_ui(chapters, "T", "1", DEFAULT_VOICE, "wav", str(tmp_path / "a"), True, *values)
    be.render_book_ui(chapters, "T", "1", DEFAULT_VOICE, "wav", str(tmp_path / "b"), False, *values)
    assert "speed" not in seen[0] and seen[1]["speed"] == 1.0


def test_render_book_ui_errors(be, book_file, tmp_path, monkeypatch):
    _, _, chapters, _, _ = be.load_book_file(str(book_file))
    defaults = slider_defaults(be)
    with pytest.raises(gr.Error, match="Upload a book"):
        be.render_book_ui(None, "", "", DEFAULT_VOICE, "wav", "", True, *defaults)
    with pytest.raises(gr.Error, match="does not exist"):
        be.render_book_ui(chapters, "T", "7", DEFAULT_VOICE, "wav", str(tmp_path / "x"), True,
                          *defaults)
    with pytest.raises(gr.Error, match="understand"):
        be.render_book_ui(chapters, "T", "abc", DEFAULT_VOICE, "wav", str(tmp_path / "x"), True,
                          *defaults)
    with pytest.raises(gr.Error, match="not found"):
        be.render_book_ui(chapters, "T", "", "Nobody", "wav", str(tmp_path / "x"), True, *defaults)
    monkeypatch.setattr("ttsvs.audio.ffmpeg_path", lambda: None)
    with pytest.raises(gr.Error, match="ffmpeg"):
        be.render_book_ui(chapters, "T", "", DEFAULT_VOICE, "mp3", str(tmp_path / "x"), True,
                          *defaults)
