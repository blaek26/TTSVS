import importlib.util
import shutil

import numpy as np
import pytest
import soundfile as sf

from ttsvs.cli import main, parse_chapter_spec

# Speed changes need librosa or ffmpeg (see ttsvs.audio.time_stretch).
needs_stretch = pytest.mark.skipif(
    importlib.util.find_spec("librosa") is None and shutil.which("ffmpeg") is None,
    reason="speed control needs librosa or ffmpeg")

BOOK_TEXT = (
    "Chapter 1\n\nIt was a dark and stormy night. The rain fell in torrents.\n\n"
    "Chapter 2\n\nThe morning came with bright sun. Birds sang in the trees.\n")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TTSVS_HOME", str(tmp_path / "home"))
    return tmp_path


@pytest.fixture
def clip(tmp_path):
    sr = 24000
    t = np.arange(sr * 6) / sr
    data = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    path = tmp_path / "clip.wav"
    sf.write(path, data, sr)
    return path


def test_voice_lifecycle(clip, capsys):
    assert main(["voices", "add", "Alice", str(clip), "--yes", "--notes", "hi"]) == 0
    capsys.readouterr()
    assert main(["voices", "list"]) == 0
    assert "Alice" in capsys.readouterr().out
    assert main(["voices", "rename", "Alice", "Bob"]) == 0
    capsys.readouterr()
    main(["voices", "list"])
    out = capsys.readouterr().out
    assert "Bob" in out and "Alice" not in out
    assert main(["voices", "remove", "Bob"]) == 0
    capsys.readouterr()
    main(["voices", "list"])
    assert "Bob" not in capsys.readouterr().out


def test_add_requires_consent(clip, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert main(["voices", "add", "Eve", str(clip)]) != 0
    main(["voices", "list"])
    assert "Eve" not in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert main(["voices", "add", "Eve", str(clip)]) == 0


def test_add_missing_file(tmp_path):
    assert main(["voices", "add", "X", str(tmp_path / "nope.wav"), "--yes"]) != 0


def test_say_to_file(clip, tmp_path):
    main(["voices", "add", "Alice", str(clip), "--yes"])
    out = tmp_path / "out.wav"
    rc = main(["say", "Hello there. This is a test.", "--voice", "Alice",
               "-o", str(out), "--engine", "dummy"])
    assert rc == 0
    data, sr = sf.read(out)
    assert len(data) > 0 and sr > 0


def test_say_default_voice_and_file(tmp_path):
    txt = tmp_path / "in.txt"
    txt.write_text("Some text from a file.")
    out = tmp_path / "o.wav"
    assert main(["say", "--file", str(txt), "-o", str(out), "--engine", "dummy"]) == 0
    assert out.exists()


def test_say_bad_voice(tmp_path, capsys):
    rc = main(["say", "hi", "--voice", "ghost", "-o", str(tmp_path / "x.wav"),
               "--engine", "dummy"])
    assert rc != 0
    assert "not found" in capsys.readouterr().err


def test_say_no_text(tmp_path):
    assert main(["say", "-o", str(tmp_path / "x.wav"), "--engine", "dummy"]) != 0


def test_doctor(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Python" in out and "data dir" in out


def test_no_command():
    assert main([]) != 0


# ------------------------------------------------------------------ helpers

def say(tmp_path, text="Hello there. This is a test.", name="out.wav", *extra):
    out = tmp_path / name
    assert main(["say", text, "-o", str(out), "--engine", "dummy", *extra]) == 0
    return out


def duration(path):
    return sf.info(str(path)).duration


def first_id(capsys):
    """The id column of the first row of `ttsvs history list`."""
    capsys.readouterr()
    assert main(["history", "list"]) == 0
    return capsys.readouterr().out.splitlines()[1].split()[0]


@pytest.fixture
def book_file(tmp_path):
    path = tmp_path / "My Book.txt"
    path.write_text(BOOK_TEXT, encoding="utf-8")
    return path


# ------------------------------------------------------------------ history

def test_history_empty(capsys):
    assert main(["history", "list"]) == 0
    assert "empty" in capsys.readouterr().out.lower()


def test_say_saves_history_and_list(tmp_path, capsys):
    say(tmp_path, "Remember this exact sentence please.")
    out = capsys.readouterr().out
    assert "Added to history as" in out
    assert main(["history", "list"]) == 0
    listing = capsys.readouterr().out
    assert "Remember this exact sentence" in listing and "(default)" in listing


def test_say_no_history(tmp_path, capsys):
    say(tmp_path, "Not kept.", "o.wav", "--no-history")
    assert "Added to history" not in capsys.readouterr().out
    assert main(["history", "list"]) == 0
    assert "empty" in capsys.readouterr().out.lower()


def test_history_records_voice_and_settings(clip, tmp_path):
    from ttsvs.history import History
    main(["voices", "add", "Alice", str(clip), "--yes"])
    say(tmp_path, "Hello Alice.", "a.wav", "--voice", "Alice")
    item = History().list()[0]
    assert item.voice_name == "Alice" and item.engine == "dummy" and item.text == "Hello Alice."
    assert item.params["speed"] == 1.0


def test_history_list_limit_and_order(tmp_path, capsys):
    for i in range(3):
        say(tmp_path, f"Message number {i}.", f"o{i}.wav")
    capsys.readouterr()
    assert main(["history", "list", "--limit", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert "Message number 2" in lines[1] and "Message number 1" in lines[2]


def test_history_export_wav(tmp_path, capsys):
    say(tmp_path, "Export me please.")
    item_id = first_id(capsys)
    dest = tmp_path / "exports" / "copy.wav"
    assert main(["history", "export", item_id, str(dest)]) == 0
    data, sr = sf.read(dest)
    assert len(data) > 0 and sr == 24000


def test_history_export_accepts_unique_id_prefix(tmp_path, capsys):
    say(tmp_path, "Prefix lookup.")
    item_id = first_id(capsys)
    dest = tmp_path / "p.wav"
    assert main(["history", "export", item_id[:12], str(dest)]) == 0
    assert dest.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="mp3 needs ffmpeg")
def test_history_export_mp3(tmp_path, capsys):
    say(tmp_path, "Export as mp3.")
    dest = tmp_path / "x.mp3"
    assert main(["history", "export", first_id(capsys), str(dest)]) == 0
    assert dest.stat().st_size > 0


def test_history_export_mp3_without_ffmpeg_is_friendly(tmp_path, capsys, monkeypatch):
    say(tmp_path, "No ffmpeg here.")
    item_id = first_id(capsys)
    monkeypatch.setattr("ttsvs.audio.ffmpeg_path", lambda: None)
    assert main(["history", "export", item_id, str(tmp_path / "x.mp3")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "ffmpeg" in err and "Traceback" not in err


def test_history_missing_item(capsys):
    assert main(["history", "export", "nope", "x.wav"]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["history", "remove", "nope"]) == 1
    assert "not found" in capsys.readouterr().err


def test_history_remove(tmp_path, capsys):
    say(tmp_path, "First one.", "a.wav")
    say(tmp_path, "Second one.", "b.wav")
    capsys.readouterr()
    main(["history", "list"])
    newest = capsys.readouterr().out.splitlines()[1].split()[0]
    assert main(["history", "remove", newest]) == 0
    main(["history", "list"])
    out = capsys.readouterr().out
    assert "First one" in out and "Second one" not in out


def test_history_clear(tmp_path, capsys, monkeypatch):
    say(tmp_path, "One.", "a.wav")
    say(tmp_path, "Two.", "b.wav")
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert main(["history", "clear"]) == 1  # declined
    main(["history", "list"])
    assert "One" in capsys.readouterr().out
    assert main(["history", "clear", "--yes"]) == 0
    assert "Removed 2" in capsys.readouterr().out
    main(["history", "list"])
    assert "empty" in capsys.readouterr().out.lower()
    assert main(["history", "clear", "--yes"]) == 0  # already empty is fine


def test_history_clear_prompt_yes(tmp_path, capsys, monkeypatch):
    say(tmp_path, "One.")
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert main(["history", "clear"]) == 0
    main(["history", "list"])
    assert "empty" in capsys.readouterr().out.lower()


# ------------------------------------------------------------------ speed

@needs_stretch
def test_say_speed_changes_length(tmp_path):
    text = "A sentence that is long enough to measure a difference in speed."
    normal = say(tmp_path, text, "n.wav")
    fast = say(tmp_path, text, "f.wav", "--speed", "2.0")
    slow = say(tmp_path, text, "s.wav", "--speed", "0.5")
    assert duration(fast) < duration(normal) * 0.65
    assert duration(slow) > duration(normal) * 1.6


def test_say_speed_out_of_range(tmp_path, capsys):
    rc = main(["say", "hi", "-o", str(tmp_path / "x.wav"), "--engine", "dummy", "--speed", "5"])
    assert rc == 1
    assert "--speed" in capsys.readouterr().err


def test_say_speed_one_needs_no_stretching(tmp_path):
    out = say(tmp_path, "Plain speed.", "o.wav", "--speed", "1.0")
    assert out.exists()


def test_say_speed_without_stretch_backend_is_friendly(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("ttsvs.audio._time_stretch_librosa", lambda audio, rate: None)
    monkeypatch.setattr("ttsvs.audio.ffmpeg_path", lambda: None)
    rc = main(["say", "Some words here.", "-o", str(tmp_path / "x.wav"), "--engine", "dummy",
               "--speed", "1.5"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.splitlines()[-1].startswith("error:") and "ffmpeg" in err and "Traceback" not in err


# ------------------------------------------------------------------ lexicon

def test_lexicon_set_list_remove(capsys):
    assert main(["lexicon", "list"]) == 0
    assert "empty" in capsys.readouterr().out.lower()
    assert main(["lexicon", "set", "TTSVS", "T T S V S"]) == 0
    assert main(["lexicon", "set", "Nguyen", "Win"]) == 0
    capsys.readouterr()
    assert main(["lexicon", "list"]) == 0
    out = capsys.readouterr().out
    assert "TTSVS" in out and "T T S V S" in out and "Nguyen" in out and "Win" in out
    assert out.index("Nguyen") < out.index("TTSVS")  # sorted by word
    assert main(["lexicon", "set", "ttsvs", "tee tee ess vee ess"]) == 0  # case-insensitive update
    capsys.readouterr()
    main(["lexicon", "list"])
    out = capsys.readouterr().out
    assert "tee tee ess" in out and "T T S V S" not in out
    assert main(["lexicon", "remove", "nguyen"]) == 0
    capsys.readouterr()
    main(["lexicon", "list"])
    assert "Nguyen" not in capsys.readouterr().out


def test_lexicon_remove_missing(capsys):
    assert main(["lexicon", "remove", "ghost"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "ghost" in err and "KeyError" not in err


def test_lexicon_applies_when_saying(tmp_path):
    short = say(tmp_path, "foo", "a.wav")
    assert main(["lexicon", "set", "foo", "a much longer replacement phrase for the word"]) == 0
    long = say(tmp_path, "foo", "b.wav")
    assert duration(long) > duration(short) * 3


# ------------------------------------------------------------------ voice defaults / consent

def test_voices_add_records_consent(clip, capsys):
    from ttsvs.voices import VoiceLibrary
    assert main(["voices", "add", "Alice", str(clip), "--yes"]) == 0
    assert VoiceLibrary().get("Alice").consent is True
    capsys.readouterr()
    main(["voices", "list"])
    out = capsys.readouterr().out
    assert "CONSENT" in out and "yes" in out


def test_voices_defaults_set_show_reset(clip, capsys):
    from ttsvs.voices import VoiceLibrary
    main(["voices", "add", "Alice", str(clip), "--yes"])
    capsys.readouterr()
    assert main(["voices", "defaults", "Alice"]) == 0
    assert "no saved defaults" in capsys.readouterr().out
    assert main(["voices", "defaults", "Alice", "--speed", "1.25", "--exaggeration", "0.7"]) == 0
    assert VoiceLibrary().get("Alice").defaults == {"speed": 1.25, "exaggeration": 0.7}
    capsys.readouterr()
    # a second call merges instead of wiping what was there
    assert main(["voices", "defaults", "Alice", "--temperature", "0.6"]) == 0
    assert VoiceLibrary().get("Alice").defaults == {
        "speed": 1.25, "exaggeration": 0.7, "temperature": 0.6}
    capsys.readouterr()
    assert main(["voices", "defaults", "alice"]) == 0  # show, by case-insensitive name
    out = capsys.readouterr().out
    assert "speed=1.25" in out and "exaggeration=0.7" in out and "temperature=0.6" in out
    assert main(["voices", "defaults", "Alice", "--reset"]) == 0
    assert VoiceLibrary().get("Alice").defaults == {}


def test_voices_defaults_validation(clip, capsys):
    main(["voices", "add", "Alice", str(clip), "--yes"])
    assert main(["voices", "defaults", "Alice", "--speed", "9"]) == 1
    assert main(["voices", "defaults", "Alice", "--exaggeration", "50"]) == 1
    assert main(["voices", "defaults", "Alice", "--reset", "--speed", "1.1"]) == 1
    assert main(["voices", "defaults", "ghost", "--speed", "1.1"]) == 1
    assert "not found" in capsys.readouterr().err


@needs_stretch
def test_say_uses_voice_defaults_and_flag_overrides(clip, tmp_path):
    main(["voices", "add", "Alice", str(clip), "--yes"])
    text = "A sentence that is long enough to measure a difference in speed."
    base = say(tmp_path, text, "base.wav", "--voice", "Alice")
    assert main(["voices", "defaults", "Alice", "--speed", "2.0"]) == 0
    fast = say(tmp_path, text, "fast.wav", "--voice", "Alice")
    assert duration(fast) < duration(base) * 0.65
    explicit = say(tmp_path, text, "explicit.wav", "--voice", "Alice", "--speed", "1.0")
    assert abs(duration(explicit) - duration(base)) < 0.05


def test_history_stores_voice_defaults_in_params(clip, tmp_path):
    from ttsvs.history import History
    main(["voices", "add", "Alice", str(clip), "--yes"])
    main(["voices", "defaults", "Alice", "--speed", "1.0"])
    say(tmp_path, "Hello.", "a.wav", "--voice", "Alice")
    assert History().list()[0].params["speed"] == 1.0


# ------------------------------------------------------------------ book

def test_parse_chapter_spec():
    assert parse_chapter_spec(None, 4) == [1, 2, 3, 4]
    assert parse_chapter_spec("", 4) == [1, 2, 3, 4]
    assert parse_chapter_spec("all", 3) == [1, 2, 3]
    assert parse_chapter_spec("1-3,5", 6) == [1, 2, 3, 5]
    assert parse_chapter_spec(" 2 , 2, 1 ", 3) == [1, 2]
    assert parse_chapter_spec("3-", 5) == [3, 4, 5]
    assert parse_chapter_spec("-2", 5) == [1, 2]
    for bad in ("0", "3-1", "x", "1-9", "7", "1,,-", "--"):
        with pytest.raises(ValueError):
            parse_chapter_spec(bad, 6)


def test_book_list(book_file, capsys):
    assert main(["book", str(book_file), "--list"]) == 0
    out = capsys.readouterr().out
    assert "My Book" in out and "Chapter 1" in out and "Chapter 2" in out
    assert "2 chapters" in out


def test_book_list_with_selection(book_file, capsys):
    assert main(["book", str(book_file), "--list", "--chapters", "2"]) == 0
    out = capsys.readouterr().out
    assert "Chapter 2" in out and "Chapter 1" not in out


def test_book_renders_one_file_per_chapter_and_resumes(book_file, tmp_path, capsys):
    out_dir = tmp_path / "audiobook"
    args = ["book", str(book_file), "-o", str(out_dir), "--engine", "dummy"]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "[1/2]" in text and "[2/2]" in text and "Done: 2 files" in text
    files = sorted(out_dir.glob("*.wav"))
    assert [f.name for f in files] == ["01 - Chapter 1.wav", "02 - Chapter 2.wav"]
    assert all(sf.info(str(f)).frames > 0 for f in files)
    assert not list(out_dir.glob("*.part*"))
    stamps = {f.name: f.stat().st_mtime_ns for f in files}

    # run again: nothing is re-rendered
    assert main(args) == 0
    again = capsys.readouterr().out
    assert again.count("already rendered, skipping") == 2
    assert {f.name: f.stat().st_mtime_ns for f in out_dir.glob("*.wav")} == stamps

    # a missing chapter file is picked up where it left off
    files[1].unlink()
    assert main(args) == 0
    assert capsys.readouterr().out.count("already rendered, skipping") == 1
    assert files[1].exists() and files[0].stat().st_mtime_ns == stamps[files[0].name]

    # --no-resume renders everything again
    assert main([*args, "--no-resume"]) == 0
    assert "already rendered, skipping" not in capsys.readouterr().out


def test_book_chapter_selection_and_voice(book_file, clip, tmp_path):
    main(["voices", "add", "Alice", str(clip), "--yes"])
    out_dir = tmp_path / "ch2"
    assert main(["book", str(book_file), "-o", str(out_dir), "--engine", "dummy",
                 "--voice", "Alice", "--chapters", "2"]) == 0
    assert [f.name for f in out_dir.glob("*.wav")] == ["02 - Chapter 2.wav"]


def test_book_default_output_folder(book_file):
    from ttsvs import config
    assert main(["book", str(book_file), "--engine", "dummy"]) == 0
    folder = config.books_dir() / "My Book"
    assert len(list(folder.glob("*.wav"))) == 2


def test_book_keeps_chapters_out_of_history(book_file, tmp_path, capsys):
    assert main(["book", str(book_file), "-o", str(tmp_path / "b"), "--engine", "dummy"]) == 0
    capsys.readouterr()
    main(["history", "list"])
    assert "empty" in capsys.readouterr().out.lower()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="mp3 needs ffmpeg")
def test_book_mp3(book_file, tmp_path):
    out_dir = tmp_path / "mp3"
    assert main(["book", str(book_file), "-o", str(out_dir), "--engine", "dummy",
                 "--format", "mp3"]) == 0
    assert len(list(out_dir.glob("*.mp3"))) == 2


def test_book_mp3_without_ffmpeg_is_friendly(book_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("ttsvs.audio.ffmpeg_path", lambda: None)
    rc = main(["book", str(book_file), "-o", str(tmp_path / "m"), "--engine", "dummy",
               "--format", "mp3"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "ffmpeg" in err and "Traceback" not in err


def test_book_errors(book_file, tmp_path, capsys):
    assert main(["book", str(tmp_path / "missing.txt"), "--list"]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["book", str(book_file), "--list", "--chapters", "9"]) == 1
    assert "does not exist" in capsys.readouterr().err
    other = tmp_path / "book.docx"
    other.write_text("x")
    assert main(["book", str(other), "--list"]) == 1
    assert "Unsupported" in capsys.readouterr().err
    assert main(["book", str(book_file), "-o", str(tmp_path / "o"), "--engine", "dummy",
                 "--voice", "ghost"]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["book", str(book_file), "-o", str(tmp_path / "o"), "--engine", "dummy",
                 "--speed", "7"]) == 1


def test_book_pdf_without_pypdf_is_friendly(tmp_path, capsys, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pypdf", None)  # makes `import pypdf` fail
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    assert main(["book", str(pdf), "--list"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "pypdf" in err and "Traceback" not in err


def test_doctor_mentions_optional_extras(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "librosa" in out and "pypdf" in out
