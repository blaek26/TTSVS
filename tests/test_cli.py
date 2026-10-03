import numpy as np
import pytest
import soundfile as sf

from ttsvs.cli import main


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


def test_build_ui_smoke():
    pytest.importorskip("gradio")
    from ttsvs.ui import build_ui
    import gradio as gr
    assert isinstance(build_ui(engine="dummy"), gr.Blocks)
