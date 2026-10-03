"""Command line interface for TTSVS."""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

CONSENT_PROMPT = "Only clone voices you have permission to use. Continue? [y/N] "


class CliError(Exception):
    """An expected, user-facing error."""


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--engine", default="chatterbox", choices=["chatterbox", "dummy"],
                   help="TTS engine (default: chatterbox)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="compute device (default: auto)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttsvs", description="Local text-to-speech with voice cloning.")
    sub = parser.add_subparsers(dest="command")

    say = sub.add_parser("say", help="synthesize text to a WAV file")
    say.add_argument("text", nargs="?", help="text to speak (or use --file)")
    say.add_argument("--file", type=Path, help="read the text from this file")
    say.add_argument("--voice", help="voice name (omit for the default voice)")
    say.add_argument("-o", "--out", type=Path, required=True, help="output WAV path")
    _add_engine_args(say)
    say.add_argument("--exaggeration", type=float)
    say.add_argument("--cfg-weight", dest="cfg_weight", type=float)
    say.add_argument("--temperature", type=float)
    say.add_argument("--play", action="store_true", help="play the result afterwards")

    voices = sub.add_parser("voices", help="manage cloned voices")
    vsub = voices.add_subparsers(dest="voices_command")
    vsub.add_parser("list", help="list voices")
    add = vsub.add_parser("add", help="add a voice from an audio clip")
    add.add_argument("name")
    add.add_argument("clip", type=Path)
    add.add_argument("--notes", default="")
    add.add_argument("--yes", "-y", action="store_true",
                     help="confirm you have permission to use this voice")
    rm = vsub.add_parser("remove", help="remove a voice")
    rm.add_argument("name")
    rn = vsub.add_parser("rename", help="rename a voice")
    rn.add_argument("name")
    rn.add_argument("new")

    ui = sub.add_parser("ui", help="launch the local web UI")
    ui.add_argument("--port", type=int, default=7860)
    _add_engine_args(ui)

    sub.add_parser("doctor", help="check the installation")
    return parser


# ---------------------------------------------------------------- helpers

def _library():
    from .voices import VoiceLibrary
    return VoiceLibrary()


def _find_voice(library, name: str):
    """Look a voice up by id or (case-insensitive) display name."""
    voices = library.list()
    for v in voices:
        if v.id == name:
            return v
    for v in voices:
        if v.name.lower() == name.lower():
            return v
    raise CliError(f"voice '{name}' not found. Run `ttsvs voices list` to see voices.")


def _duration_warning(seconds: float):
    try:
        from .voices import check_duration
        return check_duration(seconds)
    except ImportError:
        return None


# ---------------------------------------------------------------- commands

def cmd_say(args) -> int:
    if args.file:
        try:
            text = args.file.read_text(encoding="utf-8")
        except OSError as e:
            raise CliError(f"cannot read {args.file}: {e.strerror or e}")
    else:
        text = args.text
    if not text or not text.strip():
        raise CliError("no text given. Pass TEXT or --file PATH.")

    library = _library()
    voice_id = _find_voice(library, args.voice).id if args.voice else None

    params = {k: getattr(args, k) for k in ("exaggeration", "cfg_weight", "temperature")
              if getattr(args, k) is not None}

    from .synth import Synthesizer
    synth = Synthesizer(args.engine, args.device, library)

    def progress(done: int, total: int) -> None:
        print(f"\rsynthesizing {done}/{total}", end="", file=sys.stderr, flush=True)

    sr, audio = synth.speak(text, voice_id, progress=progress, **params)
    print(file=sys.stderr)
    out = _save_wav(args.out, sr, audio)
    print(f"Saved {out}")
    if args.play:
        _play(out)
    return 0


def _save_wav(path: Path, sr: int, audio) -> Path:
    import numpy as np
    import wave
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def _play(path: Path) -> None:
    import subprocess
    if sys.platform == "darwin":
        cmd = ["afplay", str(path)]
    elif sys.platform.startswith("win"):
        cmd = ["powershell", "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync()"]
    else:
        cmd = next(([c, str(path)] for c in ("paplay", "aplay", "ffplay")
                    if shutil.which(c)), None)
    if not cmd:
        print("No audio player found; open the file manually.", file=sys.stderr)
        return
    try:
        subprocess.run(cmd, check=False)
    except OSError as e:
        print(f"Could not play audio: {e}", file=sys.stderr)


def cmd_voices(args) -> int:
    library = _library()
    sub = args.voices_command
    if sub == "list":
        voices = library.list()
        if not voices:
            print("No voices yet. Add one with: ttsvs voices add NAME CLIP.wav")
            return 0
        print(f"{'NAME':<24} {'ID':<24} {'SECONDS':>8}  NOTES")
        for v in voices:
            print(f"{v.name:<24} {v.id:<24} {v.duration_s:>8.1f}  {v.notes}")
        return 0
    if sub == "add":
        if not args.clip.is_file():
            raise CliError(f"audio file not found: {args.clip}")
        if not args.yes:
            try:
                answer = input(CONSENT_PROMPT)
            except EOFError:
                answer = ""
            if answer.strip().lower() not in ("y", "yes"):
                raise CliError("cancelled: consent is required to add a voice.")
        voice = library.add(args.name, args.clip, notes=args.notes)
        print(f"Added voice '{voice.name}' (id {voice.id}, {voice.duration_s:.1f} s).")
        warning = _duration_warning(voice.duration_s)
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
        return 0
    if sub == "remove":
        v = _find_voice(library, args.name)
        library.delete(v.id)
        print(f"Removed voice '{v.name}'.")
        return 0
    if sub == "rename":
        v = _find_voice(library, args.name)
        new = library.rename(v.id, args.new)
        print(f"Renamed '{v.name}' to '{new.name}'.")
        return 0
    raise CliError("usage: ttsvs voices {list,add,remove,rename}")


def cmd_ui(args) -> int:
    try:
        from .ui import launch
    except ImportError as e:
        raise CliError(f"the UI needs gradio ({e}). Install with: pip install gradio")
    launch(port=args.port, engine=args.engine, device=args.device)
    return 0


def cmd_doctor(args) -> int:
    print(f"Python:      {sys.version.split()[0]} ({sys.executable})")
    has_torch = importlib.util.find_spec("torch") is not None
    device = "cpu"
    if has_torch:
        try:
            import torch
            print(f"torch:       {torch.__version__}")
            hip = getattr(torch.version, "hip", None)
            cuda = getattr(torch.version, "cuda", None)
            if hip:
                print(f"ROCm/HIP:    {hip} (AMD GPU build; appears as device 'cuda')")
            elif cuda:
                print(f"CUDA:        {cuda}")
            if torch.cuda.is_available():
                print(f"GPU:         {torch.cuda.get_device_name(0)}")
        except Exception as e:  # noqa: BLE001
            print(f"torch:       installed but failed to import ({e})")
    else:
        print("torch:       NOT installed (install PyTorch first, see README)")
    try:
        from .config import resolve_device
        device = resolve_device("auto")
    except Exception as e:  # noqa: BLE001
        device = f"unknown ({e})"
    print(f"device:      {device}")
    cb = importlib.util.find_spec("chatterbox") is not None
    print(f"chatterbox:  {'installed' if cb else 'NOT installed (pip install chatterbox-tts)'}")
    ff = shutil.which("ffmpeg")
    print(f"ffmpeg:      {ff or 'not found (optional, needed for mp3/m4a)'}")
    gr = importlib.util.find_spec("gradio") is not None
    print(f"gradio:      {'installed' if gr else 'NOT installed'}")
    try:
        from .config import data_dir
        print(f"data dir:    {data_dir()}")
    except Exception as e:  # noqa: BLE001
        print(f"data dir:    unknown ({e})")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    handlers = {"say": cmd_say, "voices": cmd_voices, "ui": cmd_ui, "doctor": cmd_doctor}
    try:
        return handlers[args.command](args)
    except CliError as e:
        _err(str(e))
        return 1
    except KeyboardInterrupt:
        _err("interrupted")
        return 130
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if name in ("VoiceNotFoundError", "VoiceExistsError", "ValueError",
                    "FileNotFoundError", "RuntimeError", "ImportError"):
            _err(str(e))
            return 1
        if os.environ.get("TTSVS_DEBUG"):
            raise
        _err(f"{name}: {e} (set TTSVS_DEBUG=1 for a traceback)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
