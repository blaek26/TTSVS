"""Command line interface for TTSVS."""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
from pathlib import Path

CONSENT_PROMPT = "Only clone voices you have permission to use. Continue? [y/N] "
SPEED_MIN = 0.5
SPEED_MAX = 2.0
CHARS_PER_SECOND = 15  # rough speaking rate used for duration estimates
PARAM_FLAGS = ("exaggeration", "cfg_weight", "temperature", "speed")

# Exceptions (matched by class name anywhere in the MRO, so the core modules stay
# lazily imported) that get a one-line `error:` message instead of a traceback.
_EXPECTED_ERRORS = {
    "VoiceError", "EngineError", "AudioError", "BookError", "ValueError",
    "FileNotFoundError", "OSError", "RuntimeError", "ImportError", "KeyError",
}


class CliError(Exception):
    """An expected, user-facing error."""


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--engine", default="chatterbox", choices=["chatterbox", "dummy"],
                   help="TTS engine (default: chatterbox)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="compute device (default: auto)")


def _add_param_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--exaggeration", type=float, help="expressiveness (Chatterbox)")
    p.add_argument("--cfg-weight", dest="cfg_weight", type=float,
                   help="pacing / adherence to the voice (Chatterbox)")
    p.add_argument("--temperature", type=float, help="randomness (Chatterbox)")
    p.add_argument("--speed", type=float,
                   help=f"speaking speed {SPEED_MIN} to {SPEED_MAX} (1.0 = normal; pitch is kept)")


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
    _add_param_args(say)
    say.add_argument("--no-history", action="store_true",
                     help="do not save this clip to the history")
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
    df = vsub.add_parser(
        "defaults", help="show or set a voice's default settings (used whenever you do not pass them)")
    df.add_argument("name")
    _add_param_args(df)
    df.add_argument("--reset", action="store_true", help="clear all saved defaults for the voice")

    history = sub.add_parser("history", help="browse and export past generations")
    hsub = history.add_subparsers(dest="history_command")
    hl = hsub.add_parser("list", help="list recent generations, newest first")
    hl.add_argument("--limit", type=int, default=20,
                    help="how many to show (default 20, 0 for all)")
    he = hsub.add_parser("export", help="save a past generation as .wav or .mp3")
    he.add_argument("id")
    he.add_argument("out", type=Path, help="output path; the extension picks the format")
    hr = hsub.add_parser("remove", help="delete one generation")
    hr.add_argument("id")
    hc = hsub.add_parser("clear", help="delete the whole history")
    hc.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")

    book = sub.add_parser(
        "book", help="read a .txt/.md/.epub/.pdf file aloud, one audio file per chapter")
    book.add_argument("file", type=Path)
    book.add_argument("--voice", help="voice name (omit for the default voice)")
    book.add_argument("-o", "--out", type=Path,
                      help="output folder (default: <data dir>/books/<book title>)")
    book.add_argument("--format", dest="fmt", choices=["wav", "mp3"], default="wav",
                      help="audio format (mp3 needs ffmpeg)")
    book.add_argument("--chapters", help="chapters to render, e.g. 1-3,5 (default: all)")
    book.add_argument("--no-resume", action="store_true",
                      help="render every chapter again even if its file already exists")
    book.add_argument("--list", action="store_true",
                      help="only list the detected chapters; render nothing")
    _add_engine_args(book)
    _add_param_args(book)

    lex = sub.add_parser("lexicon", help="pronunciation dictionary (whole-word replacements)")
    lsub = lex.add_subparsers(dest="lexicon_command")
    lsub.add_parser("list", help="list entries")
    ls = lsub.add_parser("set", help="add or replace an entry")
    ls.add_argument("word")
    ls.add_argument("say_as", help='how to say it, e.g. "T T S V S"')
    lr = lsub.add_parser("remove", help="remove an entry")
    lr.add_argument("word")

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


def _collect_params(args) -> dict:
    """The synthesis settings given on the command line (None ones are left out)."""
    params = {k: getattr(args, k, None) for k in PARAM_FLAGS}
    params = {k: v for k, v in params.items() if v is not None}
    speed = params.get("speed")
    if speed is not None and not SPEED_MIN <= speed <= SPEED_MAX:
        raise CliError(f"--speed must be between {SPEED_MIN} and {SPEED_MAX}.")
    return params


def estimate_minutes(chars: int) -> float:
    return chars / CHARS_PER_SECOND / 60.0


def parse_chapter_spec(spec: str | None, total: int) -> list[int]:
    """Turn "1-3,5" into [1, 2, 3, 5] (1-based, sorted, unique). Empty means all chapters."""
    if spec is None or not spec.strip() or spec.strip().lower() == "all":
        return list(range(1, total + 1))
    picked: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d*)(?:(-)(\d*))?", part)
        if not m or (not m.group(1) and not m.group(3)):
            raise ValueError(f"cannot understand chapter selection {part!r}; use e.g. 1-3,5")
        if m.group(2):
            start = int(m.group(1)) if m.group(1) else 1
            end = int(m.group(3)) if m.group(3) else total
        else:
            start = end = int(m.group(1))
        if start < 1 or end < start:
            raise ValueError(f"invalid chapter range {part!r}")
        if end > total:
            raise ValueError(
                f"chapter {end} does not exist; this book has {total} chapter{'s' if total != 1 else ''}")
        picked.update(range(start, end + 1))
    if not picked:
        raise ValueError("no chapters selected")
    return sorted(picked)


def _preview(text: str, width: int = 50) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1].rstrip() + "…"


def _fmt_params(params: dict) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                     for k, v in params.items())


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
    voice = _find_voice(library, args.voice) if args.voice else None
    voice_id = voice.id if voice else None
    params = _collect_params(args)

    from .synth import Synthesizer
    synth = Synthesizer(args.engine, args.device, library)

    def progress(done: int, total: int) -> None:
        print(f"\rsynthesizing {done}/{total}", end="", file=sys.stderr, flush=True)

    try:
        sr, audio = synth.speak(text, voice_id, progress=progress, **params)
    finally:
        print(file=sys.stderr)  # end the progress line (also before an error message)
    out = _save_wav(args.out, sr, audio)
    print(f"Saved {out}")
    if not args.no_history:
        try:
            from .history import History
            item = History().add(sr, audio, text, voice_id, voice.name if voice else None,
                                 synth.engine.name, synth.effective_params(voice_id, params))
            print(f"Added to history as {item.id}")
        except Exception as e:  # noqa: BLE001 - the clip is already saved
            print(f"warning: could not save to history: {e}", file=sys.stderr)
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


def _engine_param_specs(engine: str) -> dict:
    try:
        from .engines import get_engine
        return get_engine(engine, "cpu").supported_params()
    except Exception:  # noqa: BLE001
        return {}


def cmd_voices(args) -> int:
    library = _library()
    sub = args.voices_command
    if sub == "list":
        voices = library.list()
        if not voices:
            print("No voices yet. Add one with: ttsvs voices add NAME CLIP.wav")
            return 0
        print(f"{'NAME':<24} {'ID':<24} {'SECONDS':>8}  {'CONSENT':<8} NOTES")
        for v in voices:
            consent = "yes" if getattr(v, "consent", False) else "-"
            print(f"{v.name:<24} {v.id:<24} {v.duration_s:>8.1f}  {consent:<8} {v.notes}")
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
        voice = library.add(args.name, args.clip, notes=args.notes, consent=True)
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
    if sub == "defaults":
        v = _find_voice(library, args.name)
        params = _collect_params(args)
        if args.reset and params:
            raise CliError("use either --reset or new values, not both.")
        changed = False
        if args.reset:
            v = library.set_defaults(v.id, {})
            changed = True
        elif params:
            for key, value in params.items():
                spec = _engine_param_specs("chatterbox").get(key)
                if spec and not spec["min"] <= value <= spec["max"]:
                    raise CliError(
                        f"--{key.replace('_', '-')} must be between {spec['min']:g} and {spec['max']:g}.")
            v = library.set_defaults(v.id, {**v.defaults, **params})
            changed = True
        if changed:
            print(f"Saved defaults for '{v.name}'.")
        if v.defaults:
            print(f"Defaults for '{v.name}': {_fmt_params(v.defaults)}")
        else:
            print(f"'{v.name}' has no saved defaults (engine defaults apply). "
                  f"Set some with: ttsvs voices defaults {v.name} --speed 1.1 --exaggeration 0.6")
        return 0
    raise CliError("usage: ttsvs voices {list,add,remove,rename,defaults}")


def cmd_history(args) -> int:
    from .history import History
    history = History()
    sub = args.history_command

    def find(item_id: str):
        """Exact id, or a unique prefix of one."""
        try:
            return history.get(item_id)
        except KeyError:
            matches = [i for i in history.list() if i.id.startswith(item_id)]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise CliError(f"'{item_id}' matches {len(matches)} history items; use the full id.")
            raise CliError(f"history item '{item_id}' not found. "
                           "Run `ttsvs history list` to see ids.")

    if sub == "list":
        limit = args.limit if args.limit and args.limit > 0 else None
        items = history.list(limit)
        if not items:
            print("History is empty. Generate something with `ttsvs say` or the web UI.")
            return 0
        print(f"{'ID':<22} {'TIME (UTC)':<19} {'VOICE':<16} {'SEC':>6}  TEXT")
        for i in items:
            when = i.created_at[:19].replace("T", " ")
            print(f"{i.id:<22} {when:<19} {(i.voice_name or '(default)')[:16]:<16} "
                  f"{i.duration_s:>6.1f}  {_preview(i.text)}")
        return 0
    if sub == "export":
        item = find(args.id)
        out = history.export(item.id, args.out)
        print(f"Saved {out}")
        return 0
    if sub == "remove":
        item = find(args.id)
        history.delete(item.id)
        print(f"Removed history item {item.id}.")
        return 0
    if sub == "clear":
        count = len(history.list())
        if count == 0:
            print("History is already empty.")
            return 0
        if not args.yes:
            try:
                answer = input(f"Delete all {count} history item{'s' if count != 1 else ''}? [y/N] ")
            except EOFError:
                answer = ""
            if answer.strip().lower() not in ("y", "yes"):
                raise CliError("cancelled: nothing was deleted.")
        removed = history.clear()
        print(f"Removed {removed} history item{'s' if removed != 1 else ''}.")
        return 0
    raise CliError("usage: ttsvs history {list,export,remove,clear}")


def cmd_lexicon(args) -> int:
    from .lexicon import Lexicon
    lex = Lexicon()
    sub = args.lexicon_command
    if sub == "list":
        entries = lex.entries()
        if not entries:
            print("The pronunciation dictionary is empty. "
                  'Add a word with: ttsvs lexicon set WORD "SAY AS"')
            return 0
        width = max(len(w) for w in entries)
        for word, say_as in entries.items():
            print(f"{word:<{width}}  ->  {say_as}")
        return 0
    if sub == "set":
        lex.set(args.word, args.say_as)
        print(f"Set '{args.word}' -> '{args.say_as}'")
        return 0
    if sub == "remove":
        try:
            lex.remove(args.word)
        except KeyError:
            raise CliError(f"'{args.word}' is not in the dictionary. "
                           "Run `ttsvs lexicon list` to see entries.")
        print(f"Removed '{args.word}'.")
        return 0
    raise CliError("usage: ttsvs lexicon {list,set,remove}")


def cmd_book(args) -> int:
    from .book import chapter_filename, load_book, render_book, safe_filename
    path = args.file
    if not path.is_file():
        raise CliError(f"file not found: {path}")
    title, chapters = load_book(path)
    if not chapters:
        raise CliError(f"no readable text found in {path}.")
    wanted = set(parse_chapter_spec(args.chapters, len(chapters)))
    selected = [c for c in chapters if c.index in wanted]
    if not selected:
        raise CliError("no chapters selected.")

    if args.list:
        print(f"Title: {title}")
        print(f"{'#':>3}  {'CHARS':>8}  {'MIN':>5}  TITLE")
        for c in selected:
            print(f"{c.index:>3}  {len(c.text):>8}  {estimate_minutes(len(c.text)):>5.1f}  {c.title}")
        total = sum(len(c.text) for c in selected)
        minutes = estimate_minutes(total)
        length = "under a minute" if minutes < 1 else f"about {minutes:.0f} min"
        print(f"{len(selected)} chapter{'s' if len(selected) != 1 else ''}, {length} of speech.")
        return 0

    from .config import books_dir
    out_dir = Path(args.out) if args.out else books_dir() / safe_filename(title, "audiobook")
    library = _library()
    voice_id = _find_voice(library, args.voice).id if args.voice else None
    params = _collect_params(args)

    from .synth import Synthesizer
    synth = Synthesizer(args.engine, args.device, library)  # no history: books stay out of it

    resume = not args.no_resume
    width = max([2] + [len(str(c.index)) for c in selected])
    print(f"Rendering '{title}': {len(selected)} chapter{'s' if len(selected) != 1 else ''} "
          f"-> {out_dir}")
    if resume:
        print("(chapters already rendered in that folder are skipped; use --no-resume to redo them)")

    def progress(done: int, total: int, current: str) -> None:
        if done >= total:
            return  # the "Done:" line below says it
        existing = out_dir / chapter_filename(selected[done], args.fmt, width)
        note = "  (already rendered, skipping)" if resume and existing.is_file() else ""
        print(f"[{done + 1}/{total}] {current}{note}", flush=True)

    files = render_book(synth, selected, out_dir, voice_id, fmt=args.fmt,
                        resume=resume, progress=progress, **params)
    print(f"Done: {len(files)} file{'s' if len(files) != 1 else ''} in {out_dir}")
    return 0


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
    print(f"ffmpeg:      {ff or 'not found (optional: mp3/m4a, speed control, live streaming in the UI)'}")
    lr = importlib.util.find_spec("librosa") is not None
    print(f"librosa:     {'installed' if lr else 'not installed (optional; speed control then uses ffmpeg)'}")
    pp = importlib.util.find_spec("pypdf") is not None
    print(f"pypdf:       {'installed' if pp else 'not installed (optional; needed for PDF audiobooks)'}")
    gr = importlib.util.find_spec("gradio") is not None
    print(f"gradio:      {'installed' if gr else 'NOT installed'}")
    try:
        from .config import data_dir
        print(f"data dir:    {data_dir()}")
    except Exception as e:  # noqa: BLE001
        print(f"data dir:    unknown ({e})")
    return 0


def _error_text(e: Exception) -> str:
    if isinstance(e, KeyError) and e.args:
        return str(e.args[0])  # KeyError's own str() would add quotes
    return str(e)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    handlers = {"say": cmd_say, "voices": cmd_voices, "history": cmd_history,
                "book": cmd_book, "lexicon": cmd_lexicon, "ui": cmd_ui,
                "doctor": cmd_doctor}
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
        if any(cls.__name__ in _EXPECTED_ERRORS for cls in type(e).__mro__):
            _err(_error_text(e) or name)
            return 1
        if os.environ.get("TTSVS_DEBUG"):
            raise
        _err(f"{name}: {e} (set TTSVS_DEBUG=1 for a traceback)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
