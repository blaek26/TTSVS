"""Local Gradio web UI for TTSVS (bound to 127.0.0.1 only).

The logic lives in :class:`Backend` (plain methods, no gradio components) so it can
be tested without a browser; :func:`build_ui` only lays out components and wires
them to those methods.
"""
from __future__ import annotations

import functools
import inspect
import io
import os
import shutil
import threading
import wave
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from .cli import CHARS_PER_SECOND, estimate_minutes, parse_chapter_spec  # noqa: E402

DEFAULT_VOICE = "Default voice"
CONSENT_TEXT = "I have permission to use this voice and will not use it to deceive or impersonate."
SPEED_MIN, SPEED_MAX = 0.5, 2.0
HISTORY_SHOWN = 200  # newest items listed in the History tab
MAX_TEXT_FILE_CHARS = 500_000
SYNTH_QUEUE = "synth"  # one synthesis job at a time: the model is shared

HISTORY_HEADERS = ["Time", "Voice", "Text", "Duration (s)", "ID"]
VOICE_HEADERS = ["Name", "Seconds", "Consent", "Settings", "Notes", "Added"]
CHAPTER_HEADERS = ["#", "Title", "Characters", "Est. minutes"]
LEXICON_HEADERS = ["Word", "Say as"]

GUIDED_SCRIPT = (
    "Last Thursday, my sister Jo and I drove through the valley at dawn. "
    "Fog hung over the quiet river, and a few birds were already singing. "
    "“Do you think we’ll reach the coast by noon?” she asked. I laughed. "
    "“Maybe, if the traffic is kind!” We stopped for warm bread, sweet plums, and a "
    "cold glass of water, then kept going, eager to see the sea."
)

RECORDING_TIPS = """\
**Tips for a good recording**
- Sit in a quiet room, away from fans and echo. Soft furnishings help; a bathroom does not.
- Keep the microphone about a hand's width (15 to 20 cm) from your mouth, and stay at the same distance.
- Read the passage once at your normal pace, with natural expression. Do not whisper or shout.
- Aim for 20 to 30 seconds. Stop and start over if you stumble, then listen back before adding the voice.
- One speaker only, and no music or background noise.
"""


# ---------------------------------------------------------------- small helpers

def _friendly(exc: Exception) -> str:
    """One readable sentence for an exception, suitable for a toast."""
    name = type(exc).__name__
    if name == "VoiceExistsError":
        return f"A voice with that name already exists. {exc}"
    if name == "VoiceNotFoundError":
        return f"Voice not found. {exc}"
    if isinstance(exc, KeyError):
        return str(exc.args[0]) if exc.args else "That item no longer exists."
    if isinstance(exc, ImportError):
        return f"A required package is missing: {exc}. See the README install steps."
    return f"{exc}" if str(exc) else name


def _guarded(fn):
    """Turn any exception from a handler into a friendly gr.Error."""
    if inspect.isgeneratorfunction(fn):
        @functools.wraps(fn)
        def gen_wrapper(*args, **kwargs):
            try:
                yield from fn(*args, **kwargs)
            except gr.Error:
                raise
            except Exception as e:  # noqa: BLE001
                raise gr.Error(_friendly(e)) from e
        return gen_wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as e:  # noqa: BLE001
            raise gr.Error(_friendly(e)) from e
    return wrapper


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _pcm16(audio: np.ndarray) -> np.ndarray:
    """Float audio in [-1, 1] -> int16 (gradio would otherwise warn and may rescale floats)."""
    return (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")


def _wav_bytes(sr: int, audio: np.ndarray) -> bytes:
    """16-bit mono WAV in memory: the chunk format gr.Audio(streaming=True) accepts."""
    pcm = _pcm16(audio)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _fmt_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    return f"{minutes:.0f} min" if minutes >= 10 else f"{minutes:.1f} min"


def count_text(text: str | None) -> str:
    """The live character count and rough duration shown under the Speak box."""
    n = len(text or "")
    if n == 0:
        return "0 characters"
    return f"{n:,} characters, about {_fmt_seconds(n / CHARS_PER_SECOND)} of speech (rough estimate)"


def _local_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
    except (ValueError, OSError):
        return (iso or "")[:16].replace("T", " ")


def _one_line(text: str, width: int = 60) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= width else flat[: width - 1].rstrip() + "…"


def _fmt_params(params: dict) -> str:
    parts = []
    for key, value in (params or {}).items():
        parts.append(f"{key}={value:g}" if isinstance(value, (int, float)) else f"{key}={value}")
    return ", ".join(parts)


def _read_text_file(path: str) -> str:
    p = Path(path)
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - cp1252 only fails on a few bytes
        text = raw.decode("latin-1", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > MAX_TEXT_FILE_CHARS:
        raise gr.Error(
            f"That file is {len(text):,} characters; the Speak box takes up to "
            f"{MAX_TEXT_FILE_CHARS:,}. For something this long use the Audiobook tab.")
    return text


# ---------------------------------------------------------------- backend

class Backend:
    """Everything the UI does, as plain methods (testable without a browser)."""

    def __init__(self, engine: str = "chatterbox", device: str = "auto"):
        from .engines import get_engine
        from .history import History
        from .lexicon import Lexicon
        from .voices import VoiceLibrary

        self.engine_name = engine
        self.device = device
        self.library = VoiceLibrary()
        self.lexicon = Lexicon()
        self.history = History()
        # "speed" has its own slider (applied by the Synthesizer, never by the engine).
        self.params_spec = {k: v for k, v in get_engine(engine, device).supported_params().items()
                            if k != "speed"}
        self.streaming_supported = _ffmpeg_available()  # gr.Audio streams through ffmpeg
        self.allow_path = lambda path: None  # build_ui lets launched servers serve output files
        self._synth = None
        self._lock = threading.Lock()

    # -- shared pieces -----------------------------------------------------

    def get_synth(self):
        """Created lazily; the model itself only loads on the first generation.

        History is saved by the UI (so the item can be shown), never by the
        Synthesizer: that keeps audiobook chapters out of the history.
        """
        with self._lock:
            if self._synth is None:
                from .synth import Synthesizer
                self._synth = Synthesizer(self.engine_name, self.device, self.library,
                                          self.lexicon, self.history, save_history=False)
            return self._synth

    def voice_choices(self) -> list[str]:
        return [DEFAULT_VOICE] + [v.name for v in self.library.list()]

    def voice_names(self) -> list[str]:
        return [v.name for v in self.library.list()]

    def find_voice(self, name):
        """The Voice for a dropdown value, or None for the default voice."""
        if name in (None, "", DEFAULT_VOICE):
            return None
        return self.library.get(name)

    def voice_updates(self, main=None, manage=None, book=None, select=None):
        """Dropdown updates for every voice picker after the library changed."""
        names = self.voice_names()
        choices = [DEFAULT_VOICE] + names

        def keep(current, options, fallback):
            return current if current in options else fallback

        main_value = select if select in choices else keep(main, choices, DEFAULT_VOICE)
        manage_value = select if select in names else keep(manage, names, names[0] if names else None)
        book_value = select if select in choices else keep(book, choices, DEFAULT_VOICE)
        return (gr.update(choices=choices, value=main_value),
                gr.update(choices=names, value=manage_value),
                gr.update(choices=choices, value=book_value))

    # -- sliders / settings --------------------------------------------------

    def _clip(self, value, spec) -> float:
        lo, hi = spec.get("min", 0.0), spec.get("max", 1.0)
        return min(max(float(value), lo), hi)

    def slider_values(self, voice=None) -> list[float]:
        """Engine sliders then speed: the voice's saved defaults, else the engine defaults."""
        saved = dict(getattr(voice, "defaults", None) or {})
        values = [self._clip(saved.get(name, spec.get("default", 0.5)), spec)
                  for name, spec in self.params_spec.items()]
        speed = min(max(float(saved.get("speed", 1.0)), SPEED_MIN), SPEED_MAX)
        return values + [speed]

    def params_from_values(self, values) -> dict:
        """Slider values (engine sliders then speed) -> a params dict."""
        values = list(values)
        names = list(self.params_spec)
        if len(values) != len(names) + 1:
            raise ValueError("Unexpected number of settings.")
        params = dict(zip(names, values[:-1]))
        params["speed"] = values[-1]
        return params

    def on_voice_change(self, voice_name):
        return self.slider_values(self.find_voice(voice_name))

    @_guarded
    def save_voice_defaults(self, voice_name, *values):
        voice = self.find_voice(voice_name)
        if voice is None:
            raise gr.Error("Pick one of your cloned voices first: defaults are saved per voice.")
        params = self.params_from_values(values)
        voice = self.library.set_defaults(voice.id, params)
        return (f"Saved these settings as the defaults for **{voice.name}**: {_fmt_params(params)}.",
                self.voice_rows())

    # -- generation ------------------------------------------------------------

    def _save_history(self, synth, sr, audio, text, voice, params):
        """Add a clip to the history; returns (item or None, warning or "")."""
        try:
            voice_id = voice.id if voice else None
            item = self.history.add(sr, audio, text, voice_id, voice.name if voice else None,
                                    synth.engine.name, synth.effective_params(voice_id, params))
            return item, ""
        except Exception as e:  # noqa: BLE001 - the clip itself is fine
            return None, f" (Could not save to history: {_friendly(e)})"

    @_guarded
    def generate(self, text, voice_name, stream, *values, progress=None):
        """Generator behind the Generate button.

        Yields (live audio chunk, full clip, status text, history table). With
        streaming on, the first output gets WAV chunks as they are made; at the
        end the whole clip appears in the normal player and in History.
        """
        text = (text or "").strip()
        if not text:
            raise gr.Error("Please type some text first.")
        voice = self.find_voice(voice_name)
        voice_id = voice.id if voice else None
        params = self.params_from_values(values)
        synth = self.get_synth()
        skip = gr.skip()

        def tick(fraction, desc):
            if progress is not None:
                progress(fraction, desc=desc)

        tick(0, "Loading model (first time can take a while)...")
        yield skip, None, "Loading the model (the first time can take a while)...", skip

        if stream and self.streaming_supported:
            pieces: list[np.ndarray] = []
            sr = synth.engine.sample_rate
            for sr, chunk, i, total in synth.speak_stream(text, voice_id, **params):
                pieces.append(chunk)
                tick(i / max(total, 1), f"Synthesizing {i}/{total}")
                yield _wav_bytes(sr, chunk), skip, f"Generating... {i}/{total} chunks done", skip
            audio = np.concatenate(pieces)
        else:
            def cb(done, total):
                tick(done / max(total, 1), f"Synthesizing {done}/{total}")

            sr, audio = synth.speak(text, voice_id, progress=cb, **params)

        item, warning = self._save_history(synth, sr, audio, text, voice, params)
        saved = f" Saved to History as `{item.id}`." if item else ""
        yield (skip, (sr, _pcm16(audio)), f"Done: {len(audio) / sr:.1f} s of audio.{saved}{warning}",
               self.history_rows())

    @_guarded
    def preview_say_as(self, say_as, voice_name):
        say_as = (say_as or "").strip()
        if not say_as:
            raise gr.Error("Type the 'say as' text to preview first.")
        voice = self.find_voice(voice_name)
        sr, audio = self.get_synth().speak(say_as, voice.id if voice else None)
        return sr, _pcm16(audio)

    # -- text file ---------------------------------------------------------------

    @_guarded
    def load_text_file(self, path):
        if not path:
            return gr.skip()
        return _read_text_file(path)

    # -- history -----------------------------------------------------------------

    def history_rows(self) -> list[list]:
        return [[_local_time(i.created_at), i.voice_name or DEFAULT_VOICE.lower(),
                 _one_line(i.text), round(i.duration_s, 1), i.id]
                for i in self.history.list(HISTORY_SHOWN)]

    def history_details(self, item) -> str:
        voice = item.voice_name or "default voice"
        lines = [f"**{_local_time(item.created_at)}** &middot; voice: {voice} &middot; "
                 f"engine: {item.engine} &middot; {item.duration_s:.1f} s",
                 f"Settings: {_fmt_params(item.params) or 'engine defaults'}",
                 "", "> " + item.text.strip().replace("\n", "\n> ")]
        return "\n".join(lines)

    @_guarded
    def select_history(self, item_id):
        """-> (audio path, details markdown, selected id, export file, status)."""
        if not item_id:
            return None, "", None, gr.update(value=None, visible=False), ""
        item = self.history.get(item_id)
        return (str(item.audio_path), self.history_details(item), item.id,
                gr.update(value=None, visible=False), "")

    @_guarded
    def export_history(self, item_id):
        """-> (export file update, status). MP3 needs ffmpeg."""
        if not item_id:
            raise gr.Error("Select a history item first.")
        if not _ffmpeg_available():
            raise gr.Error("MP3 export needs ffmpeg, which was not found on this computer. "
                           "Install it (for example `sudo pacman -S ffmpeg`) and try again, "
                           "or use the download button on the player for a WAV file.")
        from . import config
        out = self.history.export(item_id, config.outputs_dir() / f"{item_id}.mp3")
        return gr.update(value=str(out), visible=True), f"Exported to `{out}`."

    @_guarded
    def use_history_settings(self, item_id):
        """-> (text, voice dropdown, *slider values (engine then speed), tab, status)."""
        if not item_id:
            raise gr.Error("Select a history item first.")
        item = self.history.get(item_id)
        note = ""
        voice_value = DEFAULT_VOICE
        if item.voice_id or item.voice_name:
            try:
                voice_value = self.library.get(item.voice_id or item.voice_name).name
            except KeyError:
                note = f" The voice '{item.voice_name or item.voice_id}' no longer exists, so the default voice is selected."
        values = []
        for name, spec in self.params_spec.items():
            values.append(self._clip(item.params.get(name, spec.get("default", 0.5)), spec))
        speed = min(max(float(item.params.get("speed", 1.0)), SPEED_MIN), SPEED_MAX)
        return (item.text, gr.update(value=voice_value), *values, speed,
                gr.Tabs(selected="speak"), f"Loaded text, voice and settings from `{item.id}`.{note}")

    @_guarded
    def delete_history(self, item_id):
        """-> (table, audio, details, selected id, export file, status)."""
        if not item_id:
            raise gr.Error("Select a history item first.")
        self.history.delete(item_id)
        return (self.history_rows(), None, "", None, gr.update(value=None, visible=False),
                "Deleted.")

    @_guarded
    def clear_history(self, confirmed):
        """-> (table, audio, details, selected id, export file, status, confirm box)."""
        if not confirmed:
            raise gr.Error("Tick the confirmation box first: clearing deletes every saved clip.")
        n = self.history.clear()
        return (self.history_rows(), None, "", None, gr.update(value=None, visible=False),
                f"Deleted {n} item{'s' if n != 1 else ''}.", False)

    # -- pronunciation -------------------------------------------------------------

    def lexicon_rows(self) -> list[list]:
        return [[w, s] for w, s in self.lexicon.entries().items()]

    @_guarded
    def lexicon_set(self, word, say_as):
        word = (word or "").strip()
        say_as = (say_as or "").strip()
        if not word:
            raise gr.Error("Type the word to replace.")
        if not say_as:
            raise gr.Error("Type how it should be said (for example 'T T S V S').")
        self.lexicon.set(word, say_as)
        return self.lexicon_rows(), f"Saved: **{word}** is spoken as “{say_as}”."

    @_guarded
    def lexicon_remove(self, word):
        word = (word or "").strip()
        if not word:
            raise gr.Error("Type or select the word to remove.")
        try:
            self.lexicon.remove(word)
        except KeyError:
            raise gr.Error(f"'{word}' is not in the dictionary.") from None
        return self.lexicon_rows(), f"Removed **{word}**.", "", ""

    # -- voices ----------------------------------------------------------------------

    def voice_rows(self) -> list[list]:
        return [[v.name, round(v.duration_s, 1),
                 "yes" if getattr(v, "consent", False) else "not recorded",
                 _fmt_params(getattr(v, "defaults", None) or {}) or "engine defaults",
                 v.notes, v.created_at[:19]]
                for v in self.library.list()]

    @_guarded
    def add_voice(self, name, audio_path, notes, consent):
        """-> (message, name box, audio, notes box, consent box, voice_dd, manage_dd, book dd, table)."""
        if not consent:
            raise gr.Error("Please confirm you have permission to use this voice.")
        if not name or not name.strip():
            raise gr.Error("Please give the voice a name.")
        if not audio_path:
            raise gr.Error("Please upload or record a voice sample.")
        voice = self.library.add(name.strip(), audio_path, notes or "", consent=True)
        msg = f"Added **{voice.name}** ({voice.duration_s:.1f} s)."
        try:
            from .voices import check_duration
            warn = check_duration(voice.duration_s)
        except ImportError:  # pragma: no cover
            warn = None
        if warn:
            msg += f"\n\nWarning: {warn}"
        return (msg, "", None, "", False, *self.voice_updates(select=voice.name), self.voice_rows())

    def preview_voice(self, name):
        if not name:
            return None
        return str(self.find_voice(name).reference_path)

    @_guarded
    def rename_voice(self, name, new, main=None, book=None):
        """-> (voice_dd, manage_dd, book dd, table, new-name box). Other pickers keep their choice."""
        if not name:
            raise gr.Error("Select a voice first.")
        if not new or not new.strip():
            raise gr.Error("Enter a new name.")
        v = self.library.rename(self.find_voice(name).id, new.strip())
        follow = lambda current: v.name if current == name else current  # noqa: E731
        return (*self.voice_updates(follow(main), v.name, follow(book)), self.voice_rows(), "")

    @_guarded
    def delete_voice(self, name, main=None, book=None):
        """-> (voice_dd, manage_dd, book dd, table, preview). Other pickers keep their choice."""
        if not name:
            raise gr.Error("Select a voice first.")
        self.library.delete(self.find_voice(name).id)
        return (*self.voice_updates(main, None, book), self.voice_rows(), None)

    # -- audiobook -----------------------------------------------------------------------

    @_guarded
    def load_book_file(self, path):
        """-> (title, chapter rows, chapters state, output folder, status)."""
        if not path:
            return "", [], None, "", ""
        from .book import load_book, safe_filename
        from . import config
        title, chapters = load_book(Path(path))
        rows = [[c.index, c.title, len(c.text), round(estimate_minutes(len(c.text)), 1)]
                for c in chapters]
        total_min = sum(estimate_minutes(len(c.text)) for c in chapters)
        out_dir = config.books_dir() / safe_filename(title, "audiobook")
        length = "under a minute" if total_min < 1 else f"about {total_min:.0f} min"
        status = (f"Found **{len(chapters)}** chapter{'s' if len(chapters) != 1 else ''}, "
                  f"{length} of speech.")
        return title, rows, chapters, str(out_dir), status

    @_guarded
    def render_book_ui(self, chapters, title, chapter_spec, voice_name, fmt, out_text,
                       use_saved, *values, progress=None):
        """-> (status, list of finished files). Skips chapters already rendered (resume)."""
        from . import config
        from .book import render_book, safe_filename
        if not chapters:
            raise gr.Error("Upload a book file first.")
        wanted = set(parse_chapter_spec(chapter_spec, len(chapters)))
        selected = [c for c in chapters if c.index in wanted]
        if not selected:
            raise gr.Error("No chapters selected.")
        voice = self.find_voice(voice_name)
        out_dir = Path(out_text.strip()).expanduser() if out_text and out_text.strip() \
            else config.books_dir() / safe_filename(title or "", "audiobook")
        params = {} if use_saved else self.params_from_values(values)
        synth = self.get_synth()

        def cb(done, total, current):
            if progress is not None:
                shown = min(done + 1, total)
                progress(done / max(total, 1), desc=f"Chapter {shown}/{total}: {current}")

        files = render_book(synth, selected, out_dir, voice.id if voice else None,
                            fmt=fmt or "wav", resume=True, progress=cb, **params)
        for f in files:
            self.allow_path(str(f))  # lets the page download files outside the data folder
        status = (f"Done: **{len(files)}** file{'s' if len(files) != 1 else ''} in `{out_dir}`. "
                  "Run it again with the same folder to pick up where it left off.")
        return status, [str(f) for f in files]


# ---------------------------------------------------------------- layout

def build_ui(engine: str = "chatterbox", device: str = "auto") -> gr.Blocks:
    be = Backend(engine, device)

    def make_param_controls() -> list:
        """Engine sliders, then the speed slider (same order as Backend.slider_values)."""
        controls = []
        for pname, spec in be.params_spec.items():
            controls.append(gr.Slider(
                minimum=spec.get("min", 0.0), maximum=spec.get("max", 1.0),
                value=spec.get("default", 0.5), step=spec.get("step", 0.05),
                label=pname, info=spec.get("help", "")))
        controls.append(gr.Slider(
            minimum=SPEED_MIN, maximum=SPEED_MAX, value=1.0, step=0.05, label="speed",
            info="1.0 is normal. Changes the pace without changing the pitch."))
        return controls

    with gr.Blocks(title="TTSVS") as demo:
        gr.Markdown("# TTSVS\nLocal text-to-speech with voice cloning. "
                    "Everything runs on this computer.")
        book_chapters = gr.State(None)  # chapters of the uploaded book

        with gr.Tabs() as tabs:
            # ------------------------------------------------------------ Speak
            with gr.Tab("Speak", id="speak"):
                text = gr.Textbox(label="Text", lines=6,
                                  placeholder="Type or paste what you want spoken...")
                with gr.Row():
                    char_count = gr.Markdown(count_text(""))
                    upload_btn = gr.UploadButton("Load a .txt file", file_types=[".txt", ".md"],
                                                 file_count="single", size="sm")
                voice_dd = gr.Dropdown(choices=be.voice_choices(), value=DEFAULT_VOICE,
                                       label="Voice")
                with gr.Accordion("Settings", open=True):
                    controls = make_param_controls()
                    with gr.Row():
                        save_defaults_btn = gr.Button("Save as this voice's defaults", size="sm")
                        defaults_msg = gr.Markdown()
                stream_cb = gr.Checkbox(
                    value=be.streaming_supported, interactive=be.streaming_supported,
                    label="Stream while generating",
                    info=("Starts playing after the first chunk while the rest is still being made."
                          if be.streaming_supported else
                          "Live playback needs ffmpeg, which was not found; the full clip appears when done."))
                gen_btn = gr.Button("Generate", variant="primary")
                status = gr.Markdown()
                stream_out = gr.Audio(label="Live playback", streaming=True, autoplay=True,
                                      interactive=False, visible=be.streaming_supported,
                                      **_download_kwargs(download=False))
                audio_out = gr.Audio(label="Result (saved to History)", type="numpy",
                                     interactive=False, **_download_kwargs())

            # ---------------------------------------------------------- History
            with gr.Tab("History", id="history"):
                gr.Markdown(f"Every generation is saved here automatically (newest {HISTORY_SHOWN} shown). "
                            "Click a row to play it.")
                history_table = gr.Dataframe(
                    headers=HISTORY_HEADERS, value=be.history_rows(), interactive=False, wrap=True,
                    datatype=["str", "str", "str", "number", "str"], max_height=320)
                refresh_btn = gr.Button("Refresh", size="sm")
                selected_item = gr.Textbox(label="Selected clip ID", interactive=False,
                                           info="Click a row above to select a clip.")
                history_audio = gr.Audio(label="Selected clip", interactive=False,
                                         **_download_kwargs())
                history_details = gr.Markdown()
                with gr.Row():
                    export_btn = gr.Button("Export MP3")
                    reuse_btn = gr.Button("Use these settings")
                    delete_item_btn = gr.Button("Delete", variant="stop")
                export_file = gr.File(label="Exported file", visible=False, interactive=False)
                history_msg = gr.Markdown()
                with gr.Row():
                    clear_confirm = gr.Checkbox(label="Yes, delete ALL history", value=False)
                    clear_btn = gr.Button("Clear all", variant="stop")

            # --------------------------------------------------------- Audiobook
            with gr.Tab("Audiobook", id="audiobook"):
                gr.Markdown(
                    "Read a whole book aloud: upload a **.txt, .md, .epub or .pdf** file (PDF needs the "
                    "`pypdf` package). It is split into chapters and each chapter becomes one audio file. "
                    "**If a job stops, run it again with the same output folder: chapters already finished "
                    "are skipped and it continues where it left off.** While a book renders, other "
                    "generations wait their turn. To stop early, close the app (Ctrl+C in the terminal).")
                book_file = gr.File(label="Book file", file_types=[".txt", ".md", ".epub", ".pdf"],
                                    type="filepath")
                book_title = gr.Textbox(label="Detected title (used for the default folder name)")
                book_table = gr.Dataframe(headers=CHAPTER_HEADERS, value=[], interactive=False,
                                          wrap=True, datatype=["number", "str", "number", "number"],
                                          max_height=300)
                book_status = gr.Markdown()
                book_spec = gr.Textbox(label="Chapters to render",
                                       placeholder="Leave empty for all, or e.g. 1-3,5")
                with gr.Row():
                    book_voice = gr.Dropdown(choices=be.voice_choices(), value=DEFAULT_VOICE,
                                             label="Voice")
                    book_fmt = gr.Radio(["wav", "mp3"], value="wav", label="Format",
                                        info="mp3 needs ffmpeg")
                book_out = gr.Textbox(label="Output folder",
                                      info="Filled in after you upload a book; edit to put the files elsewhere.")
                with gr.Accordion("Settings", open=False):
                    book_use_saved = gr.Checkbox(
                        value=True, label="Use the voice's saved settings",
                        info="Untick to use the sliders below instead.")
                    book_controls = make_param_controls()
                book_btn = gr.Button("Start", variant="primary")
                book_result = gr.Markdown()
                book_files = gr.File(label="Finished files", file_count="multiple", interactive=False)

            # ----------------------------------------------------- Pronunciation
            with gr.Tab("Pronunciation", id="pronunciation"):
                gr.Markdown(
                    "Whole-word replacements applied before speaking, for names and acronyms the model "
                    "gets wrong. Example: word **TTSVS**, say as **T T S V S**. Matching ignores case. "
                    "Click a row to edit it.")
                lex_table = gr.Dataframe(headers=LEXICON_HEADERS, value=be.lexicon_rows(), interactive=False,
                                         wrap=True, max_height=300)
                with gr.Row():
                    lex_word = gr.Textbox(label="Word")
                    lex_say = gr.Textbox(label="Say as")
                with gr.Row():
                    lex_set_btn = gr.Button("Add / Update", variant="primary")
                    lex_remove_btn = gr.Button("Remove", variant="stop")
                    lex_preview_btn = gr.Button("Preview")
                lex_msg = gr.Markdown()
                lex_audio = gr.Audio(label="Preview (uses the voice picked on the Speak tab)",
                                     interactive=False, **_download_kwargs(download=False))

            # ---------------------------------------------------------- Voices
            with gr.Tab("Voices", id="voices"):
                gr.Markdown("Add a voice from a 10 to 30 second clip: clean, one speaker, "
                            "no music. Only clone voices you have permission to use.")
                gr.Markdown("### Read this aloud while recording (about 25 seconds)")
                gr.Markdown(f"> {GUIDED_SCRIPT}")
                gr.Markdown(RECORDING_TIPS)
                new_name = gr.Textbox(label="Voice name")
                new_audio = gr.Audio(label="Voice sample (record with the microphone, or upload)",
                                     sources=["microphone", "upload"], type="filepath")
                new_notes = gr.Textbox(label="Notes (optional)")
                consent = gr.Checkbox(label=CONSENT_TEXT, value=False)
                add_btn = gr.Button("Add voice", variant="primary")
                add_msg = gr.Markdown()
                gr.Markdown("### Your voices")
                table = gr.Dataframe(headers=VOICE_HEADERS, value=be.voice_rows(), interactive=False,
                                     wrap=True)
                manage_dd = gr.Dropdown(choices=be.voice_names(), label="Selected voice")
                preview = gr.Audio(label="Preview", interactive=False)
                with gr.Row():
                    rename_to = gr.Textbox(label="New name", scale=3)
                    rename_btn = gr.Button("Rename")
                    delete_btn = gr.Button("Delete", variant="stop")

        # ------------------------------------------------------------ wiring
        def serve(path: str) -> None:
            demo.allowed_paths.append(path)

        be.allow_path = serve

        # Speak
        text.change(count_text, text, char_count, show_progress="hidden", queue=False,
                    trigger_mode="always_last", api_visibility="undocumented")
        upload_btn.upload(be.load_text_file, upload_btn, text, show_progress="hidden",
                          api_name="load_text_file")
        voice_dd.input(be.on_voice_change, voice_dd, controls, show_progress="hidden",
                       api_name="voice_defaults")
        save_defaults_btn.click(be.save_voice_defaults, [voice_dd, *controls],
                                [defaults_msg, table], api_name="save_voice_defaults")

        def toggle_stream(on):
            return gr.update(visible=bool(on) and be.streaming_supported)

        stream_cb.change(toggle_stream, stream_cb, stream_out, show_progress="hidden",
                         queue=False, api_visibility="undocumented")

        def generate(txt, voice_name, stream, *vals, progress=gr.Progress()):
            yield from be.generate(txt, voice_name, stream, *vals, progress=progress)

        gen_btn.click(generate, [text, voice_dd, stream_cb, *controls],
                      [stream_out, audio_out, status, history_table],
                      concurrency_id=SYNTH_QUEUE, concurrency_limit=1, api_name="generate")

        # History
        def history_select(evt: gr.SelectData):
            row = evt.row_value
            if not row:  # older gradio: fall back to the row position
                ids = [i.id for i in be.history.list(HISTORY_SHOWN)]
                idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                row = [None] * (len(HISTORY_HEADERS) - 1) + [ids[idx] if idx < len(ids) else None]
            return be.select_history(row[-1])

        def refresh_history():
            return be.history_rows()

        history_table.select(history_select, None,
                             [history_audio, history_details, selected_item, export_file, history_msg],
                             show_progress="hidden", api_visibility="undocumented")
        refresh_btn.click(refresh_history, None, history_table, show_progress="hidden",
                          api_name="history_refresh")
        export_btn.click(be.export_history, selected_item, [export_file, history_msg],
                         api_name="history_export")
        reuse_btn.click(be.use_history_settings, selected_item,
                        [text, voice_dd, *controls, tabs, history_msg],
                        api_name="history_use_settings")
        history_reset = [history_table, history_audio, history_details, selected_item, export_file,
                         history_msg]
        delete_item_btn.click(be.delete_history, selected_item, history_reset,
                              api_name="history_delete")
        clear_btn.click(be.clear_history, clear_confirm, [*history_reset, clear_confirm],
                        api_name="history_clear")

        # Audiobook
        book_file.change(be.load_book_file, book_file,
                         [book_title, book_table, book_chapters, book_out, book_status],
                         api_name="book_load")
        book_btn.click(be.render_book_ui,
                       [book_chapters, book_title, book_spec, book_voice, book_fmt, book_out,
                        book_use_saved, *book_controls],
                       [book_result, book_files],
                       concurrency_id=SYNTH_QUEUE, concurrency_limit=1, api_name="book_render")

        # Pronunciation
        def lex_select(evt: gr.SelectData):
            row = evt.row_value or []
            return (row[0], row[1]) if len(row) >= 2 else (gr.skip(), gr.skip())

        lex_table.select(lex_select, None, [lex_word, lex_say], show_progress="hidden",
                         api_visibility="undocumented")
        lex_set_btn.click(be.lexicon_set, [lex_word, lex_say], [lex_table, lex_msg],
                          api_name="lexicon_set")
        lex_remove_btn.click(be.lexicon_remove, lex_word, [lex_table, lex_msg, lex_word, lex_say],
                             api_name="lexicon_remove")
        lex_preview_btn.click(be.preview_say_as, [lex_say, voice_dd], lex_audio,
                              concurrency_id=SYNTH_QUEUE, concurrency_limit=1,
                              api_name="lexicon_preview")

        # Voices
        add_btn.click(be.add_voice, [new_name, new_audio, new_notes, consent],
                      [add_msg, new_name, new_audio, new_notes, consent,
                       voice_dd, manage_dd, book_voice, table], api_name="voice_add")
        manage_dd.change(be.preview_voice, manage_dd, preview, show_progress="hidden",
                         api_visibility="undocumented")
        rename_btn.click(be.rename_voice, [manage_dd, rename_to, voice_dd, book_voice],
                         [voice_dd, manage_dd, book_voice, table, rename_to], api_name="voice_rename")
        delete_btn.click(be.delete_voice, [manage_dd, voice_dd, book_voice],
                         [voice_dd, manage_dd, book_voice, table, preview], api_name="voice_delete")

        # Fill every list when the page opens (the CLI may have changed things meanwhile).
        def on_load(main, manage, book):
            return (*be.voice_updates(main, manage, book), be.voice_rows(), be.history_rows(),
                    be.lexicon_rows())

        demo.load(on_load, [voice_dd, manage_dd, book_voice],
                  [voice_dd, manage_dd, book_voice, table, history_table, lex_table],
                  show_progress="hidden", api_visibility="undocumented")

    return demo


def _download_kwargs(download: bool = True) -> dict:
    """Show (or hide) the download button across gradio versions."""
    params = inspect.signature(gr.Audio.__init__).parameters
    if "buttons" in params:  # gradio 6+
        return {"buttons": ["download"] if download else []}
    if "show_download_button" in params:
        return {"show_download_button": download}
    return {}


def launch(port: int = 7860, engine: str = "chatterbox", device: str = "auto") -> None:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    from . import config

    demo = build_ui(engine, device)
    demo.queue()
    kwargs = {"server_name": "127.0.0.1", "server_port": port, "share": False,
              "inbrowser": False,
              # History clips, voice previews and exports live in the data folder.
              "allowed_paths": [str(config.data_dir())]}
    if "show_api" in inspect.signature(demo.launch).parameters:  # removed in gradio 6
        kwargs["show_api"] = False
    demo.launch(**kwargs)
