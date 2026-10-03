# TTSVS Plan: Local Text-to-Speech with Voice Sampling

## Goal

A text-to-speech app that runs entirely on your own computer. You type text, pick a voice, and get speech back. You can add a new voice by giving it a short recording (about 10 to 30 seconds) of someone speaking, and the app will mimic that voice. Nothing is sent to the internet after the one-time model download.

## Key decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Language | Python 3.10 to 3.12 | Every serious open-source TTS model ships Python first. Works on Windows, macOS and Linux. |
| Voice model (default) | [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI (`chatterbox-tts`) | MIT licensed, zero-shot voice cloning from a single short clip, good quality, runs on NVIDIA GPU (CUDA), Apple Silicon (MPS) or plain CPU. |
| Alternative model (later) | Coqui XTTS-v2 via the maintained `coqui-tts` fork | Covers ~17 languages. Its weights are non-commercial (CPML), so it is optional, not the default. |
| Interface | Local web UI with Gradio, bound to `127.0.0.1` only, plus a command-line tool | Browser UI works the same on every OS with no native GUI toolkit to fight; the CLI is good for scripts and batch jobs. |
| Hardware | Auto-detect: CUDA, then MPS, then CPU. Override with `--device`. | CPU works everywhere but is slow (roughly several seconds per sentence); a GPU makes it near real time. |
| Privacy | No telemetry, UI never listens on the network, model files cached locally | "Runs locally" is the core requirement. After the first download it works offline. |

Open question for Blake: which OS and GPU are you on? It changes the install instructions (CUDA build of PyTorch on Windows/Linux with NVIDIA, MPS on Apple Silicon), not the code.

## Architecture

```
ttsvs/
  __init__.py
  config.py          # paths (data dir, voices dir, output dir), device selection
  audio.py           # load/resample/normalize/trim audio, save WAV/MP3
  text.py            # split long text into sentence chunks the model can handle
  voices.py          # VoiceLibrary: add/list/get/delete/rename cloned voices
  engines/
    __init__.py      # registry: get_engine(name) -> TTSEngine
    base.py          # TTSEngine abstract base class
    chatterbox.py    # Chatterbox implementation (default)
    dummy.py         # silent/sine "engine" for tests and UI work without a model
  synth.py           # Synthesizer: text + voice -> audio, chunking + stitching
  cli.py             # `ttsvs` command line
  ui.py              # Gradio web UI
tests/
pyproject.toml
README.md
scripts/            # one-click install/run scripts for Windows, macOS, Linux
```

### Shared interfaces (both workers build against these)

```python
# ttsvs/engines/base.py
class TTSEngine(ABC):
    name: str
    sample_rate: int
    def __init__(self, device: str = "auto"): ...
    @abstractmethod
    def load(self) -> None: ...            # lazy model load, idempotent
    @abstractmethod
    def synthesize(self, text: str, reference_wav: Path | None,
                   **params) -> np.ndarray: ...   # float32 mono, range [-1, 1]
    def supported_params(self) -> dict[str, dict]: ...  # name -> {default, min, max, help}

# ttsvs/voices.py
@dataclass
class Voice:
    id: str            # slug, folder name
    name: str          # display name
    reference_path: Path   # cleaned 24 kHz mono WAV
    created_at: str
    notes: str = ""
    duration_s: float = 0.0

class VoiceLibrary:
    def __init__(self, root: Path | None = None): ...   # default config.voices_dir()
    def add(self, name: str, audio_path: Path, notes: str = "") -> Voice: ...
    def list(self) -> list[Voice]: ...
    def get(self, voice_id: str) -> Voice: ...
    def delete(self, voice_id: str) -> None: ...
    def rename(self, voice_id: str, new_name: str) -> Voice: ...

# ttsvs/synth.py
class Synthesizer:
    def __init__(self, engine: str = "chatterbox", device: str = "auto",
                 library: VoiceLibrary | None = None): ...
    def speak(self, text: str, voice_id: str | None = None,
              progress: Callable[[int, int], None] | None = None,
              **params) -> tuple[int, np.ndarray]: ...   # (sample_rate, audio)
    def speak_to_file(self, text: str, out_path: Path, voice_id: str | None = None,
                      **params) -> Path: ...
```

Voices live in `<data dir>/voices/<id>/` as `reference.wav` plus `voice.json`. The data dir defaults to the OS user data folder (via `platformdirs`) and can be overridden with the `TTSVS_HOME` environment variable.

## Features, in build order

### Phase 1: working core (this PR)
1. Engine abstraction with Chatterbox and a dummy engine.
2. Voice library: import a WAV/MP3/FLAC/M4A clip, convert to mono 24 kHz, trim silence, normalize loudness, warn if shorter than ~5 s or longer than ~60 s.
3. Text chunking so long passages (paragraphs, whole articles) are synthesized sentence by sentence and stitched with short pauses.
4. CLI: `ttsvs say "text" --voice NAME -o out.wav`, `ttsvs voices add NAME clip.wav`, `ttsvs voices list`, `ttsvs voices remove NAME`, `ttsvs ui`, `ttsvs doctor` (prints device, versions, model cache).
5. Gradio UI with two tabs: **Speak** (text box, voice dropdown, expressiveness/speed controls, play and download) and **Voices** (upload or record a sample with the microphone, name it, preview, delete).
6. Unit tests using the dummy engine, so tests run without downloading models.
7. README with install steps per OS, and `scripts/` launchers.

### Phase 2: quality and convenience
- Record a sample directly in the UI with a guided script to read aloud.
- Multiple reference clips per voice, pick the best or average.
- History of generated clips; export to MP3.
- Per-voice default settings (exaggeration, pacing).
- Streaming playback while long text is still generating.

### Phase 3: reach
- XTTS-v2 engine for multilingual speech (opt-in, license notice shown).
- Read a text/EPUB/PDF file aloud to an audiobook.
- Optional packaged desktop app (PyInstaller) so non-developers can double-click.

## Ethics and safety

Voice cloning can be misused. The app will show a short consent reminder when adding a voice ("only clone voices you have permission to use") and store that the user confirmed it. Chatterbox also embeds an inaudible watermark (Resemble's Perth) in generated audio; we keep it on.

## Work split for the first build

- **Worker 1 (core engine):** `config.py`, `audio.py`, `text.py`, `voices.py`, `engines/*`, `synth.py`, `tests/` for those modules, `pyproject.toml`.
- **Worker 2 (app layer):** `cli.py`, `ui.py`, `README.md`, `scripts/`, tests for the CLI. Builds against the interfaces above and uses the dummy engine to run without models.

## Phase 2 build (second round)

Phase 1 is confirmed working on Blake's CachyOS + RX 9070 XT. This round adds the three features Blake picked plus extras.

### Features
1. **History**: every generation is saved automatically (audio + text, voice, settings, time). A History tab lists them newest first with play, download, export to MP3, re-use settings, delete and clear all. CLI: `ttsvs history list|export ID OUT|remove ID|clear`.
2. **Streaming playback**: audio starts playing after the first chunk while later chunks are still generating (UI toggle, on by default). The full clip is still saved to history at the end.
3. **Audiobooks**: read a `.txt`, `.md`, `.epub` or `.pdf` file aloud. It is split into chapters and renders one audio file per chapter (WAV or MP3) into a folder. A stopped job resumes where it left off. UI: an Audiobook tab with upload, chapter preview, voice and settings, progress, and a list of finished files. CLI: `ttsvs book FILE --voice NAME -o OUTDIR [--format mp3] [--chapters 1-3]`.
4. **Extras**:
   - **Speed control**: a speed slider (0.5x to 2x) that time-stretches the result without changing pitch.
   - **Pronunciation dictionary**: whole-word replacements applied before synthesis, for example "TTSVS" becomes "T T S V S" or "Nguyen" becomes "Win". Managed in the UI and through `ttsvs lexicon list|set WORD SAY|remove WORD`.
   - **Per-voice default settings**: save the current sliders as a voice's defaults, which load automatically when that voice is picked.
   - **Consent stored with the voice**: closes the Phase 1 gap.
   - **Guided recording**: the Voices tab shows a short paragraph to read aloud while recording with the mic.
   - **Text input**: load a .txt file into the Speak box, and show a character count with an estimated duration.

### Phase 2 interfaces

```python
# ttsvs/config.py (additions)
def history_dir() -> Path: ...      # <data dir>/history
def books_dir() -> Path: ...        # <data dir>/books (default audiobook output root)
def lexicon_path() -> Path: ...     # <data dir>/lexicon.json

# ttsvs/audio.py (addition)
def time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray: ...
    # rate > 1 = faster. Pitch preserved. Uses librosa if importable, else ffmpeg
    # atempo (chained for rates outside 0.5-2), else raises AudioError. rate == 1 returns input.

# ttsvs/voices.py (changes; voice.json without the new keys must still load)
@dataclass
class Voice:
    ...existing fields...
    consent: bool = False
    defaults: dict = field(default_factory=dict)   # engine param name -> value, may include "speed"
class VoiceLibrary:
    def add(self, name, audio_path, notes="", consent=False) -> Voice: ...
    def set_defaults(self, voice_id, params: dict) -> Voice: ...

# ttsvs/lexicon.py
class Lexicon:
    def __init__(self, path: Path | None = None): ...  # default config.lexicon_path(); missing file = empty
    def entries(self) -> dict[str, str]: ...           # sorted by word
    def set(self, word: str, say_as: str) -> None: ...
    def remove(self, word: str) -> None: ...           # KeyError if missing
    def apply(self, text: str) -> str: ...              # whole-word, case-insensitive, longest match first

# ttsvs/history.py
@dataclass
class HistoryItem:
    id: str              # sortable, e.g. 20261003-072500-ab12
    created_at: str      # ISO 8601
    text: str
    voice_id: str | None
    voice_name: str | None
    engine: str
    params: dict
    audio_path: Path     # <history dir>/<id>/audio.wav
    duration_s: float
class History:
    def __init__(self, root: Path | None = None): ...
    def add(self, sr: int, audio: np.ndarray, text: str, voice_id: str | None,
            voice_name: str | None, engine: str, params: dict) -> HistoryItem: ...
    def list(self, limit: int | None = None) -> list[HistoryItem]: ...   # newest first
    def get(self, item_id: str) -> HistoryItem: ...                      # KeyError if missing
    def delete(self, item_id: str) -> None: ...
    def clear(self) -> int: ...                                          # number removed
    def export(self, item_id: str, out_path: Path) -> Path: ...          # .wav or .mp3

# ttsvs/synth.py (changes)
class Synthesizer:
    def __init__(self, engine="chatterbox", device="auto", library=None,
                 lexicon: Lexicon | None = None, history: History | None = None,
                 save_history: bool = False): ...
    # params may include speed (float, default 1.0); it is applied per chunk via
    # audio.time_stretch and never passed to the engine. Params not given fall back to
    # the voice's saved defaults, then the engine defaults. The lexicon is applied before splitting.
    def speak_stream(self, text, voice_id=None, **params) -> Iterator[tuple[int, np.ndarray, int, int]]: ...
        # yields (sample_rate, chunk_audio, index (1-based), total) for each chunk; chunk_audio
        # already has fades and a trailing pause except for the last chunk
    def speak(self, text, voice_id=None, progress=None, **params) -> tuple[int, np.ndarray]: ...
        # built on speak_stream; when save_history is true, adds the result to history
    def speak_to_file(...): ...  # unchanged
    def effective_params(self, voice_id: str | None, params: dict) -> dict: ...  # the merged params, for UI/history

# ttsvs/book.py
@dataclass
class Chapter:
    index: int     # 1-based
    title: str
    text: str
class BookError(Exception): ...
def load_book(path: Path) -> tuple[str, list[Chapter]]: ...
    # .txt/.md: split on markdown headings or lines like "Chapter 7" / "CHAPTER VII"; else one chapter
    # .epub: stdlib zipfile + html.parser following the OPF spine; titles from <h1>/<h2>/<title>
    # .pdf: pypdf (optional extra "book"); clear BookError if not installed
    # Empty chapters are dropped.
def render_book(synth: Synthesizer, chapters: list[Chapter], out_dir: Path, voice_id=None,
                fmt: str = "wav", resume: bool = True,
                progress: Callable[[int, int, str], None] | None = None, **params) -> list[Path]: ...
    # one file per chapter: "<NN> - <safe title>.<fmt>"; resume skips files that already exist
    # (written to a .part temp name and renamed on completion, so partial files are never skipped)
    # progress(done_chapters, total_chapters, current_title)
```

### Work split for the second build
- **Worker A (core):** `config.py`, `audio.py`, `voices.py`, `lexicon.py`, `history.py`, `book.py`, `synth.py`, `pyproject.toml` (extra `book = ["pypdf"]`), and tests for all of these.
- **Worker B (app layer):** `cli.py`, `ui.py`, `README.md`, `tests/test_cli.py`, `tests/test_ui.py`.
