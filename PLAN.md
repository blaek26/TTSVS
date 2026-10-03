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
