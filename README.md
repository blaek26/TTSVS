# TTSVS

Local text-to-speech with voice cloning. Type text, pick a voice, get speech. Add a new voice from a 10 to 30 second recording and the app mimics it.

What you get:

- **Speak**: text to speech with live streaming playback, a speed control, per-voice default settings, a character count and duration estimate, and a `.txt` loader.
- **History**: every generation is saved automatically; replay, download, export to MP3, reuse the settings, delete.
- **Audiobooks**: read a `.txt`, `.md`, `.epub` or `.pdf` file aloud, one audio file per chapter, resumable.
- **Pronunciation dictionary**: fix names and acronyms with whole-word replacements.
- **Voices**: record with the microphone using a guided reading passage, or upload a clip; consent is stored with each voice.

**Privacy:** everything runs on your computer. The only network use is the one-time model download. There is no telemetry, and the web UI listens on `127.0.0.1` only (never on your network, no public share links).

## Why Chatterbox

The default engine is [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT licensed): zero-shot cloning from a single short clip, good quality, and it runs on NVIDIA GPUs (CUDA), AMD GPUs (ROCm), Apple Silicon (MPS) or plain CPU. CPU works but is slow (seconds per sentence); a GPU is close to real time. See [PLAN.md](PLAN.md) for design decisions and the roadmap.

## Install

Requires Python 3.10 to 3.12 (3.11 recommended). The one-shot installers detect your hardware and install the matching PyTorch build first:

- macOS / Linux: `scripts/install.sh`
- Windows: `powershell -ExecutionPolicy Bypass -File scripts\install.ps1`

They create `.venv` in the project folder. Then check the setup with `.venv/bin/ttsvs doctor`.

Manual install: create a venv, install PyTorch for your hardware **first** (see https://pytorch.org/get-started/locally/), then `pip install -e ".[chatterbox]"`.

Optional extras:

- `ffmpeg`: importing mp3/m4a voice samples, MP3 export, the speed control when `librosa` is missing, and live streaming playback in the web UI (without it the UI still works and shows the full clip when done). On Arch/CachyOS: `sudo pacman -S ffmpeg`.
- `librosa`: used for the speed control (pitch-preserving time stretch). It comes with Chatterbox, so you normally already have it.
- `pypdf`: needed only to read PDF audiobooks. `pip install "ttsvs[book]"`, or in the uv setup below: `uv pip install --python .venv/bin/python pypdf`. Text, Markdown and EPUB need nothing extra.

### Linux + AMD GPU (ROCm)

Tested target: CachyOS / Arch with a Radeon RX 9070 XT (RDNA4, gfx1201).

1. System packages and permissions:
   ```bash
   sudo pacman -S rocm-hip-runtime        # or the rocm-hip-sdk group
   sudo usermod -aG render,video $USER    # then log out and back in
   ```
2. Arch/CachyOS ship Python 3.13+, which Chatterbox's dependencies do not support yet. Use [`uv`](https://docs.astral.sh/uv/) (or pyenv) for Python 3.11:
   ```bash
   sudo pacman -S uv
   uv venv --python 3.11 .venv
   ```
3. Install the ROCm build of PyTorch **before** anything else (RDNA4 needs ROCm 6.4 or newer; newer `rocm6.x` index URLs may exist, check pytorch.org):
   ```bash
   uv pip install --python .venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
   # chatterbox-tts pins torch==2.6.0, whose ROCm wheels do not support RDNA4, so install it
   # without dependencies and add the rest from a shared list (keeps the newer ROCm torch):
   uv pip install --python .venv/bin/python --no-deps chatterbox-tts
   uv pip install --python .venv/bin/python -r scripts/requirements-chatterbox-nodeps.txt
   uv pip install --python .venv/bin/python -e .
   ```
   `scripts/install.sh` does all of this (including the special no-deps Chatterbox steps) automatically when it detects an AMD GPU. Override the index with `TTSVS_ROCM_INDEX=...`.
4. Verify: `.venv/bin/ttsvs doctor` should show a ROCm/HIP version and your GPU. ROCm PyTorch reports the GPU as device `cuda`, so `--device auto` and `--device cuda` both use it.

Troubleshooting: if the GPU is not detected, try `HSA_OVERRIDE_GFX_VERSION=12.0.1` (only as a last resort), and confirm you are in the `render` and `video` groups.

### Linux + NVIDIA, Windows, macOS

- NVIDIA (Linux/Windows): CUDA build of PyTorch (the installers pick `cu126`).
- Apple Silicon: the default PyTorch wheel includes MPS; `--device auto` selects it.
- No GPU: the CPU build works, just slowly.

## Quick start

Web UI (opens at http://127.0.0.1:7860):

```bash
scripts/run.sh            # Windows: scripts\run.bat
# or: ttsvs ui --port 7860
```

Command line:

```bash
ttsvs voices add alice alice_sample.wav --notes "my sample"   # asks for consent; --yes skips the prompt
ttsvs voices list
ttsvs say "Hello from my own computer." --voice alice -o hello.wav
ttsvs say --file article.txt --voice alice -o article.wav --exaggeration 0.6 --speed 1.15
ttsvs voices rename alice Alice
ttsvs voices remove Alice
ttsvs doctor
```

Omit `--voice` to use the model's default voice. The first generation downloads the model and loads it, which takes a while. Use `--engine dummy` to try the app without any model.

Voices, history, audiobooks and your pronunciation dictionary are stored in your OS user data folder; set `TTSVS_HOME` to put them elsewhere.

## The web UI

| Tab | What it does |
| --- | --- |
| **Speak** | Type or paste text (or **Load a .txt file**), pick a voice, adjust the settings, press Generate. A live character count shows a rough duration (about 15 characters per second). |
| **History** | A table of recent generations. Click a row to play it. **Export MP3**, **Use these settings** (sends text, voice and settings back to the Speak tab), **Delete**, and **Clear all** (tick the confirmation box first). |
| **Audiobook** | Upload a book, check the detected chapters, choose a voice, format and output folder, press Start. |
| **Pronunciation** | Add whole-word replacements, preview how they sound. |
| **Voices** | Record or upload a sample, see consent status and saved settings, preview, rename, delete. |

### Streaming playback

With **Stream while generating** on (the default), playback starts after the first chunk of text is ready while the rest is still being synthesized. When it finishes, the whole clip appears in the normal player (with a download button) and is saved to History. Turn it off to wait for the full clip instead. Streaming needs `ffmpeg` on your PATH.

### Speed

The **speed** slider (0.5x to 2x) time-stretches the result without changing the pitch. On the command line use `--speed 1.25`. It needs `librosa` or `ffmpeg`.

### Per-voice default settings

Pick a voice, set the sliders (expressiveness, pacing, temperature, speed) the way you like them and press **Save as this voice's defaults**. From then on, picking that voice loads those settings, and the CLI uses them whenever you do not pass the matching flag:

```bash
ttsvs voices defaults alice --exaggeration 0.7 --speed 1.1   # save (merges with what is there)
ttsvs voices defaults alice                                   # show
ttsvs voices defaults alice --reset                           # clear
```

### Recording a voice sample

The Voices tab shows a short passage (about 25 seconds, chosen to cover a wide range of sounds) to read aloud while you record with the microphone, with tips for a clean take. You can also upload a file. Adding a voice requires ticking the consent box, and the app remembers that you did.

## History

Every generation (UI or `ttsvs say`) is saved with its text, voice, settings and time. In the CLI:

```bash
ttsvs history list --limit 10
ttsvs history export 20261003-072500-ab12 take1.mp3     # .wav or .mp3 (mp3 needs ffmpeg); a unique id prefix is enough
ttsvs history remove 20261003-072500-ab12
ttsvs history clear                                      # asks first; --yes skips the question
ttsvs say "Do not keep this." -o temp.wav --no-history
```

Audiobook chapters are not added to the history.

## Audiobooks

Read a whole text aloud as an audiobook: one audio file per chapter, written to a folder.

```bash
ttsvs book story.epub --list                              # show the detected chapters and a time estimate
ttsvs book story.epub --voice alice -o ~/audiobooks/story
ttsvs book novel.txt --voice alice -o out --format mp3 --chapters 1-3,5 --speed 1.1
```

- Supported files: `.txt` and `.md` (chapters start at Markdown headings or lines like `Chapter 7` / `CHAPTER VII`; a long text with none is cut into parts), `.epub` (follows the book's reading order), and `.pdf` (needs `pypdf`, see above; scanned PDFs without a text layer cannot be read).
- Files are named `01 - Chapter title.wav` (or `.mp3`, which needs ffmpeg). Without `-o` the folder is `<data dir>/books/<book title>`.
- **Resume:** if a job stops (power cut, Ctrl+C, closing the window), run the same command again with the same output folder. Chapters that are already finished are skipped and it carries on; a chapter that was cut off halfway is redone. Use `--no-resume` to render everything again.
- The voice's saved defaults apply unless you pass flags such as `--exaggeration` or `--speed`.
- In the web UI, the **Audiobook** tab does the same thing: upload, check the chapter table (characters and estimated minutes), optionally enter chapters such as `1-3,5`, press Start, then download the finished files from the list.

## Pronunciation dictionary

For names and acronyms the model gets wrong. Entries replace whole words, ignoring case, before the text is spoken (the longest entry wins), for example `TTSVS` becomes `T T S V S` or `Nguyen` becomes `Win`.

```bash
ttsvs lexicon set TTSVS "T T S V S"
ttsvs lexicon set Nguyen "Win"
ttsvs lexicon list
ttsvs lexicon remove Nguyen
```

The same dictionary is managed in the **Pronunciation** tab, which also has a **Preview** button that speaks the "say as" text. It applies to everything you generate, including audiobooks, and is stored in `lexicon.json` in the data folder.

## Tips for a good voice sample

- 10 to 30 seconds of natural speech.
- One speaker, no background music or noise, no echo.
- Consistent, expressive but normal delivery; a decent microphone helps.
- Avoid heavy compression; WAV or FLAC is best.

## Ethics

Only clone voices you have permission to use. Do not use cloned voices to deceive or impersonate anyone. The app asks you to confirm consent when adding a voice (and stores that you did; `ttsvs voices list` and the Voices tab show it), and Chatterbox embeds an inaudible watermark (Perth) in generated audio, which we keep enabled.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests -q
```

Tests use the `dummy` engine, so no model download is needed. Project layout and roadmap are in [PLAN.md](PLAN.md).
