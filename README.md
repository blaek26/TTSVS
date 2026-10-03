# TTSVS

Local text-to-speech with voice cloning. Type text, pick a voice, get speech. Add a new voice from a 10 to 30 second recording and the app mimics it.

**Privacy:** everything runs on your computer. The only network use is the one-time model download. There is no telemetry, and the web UI listens on `127.0.0.1` only (never on your network, no public share links).

## Why Chatterbox

The default engine is [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT licensed): zero-shot cloning from a single short clip, good quality, and it runs on NVIDIA GPUs (CUDA), AMD GPUs (ROCm), Apple Silicon (MPS) or plain CPU. CPU works but is slow (seconds per sentence); a GPU is close to real time. See [PLAN.md](PLAN.md) for design decisions and the roadmap.

## Install

Requires Python 3.10 to 3.12 (3.11 recommended). The one-shot installers detect your hardware and install the matching PyTorch build first:

- macOS / Linux: `scripts/install.sh`
- Windows: `powershell -ExecutionPolicy Bypass -File scripts\install.ps1`

They create `.venv` in the project folder. Then check the setup with `.venv/bin/ttsvs doctor`.

Manual install: create a venv, install PyTorch for your hardware **first** (see https://pytorch.org/get-started/locally/), then `pip install -e ".[chatterbox]"`.

`ffmpeg` is optional and only needed to import mp3/m4a voice samples (WAV and FLAC work without it).

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
ttsvs say --file article.txt --voice alice -o article.wav --exaggeration 0.6
ttsvs voices rename alice Alice
ttsvs voices remove Alice
ttsvs doctor
```

Omit `--voice` to use the model's default voice. The first generation downloads the model and loads it, which takes a while. Use `--engine dummy` to try the app without any model.

Voices and outputs are stored in your OS user data folder; set `TTSVS_HOME` to put them elsewhere.

## Tips for a good voice sample

- 10 to 30 seconds of natural speech.
- One speaker, no background music or noise, no echo.
- Consistent, expressive but normal delivery; a decent microphone helps.
- Avoid heavy compression; WAV or FLAC is best.

## Ethics

Only clone voices you have permission to use. Do not use cloned voices to deceive or impersonate anyone. The app asks you to confirm consent when adding a voice, and Chatterbox embeds an inaudible watermark (Perth) in generated audio, which we keep enabled.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests -q
```

Tests use the `dummy` engine, so no model download is needed. Project layout and roadmap are in [PLAN.md](PLAN.md).
