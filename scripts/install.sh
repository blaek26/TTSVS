#!/usr/bin/env bash
# TTSVS installer for macOS / Linux. Creates .venv, installs the right PyTorch build, then TTSVS.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"  # where the uv installer puts uv

ROCM_INDEX="${TTSVS_ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.4}"
CUDA_INDEX="${TTSVS_CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"
CPU_INDEX="https://download.pytorch.org/whl/cpu"

# Pick the torch flavor.
FLAVOR=cpu
if [ "$(uname -s)" = "Darwin" ]; then
  FLAVOR=mac
elif command -v nvidia-smi >/dev/null 2>&1; then
  FLAVOR=cuda
elif [ -d /opt/rocm ] || (command -v lspci >/dev/null 2>&1 && lspci | grep -iE 'VGA|Display|3D' | grep -qiE 'AMD|ATI|Radeon'); then
  FLAVOR=rocm
fi
FLAVOR="${TTSVS_FLAVOR:-$FLAVOR}"  # override: rocm, cuda, mac or cpu
echo "==> Detected: $FLAVOR"

# Create venv with Python 3.10-3.12. Chatterbox's dependency pins break on 3.13+ (Arch/CachyOS
# ship 3.14), so prefer uv, which downloads a private Python 3.11 without touching the system.
venv_ok() {
  [ -x .venv/bin/python ] && .venv/bin/python -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'
}
if [ -d .venv ] && ! venv_ok; then
  echo "==> Existing .venv uses an unsupported Python; recreating it"
  rm -rf .venv
fi

# No uv and no Python 3.10-3.12 on the system: fetch uv into ~/.local/bin (no sudo needed).
if ! command -v uv >/dev/null 2>&1 && ! command -v python3.11 >/dev/null 2>&1 \
   && ! command -v python3.12 >/dev/null 2>&1 && ! command -v python3.10 >/dev/null 2>&1; then
  echo "==> Installing uv (fetches a private Python 3.11 for this project)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv --python 3.11 .venv
  PIP=(uv pip install --python .venv/bin/python)
else
  PYBIN=""
  for p in python3.11 python3.12 python3.10; do
    if command -v "$p" >/dev/null 2>&1; then PYBIN="$p"; break; fi
  done
  if [ -z "$PYBIN" ]; then
    echo "Need Python 3.10-3.12, but only a newer Python was found."
    echo "Install uv (it fetches Python 3.11 for this project only), then re-run this script:"
    echo "  Arch/CachyOS:  sudo pacman -S uv"
    echo "  Other:         curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
  [ -d .venv ] || "$PYBIN" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  PIP=(.venv/bin/python -m pip install)
fi

case "$FLAVOR" in
  rocm) echo "==> Installing PyTorch (ROCm)"; "${PIP[@]}" torch torchaudio --index-url "$ROCM_INDEX" ;;
  cuda) echo "==> Installing PyTorch (CUDA)"; "${PIP[@]}" torch torchaudio --index-url "$CUDA_INDEX" ;;
  mac)  echo "==> Installing PyTorch (macOS, MPS)"; "${PIP[@]}" torch torchaudio ;;
  *)    echo "==> Installing PyTorch (CPU)"; "${PIP[@]}" torch torchaudio --index-url "$CPU_INDEX" ;;
esac

if [ "$FLAVOR" = "rocm" ]; then
  # chatterbox-tts pins torch==2.6.0 whose ROCm wheels lack RDNA4 support, so install it
  # without deps (keeping the newer ROCm torch) plus its other deps from our list.
  echo "==> Installing Chatterbox (no-deps, keeping ROCm torch)"
  "${PIP[@]}" --no-deps chatterbox-tts
  "${PIP[@]}" -r scripts/requirements-chatterbox-nodeps.txt
  echo "==> Installing TTSVS"
  "${PIP[@]}" -e .
else
  echo "==> Installing TTSVS and Chatterbox"
  "${PIP[@]}" -e ".[chatterbox]"
fi

echo
echo "Done. Check the setup with:  .venv/bin/ttsvs doctor"
echo "Start the UI with:           scripts/run.sh"
