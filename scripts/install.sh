#!/usr/bin/env bash
# TTSVS installer for macOS / Linux. Creates .venv, installs the right PyTorch build, then TTSVS.
set -euo pipefail
cd "$(dirname "$0")/.."

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
echo "==> Detected: $FLAVOR"

# Create venv: prefer uv with Python 3.11 (Arch/CachyOS ship 3.13+, too new for Chatterbox).
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv --python 3.11 .venv
  PIP=(uv pip install --python .venv/bin/python)
else
  PYBIN=""
  for p in python3.11 python3.12 python3.10 python3; do
    if command -v "$p" >/dev/null 2>&1; then PYBIN="$p"; break; fi
  done
  [ -n "$PYBIN" ] || { echo "No Python found. Install Python 3.10-3.12 (or uv)."; exit 1; }
  "$PYBIN" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' || \
    echo "WARNING: $PYBIN is not 3.10-3.12; Chatterbox may fail to install. Consider installing uv (https://docs.astral.sh/uv/)."
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
