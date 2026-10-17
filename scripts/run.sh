#!/usr/bin/env bash
# Launch the TTSVS web UI (http://127.0.0.1:7860). Extra args are passed to `ttsvs ui`.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/ttsvs ] || { echo "Not installed yet. Run scripts/install.sh first."; exit 1; }
exec .venv/bin/ttsvs ui "$@"
