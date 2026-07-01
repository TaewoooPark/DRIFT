#!/usr/bin/env bash
# DRIFT installer — macOS (Apple GPU / MPS) and Linux (NVIDIA / CUDA).
#
# The PyPI `torch` wheel is platform-correct on these OSes: the macOS arm64
# wheel ships MPS, the Linux x86_64 wheel ships CUDA — so a plain install picks
# the right GPU backend automatically. (Windows needs the CUDA index; use
# scripts/install.ps1 there.)
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found — install it first:  https://github.com/astral-sh/uv" >&2
  exit 1
fi

echo "[1/3] creating .venv (Python 3.12) …"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/3] installing DRIFT + dependencies …"
uv pip install -e .

echo "[3/3] done."
export PYTORCH_ENABLE_MPS_FALLBACK=1
cat <<'NEXT'

  installed ✓   next:

    source .venv/bin/activate
    export PYTORCH_ENABLE_MPS_FALLBACK=1   # macOS only; harmless elsewhere
    drift doctor                            # check environment + device
    drift up 2                              # try it on THIS machine (2 local nodes)

  to join a cluster:
    drift node          # on each worker machine (auto device, LAN-announced)
    drift run           # on the head machine (auto-discovers the workers)

NEXT
