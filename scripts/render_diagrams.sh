#!/usr/bin/env bash
# Render the generated diagram HTML to mono-black PNGs via headless Chrome (2x).
#   python scripts/make_diagrams.py && scripts/render_diagrams.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
BUILD="docs/img/_build"
OUT="docs/img"
mkdir -p "$OUT"

PY="${PYTHON:-python3}"
"$PY" - <<'PY' | while IFS=' ' read -r name w h; do
import json
for m in json.load(open("docs/img/_build/manifest.json")):
    print(m["name"], m["w"], m["h"])
PY
  "$CHROME" --headless=new --disable-gpu --no-first-run --no-default-browser-check \
    --user-data-dir=/tmp/drift-chrome-diag --hide-scrollbars \
    --force-device-scale-factor=2 --default-background-color=000000ff \
    --window-size="${w},${h}" \
    --screenshot="$OUT/${name}.png" "file://$PWD/$BUILD/${name}.html" >/dev/null 2>&1
  echo "rendered $OUT/${name}.png (${w}x${h} @2x)"
done
