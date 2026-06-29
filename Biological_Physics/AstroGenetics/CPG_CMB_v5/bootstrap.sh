#!/usr/bin/env bash
# bootstrap.sh — CPG_CMB cold start. Finds (or installs) a suitable Python 3.9+,
# then hands off to preflight.py, which installs every remaining dependency itself.
#
#   ./bootstrap.sh            # core chain
#   ./bootstrap.sh --idat     # also install methylprep for raw IDAT decode
#
# Safe to re-run; it only acts on what is missing.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIN_MAJOR=3; MIN_MINOR=9

ok_python() {  # $1 = interpreter; succeeds if it is >= 3.9
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= ('"$MIN_MAJOR"','"$MIN_MINOR"') else 1)' 2>/dev/null
}

# 1. find a good python already on the machine (prefer the newest)
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && ok_python "$c"; then PY="$(command -v "$c")"; break; fi
done

# 2. none found -> install one with the platform's package manager
if [ -z "$PY" ]; then
  echo "[bootstrap] no Python >= ${MIN_MAJOR}.${MIN_MINOR} found; attempting to install one..."
  if command -v brew >/dev/null 2>&1; then
    brew install python@3.12 && PY="$(command -v python3.12 || command -v python3)"
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y python3 python3-pip python3-venv
    PY="$(command -v python3)"
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y python3 python3-pip && PY="$(command -v python3)"
  else
    echo "[bootstrap] could not auto-install Python. Install Python >= ${MIN_MAJOR}.${MIN_MINOR} and re-run." >&2
    exit 2
  fi
fi

echo "[bootstrap] using Python: $PY ($("$PY" --version 2>&1))"
# 3. make sure pip exists for that interpreter
"$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade || true

# 4. hand off to the Python preflight (installs packages, decompresses atlas, checks mapping)
exec "$PY" "$HERE/preflight.py" --root "$HERE" "$@"
