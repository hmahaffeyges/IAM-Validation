#!/bin/bash
# Push only if the canon check passes. Use this (or MethylPhys/chain/guarded_push.sh) for every push; a plain `git push` skips the check.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
python3 CANON/canon_check.py || { echo "checked_push.sh: CANON CHECK FAILED - push refused"; exit 1; }
git fetch -q origin main
python3 CANON/repro_check.py || { echo "checked_push.sh: REPRODUCIBILITY CHECK FAILED - push refused"; exit 1; }
n=$(grep -rl "COMMISSIONING-RETURN" docs/book --include=*.tex 2>/dev/null | wc -l | tr -d " ")
[ "$n" -gt 0 ] && echo "REMINDER: $(grep -rh "COMMISSIONING-RETURN" docs/book --include=*.tex | wc -l | tr -d " ") figure(s) wait to go back into the book at commissioning (development/COMMISSIONING_RETURNS.md)."
git push "$@"
