#!/bin/bash
# Push only if the canon check passes. Use this (or MethylPhys/chain/guarded_push.sh) for every push; a plain `git push` skips the check.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
python3 CANON/canon_check.py || { echo "checked_push.sh: CANON CHECK FAILED - push refused"; exit 1; }
git push "$@"
