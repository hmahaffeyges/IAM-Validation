#!/bin/bash
# Push only if the canon check passes. Use this (or MethylPhys/chain/guarded_push.sh) for every push; a plain `git push` skips the check.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
python3 CANON/canon_check.py || { echo "checked_push.sh: CANON CHECK FAILED - push refused"; exit 1; }
git fetch -q origin main
python3 CANON/repro_check.py || { echo "checked_push.sh: REPRODUCIBILITY CHECK FAILED - push refused"; exit 1; }
python3 CANON/status_check.py || { echo "checked_push.sh: STATUS CHECK FAILED - an old status is still printed somewhere - push refused"; exit 1; }
python3 CANON/book_toolbox.py --check || { echo "checked_push.sh: BOOK TOOLBOX CHECK FAILED - run python3 CANON/book_toolbox.py - push refused"; exit 1; }
python3 CANON/chain_gate_controls.py > /dev/null && python3 CANON/chain_gate.py check || { echo "checked_push.sh: CHAIN GATE FAILED - a logged test did not run through the chain (rule 23b) - push refused"; exit 1; }
python3 CANON/commissioning_check.py || { echo "checked_push.sh: COMMISSIONING CHECK FAILED - a commissioned instrument is missing from a required document, or a stale statement remains - push refused"; exit 1; }
python3 CANON/ascore_db.py --check || { echo "checked_push.sh: A-SCORE DATABASE CHECK FAILED - run python3 CANON/ascore_db.py - push refused"; exit 1; }
python3 CANON/catalogue.py --check || { echo "checked_push.sh: CATALOGUE CHECK FAILED - INSTRUMENTS.md or the README catalogue differs from CANON/catalogue.json - push refused"; exit 1; }
python3 CANON/bib_check.py > /dev/null || { python3 CANON/bib_check.py | tail -20; echo "checked_push.sh: BIBLIOGRAPHY CHECK FAILED - a citation does not match its CrossRef record - push refused"; exit 1; }
n=$(grep -rl "COMMISSIONING-RETURN" docs/book --include=*.tex 2>/dev/null | wc -l | tr -d " ")
[ "$n" -gt 0 ] && echo "REMINDER: $(grep -rh "COMMISSIONING-RETURN" docs/book --include=*.tex | wc -l | tr -d " ") figure(s) wait to go back into the book at commissioning (development/COMMISSIONING_RETURNS.md)."
git push "$@"
