#!/bin/bash
# The ONLY sanctioned way to score a cancer in DEV-FINGERPRINT-02 (author rule 2026-10-11: one cancer at a time, learn before the next).
# Refuses unless every earlier scored_*.json is named on a "**Reviewed:**" line of DEV_FINGERPRINT_02_OUTCOME.md (the result understood and written
# up), so no second cancer file is read while an earlier result is unexplained. Usage: bash score_one.sh SCORER(02b|gain_02c) PAIR PATD [BETD]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); TOP=$(git -C "$HERE" rev-parse --show-toplevel); OUT="$TOP/Biological_Physics/MethylPhys/doors/DEV_FINGERPRINT_02_OUTCOME.md"
for f in "$HERE"/scored_*.json; do [ -e "$f" ] || continue; b=$(basename "$f")
  grep -q "^\*\*Reviewed:\*\*.*$b" "$OUT" 2>/dev/null || { echo "REFUSED: $b is scored but not reviewed in DEV_FINGERPRINT_02_OUTCOME.md (one cancer at a time)"; exit 4; }; done
case $1 in 02b) python3 "$HERE/score_fingerprint_02b.py" score "$2" "$3" "$4";; 02b2) python3 "$HERE/score_fingerprint_02b2.py" score "$2" "$3" "$4";; gain_02c) python3 "$HERE/score_gain_02c.py" score "$3" "$2";; *) echo "unknown scorer"; exit 2;; esac
