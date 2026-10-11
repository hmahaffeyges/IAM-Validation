#!/bin/bash
# The ONLY sanctioned way to score a cancer in DEV-FINGERPRINT-02 (author rule 2026-10-11: one cancer at a time, learn before the next).
# Refuses unless every earlier scored_*.json is named on a "**Reviewed:**" line of DEV_FINGERPRINT_02_OUTCOME.md (the result understood and written
# up), so no second cancer file is read while an earlier result is unexplained. Usage: bash score_one.sh pair PAIR PATD BETD   (both arms of one pair) | SCORER(02b|02b2|gain_02c) PAIR PATD [BETD]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); TOP=$(git -C "$HERE" rev-parse --show-toplevel); OUT="$TOP/Biological_Physics/MethylPhys/doors/DEV_FINGERPRINT_02_OUTCOME.md"
for f in "$HERE"/scored_*.json; do [ -e "$f" ] || continue; b=$(basename "$f")
  grep -q "^\*\*Reviewed:\*\*.*$b" "$OUT" 2>/dev/null || { echo "REFUSED: $b is scored but not reviewed in DEV_FINGERPRINT_02_OUTCOME.md (one cancer at a time)"; exit 4; }; done
# pair mode: both arms of ONE pair in one step (loss: the pair's own sealed scorer, from its seal; gain: score_gain_02c.py)
if [ "$1" = "pair" ]; then
  SC=$(python3 -c "import json,sys;print('score_fingerprint_02b2.py' if sys.argv[1].startswith('t47d') else 'score_fingerprint_02b.py')" "$2")
  python3 "$HERE/$SC" score "$2" "$3" "$4"; python3 "$HERE/score_gain_02c.py" score "$3" "$2"; exit 0; fi
case $1 in 02b) python3 "$HERE/score_fingerprint_02b.py" score "$2" "$3" "$4";; 02b2) python3 "$HERE/score_fingerprint_02b2.py" score "$2" "$3" "$4";; gain_02c) python3 "$HERE/score_gain_02c.py" score "$3" "$2";; *) echo "unknown scorer"; exit 2;; esac
