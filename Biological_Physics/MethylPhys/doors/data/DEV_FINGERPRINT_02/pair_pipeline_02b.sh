#!/bin/bash
# DEV-FINGERPRINT-02, one pair under rule 02b, normal side only: prep_normal -> planted test of score_fingerprint_02b.py on this pair (NULL, RUN05,
# RUN01, SCAT) -> seal -> note section -> commit and push through the gate. A pair whose planted test fails is not sealed.
# Usage: bash pair_pipeline_02b.sh PAIR PATDIR BETADIR SCRATCH
set -euo pipefail
PAIR=$1; PATD=$2; BETD=$3; SCR=$4; HERE=$(cd "$(dirname "$0")" && pwd); TOP=$(git -C "$HERE" rev-parse --show-toplevel)
CELL=$(python3 -c "import sys;s=open('$HERE/score_fingerprint_02b.py').read();P=eval(s[s.index('PAIRS = {')+8:s.index('STEP, PAIR')].strip());print(P[sys.argv[1]][0])" "$PAIR")
TAG=$(echo "$CELL" | tr ' ' '_')
[ -s "$HERE/normal_$TAG/runloss_sim.txt" ] || bash "$HERE/prep_normal.sh" "$CELL" "$(dirname "$PATD")" > /dev/null
[ -s "$HERE/plant02b_${PAIR}_output.json" ] || python3 "$HERE/plant_fingerprint_02b.py" "$PATD" "$BETD" "$SCR/$PAIR" "$PAIR" | tail -6
grep -q '"PASS": false' "$HERE/plant02b_${PAIR}_output.json" && { echo "PLANTED TEST FAILED for $PAIR: not sealed"; exit 3; }
python3 "$HERE/score_fingerprint_02b.py" seal "$PAIR" "$PATD" "$BETD" > /dev/null
python3 "$HERE/note_section_02b.py" "$PAIR" >> "$TOP/Biological_Physics/MethylPhys/doors/DEV_FINGERPRINT_02.md"
printf '\n### %s · DEV-FINGERPRINT-02 pair %s sealed under rule 02b\nNormal side, planted test (NULL, RUN05, RUN01, SCAT) and seal by pair_pipeline_02b.sh; section written by note_section_02b.py from the outputs.\n' "$(TZ=America/Los_Angeles date +%F)" "$PAIR" >> "$TOP/development/METHYLPHYS_DEVELOPMENT_LOG.md"
cd "$TOP" && git add -A --sparse "$HERE" Biological_Physics/MethylPhys/doors/DEV_FINGERPRINT_02.md development
git -c user.name="Heath W. Mahaffey" -c user.email="hmahaffeyges@users.noreply.github.com" commit -q -m "DEV-FINGERPRINT-02 pair $PAIR SEALED under rule 02b (normal side only) before any cancer file is aligned. LOG."
GIT_TERMINAL_PROMPT=0 bash CANON/checked_push.sh -q origin HEAD:main 2>&1 | grep -E "check: |REFUSED|FAILED" | head -4
git log --oneline -1 | cut -c1-12
