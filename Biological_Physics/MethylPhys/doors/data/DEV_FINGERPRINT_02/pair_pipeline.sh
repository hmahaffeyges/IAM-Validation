#!/bin/bash
# DEV-FINGERPRINT-02, one pair, normal side only, in order: prep_normal (copy error, readability, planted-loss response, run-loss simulation) ->
# planted test of the scorer on this pair -> power -> seal -> note section -> commit and push through the gate. Stops at the first failure;
# a pair whose planted test does not pass is not sealed. Usage: bash pair_pipeline.sh PAIR PATDIR BETADIR SCRATCH
set -euo pipefail
PAIR=$1; PATD=$2; BETD=$3; SCR=$4; HERE=$(cd "$(dirname "$0")" && pwd); TOP=$(git -C "$HERE" rev-parse --show-toplevel)
CELL=$(python3 -c "import sys;s=open('$HERE/score_fingerprint_02.py').read();P=eval(s[s.index('PAIRS = {')+8:s.index('STEP, PAIR')].strip());print(P[sys.argv[1]][0])" "$PAIR")
TAG=$(echo "$CELL" | tr ' ' '_')
[ -s "$HERE/normal_$TAG/runloss_sim.txt" ] || bash "$HERE/prep_normal.sh" "$CELL" "$(dirname "$PATD")" > /dev/null
PL=$([ "$PAIR" = breast_mcf10a ] && echo plant_fingerprint_02_output.json || echo "plant_${PAIR}_output.json")
[ -s "$HERE/$PL" ] || python3 "$HERE/plant_fingerprint_02.py" "$PATD" "$BETD" "$SCR/$PAIR" "$PAIR" | tail -4
grep -q '"PASS": false' "$HERE/$PL" && { echo "PLANTED TEST FAILED for $PAIR: not sealed"; exit 3; }
python3 "$HERE/power_fingerprint_02.py" "$PATD" "$PAIR" > "$HERE/power_${PAIR}_output.txt"; cat "$HERE/power_${PAIR}_output.txt"
python3 "$HERE/score_fingerprint_02.py" seal "$PAIR" "$PATD" "$BETD" > /dev/null
[ "$PAIR" = breast_mcf10a ] || python3 "$HERE/note_section.py" "$PAIR" >> "$TOP/Biological_Physics/MethylPhys/doors/DEV_FINGERPRINT_02.md"
printf '\n### %s · DEV-FINGERPRINT-02 pair %s sealed\nNormal side and planted test done by pair_pipeline.sh; section in DEV_FINGERPRINT_02.md written by note_section.py from the outputs.\n' "$(TZ=America/Los_Angeles date +%F)" "$PAIR" >> "$TOP/development/METHYLPHYS_DEVELOPMENT_LOG.md"
cd "$TOP" && git add -A --sparse "$HERE" Biological_Physics/MethylPhys/doors/DEV_FINGERPRINT_02.md development
git -c user.name="Heath W. Mahaffey" -c user.email="hmahaffeyges@users.noreply.github.com" commit -q -m "DEV-FINGERPRINT-02 pair $PAIR SEALED (normal side only) before any cancer file is aligned: planted test, power, seal, note. LOG."
GIT_TERMINAL_PROMPT=0 bash CANON/checked_push.sh -q origin HEAD:main 2>&1 | grep -E "check: |REFUSED|FAILED" | head -4
git log --oneline -1 | cut -c1-12
