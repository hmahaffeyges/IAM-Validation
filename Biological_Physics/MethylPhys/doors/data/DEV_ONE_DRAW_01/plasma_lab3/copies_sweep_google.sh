#!/bin/bash
# DEV-ONE-DRAW-01 (plan 60.27): one grid point of the copies sweep on a Google machine (the two largest points lost their GitHub runners twice).
# SOURCED by the launcher's job (W, R, B, log come from it; return, not exit); ROW is set by the job. Saves as it goes: the point's result is put in
# the bucket the moment it exists, and a relaunch that finds it there stops at once (skip-done). Same script and grid as the GitHub jobs.
set -u; O=$B/copies_sweep; mkdir -p $W/out
if gcloud storage ls "$O/copies_sweep_r$ROW.json" > /dev/null 2>&1; then log "already read: row $ROW in the bucket"; return 0; fi
cd $R/Biological_Physics/MethylPhys/doors/data/DEV_ONE_DRAW_01/plasma_lab3 && python3 copies_sweep.py $ROW $W/cs $W/out/copies_sweep_r$ROW.json > $W/sweep.log 2>&1
rc=$?; [ -f $W/out/copies_sweep_r$ROW.json ] && gcloud storage cp -q $W/out/copies_sweep_r$ROW.json $O/
log "row $ROW rc $rc: $(tail -2 $W/sweep.log | tr '\n' ' ' | cut -c1-250)"
