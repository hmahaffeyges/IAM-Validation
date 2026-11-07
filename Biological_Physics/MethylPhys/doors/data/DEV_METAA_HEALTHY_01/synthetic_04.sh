#!/bin/bash
# DEV-METAA-HEALTHY-01 synthetic test 4, one draw per Google machine (sourced by the launcher's job, which sets DRAW; W, R, B, log given).
# Saves as it goes: the draw's result goes to the bucket the moment it exists; a relaunch that finds it stops ('already read').
set -u
if gcloud storage ls "$B/synthetic_04_draw${DRAW}.json" > /dev/null 2>&1; then log "already read: draw $DRAW is in the bucket"; return 0; fi
cd $R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01 && python3 synthetic_04.py $W/s4 $DRAW > $W/s4_$DRAW.log 2>&1; rc=$?
gcloud storage cp -q $W/s4_$DRAW.log $B/ > /dev/null 2>&1; [ -f $W/s4/synthetic_04_draw${DRAW}.json ] && gcloud storage cp -q $W/s4/synthetic_04_draw${DRAW}.json $B/ > /dev/null 2>&1
log "draw $DRAW rc $rc: $(tail -1 $W/s4_$DRAW.log | cut -c1-200)"
