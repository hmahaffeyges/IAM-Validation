#!/bin/bash
# DEV-METAA-HEALTHY-01 test 7 identity check, one draw per Google machine (sourced by the launcher's job, which sets DRAW; W, R, B, log given).
# Saves as it goes: the draw's result goes to the bucket the moment it exists; a relaunch that finds it stops ('already read').
set -u
if gcloud storage ls "$B/synthetic_07_default_draw${DRAW}.json" > /dev/null 2>&1; then log "already read: draw $DRAW is in the bucket"; return 0; fi
cd $R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01 && python3 synthetic_07_default.py $W/s7d $DRAW > $W/s7dd_$DRAW.log 2>&1; rc=$?
gcloud storage cp -q $W/s7dd_$DRAW.log $B/ > /dev/null 2>&1; [ -f $W/s7d/synthetic_07_default_draw${DRAW}.json ] && gcloud storage cp -q $W/s7d/synthetic_07_default_draw${DRAW}.json $B/ > /dev/null 2>&1
log "draw $DRAW rc $rc: $(tail -1 $W/s7dd_$DRAW.log | cut -c1-200)"
