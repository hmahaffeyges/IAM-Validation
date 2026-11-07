#!/bin/bash
# The test 6 capture diagnostic on a Google machine (SOURCED by the launcher's job; W, R, B, log given). Saves its outputs to the bucket.
set -u
if gcloud storage ls "$B/synthetic_06_capture.json" > /dev/null 2>&1; then log "already read: the capture is in the bucket"; return 0; fi
gcloud storage cp -q "$B/synthetic_06_diag_draw1.json" $W/d1.json > /dev/null 2>&1; log "diag draw 1 $(stat -c %s $W/d1.json 2>/dev/null || echo missing)"
cd $R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01
( python3 synthetic_06_capture.py $W/d1.json $W/cap > $W/cap.log 2>&1; echo $? > $W/cap_rc ) &
while [ ! -f $W/cap_rc ]; do sleep 240; log "capture: $(grep -c 'line_on_captured' $W/cap.log) of 2 arrays"; done
for f in $W/cap/synthetic_06_capture.json $W/cap/cap_low.parquet $W/cap/cap_center.parquet $W/cap.log; do [ -f $f ] && gcloud storage cp -q $f $B/ > /dev/null 2>&1; done
log "capture rc $(cat $W/cap_rc)"
