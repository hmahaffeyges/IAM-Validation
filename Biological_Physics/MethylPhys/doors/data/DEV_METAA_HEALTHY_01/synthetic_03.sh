#!/bin/bash
# DEV-METAA-HEALTHY-01 synthetic test 3 on a Google machine (sourced by the launcher's job; W, R, B, log given). Saves as it goes: the summary
# goes to the bucket the moment it exists, and a relaunch that finds it stops ('already read').
set -u
if gcloud storage ls "$B/synthetic_03_summary.txt" > /dev/null 2>&1; then log "already read: the summary is in the bucket"; return 0; fi
cd $R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01 && python3 synthetic_03.py $W/s3 > $W/s3.log 2>&1; rc=$?
gcloud storage cp -q $W/s3.log $B/ > /dev/null 2>&1; [ -f $W/s3/synthetic_03_summary.txt ] && gcloud storage cp -q $W/s3/synthetic_03_summary.txt $B/ > /dev/null 2>&1
log "synthetic 3 rc $rc: $(tail -1 $W/s3.log | cut -c1-200)"
