#!/bin/bash
# erase_test.yml's run step on a Google spot machine (2026-11-13, PHY-086: GitHub's machines were all reading the referee test). Same
# environment and the same erase_gh.sh; checkpoints on (CKPT_ON=1: a reclaimed machine resumes from the saved reads or alignment).
# Set by the launcher: GSM, LIST (repository-relative sample list), MAXP (a pair cap for the canary, or empty for the list's default).
# Uploads eout/ (count JSON, bundle, logs) and run.log to $B/out/$GSM/ and writes rc to the log.
MP=$R/Biological_Physics/MethylPhys; E=$MP/doors/data/DEV_ERASE_01
. $W/v/bin/activate
export INDEX_TAG=hg19-bwameth-index-v1 INDEX_TAG_MM10=mm10-bwameth-index-v1 GITHUB_REPOSITORY=hmahaffeyges/IAM-Validation MP WORK=$W/ew OUT=$W/eout NW=4 FULL=0 SAMPLES=$R/$LIST
[ -n "${MAXP:-}" ] && export MAXPAIRS=$MAXP
mkdir -p $W/eout
CKPT=$([ "${CKPT_ON:-0}" = 1 ] && echo $B/ckpt/$GSM) bash $E/erase_gh.sh $GSM > $W/run.log 2>&1; rc=$?
log "$GSM rc $rc: $(grep -E 'fetched pairs|aligned in|pat in' $W/run.log | tr '\n' ' ' | cut -c1-250)"
for x in $W/eout/* $W/run.log; do [ -f "$x" ] && gcloud storage cp -q "$x" $B/out/$GSM/ > /dev/null 2>&1; done
[ $rc -eq 0 ] && gcloud storage rm -r -q "$B/ckpt/$GSM" > /dev/null 2>&1
log "$GSM done rc $rc"
