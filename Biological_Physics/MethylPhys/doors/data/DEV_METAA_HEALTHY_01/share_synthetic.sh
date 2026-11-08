#!/bin/bash
# The share synthetic, one (cell, share) on a Google machine; SOURCED by the launcher's job (W, R, B, log, up given; CI, SI set). Saves its result
# to the bucket; a result already there is not read again.
set -u; D=$R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01; F=share_synthetic_c${CI}_s${SI}.json
if gcloud storage ls $B/$F > /dev/null 2>&1; then log "already read: $F"; return 0 2>/dev/null || exit 0; fi
( cd $D && python3 share_synthetic.py $W/ss $CI $SI > $W/ss.log 2>&1; echo $? > $W/ss_rc ) &
while [ ! -f $W/ss_rc ]; do sleep 240; log "share synthetic c$CI s$SI: $(grep -c 'rc' $W/ss.log) lines"; done
[ -f $W/ss/$F ] && gcloud storage cp -q $W/ss/$F $B/ > /dev/null 2>&1; mv $W/ss.log $W/ss_c${CI}_s${SI}.log; gcloud storage cp -q $W/ss_c${CI}_s${SI}.log $B/ > /dev/null 2>&1
log "share synthetic c$CI s$SI rc $(cat $W/ss_rc)"
