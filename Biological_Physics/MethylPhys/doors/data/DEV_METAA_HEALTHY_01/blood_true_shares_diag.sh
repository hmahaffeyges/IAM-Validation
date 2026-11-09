#!/bin/bash
# The true-shares diagnostic (blood_true_shares_diag.py) on a Google machine: PART set by the launcher; W, R, B, log and up given by it.
set -u; D=$R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01; O=$B/true_shares_diag; mkdir -p $W/vals $W/td
if gcloud storage ls "$O/true_shares_diag_p${PART}.json.done" > /dev/null 2>&1; then log "already read: part $PART is in the bucket"; exit 0; fi
gcloud storage cp -q "$O/true_shares_diag_p${PART}.json" $W/td/ > /dev/null 2>&1 && log "already read (part of it)"
gcloud storage cp -q "$(dirname $B)/rehearsal/bv2_donors/betas/*_betas.parquet" $W/vals/ > /dev/null 2>&1; log "values fetched: $(ls $W/vals | wc -l)"
( cd $R && python3 -u $D/blood_true_shares_diag.py $W/vals $W/td $PART > $W/td.log 2>&1; echo $? > $W/td_rc ) &
while [ ! -f $W/td_rc ]; do sleep 180; ls $W/td/*.json > /dev/null 2>&1 && gcloud storage cp -q $W/td/*.json $O/ > /dev/null 2>&1; log "p$PART: $(tail -1 $W/td.log | cut -c1-80)"; done
ls $W/td/*.json > /dev/null 2>&1 && gcloud storage cp -q $W/td/*.json $O/ > /dev/null 2>&1; log "p$PART rc $(cat $W/td_rc): $(tail -2 $W/td.log | tr '\n' ' ' | cut -c1-200)"; up $W/td.log
[ "$(cat $W/td_rc)" = 0 ] && echo done | gcloud storage cp -q - $O/true_shares_diag_p${PART}.json.done
