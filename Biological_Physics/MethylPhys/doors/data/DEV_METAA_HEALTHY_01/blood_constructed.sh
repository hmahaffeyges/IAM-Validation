#!/bin/bash
# The constructed bloods (blood_constructed.py) on a Google machine: MODE and PART set by the launcher; W, R, B, log and up given by it. Fetches the
# atlas donors' kept values (B/../rehearsal/bv2_donors/betas/), reads its 13 bloods, and puts its file in the bucket as it goes.
set -u; D=$R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01; O=$B/metaa_healthy/blood_constructed; mkdir -p $W/vals $W/bc
gcloud storage cp -q "$B/rehearsal/bv2_donors/betas/*_betas.parquet" $W/vals/ > /dev/null 2>&1; log "values fetched: $(ls $W/vals | wc -l)"
( cd $R && python3 -u $D/blood_constructed.py $W/vals $W/bc $MODE $PART > $W/bc.log 2>&1; echo $? > $W/bc_rc ) &
while [ ! -f $W/bc_rc ]; do sleep 240; ls $W/bc/*.json > /dev/null 2>&1 && gcloud storage cp -q $W/bc/*.json $O/ > /dev/null 2>&1; log "$MODE p$PART: $(tail -1 $W/bc.log | cut -c1-80)"; done
ls $W/bc/*.json > /dev/null 2>&1 && gcloud storage cp -q $W/bc/*.json $O/ > /dev/null 2>&1; log "$MODE p$PART rc $(cat $W/bc_rc)"; up $W/bc.log
[ "$(cat $W/bc_rc)" = 0 ] && echo done | gcloud storage cp -q - $O/blood_constructed_${MODE}_p${PART}.json.done
