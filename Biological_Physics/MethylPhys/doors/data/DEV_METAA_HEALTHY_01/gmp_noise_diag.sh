#!/bin/bash
# gmp_noise_diag.py on a Google machine (W, R, B, log and up given by the launcher). Saves its file to the bucket as it goes; a relaunch resumes.
set -u; D=$R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01; O=$B/gmp_noise_diag; mkdir -p $W/vals $W/gn
if gcloud storage ls "$O/gmp_noise_diag.json.done" > /dev/null 2>&1; then log "already read: in the bucket"; exit 0; fi
gcloud storage cp -q "$O/gmp_noise_diag.json" $W/gn/ > /dev/null 2>&1 && log "already read (part of it)"
for g in $(tail -n +2 $D/gmp_noise_donors.csv | cut -d, -f1); do gcloud storage cp -q "$(dirname $B)/rehearsal/bv2_donors/betas/${g}_betas.parquet" $W/vals/ > /dev/null 2>&1; done; log "values fetched: $(ls $W/vals | wc -l)"
( cd $R && python3 -u $D/gmp_noise_diag.py $W/vals $W/gn > $W/gn.log 2>&1; echo $? > $W/gn_rc ) &
while [ ! -f $W/gn_rc ]; do sleep 180; ls $W/gn/*.json > /dev/null 2>&1 && gcloud storage cp -q $W/gn/*.json $O/ > /dev/null 2>&1; log "$(tail -1 $W/gn.log | cut -c1-80)"; done
gcloud storage cp -q $W/gn/*.json $O/ > /dev/null 2>&1; log "rc $(cat $W/gn_rc): $(tail -2 $W/gn.log | tr '\n' ' ' | cut -c1-200)"; up $W/gn.log
[ "$(cat $W/gn_rc)" = 0 ] && echo done | gcloud storage cp -q - $O/gmp_noise_diag.json.done
