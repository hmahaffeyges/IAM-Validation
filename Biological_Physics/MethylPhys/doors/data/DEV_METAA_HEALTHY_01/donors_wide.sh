#!/bin/bash
# Each cell's healthy width from the atlas's own donors (array step 4), one shard of 40 on a Google machine: IAM-Atlas v2.1's 456 build arrays
# (atlas_donors_samples.csv) read by the one command, through run_series.py read exactly as series.yml runs it on GitHub (pass 1, no
# references), with --full-chain --cell-panel atlas --dev-line-wide. SOURCED by the launcher's job (W, R, B, log, up given; SHARD set). Every
# bundle is put in the bucket as it is written; a shard already done is not read again.
set -u; MP=$R/Biological_Physics/MethylPhys; F=donors_wide_s${SHARD}; O=$B/donors_wide
if gcloud storage ls $O/$F.done > /dev/null 2>&1; then log "already read: $F"; return 0 2>/dev/null || exit 0; fi
( cd $R && python3 -u $MP/chain/MethylPhys_Interface/run_series.py read $MP/doors/data/DEV_METAA_HEALTHY_01/atlas_donors_samples.csv $W/dw $SHARD 40 \
    --specimen "sorted cells" --pass 1 -- --full-chain --cell-panel atlas --dev-line-wide > $W/dw.log 2>&1; echo $? > $W/dw_rc ) &
while [ ! -f $W/dw_rc ]; do sleep 240; ls $W/dw/pass1/*.json > /dev/null 2>&1 && gcloud storage cp -q -n $W/dw/pass1/*.json $O/pass1/ > /dev/null 2>&1; done
ls $W/dw/pass1/*.json > /dev/null 2>&1 && gcloud storage cp -q -n $W/dw/pass1/*.json $O/pass1/ > /dev/null 2>&1; cp $W/dw.log $W/$F.log; gcloud storage cp -q $W/$F.log $O/ > /dev/null 2>&1
[ "$(cat $W/dw_rc)" = 0 ] && { echo done > $W/d; gcloud storage cp -q $W/d $O/$F.done > /dev/null 2>&1; }
log "donors wide shard $SHARD rc $(cat $W/dw_rc): $(tail -1 $W/dw.log | cut -c1-200)"
