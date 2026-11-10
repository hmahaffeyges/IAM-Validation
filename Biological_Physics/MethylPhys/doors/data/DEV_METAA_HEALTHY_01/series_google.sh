#!/bin/bash
# A series of arrays read by the one command on a Google machine, one shard (2026-11-08; donors_wide.sh made general for the rehearsal, array
# step 6): run_series.py read exactly as series.yml runs it on GitHub (pass 1, no references), with --full-chain --cell-panel atlas
# --dev-line-wide. SOURCED by the launcher's job (W, R, B, log, up given; LIST, TAG, SPEC, NSH, SHARD set). Every bundle is put in the bucket as
# it is written (B/TAG/pass1/); PLANT, when set, is passed as --plant (2026-11-10: neutrophil_atlas for Group-A); with KEEP=1 each array's calibrated values too (B/TAG/betas/); a shard already done is not read again.
set -u; MP=$R/Biological_Physics/MethylPhys; F=${TAG}_s${SHARD}; O=$B/$TAG
if gcloud storage ls $O/$F.done > /dev/null 2>&1; then log "already read: $F"; return 0 2>/dev/null || exit 0; fi
( cd $R && python3 -u $MP/chain/MethylPhys_Interface/run_series.py read $R/$LIST $W/sg $SHARD $NSH --specimen "$SPEC" --pass 1 $([ "${KEEP:-0}" = 1 ] && echo "--keep-values $W/keep") $([ -n "${PLANT:-}" ] && echo "--plant $PLANT") -- --full-chain --cell-panel atlas --dev-line-wide > $W/sg.log 2>&1; echo $? > $W/sg_rc ) &
while [ ! -f $W/sg_rc ]; do sleep 240; ls $W/sg/pass1/*.json > /dev/null 2>&1 && gcloud storage cp -q -n $W/sg/pass1/*.json $O/pass1/ > /dev/null 2>&1; [ "${KEEP:-0}" = 1 ] && ls $W/keep/*_betas.parquet > /dev/null 2>&1 && gcloud storage cp -q -n $W/keep/*_betas.parquet $O/betas/ > /dev/null 2>&1; log "$F: $(ls $W/sg/pass1/*.json 2>/dev/null | wc -l | tr -d ' ') bundles"; done
ls $W/sg/pass1/*.json > /dev/null 2>&1 && gcloud storage cp -q -n $W/sg/pass1/*.json $O/pass1/ > /dev/null 2>&1
[ "${KEEP:-0}" = 1 ] && ls $W/keep/*_betas.parquet > /dev/null 2>&1 && gcloud storage cp -q -n $W/keep/*_betas.parquet $O/betas/ > /dev/null 2>&1   # 2026-11-08: KEEP=1 keeps each array's calibrated values (for building bloods from the atlas's donors)
cp $W/sg.log $W/$F.log; gcloud storage cp -q $W/$F.log $O/ > /dev/null 2>&1
[ "$(cat $W/sg_rc)" = 0 ] && { echo done > $W/d; gcloud storage cp -q $W/d $O/$F.done > /dev/null 2>&1; }
log "$TAG shard $SHARD rc $(cat $W/sg_rc): $(tail -1 $W/sg.log | cut -c1-200)"
