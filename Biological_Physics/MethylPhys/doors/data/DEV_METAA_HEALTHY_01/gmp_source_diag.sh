#!/bin/bash
# gmp_source_diag.py on a Google machine, one part of six (PART set by the launcher; W, R, B, log and up given by it). Saves as it goes.
set -u; D=$R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01; O=$B/gmp_source_diag; mkdir -p $W/vals
if [ "${CURVE:-0}" = 1 ]; then O=$B/${CURVE_OUT:-gmp_curve_diag}; gcloud storage ls "$O/gmp_curve_diag_p${PART}.json.done" > /dev/null 2>&1 && { log "already read"; exit 0; }
  gcloud storage cp -q "$O/gmp_curve_diag_p${PART}.json" $W/ > /dev/null 2>&1; for g in $(tail -n +2 $D/gmp_source_donors.csv | cut -d, -f1); do gcloud storage cp -q "$(dirname $B)/rehearsal/bv2_donors/betas/${g}_betas.parquet" $W/vals/ > /dev/null 2>&1; done
  ( cd $R && python3 -u $D/gmp_source_diag.py $W/vals $W/gmp_curve_diag_p${PART}.json $PART curve > $W/gc.log 2>&1; echo $? > $W/gc_rc ) &
  while [ ! -f $W/gc_rc ]; do sleep 120; [ -f $W/gmp_curve_diag_p${PART}.json ] && gcloud storage cp -q $W/gmp_curve_diag_p${PART}.json $O/ > /dev/null 2>&1; log "c$PART: $(tail -1 $W/gc.log | cut -c1-80)"; done
  rc=$(cat $W/gc_rc); gcloud storage cp -q $W/gmp_curve_diag_p${PART}.json $O/ > /dev/null 2>&1; log "curve rc $rc: $(tail -2 $W/gc.log | tr '\n' ' ' | cut -c1-200)"; up $W/gc.log
  [ "$rc" = 0 ] && echo done | gcloud storage cp -q - $O/gmp_curve_diag_p${PART}.json.done; exit 0; fi
if [ "${ERR:-0}" = 1 ]; then gcloud storage cp -q "$O/gmp_source_diag_p${PART}.json" $W/ > /dev/null 2>&1; for g in $(tail -n +2 $D/gmp_source_donors.csv | cut -d, -f1); do gcloud storage cp -q "$(dirname $B)/rehearsal/bv2_donors/betas/${g}_betas.parquet" $W/vals/ > /dev/null 2>&1; done
  cd $R && python3 -u $D/gmp_source_diag.py $W/vals $W/gmp_source_diag_p${PART}.json $PART errors > $W/gse.log 2>&1; rc=$?; gcloud storage cp -q $W/gmp_source_diag_p${PART}.json $O/ > /dev/null 2>&1; log "errors rc $rc: $(tail -1 $W/gse.log | cut -c1-150)"; up $W/gse.log; exit 0; fi
if gcloud storage ls "$O/gmp_source_diag_p${PART}.json.done" > /dev/null 2>&1; then log "already read: in the bucket"; exit 0; fi
gcloud storage cp -q "$O/gmp_source_diag_p${PART}.json" $W/ > /dev/null 2>&1 && log "already read (part of it)"
for g in $(tail -n +2 $D/gmp_source_donors.csv | cut -d, -f1); do gcloud storage cp -q "$(dirname $B)/rehearsal/bv2_donors/betas/${g}_betas.parquet" $W/vals/ > /dev/null 2>&1; done; log "values fetched: $(ls $W/vals | wc -l)"
( cd $R && python3 -u $D/gmp_source_diag.py $W/vals $W/gmp_source_diag_p${PART}.json $PART > $W/gs.log 2>&1; echo $? > $W/gs_rc ) &
while [ ! -f $W/gs_rc ]; do sleep 120; [ -f $W/gmp_source_diag_p${PART}.json ] && gcloud storage cp -q $W/gmp_source_diag_p${PART}.json $O/ > /dev/null 2>&1; log "p$PART: $(tail -1 $W/gs.log | cut -c1-80)"; done
rc=$(cat $W/gs_rc); gcloud storage cp -q $W/gmp_source_diag_p${PART}.json $O/ > /dev/null 2>&1; log "rc $rc: $(tail -2 $W/gs.log | tr '\n' ' ' | cut -c1-200)"; up $W/gs.log
[ "$rc" = 0 ] && echo done | gcloud storage cp -q - $O/gmp_source_diag_p${PART}.json.done
