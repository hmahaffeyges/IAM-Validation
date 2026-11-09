#!/bin/bash
# The 18 sorted libraries read by the one command with the molecules' outer calls left out (2026-11-09; development, spent, their merged
# molecule files already in the build bucket, no alignment): rows FIRST to LAST of trim_read_samples.csv, each read with --dev-seq-trim-ends
# for each N in TRIMS (0 = as read before), the default chain otherwise. Each bundle is put in the bucket as it is written; already read are skipped.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; SDIR=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas
ROOT=${B%%/seq_atlas/*}; O=$B/trim_read; mkdir -p $W/tr; gcloud storage ls "$O/" 2>/dev/null | sed 's#.*/##' > $W/done.txt
sed -n "$((FIRST + 1)),$((LAST + 1))p" $SDIR/trim_read_samples.csv > $W/rows.csv
while IFS=, read -r G SPEC PAT; do
  for N in ${TRIMS:-0 1 2}; do
    grep -qx "${G}_t$N.json" $W/done.txt && { echo "already read: ${G}_t$N"; log "already read: ${G}_t$N"; continue; }
    [ -s $W/tr/$G.pat.gz ] || gcloud storage cp -q "$ROOT/$PAT" $W/tr/$G.pat.gz > /dev/null 2>&1 || { log "$G: pat not copied"; break; }
    T=""; [ "$N" -gt 0 ] && T="--dev-seq-trim-ends $N"
    (cd $IF && python3 run_sample.py --pat $W/tr/$G.pat.gz --dev-seq-values 1 --specimen "$SPEC" --id ${G}_t$N --out $W/tr/${G}_t$N.html --bundle $W/tr/${G}_t$N.json --no-info-sphere $T > $W/tr/${G}_t$N.log 2>&1) \
      && gcloud storage cp -q $W/tr/${G}_t$N.json $O/ > /dev/null 2>&1 && log "read ${G}_t$N" || log "${G}_t$N FAILED: $(tail -2 $W/tr/${G}_t$N.log | tr '\n' ' ' | cut -c1-200)"
  done; rm -f $W/tr/$G.pat.gz
done < $W/rows.csv
log "rows $FIRST to $LAST done"
