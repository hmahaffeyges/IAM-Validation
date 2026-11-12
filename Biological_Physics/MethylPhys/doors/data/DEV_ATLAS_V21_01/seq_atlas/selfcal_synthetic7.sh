#!/bin/bash
# The self-calibration synthetic, seventh form (2026-11-12; the author's distance-relay zones: the library and the atlas both read past each side's reach along the molecule, marked side past one call, unmarked side past three; nothing corrected): row ROW of selfcal_synthetic_samples.csv (one spent sorted
# library from each of three laboratories, its merged molecule file already in the build bucket), read by the one command with
# --dev-seq-self-cal, and with the row's planted errors (--dev-plant-ends, TESTS ONLY) when it has any. Each bundle is put in the bucket as it
# is written; already read are skipped.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; SDIR=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas
ROOT=${B%%/seq_atlas/*}; O=$B/selfcal_synthetic7; mkdir -p $W/sc
gcloud storage ls "$O/" 2>/dev/null | sed 's#.*/##' > $W/done.txt
python3 -c "import csv,sys; r=list(csv.DictReader(open('$SDIR/selfcal_synthetic_samples.csv')))[$ROW]; print('|'.join([r['gsm'],r['specimen'],r['pat'],r['cond'],r['plant']]))" > $W/row.txt
IFS='|' read -r G SPEC PAT C PL < $W/row.txt; N=${G}_$C
if grep -qx "$N.json" $W/done.txt; then log "already read: $N"; else
  gcloud storage cp -q "$ROOT/$PAT" $W/sc/$G.pat.gz > /dev/null 2>&1 || { log "$G: pat not copied"; exit 1; }
  P=""; [ -n "$PL" ] && P="--dev-plant-ends $PL"
  (cd $IF && python3 run_sample.py --pat $W/sc/$G.pat.gz --dev-seq-values 1 --dev-seq-self-cal --dev-seq-self-cal-reach 3 $P --specimen "$SPEC" --id $N --out $W/sc/$N.html --bundle $W/sc/$N.json --no-info-sphere > $W/sc/$N.log 2>&1) \
    && gcloud storage cp -q $W/sc/$N.json $O/ > /dev/null 2>&1 && log "read $N" || log "$N FAILED: $(tail -3 $W/sc/$N.log | tr '\n' ' ' | cut -c1-300)"
fi
log "row $ROW done"
