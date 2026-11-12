#!/bin/bash
# The eighth self-calibration form on the 18 spent sorted libraries of the trim reads (trim_read_samples.csv; three laboratories; no plants)
# (2026-11-12; development, the real-data side of bar 4): row ROW read by the one command with --dev-seq-self-cal --dev-seq-self-cal-reach 3
# --dev-seq-end-cal. Question: do the three laboratories' libraries of one cell type now read alike? Each bundle is put in the bucket as it
# is written; already read are skipped.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; SDIR=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas
ROOT=${B%%/seq_atlas/*}; O=$B/selfcal8_real; mkdir -p $W/sc
gcloud storage ls "$O/" 2>/dev/null | sed 's#.*/##' > $W/done.txt
python3 -c "import csv,sys; r=list(csv.DictReader(open('$SDIR/trim_read_samples.csv')))[$ROW]; print('|'.join([r['gsm'],r['specimen'],r['pat']]))" > $W/row.txt
IFS='|' read -r G SPEC PAT < $W/row.txt; N=$G
if grep -qx "$N.json" $W/done.txt; then log "already read: $N"; else
  gcloud storage cp -q "$ROOT/$PAT" $W/sc/$G.pat.gz > /dev/null 2>&1 || { log "$G: pat not copied"; exit 1; }
  (cd $IF && python3 run_sample.py --pat $W/sc/$G.pat.gz --dev-seq-values 1 --dev-seq-self-cal --dev-seq-self-cal-reach 3 --dev-seq-end-cal --specimen "$SPEC" --id $N --out $W/sc/$N.html --bundle $W/sc/$N.json --no-info-sphere > $W/sc/$N.log 2>&1) \
    && gcloud storage cp -q $W/sc/$N.json $O/ > /dev/null 2>&1 && log "read $N" || log "$N FAILED: $(tail -3 $W/sc/$N.log | tr '\n' ' ' | cut -c1-300)"
fi
log "row $ROW done"
