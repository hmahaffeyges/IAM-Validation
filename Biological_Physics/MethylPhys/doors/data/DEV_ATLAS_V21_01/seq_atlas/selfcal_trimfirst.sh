#!/bin/bash
# The self-calibration diagnosis, trim first (2026-11-12; diagnosis, nothing scored as a form): row ROW of selfcal_synthetic_samples.csv read
# by the one command with its planted errors (--dev-plant-ends, TESTS ONLY; unplanted rows planted with 0,0,0 so every row's file is built
# the same way), then the outermost call at each molecule end left out before ANY reading, the Deconvolver included (--dev-seq-trim-ends 1),
# against the atlas rebuilt the same way (atlas_trim/atlas57t1, --dev-seq-atlas-dir). The plants sit on the outermost call, so if the
# seventh form's misses came through the Deconvolver, planted rows now read as their unplanted row. Already read are skipped.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; SDIR=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas
ROOT=${B%%/seq_atlas/*}; O=$B/selfcal_trimfirst; mkdir -p $W/sc $W/atl
gcloud storage ls "$O/" 2>/dev/null | sed 's#.*/##' > $W/done.txt
python3 -c "import csv,sys; r=list(csv.DictReader(open('$SDIR/selfcal_synthetic_samples.csv')))[$ROW]; print('|'.join([r['gsm'],r['specimen'],r['pat'],r['cond'],r['plant']]))" > $W/row.txt
IFS='|' read -r G SPEC PAT C PL < $W/row.txt; N=${G}_$C
if grep -qx "$N.json" $W/done.txt; then log "already read: $N"; else
  gcloud storage cp -q $ROOT/seq_atlas/fpos/atlas_trim/atlas57t1/array_sites.tgz $ROOT/seq_atlas/fpos/atlas_trim/atlas57t1/deconv_reference_v1.parquet $W/atl/ > /dev/null 2>&1 \
    && (cd $W/atl && tar xzf array_sites.tgz) || { log "rebuilt atlas not copied"; exit 1; }
  gcloud storage cp -q "$ROOT/$PAT" $W/sc/$G.pat.gz > /dev/null 2>&1 || { log "$G: pat not copied"; exit 1; }
  [ -n "$PL" ] || PL="0,0,0"
  (cd $IF && python3 run_sample.py --pat $W/sc/$G.pat.gz --dev-seq-values 1 --dev-plant-ends $PL --dev-seq-trim-ends 1 --dev-seq-atlas-dir $W/atl --specimen "$SPEC" --id $N --out $W/sc/$N.html --bundle $W/sc/$N.json --no-info-sphere > $W/sc/$N.log 2>&1) \
    && gcloud storage cp -q $W/sc/$N.json $O/ > /dev/null 2>&1 && log "read $N" || log "$N FAILED: $(tail -3 $W/sc/$N.log | tr '\n' ' ' | cut -c1-300)"
fi
log "row $ROW done"
