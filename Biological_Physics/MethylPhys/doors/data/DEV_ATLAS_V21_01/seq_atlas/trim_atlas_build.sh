#!/bin/bash
# The IAM Sequencing Atlas rebuilt from its donors' trimmed counts (2026-11-09; development): the 156 donors of seq_atlas_inputs.csv (57 cells,
# the 3 granulocyte donors among them), each counted with one call left out at each molecule end (atlas_trim_donors.sh, $B/atlas_trim/t1/),
# built by the one command exactly as the published atlas (run_sample.py --build seq-atlas, pieces of 250,000 CpGs), every cell in one folder.
# PART=k of NPARTS: builds the pieces k, k+NPARTS, ... into $B/atlas_trim/pieces57t1/ (a piece already there is skipped).
# PART=final: from all 113 pieces, the in-mix columns at IAM-Atlas v2.1's sites (--build seq-array-cols) and the Deconvolver's reference
# (--build seq-deconv-ref), put in $B/atlas_trim/atlas57t1/ as array_sites.tgz and deconv_reference_v1.parquet.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; SDIR=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas
O=$B/atlas_trim; HG19=28217448; PIECE=250000; mkdir -p $W/lb $W/pc
if [ "$PART" = final ]; then
  gcloud storage cp -q "$O/pieces57t1/*.parquet" $W/pc/ > /dev/null 2>&1; n=$(ls $W/pc/*.parquet | wc -l); log "pieces: $n"
  [ "$n" -eq 113 ] || { log "only $n of 113 pieces; not built"; exit 1; }
  (cd $IF && python3 run_sample.py --build seq-array-cols --build-samples $W/pc --build-out $W/atlas57t1/array_sites --bundle $W/array_cols.json > $W/ac.log 2>&1) || { log "array cols FAILED: $(tail -2 $W/ac.log)"; exit 1; }
  (cd $IF && python3 run_sample.py --build seq-deconv-ref --build-samples $W/pc --build-out $W/atlas57t1/deconv_reference_v1.parquet --bundle $W/deconv_ref.json > $W/dr.log 2>&1) || { log "deconv ref FAILED: $(tail -2 $W/dr.log)"; exit 1; }
  (cd $W/atlas57t1 && tar czf ../array_sites.tgz array_sites)
  gcloud storage cp -q $W/array_sites.tgz $W/atlas57t1/deconv_reference_v1.parquet $W/array_cols.json $W/deconv_ref.json $O/atlas57t1/ > /dev/null 2>&1 \
    && log "built: $(tail -1 $W/ac.log | cut -c1-150) | $(tail -1 $W/dr.log | cut -c1-150)"; log "rows final to final done"; exit 0
fi
gcloud storage cp -q "$O/t1/*.lbeta" $W/lb/ > /dev/null 2>&1; nl=$(ls $W/lb/*.lbeta | wc -l); log "trimmed count files: $nl"
[ "$nl" -eq 156 ] || { log "only $nl of 156 donors counted; not built"; exit 1; }
python3 - $SDIR/seq_atlas_inputs.csv $SDIR/gran_inputs.csv $W/lb $W/inputs_t1.csv <<'PY'
import sys, os, pandas as pd
S = pd.concat([pd.read_csv(sys.argv[1], dtype=str), pd.read_csv(sys.argv[2], dtype=str)], ignore_index=True).drop_duplicates("gsm"); assert len(S) == 156 and S.cell.nunique() == 57, (len(S), S.cell.nunique())
S["file"] = [os.path.join(sys.argv[3], g + ".lbeta") for g in S.gsm]; assert all(os.path.exists(f) for f in S.file); S.to_csv(sys.argv[4], index=False); print(len(S), S.cell.nunique())
PY
gcloud storage ls "$O/pieces57t1/" 2>/dev/null | sed 's#.*/##' > $W/done.txt; k=0
for P0 in $(seq 0 $PIECE $((HG19 - 1))); do
  if [ $((k % NPARTS)) -eq $PART ]; then
    P1=$((P0 + PIECE)); [ $P1 -gt $HG19 ] && P1=$HG19; N=piece_$(printf %08d $P0)_$(printf %08d $P1)
    if grep -qx "$N.parquet" $W/done.txt; then echo "already read: $N"; log "already read: $N"
    else (cd $IF && python3 run_sample.py --build seq-atlas --build-samples $W/inputs_t1.csv --build-block $P0:$P1 --build-out $W/$N.parquet --bundle $W/$N.json > $W/$N.log 2>&1) \
      && gcloud storage cp -q $W/$N.parquet $W/$N.json $O/pieces57t1/ > /dev/null 2>&1 && log "built $N" || log "$N FAILED: $(tail -2 $W/$N.log | tr '\n' ' ' | cut -c1-200)"; rm -f $W/$N.parquet; fi
  fi; k=$((k + 1))
done
log "rows $PART to $PART done"
