#!/bin/bash
# Arrays read again by the one command from their kept values (2026-11-09; development, spent, no download, no recalibration): rows FIRST to
# LAST of LISTF (gsm, lab, values = the bucket key of the array's kept Stage 1 values, specimen), each read with the default chain
# (run_sample.py --betas; the information sphere off for speed) and its bundle put in the bucket under OUTD as it is written; already read are skipped.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; ROOT=${B%%/seq_atlas/*}; O=$B/${OUTD:-values_read}; mkdir -p $W/vr
gcloud storage ls "$O/" 2>/dev/null | sed 's#.*/##' > $W/done.txt
sed -n "$((FIRST + 1)),$((LAST + 1))p" $R/$LISTF > $W/rows.csv
while IFS=, read -r G LAB VK SPEC; do
  grep -qx "$G.json" $W/done.txt && { echo "already read: $G"; log "already read: $G"; continue; }
  gcloud storage cp -q "$ROOT/$VK" $W/vr/$G.parquet > /dev/null 2>&1 || { log "$G: values not copied"; continue; }
  python3 -c "import pandas as pd, sys; X = pd.read_parquet(sys.argv[1]); x = X.iloc[:, 0] if X.shape[1] == 1 else X.set_index(X.columns[0]).iloc[:, 0]; x.rename('beta').to_frame().to_csv(sys.argv[2], index_label='probe')" $W/vr/$G.parquet $W/vr/$G.csv
  (cd $IF && python3 run_sample.py --betas $W/vr/$G.csv --specimen "$SPEC" --id $G --out $W/vr/$G.html --bundle $W/vr/$G.json --no-info-sphere > $W/vr/$G.log 2>&1) \
    && gcloud storage cp -q $W/vr/$G.json $O/ > /dev/null 2>&1 && log "read $G ($LAB)" || log "$G FAILED: $(tail -2 $W/vr/$G.log | tr '\n' ' ' | cut -c1-200)"
  rm -f $W/vr/$G.parquet $W/vr/$G.csv $W/vr/$G.html
done < $W/rows.csv
log "rows $FIRST to $LAST done"
