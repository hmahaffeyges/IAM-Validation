#!/bin/bash
# The IAM Sequencing Atlas's inner level past an N-call reach (2026-11-12; development; the seventh self-calibration form): the 156 donors of
# seq_atlas_inputs.csv and gran_inputs.csv, each counted with N calls left out at each molecule end (atlas_trim_donors.sh TRIMS=N, $B/atlas_trim/tN/),
# then each donor's counts at its cell's Met-A identity sites (run_sample.py --build seq-metaa-donors) and the inner reference
# (run_sample.py --build seq-metaa-inner), exactly as metaa_inner_reference_v1 was built from the one-call counts. Writes
# $B/atlas_trim/innerN/metaa_inner_reference_tN.parquet and its record.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; SDIR=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas
O=$B/atlas_trim; mkdir -p $W/lb
gcloud storage cp -q "$O/t$N/*.lbeta" $W/lb/ > /dev/null 2>&1; nl=$(ls $W/lb/*.lbeta | wc -l); log "count files past $N calls: $nl"
[ "$nl" -eq 156 ] || { log "only $nl of 156 donors counted; not built"; exit 1; }
python3 - $SDIR/seq_atlas_inputs.csv $SDIR/gran_inputs.csv $W/lb $W/inputs_t$N.csv <<'PY'
import sys, os, pandas as pd
S = pd.concat([pd.read_csv(sys.argv[1], dtype=str), pd.read_csv(sys.argv[2], dtype=str)], ignore_index=True).drop_duplicates("gsm"); assert len(S) == 156 and S.cell.nunique() == 57, (len(S), S.cell.nunique())
S["file"] = [os.path.join(sys.argv[3], g + ".lbeta") for g in S.gsm]; assert all(os.path.exists(f) for f in S.file); S.to_csv(sys.argv[4], index=False); print(len(S), S.cell.nunique())
PY
(cd $IF && python3 run_sample.py --build seq-metaa-donors --build-samples $W/inputs_t$N.csv --build-out $W/donors_t$N.parquet --bundle $W/donors_t$N.json > $W/dn.log 2>&1) || { log "donor counts FAILED: $(tail -2 $W/dn.log | tr '\n' ' ' | cut -c1-200)"; exit 1; }
(cd $IF && python3 run_sample.py --build seq-metaa-inner --build-input $W/donors_t$N.parquet --build-out $W/metaa_inner_reference_t$N.parquet --bundle $W/metaa_inner_reference_t$N.json > $W/in.log 2>&1) || { log "inner FAILED: $(tail -2 $W/in.log | tr '\n' ' ' | cut -c1-200)"; exit 1; }
gcloud storage cp -q $W/metaa_inner_reference_t$N.parquet $W/metaa_inner_reference_t$N.json $O/inner$N/ > /dev/null 2>&1 && log "built: $(tail -1 $W/in.log | cut -c1-200)"; log "inner $N done"
