#!/bin/bash
# The IAM Sequencing Atlas's donors counted from their molecules with the trim a sample is read with (2026-11-09; development): rows FIRST to
# LAST of seq_atlas_inputs.csv; each donor's hg19 molecule file (.pat.gz, beside its published .beta on GEO) is fetched and counted by the one
# command (run_sample.py --build seq-pat-lbeta) for each N in TRIMS into $B/atlas_trim/t$N/<gsm>.lbeta, put in the bucket as it is written;
# already counted are skipped. CHECK=1 also compares the N=0 count with the published .beta (equal where that file's count is below 255).
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; SDIR=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas
O=$B/atlas_trim; mkdir -p $W/at; gcloud storage ls -r "$O/" 2>/dev/null | grep lbeta | sed 's#.*/atlas_trim/##' > $W/done.txt
sed -n "$((FIRST + 1)),$((LAST + 1))p" $SDIR/${INF:-seq_atlas_inputs.csv} > $W/rows.csv   # INF: gran_inputs.csv for the granulocytes
while IFS=, read -r G TITLE CELL HALF FILE; do
  for N in ${TRIMS:-1}; do
    grep -qx "t$N/$G.lbeta" $W/done.txt && { echo "already read: $G t$N"; log "already read: $G t$N"; continue; }
    T=""; [ "$N" -gt 0 ] && T="--dev-seq-trim-ends $N"
    (cd $IF && python3 run_sample.py --build seq-pat-lbeta --geo-pat $G $T --build-out $W/at/${G}_t$N.lbeta --bundle $W/at/${G}_t$N.json > $W/at/${G}_t$N.log 2>&1) \
      && gcloud storage cp -q $W/at/${G}_t$N.lbeta $O/t$N/$G.lbeta > /dev/null 2>&1 && gcloud storage cp -q $W/at/${G}_t$N.json $O/t$N/$G.json > /dev/null 2>&1 \
      && log "counted $G t$N: $(tail -1 $W/at/${G}_t$N.log | cut -c1-160)" || log "$G t$N FAILED: $(tail -2 $W/at/${G}_t$N.log | tr '\n' ' ' | cut -c1-200)"
    if [ "${CHECK:-0}" = 1 ] && [ "$N" = 0 ]; then
      curl -sfL --retry 6 -o $W/at/$G.beta "$FILE" && python3 - $W/at/${G}_t0.lbeta $W/at/$G.beta > $W/at/check.txt 2>&1 <<'PY'
import sys, numpy as np
a = np.fromfile(sys.argv[1], np.uint16).reshape(-1, 2).astype(np.int64); b = np.fromfile(sys.argv[2], np.uint8).reshape(-1, 2).astype(np.int64)
ok = a[:, 1] < 255; d = np.abs(a[ok] - b[ok]).max(axis=1)
print(f"CpGs {len(a):,} vs {len(b):,}; below the 255 cap {int(ok.sum()):,}; differing {int((d > 0).sum()):,}; largest difference {int(d.max())}")
PY
      log "CHECK $G: $(cat $W/at/check.txt | tail -1)"; gcloud storage cp -q $W/at/check.txt $O/check_$G.txt > /dev/null 2>&1
    fi
    rm -f $W/at/${G}_t$N.lbeta
  done; rm -f $W/at/$G.beta
done < $W/rows.csv
log "rows $FIRST to $LAST done"
