#!/bin/bash
# DEV-ONE-DRAW-01, GSE186007: each plasma's dispersion index at the library-floor sites (plasma_lab3_copies.py one), from its published
# coverage file converted by the chain's own --build seq-table, one plasma per GitHub job (cov_read.yml; its result is the job's artifact).
# Usage: plasma_lab3_copies.sh LIST FIRST LAST (rows of the list; W, R and CPG_BED set by the workflow)
set -u; mkdir -p $W/out; H=$R/Biological_Physics/MethylPhys/doors/data/DEV_ONE_DRAW_01/plasma_lab3; BED=${CPG_BED}
cd $R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface
A=${2:-1}; Z=${3:-100000}
tail -n +2 "$1" | sed -n "${A},${Z}p" | while IFS=',' read -r G T GRP AGE SEX SPEC FMT GEN URL FLAGS; do
  [ "$GRP" = HEA ] || { echo "$G is not a healthy plasma: not read"; continue; }
  ok=0; for t in 1 2 3 4 5 6; do curl -fsSL --retry 5 --retry-delay 20 -o $W/$G.cov.gz "$URL" && gzip -t $W/$G.cov.gz 2>/dev/null && { ok=1; break; }; rm -f $W/$G.cov.gz; sleep 60; done
  [ $ok = 1 ] || { echo "$G download FAILED"; continue; }
  python3 run_sample.py --build seq-table --build-input $W/$G.cov.gz --build-out $W/$G.lbeta --build-samples "$FMT:$GEN:$BED" > $W/$G.t.log 2>&1 || { echo "$G table FAILED"; tail -3 $W/$G.t.log; continue; }
  python3 $H/plasma_lab3_copies.py one $W/$G.lbeta $G $W/out/${G}_copies.json
  rm -f $W/$G.cov.gz $W/$G.lbeta
done
