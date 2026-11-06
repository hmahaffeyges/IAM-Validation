#!/bin/bash
# Plasma from one laboratory, healthy and cancer (GSE186007: 34 healthy, 50 colorectal, 16 pancreatic cancer plasmas; written and committed
# before any is read; record Biological_Physics/MethylPhys/doors/DEV_ONE_DRAW_01.md). Each plasma: its Bismark coverage file on hg19 -> .lbeta
# on the hg19 numbering (--build seq-table, no alignment) -> the full chain with the plasma passes' flags (the list's rsflags column). The same
# steps as lab2_cd8_read.sh. Each finished plasma is uploaded at once, and one already in the bucket is skipped, so a restart resumes.
# Usage (inside a machine job, after the launcher's preamble: $W, $R, $B, log, up): bash plasma_lab3_read.sh SAMPLES.csv [FIRST LAST]
set -u; mkdir -p $W/out; S=$W/s; mkdir -p $S
U=https://github.com/hmahaffeyges/IAM-Validation/releases/download/hg19-bwameth-index-v1
curl -fsSL --retry 5 -o $S/idx_sums $U/SHA256SUMS && for q in $(awk '$2=="parts"{for(i=3;i<=NF;i++) print $i}' $S/idx_sums); do curl -fsSL --retry 5 "$U/$q" || exit 1; done \
  | tee >(sha256sum | cut -d' ' -f1 > $S/idx_got) | zstd -dc -T4 | tar -C $S -xf - --wildcards '*references/hg19/CpG.bed.gz'; sleep 3
[ "$(cat $S/idx_got)" = "$(awk '$2=="stream"{print $1}' $S/idx_sums)" ] || { log "index stream checksum FAILED"; exit 1; }
BED=$(find $S -name CpG.bed.gz | head -1); N=$(zcat $BED | wc -l); log "hg19 dictionary $N CpGs (stream checksum verified)"; [ "$N" = 28217448 ] || exit 1
cd $R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface
A=${2:-1}; Z=${3:-100000}
tail -n +2 "$1" | sed -n "${A},${Z}p" | while IFS=',' read -r G T GRP AGE SEX SPEC FMT GEN URL FLAGS; do
  gcloud storage ls "$B/out/${G}_bundle.json" > /dev/null 2>&1 && { log "$G already read"; continue; }
  ok=0; for t in 1 2 3 4 5 6; do curl -fsSL --retry 5 --retry-delay 20 -o $W/$G.cov.gz "$URL" && gzip -t $W/$G.cov.gz 2>/dev/null && { ok=1; break; }; rm -f $W/$G.cov.gz; sleep 60; done
  [ $ok = 1 ] || { log "$G download FAILED"; continue; }
  python3 run_sample.py --build seq-table --build-input $W/$G.cov.gz --build-out $W/$G.lbeta --build-samples "$FMT:$GEN:$BED" > $W/$G.t.log 2>&1 || { log "$G table FAILED: $(tail -2 $W/$G.t.log | tr '\n' ' ' | cut -c1-200)"; continue; }
  python3 run_sample.py --dev-seq-beta $W/$G.lbeta --dev-seq-values 1 --specimen "$SPEC" --id $G $FLAGS --out $W/out/${G}.html --bundle $W/out/${G}_bundle.json > $W/$G.r.log 2>&1
  for x in $W/out/${G}_bundle.json $W/$G.t.log $W/$G.r.log; do [ -f "$x" ] && gcloud storage cp -q $x $B/out/ > /dev/null 2>&1; done
  log "$G read: $(tail -1 $W/$G.r.log | cut -c1-150)"; rm -f $W/$G.cov.gz $W/$G.lbeta
done
