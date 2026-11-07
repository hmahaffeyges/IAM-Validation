#!/bin/bash
# (xl_class, 2026-11-07: a copy with --dev-floor-class added, for the cross-laboratory rehearsal on spent data) A second outside laboratory's healthy CD8 T cells read by the one command (development; written and committed before any is read; see
# Biological_Physics/MethylPhys/doors/DEV_ATLAS_V21_01.md). Runs on a machine after the launcher's preamble ($W work folder, $R the repository,
# log and up from the preamble). Each sample: GSE245832 Bismark coverage (GRCh38) -> .lbeta on the hg19 numbering (--build seq-table) -> the
# full chain against the IAM Sequencing Atlas, with the same flags as the first outside-laboratory test.
# Usage (inside a machine job): bash lab2_cd8_read.sh SAMPLES.csv
set -u; L2=$(cd "$(dirname "$0")" && pwd); mkdir -p $W/out; S=$W/s; mkdir -p $S
U=https://github.com/hmahaffeyges/IAM-Validation/releases/download/hg19-bwameth-index-v1
curl -fsSL --retry 5 -o $S/idx_sums $U/SHA256SUMS && for q in $(awk '$2=="parts"{for(i=3;i<=NF;i++) print $i}' $S/idx_sums); do curl -fsSL --retry 5 "$U/$q" || exit 1; done \
  | tee >(sha256sum | cut -d' ' -f1 > $S/idx_got) | zstd -dc -T4 | tar -C $S -xf - --wildcards '*references/hg19/CpG.bed.gz'; sleep 3
[ "$(cat $S/idx_got)" = "$(awk '$2=="stream"{print $1}' $S/idx_sums)" ] || { log "index stream checksum FAILED"; exit 1; }
BED=$(find $S -name CpG.bed.gz | head -1); N=$(zcat $BED | wc -l); log "hg19 dictionary $N CpGs (stream checksum verified)"; [ "$N" = 28217448 ] || exit 1
cd $R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface
tail -n +2 "$1" | while IFS=',' read -r G DON CELL GRP SPEC FMT GEN URL; do
  ok=0; for t in 1 2 3 4 5 6; do curl -fsSL --retry 5 --retry-delay 20 -o $W/$G.cov.gz "$URL" && gzip -t $W/$G.cov.gz 2>/dev/null && { ok=1; break; }; rm -f $W/$G.cov.gz; sleep 60; done
  [ $ok = 1 ] || { log "$G download FAILED"; continue; }
  python3 run_sample.py --build seq-table --build-input $W/$G.cov.gz --build-out $W/$G.lbeta --build-samples "$FMT:$GEN:$BED" > $W/$G.t.log 2>&1 || { log "$G table FAILED: $(tail -2 $W/$G.t.log | tr '\n' ' ' | cut -c1-200)"; continue; }
  log "$G table: $(tail -1 $W/$G.t.log | cut -c1-200)"
  python3 run_sample.py --dev-seq-beta $W/$G.lbeta --dev-seq-values 1 --specimen "$SPEC" --id $G --full-chain --dev-seq-ref auto --dev-seq-level donors --dev-terr-counted \
    --dev-metaa-sites --out $W/out/${G}.html --bundle $W/out/${G}_bundle.json > $W/$G.r.log 2>&1
  up $W/out/${G}_bundle.json $W/out/${G}_metaa_sites.csv.gz $W/$G.t.log $W/$G.r.log; log "$G read: $(tail -1 $W/$G.r.log | cut -c1-150)"; rm -f $W/$G.cov.gz $W/$G.lbeta
done
