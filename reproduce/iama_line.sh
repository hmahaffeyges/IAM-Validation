#!/bin/bash
# IAM-A's self-reference line rebuilt from public files (the clean room PHY-055, as a reproduce step). Counts Loyfer's 153 healthy-cell
# samples streamed from GEO and five ENCODE whole-genome bisulfite files (ENCODE's public bucket, over HTTPS) through chain Stage Q2, on
# ENCODE's GRCh38 analysis set, then compares the counts, the line and the shape reference with the committed ones.
# Run by reproduce.py (REPO, WORK, PY, MP set). Workers: $NW (default: all cores).
set -euo pipefail; NW=${NW:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}; Q=$MP/doors/data/DEV_CHAIN_Q2_01
mkdir -p $WORK/q2 && cd $WORK/q2 && rm -rf chain && cp -r $MP/chain ./chain && cp $Q/box_q2_line.py $Q/loyfer_samples.csv $Q/encode_samples.tsv .
$PY box_q2_line.py loyfer $NW
[ -s ref.fa ] || { curl -sfL --retry 5 -o ref.fa.gz "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz" && gunzip -f ref.fa.gz; }
$PY -c "import pysam; pysam.faidx('ref.fa')"
tail -n +2 encode_samples.tsv | while IFS=$'\t' read cell lc exp acc url; do
  [ -s out/$acc.json ] && continue
  curl -sfL --retry 5 -o $acc.bam "https://encode-public.s3.amazonaws.com/${url#s3://encode-public/}" && $PY -c "import pysam; pysam.index('$acc.bam')"
  head -1 encode_samples.tsv > one.tsv; grep -P "\t$acc\t" encode_samples.tsv >> one.tsv; cp encode_samples.tsv all.tsv; cp one.tsv encode_samples.tsv
  OMP_NUM_THREADS=1 $PY box_q2_line.py encode $NW; cp all.tsv encode_samples.tsv; rm -f $acc.bam $acc.bam.bai
done
$PY $MP/doors/data/CLEANROOM_IAMA_01/cleanroom_iama.py compare $REPO out
