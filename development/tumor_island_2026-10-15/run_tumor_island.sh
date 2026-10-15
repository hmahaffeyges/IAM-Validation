#!/bin/bash
# DEV-BACKBONE-01 island arm, real counts (2026-10-15; author approved). The six early-onset colorectal pairs (GSE284325, PRJNA1198593),
# patients 1-6, tumor and adjacent normal: first 10 million read-1 reads per sample streamed from ENA, Trim Galore, Bismark directional
# (bowtie2, the original hg38 index and tools from s3 downloads/tumour_work/tools.tgz), extract_island.py. One sample at a time.
set -uo pipefail
W=~/tum; T=$W/mm/bin; cd $W
if [ ! -x $T/bismark ]; then cat tools.tgz.part_0* | tar xzf - && rm -f tools.tgz.part_0*; fi
export PATH=$T:$PATH; bismark --version | sed -n 2p
OUT=$W/out; mkdir -p $OUT
while IFS=$'\t' read -r name srr url; do
  [ -s $OUT/$name.parquet ] && { echo "$name cached"; continue; }
  t0=$(date +%s); mkdir -p $W/w && cd $W/w && rm -rf ./*
  curl -s --retry 5 "https://$url" | pigz -dc | head -n 40000000 | pigz -p 4 > r1.fq.gz
  trim_galore -q 20 --cores 4 --gzip r1.fq.gz > trim.log 2>&1
  bismark --genome $W/ref --score_min L,0,-0.2 --parallel 6 -p 2 --bowtie2 -o . r1_trimmed.fq.gz > bismark.log 2>&1
  python $W/extract_island.py r1_trimmed_bismark_bt2.bam $W/ref/hg38.fa $OUT/$name.parquet parity > $OUT/${name}_extract.log 2>&1
  cp *_SE_report.txt $OUT/${name}_bismark_report.txt 2>/dev/null
  ~/env/bin/python -c "import boto3;s=boto3.client('s3');[s.upload_file('$OUT/$name'+x,'methylphys-data-945451304272-us-west-2-an','results/DEV_BACKBONE_01/island/$name'+x) for x in ('.parquet','_extract.log','_bismark_report.txt')]"
  echo "$name done $(( $(date +%s)-t0 ))s $(tail -1 $OUT/${name}_extract.log | cut -c1-200)"
  cd $W
done < samples.tsv
echo ALL_DONE
