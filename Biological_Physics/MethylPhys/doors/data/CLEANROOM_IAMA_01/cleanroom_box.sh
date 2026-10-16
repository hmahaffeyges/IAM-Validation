#!/bin/bash
# PHY-055 IAM-A clean room, part A, on a self-terminating machine. Fresh clone at COMMIT, empty folders, public data only.
set -uo pipefail
COMMIT=$1; B=methylphys-data-945451304272-us-west-2-an; P=results/CLEANROOM_IAMA_01/; PY=~/env/bin/python
st(){ printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> ~/cr_status.txt; $PY -c "import boto3;boto3.client('s3').upload_file('$HOME/cr_status.txt','$B','${P}status.txt')"; }
rm -rf ~/cr && mkdir -p ~/cr/run && cd ~/cr
git clone -q https://github.com/hmahaffeyges/IAM-Validation.git repo && cd repo && git checkout -q $COMMIT && cd ~/cr/run || { st "clone FAILED"; exit 1; }
cp -r ../repo/Biological_Physics/MethylPhys/chain ./chain
cp ../repo/Biological_Physics/MethylPhys/doors/data/DEV_CHAIN_Q2_01/{box_q2_line.py,loyfer_samples.csv,encode_samples.tsv} .
st "clone $COMMIT ok; counting Loyfer from GEO"
for pass in 1 2 3 4; do $PY box_q2_line.py loyfer 14 >> loyfer.log 2>&1; n=$(ls out/GSM*.json 2>/dev/null | wc -l); st "pass $pass: $n of 153"; [ $n -ge 153 ] && break; sleep 120; done
curl -sL -o ref.fa.gz "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz" && gunzip -f ref.fa.gz && samtools faidx ref.fa 2>/dev/null || $PY -c "import pysam;pysam.faidx('ref.fa')"
tail -n +2 encode_samples.tsv | while IFS=$'\t' read cell lc exp acc url; do
  st "encode $acc"; $PY -c "import boto3;from botocore import UNSIGNED;from botocore.config import Config;b,k='$url'[5:].split('/',1);boto3.client('s3',config=Config(signature_version=UNSIGNED),region_name='us-west-2').download_file(b,k,'$acc.bam')"
  head -1 encode_samples.tsv > one.tsv; grep -P "\t$acc\t" encode_samples.tsv >> one.tsv; cp encode_samples.tsv all.tsv; cp one.tsv encode_samples.tsv
  OMP_NUM_THREADS=1 $PY box_q2_line.py encode 32 >> encode.log 2>&1; cp all.tsv encode_samples.tsv; rm -f $acc.bam*
done
$PY ../repo/Biological_Physics/MethylPhys/doors/data/CLEANROOM_IAMA_01/cleanroom_iama.py compare ../repo out > compare.txt 2>&1
cp ../repo/Biological_Physics/MethylPhys/doors/data/CLEANROOM_IAMA_01/cleanroom_iama_result.json . 2>/dev/null
$PY -c "
import boto3,glob,os;s=boto3.client('s3')
for f in glob.glob('out/*.json')+['compare.txt','loyfer.log','encode.log','cleanroom_iama_result.json']:
    os.path.exists(f) and s.upload_file(f,'$B','${P}'+('out/' if f.startswith('out/') else '')+os.path.basename(f))"
st "finished: $(tail -1 compare.txt | cut -c1-200)"
