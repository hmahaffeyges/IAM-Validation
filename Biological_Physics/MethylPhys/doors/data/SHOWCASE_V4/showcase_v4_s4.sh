#!/bin/bash
# Showcase scenario 4 on an AWS machine (2026-10-19): fresh public clone at <commit>, the chain's pinned environment (reproduce/setup.sh),
# the three GEO arrays, the IAM-Atlas v2.1 summary from S3, then scenario4_colon.py. Results to S3 results/SHOWCASE_V4/s4/ as made.
set -u; C=${1:?commit}; W=$HOME/s4; rm -rf $W; mkdir -p $W/idat $W/out; cd $W; B=methylphys-data-945451304272-us-west-2-an; K=results/SHOWCASE_V4/s4; PYB=$HOME/env/bin/python
put(){ $PYB -c "import boto3,sys;boto3.client('s3').upload_file(sys.argv[1],'$B','$K/'+sys.argv[2])" "$1" "$2"; }
st(){ echo "$(date -u +%FT%TZ) $*" >> $W/status.txt; put $W/status.txt status.txt; }
st "clone $C"; git clone -q https://github.com/hmahaffeyges/IAM-Validation.git repo && git -C repo checkout -q $C || { st "FAILED clone"; exit 1; }
cd repo && bash Biological_Physics/MethylPhys/reproduce/setup.sh > $W/setup.log 2>&1 || { st "FAILED setup: $(tail -2 $W/setup.log | tr '\n' ' ')"; exit 1; }; st "setup: $(tail -1 $W/setup.log)"
PY=$PWD/Biological_Physics/MethylPhys/reproduce/.venv/bin/python
for g in GSM4589141 GSM4589144 GSM4589145; do
  u=https://ftp.ncbi.nlm.nih.gov/geo/samples/${g:0:7}nnn/$g/suppl/
  for f in $(curl -sfL --retry 5 $u | grep -o "${g}_[A-Za-z0-9_]*_\(Grn\|Red\)\.idat\.gz" | sort -u); do curl -sfL --retry 8 --retry-delay 20 -o $W/idat/$f $u$f && gzip -t $W/idat/$f && gunzip -f $W/idat/$f || { st "FAILED download $f"; exit 1; }; done
done; st "idats: $(ls $W/idat | wc -l) files"
$PYB -c "import boto3;boto3.client('s3').download_file('$B','results/ATLAS_V21/assembly/atlas_v21_summary.parquet','$W/atlas_v21_summary.parquet')" || { st "FAILED atlas summary"; exit 1; }; st "atlas summary in"
OMP_NUM_THREADS=2 $PY Biological_Physics/MethylPhys/doors/data/SHOWCASE_V4/scenario4_colon.py $W/atlas_v21_summary.parquet $W/idat $W/out > $W/s4.log 2>&1; rc=$?
put $W/s4.log s4.log; [ -f $W/out/scenario4_colon.json ] && put $W/out/scenario4_colon.json scenario4_colon.json
st "finished rc $rc: $(tail -1 $W/s4.log | cut -c1-200)"
