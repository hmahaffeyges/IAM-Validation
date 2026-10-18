#!/bin/bash
# Chain v4 showcase, run 2: scenario 3 only (OTH-073, 2026-10-18): public samples picked from GEO labels alone by the rules sealed in development/TEST_PLAN_TRACKS_2026-10-17.md
# (Parts 6 and 7; picks in selected.json), each read ONCE through the one command on a clean install. A showcase of what the chain does, not a test:
# nothing is scored. Every report (HTML, both PDFs, bundle, log) is uploaded to S3 as it is made. Run: bash showcase_v4.sh <commit>
set -u; C=${1:?commit}; W=$HOME/showcase; rm -rf $W; mkdir -p $W/idat $W/out; cd $W
BK=methylphys-data-945451304272-us-west-2-an; P=results/SHOWCASE_V4/run2
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a $W/status.txt; ~/env/bin/python -c "import boto3;boto3.client('s3').upload_file('$W/status.txt','$BK','$P/status.txt')" 2>/dev/null; }
up(){ ~/env/bin/python -c "import boto3,os;s=boto3.client('s3');[s.upload_file(os.path.join('$W/out',f),'$BK','$P/'+f) for f in os.listdir('$W/out')]"; }
st "clone"; git clone -q https://github.com/hmahaffeyges/IAM-Validation.git repo && cd repo && git checkout -q $C || { st "FAILED clone"; exit 1; }
st "environment"; curl -LsSf https://astral.sh/uv/install.sh | sh > $W/uv.log 2>&1; export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.11 $W/venv >> $W/uv.log 2>&1 && uv pip install -q -p $W/venv/bin/python -r Biological_Physics/MethylPhys/chain/requirements.txt >> $W/uv.log 2>&1 || { st "FAILED environment"; exit 1; }
RS=Biological_Physics/MethylPhys/chain/MethylPhys_Interface/run_sample.py; PY=$W/venv/bin/python; L=$W/out/ledger.jsonl
get(){ for g in "$@"; do u="https://ftp.ncbi.nlm.nih.gov/geo/samples/${g:0:${#g}-3}nnn/$g/suppl/"
  for f in $(curl -s $u | grep -o "${g}_[^\"]*_\(Grn\|Red\)\.idat\.gz" | sort -u); do curl -s -o $W/idat/$f $u$f; done; done; }
grn(){ ls $W/idat/${1}_*Grn.idat.gz 2>/dev/null | head -1; }; red(){ ls $W/idat/${1}_*Red.idat.gz 2>/dev/null | head -1; }
one(){ id=$1; g=$2; sp=$3; shift 3; $PY $RS --grn "$(grn $g)" --red "$(red $g)" --specimen "$sp" --id $id "$@" --out $W/out/$id.html --ledger $L > $W/out/$id.log 2>&1; st "$id ($sp) rc $?"; up; }
MIX="GSM8618257 GSM8618258 GSM8618259"
get $MIX; st "downloaded $(ls $W/idat | wc -l) IDAT files"
# run 2 (2026-10-18): the banner first named the two cell lines with a word the report refuses to print (its claim guard); reworded
# 3: known mixtures of nine cell types (GSE281305), outside the Deconvolver's commissioned arena (blood-cell shares), labeled so
BN="SHOWCASE, OUTSIDE THE COMMISSIONED SCOPE: a DNA mixture of nine cell types, including two cultured cell lines, fibroblasts and endothelial cells (GSE281305). The IAM-Deconvolver is commissioned on blood-cell shares; this shows what it reports on cells it was not built for. Not a clinical result."
n=1; for g in $MIX; do one S3_GSE281305_mix$n $g "constructed DNA mixture" --banner "$BN"; n=$((n+1)); done
st "finished: $(ls $W/out | grep -c 'researcher.pdf') researcher PDFs, $(ls $W/out | grep -c '\.html$') reports"; exit 0
