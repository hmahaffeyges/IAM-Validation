#!/bin/bash
# Chain v4 example reports on REAL arrays (OTH-069, 2026-10-18). Run on a clean machine: bash report_examples_v4.sh <commit>
# Four healthy whole-blood EPIC v1 arrays from one study (GSE235717; already read in the C-Score band, DEV_CSCORE_TARE_01, so no unread data
# is spent). Pass 1: each array through the chain alone. Then the first three are the same-run healthy references (their Met-A and C-Score
# from pass 1) and the fourth is read against them, so Met-A and the C-Score both give tared readings. Every report (HTML and both PDFs) is
# uploaded to S3 as it is made. Nothing is scored here: these are examples of the report.
set -u; C=${1:?commit}; W=$HOME/examples; rm -rf $W; mkdir -p $W/idat $W/out; cd $W
B=s3://methylphys-data-945451304272-us-west-2-an/results/REPORT_EXAMPLES_V4
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a $W/status.txt; aws s3 cp -q $W/status.txt $B/status.txt 2>/dev/null || ~/env/bin/python -c "import boto3;boto3.client('s3').upload_file('$W/status.txt','methylphys-data-945451304272-us-west-2-an','results/REPORT_EXAMPLES_V4/status.txt')"; }
up(){ ~/env/bin/python -c "import boto3,os,sys;s=boto3.client('s3');[s.upload_file(os.path.join('$W/out',f),'methylphys-data-945451304272-us-west-2-an','results/REPORT_EXAMPLES_V4/'+f) for f in os.listdir('$W/out')]"; }
st "clone"; git clone -q https://github.com/hmahaffeyges/IAM-Validation.git repo && cd repo && git checkout -q $C || { st "FAILED clone"; exit 1; }
st "environment"; curl -LsSf https://astral.sh/uv/install.sh | sh > $W/uv.log 2>&1; export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.11 $W/venv >> $W/uv.log 2>&1 && uv pip install -q -p $W/venv/bin/python -r Biological_Physics/MethylPhys/chain/requirements.txt >> $W/uv.log 2>&1 || { st "FAILED environment"; exit 1; }
GSMS="GSM7508426 GSM7508427 GSM7508428 GSM7508429"
for g in $GSMS; do u="https://ftp.ncbi.nlm.nih.gov/geo/samples/${g:0:7}nnn/$g/suppl/"
  for f in $(curl -s $u | grep -o "${g}_[^\"]*_\(Grn\|Red\)\.idat\.gz" | sort -u); do curl -s -o $W/idat/$f $u$f; done; done
st "downloaded $(ls $W/idat | wc -l) IDAT files"
RS=Biological_Physics/MethylPhys/chain/MethylPhys_Interface/run_sample.py; PY=$W/venv/bin/python
for g in $GSMS; do G=$(ls $W/idat/${g}_*Grn.idat.gz); R=$(ls $W/idat/${g}_*Red.idat.gz)
  $PY $RS --grn $G --red $R --specimen "whole blood" --id ${g}_pass1 --out $W/out/${g}_pass1.html --ledger $W/out/ledger.jsonl > $W/out/${g}_pass1.log 2>&1; st "pass 1 $g rc $?"; up; done
$PY - "$W/out" <<'PY'
import json, sys, os
o=sys.argv[1]; rows=[]
for g in ("GSM7508426","GSM7508427","GSM7508428"):
    b=json.load(open(os.path.join(o,f"{g}_pass1_bundle.json"))); rows.append((g,(b.get("met_a") or {}).get("A"),(b.get("met_a_cscore") or {}).get("C")))
open(os.path.join(o,"same_run_refs.csv"),"w").write("id,A,C\n"+"".join(f"{g},{a},{c}\n" for g,a,c in rows)); print(rows)
PY
st "references: $(tail -n +2 $W/out/same_run_refs.csv | tr '\n' ' ')"
g=GSM7508429; G=$(ls $W/idat/${g}_*Grn.idat.gz); R=$(ls $W/idat/${g}_*Red.idat.gz)
$PY $RS --grn $G --red $R --specimen "whole blood" --id GSE235717_${g} --ref-table $W/out/same_run_refs.csv --out $W/out/GSE235717_${g}.html --ledger $W/out/ledger.jsonl > $W/out/GSE235717_${g}.log 2>&1; rc=$?
up; st "finished rc $rc: $(ls $W/out | grep -c pdf) PDFs"; exit $rc
