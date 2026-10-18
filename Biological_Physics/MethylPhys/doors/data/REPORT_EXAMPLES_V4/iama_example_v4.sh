#!/bin/bash
# Chain v4 example report for IAM-A on REAL sequencing (OTH-069, 2026-10-18). Run on a clean machine: bash iama_example_v4.sh <commit>
# One healthy CD8 T-cell sample of IAM-A's sealed grand test (GSM4293895, SRR11007296; read and scored 2026-10-17, so no unread data is spent),
# its committed Bismark alignment from that run (S3) and the reference genome it was aligned to (the alignment tools bundle), read through the
# ONE command (run_sample.py --bam). The test: the reading must equal the committed grand-test reading of this sample (A 0.9083, Normal), and
# the report and both PDFs must be written. Every output is uploaded to S3 as it is made.
set -u; C=${1:?commit}; W=$HOME/iama; rm -rf $W; mkdir -p $W/out; cd $W
BK=methylphys-data-945451304272-us-west-2-an; P=results/REPORT_EXAMPLES_V4/iama
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a $W/status.txt; ~/env/bin/python -c "import boto3;boto3.client('s3').upload_file('$W/status.txt','$BK','$P/status.txt')" 2>/dev/null; }
up(){ ~/env/bin/python -c "import boto3,os;s=boto3.client('s3');[s.upload_file(os.path.join('$W/out',f),'$BK','$P/'+f) for f in os.listdir('$W/out')]"; }
st "clone"; git clone -q https://github.com/hmahaffeyges/IAM-Validation.git repo && cd repo && git checkout -q $C || { st "FAILED clone"; exit 1; }
st "environment"; curl -LsSf https://astral.sh/uv/install.sh | sh > $W/uv.log 2>&1; export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.11 $W/venv >> $W/uv.log 2>&1 && uv pip install -q -p $W/venv/bin/python -r Biological_Physics/MethylPhys/chain/requirements.txt >> $W/uv.log 2>&1 || { st "FAILED environment"; exit 1; }
PY=$W/venv/bin/python; $PY -c "import pysam; print('pysam', pysam.__version__)" | tee -a $W/status.txt
st "downloading the alignment and the reference bundle"
$PY - <<PYE || { st "FAILED download"; exit 1; }
import boto3; s=boto3.client("s3")
for k,f in (("results/DEV_IAMA_GRAND_01/bam/SRR11007296.bam","$W/SRR11007296.bam"),("results/DEV_IAMA_GRAND_01/bam/SRR11007296.bam.bai","$W/SRR11007296.bam.bai")): s.download_file("$BK",k,f)
for i in ("00","01","02"): s.download_file("$BK",f"downloads/tumour_work/tools.tgz.part_{i}",f"$W/tools.tgz.part_{i}")
PYE
mkdir -p $W/tools && cat $W/tools.tgz.part_0* | tar xz -C $W/tools && rm -f $W/tools.tgz.part_0*
FA=$(find $W/tools -path '*ref*' \( -name '*.fa' -o -name '*.fasta' \) | head -1); [ -s "$FA" ] || { st "FAILED: no reference fasta in the bundle"; exit 1; }
[ -s "$FA.fai" ] || $PY -c "import pysam;pysam.faidx('$FA')"; st "reference $(basename $FA); counting with $(nproc) workers"
RS=Biological_Physics/MethylPhys/chain/MethylPhys_Interface/run_sample.py
$PY $RS --bam $W/SRR11007296.bam --ref-fasta $FA --assembly hg38 --workers $(nproc) --seq-cell "cd8 t cells" --specimen "sorted CD8 T cells" \
  --id GSM4293895_CD8 --out $W/out/GSM4293895_CD8.html --ledger $W/out/ledger.jsonl > $W/out/GSM4293895_CD8.log 2>&1; rc=$?
$PY - "$W/out/GSM4293895_CD8_bundle.json" Biological_Physics/MethylPhys/doors/data/DEV_IAMA_GRAND_01/grand_scored.json <<'PYE' | tee -a $W/status.txt
import json, sys, os
b = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}; q = b.get("iam_a") or {}
g = [r for r in json.load(open(sys.argv[2]))["rows"] if r["sample"] == "GSM4293895"][0]
print("one command:", q.get("A"), q.get("state"), "| committed grand test:", round(g["A"], 4), "| equal:", q.get("A") is not None and round(float(q["A"]), 4) == round(g["A"], 4))
PYE
up; st "finished rc $rc: $(ls $W/out | grep -c pdf) PDFs"; exit $rc
