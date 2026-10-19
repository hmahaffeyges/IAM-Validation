#!/bin/bash
# Test of Biological_Physics/MethylPhys/reproduce/ from a fresh public clone (OTH-076, 2026-10-18): clone at the commit, remove the methylprep manifests so Stage 1 must
# fetch them as a researcher's machine would, bash Biological_Physics/MethylPhys/reproduce/setup.sh, then each test in order. Every verdict, log and status line goes to S3
# as it is made (the machine's own Python; the clean environment has no boto3 by design). Run: bash reproduce_fresh.sh <commit> <test ...>
set -u; C=${1:?commit}; shift; W=$HOME/rp; rm -rf $W; mkdir -p $W; cd $W
BK=methylphys-data-945451304272-us-west-2-an; P=${OUTP:-results/REPRODUCE_V1}
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a $W/status.txt; ~/env/bin/python -c "import boto3;boto3.client('s3').upload_file('$W/status.txt','$BK','$P/status.txt')" 2>/dev/null; }
up(){ ~/env/bin/python - "$W/repo/Biological_Physics/MethylPhys/reproduce/out" "$BK" "$P/out" <<'PY'
import boto3, os, sys
s=boto3.client("s3"); r,b,p=sys.argv[1:4]
for d,_,fs in os.walk(r):
    for f in fs:
        q=os.path.join(d,f)
        if any(x in q for x in ("/seq/e_", "/seq/mamba", "/seq/bin", "/seq/wgbs_tools", "/q2/chain/")): continue   # installed tools and the chain copy
        if os.path.getsize(q)<5_000_000 and (f.endswith((".json",".log",".csv",".txt"))): s.upload_file(q,b,p+"/"+os.path.relpath(q,r))
PY
}
st "clone $C"; git clone -q https://github.com/hmahaffeyges/IAM-Validation.git repo && cd repo && git checkout -q $C || { st "FAILED clone"; exit 1; }
rm -rf ~/.methylprep_manifest_files; st "methylprep manifests removed"
bash Biological_Physics/MethylPhys/reproduce/setup.sh > $W/setup.log 2>&1 && st "setup: $(tail -1 $W/setup.log)" || { st "FAILED setup: $(tail -3 $W/setup.log | tr '\n' ' ')"; exit 1; }
# every 5 minutes: the runner's progress lines and the last line of each step log, so a watcher sees a long step working (checklist 7)
( while sleep 300; do { tail -3 $W/repo/Biological_Physics/MethylPhys/reproduce/out/progress.txt; for f in $W/repo/Biological_Physics/MethylPhys/reproduce/out/*/logs/*.log; do [ -f $f ] && echo "$(basename $(dirname $(dirname $f)))/$(basename $f): $(tail -c 160 $f | tr '\n' ' ')"; done; } > $W/progress.txt 2>/dev/null
    ~/env/bin/python -c "import boto3;boto3.client('s3').upload_file('$W/progress.txt','$BK','$P/progress.txt')" 2>/dev/null; done ) & PROG=$!
for t in "$@"; do st "start $t"; python3 Biological_Physics/MethylPhys/reproduce/reproduce.py $t > $W/$t.txt 2>&1; rc=$?; up; st "$t rc $rc: $(tail -1 $W/$t.txt)"; done
kill $PROG 2>/dev/null; st "finished"
