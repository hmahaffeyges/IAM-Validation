#!/bin/bash
# Chain v4 one-command test from a fresh public clone (OTH-068, 2026-10-17). Run on a clean machine with internet access:
#   bash cleanclone_v4.sh <commit>
# 1. clone the public repository and check out <commit>; 2. a new Python 3.11 environment with ONLY chain/requirements.txt;
# 3. one command: python Biological_Physics/MethylPhys/chain/release_check_v4.py. Nothing is read from our storage.
set -u; C=${1:?commit}; W=$HOME/cleanclone; rm -rf $W; mkdir -p $W; cd $W
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a $W/status.txt; }
st "clone"; git clone -q https://github.com/hmahaffeyges/IAM-Validation.git repo && cd repo && git checkout -q $C || { st "FAILED clone"; exit 1; }
st "environment"; curl -LsSf https://astral.sh/uv/install.sh | sh > $W/uv.log 2>&1; export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.11 $W/venv >> $W/uv.log 2>&1 && uv pip install -q -p $W/venv/bin/python -r Biological_Physics/MethylPhys/chain/requirements.txt >> $W/uv.log 2>&1 || { st "FAILED environment"; exit 1; }
$W/venv/bin/python -c "import methylprep, numpy, pandas, scipy, matplotlib, pyarrow; print('python', __import__('sys').version.split()[0], 'methylprep', methylprep.__version__, 'pandas', pandas.__version__)" | tee -a $W/status.txt
st "release check v4"; $W/venv/bin/python Biological_Physics/MethylPhys/chain/release_check_v4.py --json $W/release_check_v4.json > $W/release_check_v4.log 2>&1; rc=$?
st "finished rc $rc: $(tail -1 $W/release_check_v4.log)"; exit $rc
