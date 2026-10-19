#!/bin/bash
# A researcher's clone of Biological_Physics only (2026-10-19, author: "only the biophysics folder would need to be cloned"). On a clean
# machine with internet access:  bash cleanclone_bio.sh <commit>
# 1. a partial, sparse clone: git downloads only Biological_Physics and the repository's top-level files (README, LICENSE, CITATION), and
#    the test confirms no other folder came down; 2. a new Python 3.11 environment from chain/requirements.txt only; 3. the chain v4 release
#    check; 4. one reproduction (the C-Score band, public GEO data) compared number for number with the committed result. Every step must pass.
set -u; C=${1:?commit}; W=$HOME/cleanclone; rm -rf $W; mkdir -p $W; cd $W
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a $W/status.txt; }
st "clone (Biological_Physics only)"
git clone -q --filter=blob:none --sparse https://github.com/hmahaffeyges/IAM-Validation.git repo && cd repo && git sparse-checkout set Biological_Physics && git checkout -q $C || { st "FAILED clone"; exit 1; }
dirs=$(find . -mindepth 1 -maxdepth 1 -type d ! -name .git -printf "%f " ); st "top-level folders present: $dirs| checkout size $(du -sh --exclude=.git . | cut -f1)"
[ "$(echo $dirs)" = "Biological_Physics" ] || { st "FAILED: folders other than Biological_Physics came down"; exit 1; }
st "environment"; curl -LsSf https://astral.sh/uv/install.sh | sh > $W/uv.log 2>&1; export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.11 $W/venv >> $W/uv.log 2>&1 && uv pip install -q -p $W/venv/bin/python -r Biological_Physics/MethylPhys/chain/requirements.txt >> $W/uv.log 2>&1 || { st "FAILED environment"; exit 1; }
$W/venv/bin/python -c "import methylprep, pandas; print('python', __import__('sys').version.split()[0], 'methylprep', methylprep.__version__, 'pandas', pandas.__version__)" | tee -a $W/status.txt
st "release check v4"; $W/venv/bin/python Biological_Physics/MethylPhys/chain/release_check_v4.py --json $W/release_check_v4.json > $W/release_check_v4.log 2>&1; rc=$?
st "release check rc $rc: $(tail -1 $W/release_check_v4.log)"
[ $rc -eq 0 ] || exit $rc
st "reproduce setup"; bash Biological_Physics/MethylPhys/reproduce/setup.sh > $W/setup.log 2>&1 || { st "FAILED setup: $(tail -2 $W/setup.log | tr '\n' ' ')"; exit 1; }
st "reproduce cscore_band"; python3 Biological_Physics/MethylPhys/reproduce/reproduce.py cscore_band > $W/cscore_band.txt 2>&1; r2=$?
cp Biological_Physics/MethylPhys/reproduce/out/cscore_band/verdict.json $W/cscore_band_verdict.json 2>/dev/null
st "finished rc $r2: release check $(tail -1 $W/release_check_v4.log | cut -c1-80) | cscore_band $(tail -1 $W/cscore_band.txt | cut -c1-80)"; exit $r2
