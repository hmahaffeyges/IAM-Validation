#!/bin/bash
# The IAM Sequencing Atlas v1.2 deposit (atlas + its globe + the noise maps under the atlas's DOI); SOURCED by the Google launcher (W, R, B, log and
# up come from it; return, not exit). Saves as it goes: zenodo_seq_atlas_v12.py skips every file already in the draft ('already read (in the draft)').
# Usage: V12_ZA=<atlas draft files URL> GBK=<bucket name>, then source this file.
cd $R && python3 Biological_Physics/MethylPhys/deposit/zenodo_seq_atlas_v12.py $GBK $V12_ZA $W/v12 > $W/v12.log 2>&1; rc=$?
log "v12 deposit rc $rc: $(tail -3 $W/v12.log | tr '\n' ' ' | cut -c1-300)"; up $W/v12.log
