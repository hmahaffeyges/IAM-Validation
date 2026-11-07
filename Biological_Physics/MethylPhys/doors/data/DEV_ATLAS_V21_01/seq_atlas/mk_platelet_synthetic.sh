#!/bin/bash
# plan 60.24: the platelet test's synthetic on GitHub (cov_read.yml gives W, R and CPG_BED; the list and rows are not used). Result in $W/out.
set -u; mkdir -p $W/out; cd $R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas
python3 mk_platelet_synthetic.py $W/mksyn $W/out/mk_platelet_synthetic.txt 2>&1 | tail -8
