#!/bin/bash
# One grid row of the naive and memory CD4 synthetic per GitHub job (cov_read.yml gives W and R; the list is subset_grid.csv, FIRST = LAST = the row).
set -u; mkdir -p $W/out; cd $R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/label_rule
python3 subset_synthetic.py $2 $W/ss $W/out/subset_r$2.json 2>&1 | tail -3
