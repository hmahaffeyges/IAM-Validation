#!/bin/bash
# DEV-ONE-DRAW-01: one grid point of the copies sweep synthetic per GitHub job (cov_read.yml gives W and R; the list is copies_sweep_grid.csv and
# FIRST = LAST = the grid row). Result in $W/out.
set -u; mkdir -p $W/out; cd $R/Biological_Physics/MethylPhys/doors/data/DEV_ONE_DRAW_01/plasma_lab3
python3 copies_sweep.py $2 $W/cs $W/out/copies_sweep_r$2.json 2>&1 | tail -4
