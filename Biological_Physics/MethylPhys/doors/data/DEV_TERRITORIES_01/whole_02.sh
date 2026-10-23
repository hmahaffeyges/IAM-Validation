#!/bin/bash
# DEV-TERRITORIES-01 test 2 (written 2026-10-23 before it runs; development, tuning, nothing scored): Met-A as a two-level instrument through
# run_sample.py on the healthy whole blood GSE235717 GSM7508429 (spent): the whole-sample reading (every atlas site, met_a_whole), each cell's
# reading (identity sites, met_a_each_cell) and the territories, healthy and with plants made by the chain. Expected, written before it runs:
# every plant raises the whole-sample reading above healthy; an island plant moves its 'islands' part most; a neutrophil identity-site loss moves
# neutrophils' own per-cell reading most. Usage: bash whole_02.sh BETAS_CSV OUTDIR
set -u; B=$1; O=$2; mkdir -p $O; H=$(cd "$(dirname "$0")/../../../chain/MethylPhys_Interface" && pwd)
run() { ( cd $H && python3 run_sample.py --betas "$B" --specimen "whole blood" --id w_$1 --out $O/$1.html --bundle $O/$1.json --no-info-sphere $2 > $O/$1.log 2>&1 ); echo "$1 exit $?"; }
run healthy ""
run neu_identity_05 "--dev-plant neutrophils:0.05"
run neu_all_05 "--dev-plant neutrophils:0.05:all"
run islands_10 "--dev-plant-territory islands:0.2:0.1"
run domains_10 "--dev-plant-territory domains:0.2:0.1"
python3 - "$O" <<'PY'
import json, sys, os
O = sys.argv[1]; L = ["DEV-TERRITORIES-01 test 2: Met-A as a two-level instrument, through run_sample.py on GSM7508429 (healthy whole blood), untared"]
for p in ("healthy", "neu_identity_05", "neu_all_05", "islands_10", "domains_10"):
    B = json.load(open(os.path.join(O, p + ".json"))); W = B.get("met_a_whole") or {}; bp = W.get("by_place") or {}
    ec = {c["cell"]: c.get("A") for c in (B.get("met_a_each_cell") or {}).get("cells", []) if c.get("A") is not None}
    T = (B.get("territories") or {}).get("sets") or {}
    L.append(f"{p:16} | whole A {W.get('A')} (se {W.get('se_over_sites')}, {W.get('sites')} sites, a 1 % loss moves {W.get('shift_per_1pct_loss')}) | "
             + ", ".join(f"{k} {v['A']}" for k, v in bp.items() if v) + f" | neutrophils' own {ec.get('neutrophils')} | territories "
             + ", ".join(f"{k} {v.get('A')}" for k, v in T.items()))
open(os.path.join(O, "whole_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
PY
