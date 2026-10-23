#!/bin/bash
# DEV-TERRITORIES-01 synthetic test 1 (written and committed 2026-10-23 before it runs; plan 48; development, nothing scored).
# The territory readings (run_sample.py --dev-territories) on a healthy whole blood (GSE235717 GSM7508429, spent), then with changes planted by
# the chain itself (--dev-plant-territory): island gains and domain losses, each at 20 % of the set's sites, at two sizes.
# Expected, written before it runs: healthy reads A near 1 with level change near 0 in both sets; an island gain raises the islands' A and gives a
# positive level change, and leaves the domains about unchanged; a domain loss raises the domains' A with a negative level change and leaves the
# islands about unchanged; the larger plant moves further. Usage: bash synthetic_01.sh BETAS_CSV OUTDIR
set -u; B=$1; O=$2; mkdir -p $O; H=$(cd "$(dirname "$0")/../../../chain/MethylPhys_Interface" && pwd)
for p in none islands:0.2:0.1 islands:0.2:0.3 domains:0.2:0.1 domains:0.2:0.3; do
  t=${p//:/_}; x=""; [ $p != none ] && x="--dev-plant-territory $p"
  ( cd $H && python3 run_sample.py --betas "$B" --specimen "whole blood" --id terr_$t --out $O/$t.html --bundle $O/$t.json --dev-territories --no-info-sphere $x > $O/$t.log 2>&1 ); echo "$p exit $?"
done
python3 - "$O" <<'PY'
import json, sys, glob, os
O = sys.argv[1]; L = ["DEV-TERRITORIES-01 synthetic test 1: territory readings through run_sample.py on GSM7508429 (healthy whole blood), untared"]
for p in ("none", "islands_0.2_0.1", "islands_0.2_0.3", "domains_0.2_0.1", "domains_0.2_0.3"):
    T = (json.load(open(os.path.join(O, p + ".json"))).get("territories") or {}).get("sets") or {}
    L.append(f"{p:16} | " + " | ".join(f"{k}: A {v.get('A')} (se {v.get('se_over_sites')}), level change {v.get('level_change')}, {v.get('measured')} of {v.get('set_size')} sites" for k, v in T.items()))
open(os.path.join(O, "synthetic_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
PY
