#!/bin/bash
# DEV-ERASE-01 planted check runner (committed with planted_real_erase_01.py, before it runs). Base arms first; the doubled arm needs eps_base.
set -e; export CHAIN=$PWD PYTHONUNBUFFERED=1
G=/home/ubuntu/data/round2iama/GSM5652313_Blood-Granulocytes-Z000000TZ.hg38.pat.gz; T=/mnt/scratch/run2/loyfer/GSM5652281_Blood-T-CD4-Z000000UM.pat.gz
PY=~/env/bin/python; S=planted_real_erase_01.py
( $PY $S $G gran base 1 > r_gran_base.tsv ) & ( $PY $S $T cd4 base 2 > r_cd4_base.tsv ) & ( $PY $S $G gran halved 3 > r_gran_halved.tsv ) & ( $PY $S $T cd4 halved 4 > r_cd4_halved.tsv ) & wait
( $PY $S $G gran doubled 5 $(cut -f3 r_gran_base.tsv) > r_gran_doubled.tsv ) & ( $PY $S $T cd4 doubled 6 $(cut -f3 r_cd4_base.tsv) > r_cd4_doubled.tsv ) & wait
cat r_*.tsv > planted_real_erase_01_raw.tsv
$PY - <<'PY'
import pandas as pd
R = pd.read_csv("planted_real_erase_01_raw.tsv", sep="\t", header=None, names=["file", "arm", "eps", "AB", "opp", "err"]).set_index(["file", "arm"])
L = ["DEV-ERASE-01 planted check on real healthy molecules, run once (planted_real_erase_01.py)"]; ok = True
for f in ("gran", "cd4"):
    b = R.loc[(f, "base")]; h = R.loc[(f, "halved")]; d = R.loc[(f, "doubled")]
    p1, p2, p3 = h.eps / b.eps, d.eps / b.eps, b.AB
    m1, m3 = 0.45 <= p1 <= 0.55, 0.95 <= p3 <= 1.05; ok &= m1 and m3
    L += [f"{f}: base eps {b.eps:.5f} ({int(b.opp):,} opportunities, {int(b.err):,} isolated errors)",
          f"  P1 halved/base {p1:.4f} (bar 0.45-0.55) -> {'met' if m1 else 'NOT met'}",
          f"  P2 doubled/base {p2:.4f} (reported, no bar)",
          f"  P3 base halves A/B {p3:.4f} (bar 0.95-1.05) -> {'met' if m3 else 'NOT met'}"]
L.append(f"PLANTED CHECK: {'MET' if ok else 'NOT MET'}")
open("planted_real_erase_01.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
PY
