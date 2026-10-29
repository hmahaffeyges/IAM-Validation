"""DEV-METAA-ATLAS-CURVE-01, Met-A on sequencing over healthy Met-A of its own cell: depth check (written and committed before it runs).
On the 153 healthy readings already made by the one command (metaa_seq_readings.csv: gsm, cell, A, depth = held-error chances), each
sample's A is divided by the median A of the other healthy samples of its own cell (own sample left out; a cell with no other sample is not
read). The ratio's job: a healthy sample reads 1 whatever its depth. Decision, stated before: |Spearman rho| of the ratio against depth at most
0.2 -> one reference per cell serves every depth; above 0.2 -> the reference must be matched to depth. Also reported: the ratio's median and its
2.5th to 97.5th percentile. Nothing is fitted. Usage: python3 ratio_depth.py OUT.txt"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); d = pd.read_csv(os.path.join(H, "metaa_seq_readings.csv")).dropna(subset=["A"])
r = []
for i, x in d.iterrows():
    o = d[(d.cell == x.cell) & (d.gsm != x.gsm)].A
    if len(o): r.append((x.gsm, x.cell, x.A / o.median(), x.depth, x.A))
R = pd.DataFrame(r, columns=["gsm", "cell", "ratio", "depth", "A"]); rho, p = spearmanr(R.ratio, R.depth); rho0, p0 = spearmanr(R.A, R.depth)
L = [f"read {len(R)} of {len(d)} (a cell needs another healthy sample)",
     f"ratio: median {R.ratio.median():.4f}, 2.5th to 97.5th percentile {R.ratio.quantile(.025):.4f} to {R.ratio.quantile(.975):.4f}",
     f"ratio against depth: Spearman rho {rho:+.3f} (p {p:.2g}); raw A against depth on the same samples: rho {rho0:+.3f}",
     f"decision (|rho| at most 0.2 -> one reference per cell): {'one reference per cell' if abs(rho) <= 0.2 else 'the reference must be matched to depth'}"]
R.to_csv(os.path.join(H, "ratio_depth_readings.csv"), index=False); open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
