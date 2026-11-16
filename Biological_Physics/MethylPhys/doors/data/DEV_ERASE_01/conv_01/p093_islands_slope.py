#!/usr/bin/env python3
"""PHY-093 islands against each library's meter (register T1, A17; 2026-11-16). Reads p093_read.csv (53 libraries, the reading's per-library
counts: islands stray-mark error isl_err and the library's meter conv_fail + tc_err). OLS slope and intercept of isl_err on meter, 95 % range
from 2000 bootstrap resamples of libraries (seed 1). Per type: pooled islands minus the type's median meter (the reading script's convention).
Prints and writes p093_islands_slope_summary.txt. No chain stage is imported: a statistic of the chain's own committed counts."""
import csv, os, random, statistics as st
H = os.path.dirname(os.path.abspath(__file__))
R = [r for r in csv.DictReader(open(os.path.join(H, "p093_read.csv"))) if r["isl_opp"]]
x = [float(r["meter"]) for r in R]; y = [float(r["isl_err"]) for r in R]
def ols(x, y):
    mx, my = st.mean(x), st.mean(y); b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x); return b, my - b * mx
b, a = ols(x, y); rng = random.Random(1); bs = []
for _ in range(2000):
    idx = [rng.randrange(len(x)) for _ in x]; bs.append(ols([x[i] for i in idx], [y[i] for i in idx])[0])
bs.sort()
L = [f"PHY-093 islands vs meter, {len(R)} libraries: slope {b:.2f} ({bs[50]:.2f}-{bs[1949]:.2f}), intercept {a:.4f}"]
for c in ("B", "CD4 T", "CD8 T", "monocytes"):
    g = [r for r in R if r["cell"] == c]
    pi = sum(float(r["isl_iso"]) for r in g) / sum(float(r["isl_opp"]) for r in g); mm = st.median(float(r["meter"]) for r in g)
    ge = st.mean(float(r["gain_err"]) - float(r["meter"]) for r in g)
    L.append(f"  {c}: n {len(g)}, writer at islands {pi - mm:.4f}, writer genome-wide {ge:.4f}")
open(os.path.join(H, "p093_islands_slope_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
