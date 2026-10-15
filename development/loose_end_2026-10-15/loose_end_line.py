"""Whiteboard part 7 (2026-10-15): the loose end read off the line through 56 healthy cell types. Nothing tuned.
Inputs (committed): DEV_ARCH_01/arch_vs_held.csv (held error, loose share per cell) and PROC_CHANNEL_01/channel_cells_genomewide.csv
(island gain error 'denovo'). Robust line held_err = a + b * loose_share (Theil-Sen); a is the firm end (L = 0), a + b the loose end (L = 1);
bootstrap over cells (5,000, seed 1); leave-one-class-out firm end. Writes loose_end_line.json."""
import os, json, numpy as np, pandas as pd
from scipy.stats import theilslopes, linregress, spearmanr
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "..", "Biological_Physics", "MethylPhys", "doors", "data")
A = pd.read_csv(os.path.join(R, "DEV_ARCH_01", "arch_vs_held.csv")).rename(columns={"Unnamed: 0": "cell"})
C = pd.read_csv(os.path.join(R, "PROC_CHANNEL_01", "channel_cells_genomewide.csv"))
M = A.merge(C[["cell", "denovo"]], on="cell"); x = M.loose_share.values; y = M.held_err.values
s, i, _, _ = theilslopes(y, x); lr = linregress(x, y)
rng = np.random.default_rng(1); F, L = [], []
for _ in range(5000):
    k = rng.integers(0, len(x), len(x)); s_, i_, _, _ = theilslopes(y[k], x[k]); F.append(i_); L.append(i_ + s_)
res = y - (i + s * x)
O = dict(cells=len(M), firm_end=round(i, 4), loose_end=round(i + s, 4), firm_end_95=[round(v, 4) for v in np.percentile(F, [2.5, 97.5])],
         loose_end_95=[round(v, 4) for v in np.percentile(L, [2.5, 97.5])], r=round(lr.rvalue, 3), implied_D_of_firm_end=round(1 / i - 1, 1),
         residual_abs_median=round(float(np.median(abs(res))), 4), within_0_003=round(float((abs(res) <= 0.003).mean()), 2),
         leave_one_class_out_firm_end={k: round(theilslopes(y[M.klass != k], x[M.klass != k])[1], 4) for k in M.klass.unique()},
         island_median=round(float(M.denovo.median()), 4), island_range=[round(float(M.denovo.min()), 4), round(float(M.denovo.max()), 4)],
         island_vs_loose_share_rho=round(spearmanr(M.denovo, M.loose_share)[0], 2),
         island_odds_over_firm_end=round(float(np.median((1 / M.denovo - 1) / (1 / i - 1))), 2))
json.dump(O, open(os.path.join(H, "loose_end_line.json"), "w"), indent=1); print(json.dumps(O))
