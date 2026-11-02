"""How many sites of each cell have a zero or near-zero width (person-to-person spread and the range of the average, as the report globe
combines them: sqrt(donor_sd^2 + ((mu_hi - mu_lo) / 3.92)^2)) in the IAM Sequencing Atlas's sphere columns and in IAM-Atlas v2.1.
Usage: python3 zero_widths.py SPHERE_COLS_DIR OUT.csv OUT.txt"""
import os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
CH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain"); V21 = os.path.join(CH, "..", "atlas", "v2.1")
def cols(d):
    out = {}
    for k in (1, 2, 3):
        T = {s: pq.read_table(os.path.join(d, f"{s}_part{k}.parquet")) for s in ("mu", "mu_lo", "mu_hi", "donor_sd")}
        for c in T["mu"].column_names:
            if all(c in T[s].column_names for s in T):
                v = {s: T[s].column(c).to_numpy() for s in T}; f = {s: np.where(x == 65535, np.nan, x.astype("float64") / 1e4) for s, x in v.items()}
                out[c] = np.sqrt(f["donor_sd"] ** 2 + ((f["mu_hi"] - f["mu_lo"]) / 3.92) ** 2)
    return out
rows = []
for lab, d in (("sequencing", sys.argv[1]), ("v2.1", V21)):
    for c, w in cols(d).items():
        ok = np.isfinite(w); rows.append(dict(atlas=lab, cell=c, sites=int(ok.sum()), zero=int((w[ok] == 0).sum()), under_0005=int((w[ok] < 0.005).sum()), smallest=round(float(np.nanmin(w)), 4)))
D = pd.DataFrame(rows); D.to_csv(sys.argv[2], index=False); L = []
for lab, g in D.groupby("atlas"): L.append(f"{lab}: {len(g)} cells; sites with zero width {int(g.zero.sum())}, under 0.005 {int(g.under_0005.sum())}; smallest width {g.smallest.min():.4f}")
n = D[(D.atlas == "sequencing") & (D.cell == "cortical neurons")]
if len(n): L.append(f"cortical neurons, sequencing: zero {int(n.zero.iloc[0])}, under 0.005 {int(n.under_0005.iloc[0])} ({int(n.under_0005.iloc[0] - n.zero.iloc[0])} above zero)")
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
