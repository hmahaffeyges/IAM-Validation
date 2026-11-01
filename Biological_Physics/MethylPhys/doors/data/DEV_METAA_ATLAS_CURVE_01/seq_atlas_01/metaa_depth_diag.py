"""Pass 4, the depth trend of sorted-cell Met-A on the healthy inputs: a diagnostic (added after pass 4 was scored, disclosed; nothing scored).
From committed chain outputs only: the per-donor counts (metaa_donor_counts_v1.parquet, run_sample.py --build seq-metaa-donors), pass 4's
readings (pass4_sorted_by_depth.csv), pass 3's own-cell shares and IAM-A (healthy_fullchain3_readings.csv). For each healthy input: its
disorder per read at the identity sites at a fixed depth of 10 reads (each site's reads thinned to 10 without replacement, the exact
hypergeometric expectation, as donor_pair does); its off-pattern reads where the cell's other donors agree (methylated reads where the others
are at most 1 % methylated; unmethylated reads where they are at least 99 %); its median depth. For each pass 4 sample, the same against the
donors it was read against (its cell, the reference half): whether Met-A follows the ratio of disorder per read, and what that ratio follows.
Usage: python3 metaa_depth_diag.py OUT.txt"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, hypergeom
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", "..", ".."))
CH = os.path.join(MP, "chain"); sys.path.insert(0, CH); import stage_metaa_atlas as SA
T = pd.read_parquet(os.path.join(CH, "Runtime Matrices", "Seq_Atlas", "metaa_donor_counts", "metaa_donor_counts_v1.parquet")); T[["m", "n"]] = T[["m", "n"]].astype("int64")
M0 = 10; j = np.arange(M0 + 1); Hj = np.asarray(SA._H(pd.Series(j / M0)), "float64")
X = T[T.n >= M0].copy(); X["h10"] = hypergeom.pmf(j[None, :], X.n.to_numpy()[:, None], X.m.to_numpy()[:, None], M0) @ Hj
tot = T.groupby(["cell", "probe"])[["m", "n"]].transform("sum"); T["om"] = tot.m - T.m; T["on"] = tot.n - T.n; T["olev"] = T.om / T.on.where(T.on > 0)
U = T[(T.olev <= 0.01) & (T.on >= 20)]; K = T[(T.olev >= 0.99) & (T.on >= 20)]
D = pd.DataFrame({"h10": X.groupby("gsm").h10.mean(), "depth": T[T.n > 0].groupby("gsm").n.median(),
                  "off_unmarked": U.groupby("gsm").m.sum() / U.groupby("gsm").n.sum(), "off_marked": (K.groupby("gsm").n.sum() - K.groupby("gsm").m.sum()) / K.groupby("gsm").n.sum()})
I = pd.read_csv(os.path.join(MP, "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", "seq_atlas_inputs.csv"))
for c in ("gsm", "cell", "half"): I[c] = I[c].astype(str).str.strip()
I["half"] = I.half.str.upper(); D = D.join(I.set_index("gsm")[["cell", "half"]])
R3 = pd.read_csv(os.path.join(H, "healthy_fullchain3_readings.csv"))
for k, name in (("own", "own-cell share"), ("iama", "IAM-A")): D[k] = R3[R3.reading == name].drop_duplicates("gsm").set_index("gsm").A.reindex(D.index)
S4 = pd.read_csv(os.path.join(H, "healthy_fullchain4_samples.csv")); S4["gsm"] = S4.gsm.str.strip(); S4["refhalf"] = S4.rsflags.astype(str).str.extract(r"--dev-seq-ref-half (\w)")[0]
d = pd.read_csv(os.path.join(H, "pass4_sorted_by_depth.csv")).merge(S4[["gsm", "refhalf"]], on="gsm"); rows = []
for r in d.itertuples():
    q = D[(D.cell == r.cell) & (D.half == r.refhalf) & (D.index != r.gsm)]
    if r.gsm not in D.index or q.empty: continue
    s = D.loc[r.gsm]; rows.append(dict(gsm=r.gsm, cell=r.cell, A=r.A, ratio_h10=s.h10 / q.h10.mean(), d_depth=s.depth - q.depth.mean(), d_off_unmarked=s.off_unmarked - q.off_unmarked.mean(),
                                         d_off_marked=s.off_marked - q.off_marked.mean(), d_own=s.own - q.own.mean(), d_iama=s.iama - q.iama.mean(), refs=len(q)))
P = pd.DataFrame(rows); P.to_csv(os.path.join(H, "metaa_depth_diag.csv"), index=False); D.to_csv(os.path.join(H, "metaa_depth_diag_donors.csv"))
rho = lambda a, b: spearmanr(a, b, nan_policy="omit").correlation
d["A_c"] = d.A - d.groupby("cell").A.transform("median"); d["dep_r"] = d.groupby("cell").depth.rank(pct=True); mlt = d.groupby("cell").filter(lambda g: len(g) >= 2)
sg = [np.sign(spearmanr(g.depth, g.A).correlation) for _, g in mlt.groupby("cell") if g.depth.nunique() > 1]
L = [f"Pass 4 depth trend, diagnostic (committed chain outputs only): {len(P)} samples, {len(D)} donors",
     f"within a cell: Met-A against depth rank rho {rho(mlt.dep_r, mlt.A_c):+.3f} over {len(mlt)} samples; the deeper sample reads lower in {sum(x < 0 for x in sg)} of {len(sg)} cells",
     f"Met-A against the ratio of disorder per read (sample over its reference donors, 10 reads): rho {rho(P.ratio_h10, P.A):+.3f}; medians {P.A.median():.4f} and {P.ratio_h10.median():.4f}",
     f"Met-A against: depth minus reference depth rho {rho(P.d_depth, P.A):+.3f}; off-pattern reads, unmarked side, difference {rho(P.d_off_unmarked, P.A):+.3f}; marked side {rho(P.d_off_marked, P.A):+.3f}; own-cell share difference {rho(P.d_own, P.A):+.3f}; IAM-A difference {rho(P.d_iama, P.A):+.3f}",
     f"depth difference against: ratio of disorder per read {rho(P.d_depth, P.ratio_h10):+.3f}; off-pattern difference, unmarked {rho(P.d_depth, P.d_off_unmarked):+.3f}, marked {rho(P.d_depth, P.d_off_marked):+.3f}",
     f"across donors: disorder per read against depth {rho(D.depth, D.h10):+.3f}; IAM-A against disorder per read {rho(D.iama, D.h10):+.3f}, against depth {rho(D.iama, D.depth):+.3f}; own-cell share against depth {rho(D.own, D.depth):+.3f}",
     f"donor medians: disorder per read {D.h10.median():.4f}; off-pattern unmarked {D.off_unmarked.median():.4f}, marked {D.off_marked.median():.4f}; depth {D.depth.median():.0f}"]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
