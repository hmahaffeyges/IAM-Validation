"""DEV-BREACH-01 planted test of bars B1 and B2 (written 2026-10-13, BEFORE it was run; no GSE281307 array is read).

Reading, as drafted: M = mean binary entropy at the 6,000 sealed composition-blind breast sites, divided by the median M of the arrays on
its own slide. B1: "progressors" above "non-progressors", one-sided Mann-Whitney p < 0.05, 40 against 46. B2: slides split in half; the
line that best separates the groups on half A (Youden) must put more "progressors" than "non-progressors" above it on half B (one-sided
Fisher, p < 0.05).

# NO CHAIN: the sealed reading's arithmetic on the chain's Stage 1 betas; the plant is plant_real_normals.py's rule (methylated sites
# lose a share d, unmethylated gain d (1 - beta)).
Arrays: the 97 spent TCGA-BRCA normals, plates with at least 3 normals standing in for slides. Each draw: 86 arrays drawn, 40 labeled
"progressor" at random (planted with d), 46 "non-progressor"; plates split in half at random for B2. 1,000 draws per case.
Cases NULL, d = 0.005, 0.01, 0.02. BARS: P1 NULL meets B1 in at most 10 % of draws; P2 d = 0.01 meets B1 in at least 80 %;
P3 d = 0.01 meets B2 in at least 50 % (B2 uses half the data twice, so it is weaker by design; the value is reported, not tuned).
Usage: python3 planted_breach_01.py BETADIR
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
HERE = os.path.dirname(os.path.abspath(__file__)); BD = sys.argv[1]; B = os.path.join(HERE, "..", "DEV_TISSUE_BREAST_01")
S = json.load(open(os.path.join(B, "sealed_breast_tissue.json"))); HI, LO = S["sites_methylated"], S["sites_unmethylated"]
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
N = pd.read_csv(os.path.join(B, "null_normals_output.csv")).set_index("sample"); pl = N.plate; N = N[pl.map(pl.value_counts()) >= 3]
X = {}
for s in N.index:
    b = pd.read_parquet(os.path.join(BD, s + ".parquet")); b = b["beta"] if "beta" in b.columns else b.iloc[:, 0]
    X[s] = (b.reindex(HI).values, b.reindex(LO).values)
ok_h = np.all([np.isfinite(v[0]) for v in X.values()], 0); ok_l = np.all([np.isfinite(v[1]) for v in X.values()], 0)
X = {s: (v[0][ok_h], v[1][ok_l]) for s, v in X.items()}
Mof = lambda hi, lo: float(np.concatenate([H(hi), H(lo)]).mean())
rg = np.random.default_rng(20261013); rows = []
for case, d in (("NULL", 0.0), ("d0.005", 0.005), ("d0.01", 0.01), ("d0.02", 0.02)):
    b1 = []; b2 = []
    for _ in range(1000):
        pick = list(rg.choice(N.index, 86, replace=False)); prog = set(rg.choice(pick, 40, replace=False))
        m = pd.Series({s: Mof(X[s][0] * (1 - d), X[s][1] + d * (1 - X[s][1])) if s in prog else Mof(*X[s]) for s in pick})
        plate = N.plate[pick]; r = m / m.groupby(plate).transform("median"); y = pd.Series({s: s in prog for s in pick})
        b1.append(mannwhitneyu(r[y], r[~y], alternative="greater").pvalue < 0.05)
        P = plate.unique(); A = set(rg.choice(P, len(P) // 2, replace=False)); ia = plate.isin(A)
        ra, ya, rb, yb = r[ia], y[ia], r[~ia], y[~ia]
        if ya.sum() == 0 or (~ya).sum() == 0 or yb.sum() == 0 or (~yb).sum() == 0: b2.append(False); continue
        th = max(np.unique(ra), key=lambda t: (ra[ya] >= t).mean() - (ra[~ya] >= t).mean())
        tab = [[int((rb[yb] >= th).sum()), int((rb[yb] < th).sum())], [int((rb[~yb] >= th).sum()), int((rb[~yb] < th).sum())]]
        b2.append(fisher_exact(tab, alternative="greater")[1] < 0.05)
    rows.append(dict(case=case, d=d, meets_B1=float(np.mean(b1)), meets_B2=float(np.mean(b2))))
T = pd.DataFrame(rows).set_index("case"); T.to_csv(os.path.join(HERE, "planted_breach_01_output.csv"))
p1, p2, p3 = T.loc["NULL", "meets_B1"] <= .10, T.loc["d0.01", "meets_B1"] >= .80, T.loc["d0.01", "meets_B2"] >= .50
L = [f"normals {len(X)} on {N.plate.nunique()} plates | sites {int(ok_h.sum())} + {int(ok_l.sum())}"]
L += [f"  {c:<7} meets B1 in {r.meets_B1:.1%}, B2 in {r.meets_B2:.1%} of draws" for c, r in T.iterrows()]
L.append(f"P1 null (B1 <= 10 %): {'MET' if p1 else 'NOT MET'} | P2 1 % (B1 >= 80 %): {'MET' if p2 else 'NOT MET'} | P3 1 % (B2 >= 50 %): {'MET' if p3 else 'NOT MET'}"
         f" | NULL meets B2 in {T.loc['NULL', 'meets_B2']:.1%}")
open(os.path.join(HERE, "planted_breach_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
