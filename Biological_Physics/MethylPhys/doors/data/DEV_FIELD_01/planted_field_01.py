"""DEV-FIELD-01 planted test of bar F1 (written 2026-10-13, BEFORE it was run; no GSE287331 array is read).

F1, as drafted: in 31 women, R = Met-A_blind(other quadrant) / Met-A_blind(her other breast), both on one slide; F1 MET if R > 1 in
more women than chance (one-sided sign test, p < 0.05, which at n = 31 needs at least 21 of 31).

The reading here is the sealed one of DEV-TISSUE-BREAST-01, unchanged: Met-A_blind = mean binary entropy at the 6,000 composition-blind
sites listed in sealed_breast_tissue.json (3,000 methylated, 3,000 unmethylated), H as in score_breast_tissue.py.

# NO CHAIN: the sealed scorer's own arithmetic (mean entropy at a fixed site list) on the chain's Stage 1 betas. The planted change is
# the planting rule of plant_real_normals.py (DEV-TISSUE-BREAST-01 synthetic test 4): methylated identity sites lose a share d,
# unmethylated ones gain d (1 - beta), applied here to the blind sites of one array.

Arrays: the 97 TCGA-BRCA solid-tissue normals (spent; development use). Stand-in for "one woman, one slide": two DIFFERENT women on one
TCGA plate. That carries person-to-person spread the real comparison does not, so this test is CONSERVATIVE.
Each draw: within every plate, shuffle and pair the arrays; take 31 pairs at random; the first member of each pair is the "other
quadrant" and gets the plant; R per pair; sign test. 1,000 draws per case. Cases: NULL (d = 0), d = 0.005, 0.01, 0.02, 0.05.
BARS
  P1 null honest   NULL meets F1 in at most 10 % of draws
  P2 sensitivity   d = 0.01 meets F1 in at least 80 % of draws
If P2 fails, the note states the smallest d that reaches 80 %, and the design's sensitivity is that, not 1 %.
Usage: python3 planted_field_01.py BETADIR    (the TCGA normal betas, one parquet per sample)
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__)); BD = sys.argv[1]
B = os.path.join(HERE, "..", "DEV_TISSUE_BREAST_01")
S = json.load(open(os.path.join(B, "sealed_breast_tissue.json"))); HI, LO = S["sites_methylated"], S["sites_unmethylated"]
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
N = pd.read_csv(os.path.join(B, "null_normals_output.csv")).set_index("sample")
X = {}
for s in N.index:
    b = pd.read_parquet(os.path.join(BD, s + ".parquet")); b = b["beta"] if "beta" in b.columns else b.iloc[:, 0]
    X[s] = (b.reindex(HI).values, b.reindex(LO).values)
ok_hi = np.all([np.isfinite(v[0]) for v in X.values()], 0); ok_lo = np.all([np.isfinite(v[1]) for v in X.values()], 0)
X = {s: (v[0][ok_hi], v[1][ok_lo]) for s, v in X.items()}
def meta(hi, lo): return float(np.concatenate([H(hi), H(lo)]).mean())
def plant(hi, lo, d): return hi * (1 - d), lo + d * (1 - lo)
NP, CRIT = 31, int(next(k for k in range(32) if binom.sf(k - 1, 31, 0.5) < 0.05))
rg = np.random.default_rng(20261013); plates = N.plate
rows = []
for case, d in (("NULL", 0.0), ("d0.005", 0.005), ("d0.01", 0.01), ("d0.02", 0.02), ("d0.05", 0.05)):
    hits = []; ks = []
    for _ in range(1000):
        pairs = []
        for p, g in plates.groupby(plates):
            m = list(rg.permutation(g.index)); pairs += [(m[i], m[i + 1]) for i in range(0, len(m) - 1, 2)]
        pick = [pairs[i] for i in rg.choice(len(pairs), NP, replace=False)]
        k = 0
        for a, b in pick:
            ha, la = plant(*X[a], d); k += meta(ha, la) / meta(*X[b]) > 1
        ks.append(k); hits.append(k >= CRIT)
    rows.append(dict(case=case, d=d, draws=1000, meets_F1=float(np.mean(hits)), median_k=float(np.median(ks))))
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "planted_field_01_output.csv"), index=False)
p1 = T.set_index("case").loc["NULL", "meets_F1"] <= 0.10; p2 = T.set_index("case").loc["d0.01", "meets_F1"] >= 0.80
reach = T[(T.case != "NULL") & (T.meets_F1 >= 0.80)].d.min()
L = [f"sites used: {int(ok_hi.sum())} methylated + {int(ok_lo.sum())} unmethylated of the 6,000 sealed | normals {len(X)} | "
     f"same-plate pairs available per draw: up to {sum(len(g) // 2 for _, g in plates.groupby(plates))} | F1 needs >= {CRIT} of {NP}"]
L += [f"  {r.case:<7} meets F1 in {r.meets_F1:.1%} of draws (median {r.median_k:.0f} of {NP} pairs read R > 1)" for r in T.itertuples()]
L.append(f"P1 null honest (<= 10 %): {'MET' if p1 else 'NOT MET'} | P2 1 % loss found (>= 80 %): {'MET' if p2 else 'NOT MET'}"
         f" | smallest planted d reaching 80 %: {reach if reach == reach else 'none tried'}")
open(os.path.join(HERE, "planted_field_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
