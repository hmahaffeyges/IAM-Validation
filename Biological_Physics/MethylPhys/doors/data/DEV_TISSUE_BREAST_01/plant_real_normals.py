"""DEV-TISSUE-BREAST-01 synthetic test 4 (2026-10-11; built on the REAL normal arrays, no tumour array read). Fake tumour from real normal j:
beta = (1 - a) beta_j + a beta_lum, a ~ U(0.3, 0.6) (epithelium-rich shift; beta_lum = atlas luminal posterior mean), with a planted change d in the
luminal part only (unmethylated luminal identity sites gain d (1 - beta), methylated ones lose a share d). Read as null_normals.py (composition-aware
Met-A) and tared by the median of the OTHER normals on j's plate (plates with >= 3 normals). Cases: NULL (a = 0, d = 0), SHIFT (composition only),
d = 0.02 / 0.05 / 0.10. Bar = the 95th percentile of the plate-tared normals (plate_tare_check). Per case: mean, SD, share above the bar.
Writes plant_real_normals_output.csv. Usage: python3 plant_real_normals.py POSTDIR BETADIR"""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import nnls
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = sys.argv[:3]
src = open(os.path.join(HERE, "null_normals.py")).read(); src = src[:src.index("rows = []")]; exec(src)   # same panel, sites, loci, H, Bt
T = pd.read_csv(os.path.join(HERE, "null_normals_output.csv")).set_index("sample"); rg = np.random.default_rng(20261011)
lum = Mu["luminal"].values; S_ = Mu.index.get_indexer(S); hi_ = Mu.index.get_indexer(hi); lo_ = Mu.index.get_indexer(lo); inf_ = Mu.index.get_indexer(inf)
def comp_of(b):
    w, _ = nnls(Mu.values[inf_], b[inf_]); w = w / w.sum(); return H(b[S_]).mean() / H(Mu.values @ w)[S_].mean()
base = {s: T.loc[s, "comp"] for s in T.index}; plates = T.plate; ok = plates.map(plates.value_counts()) >= 3
X = {s: Bt[s].loc[common].values for s in T.index[ok]}
tared = np.array([base[s] / np.median([base[o] for o in T.index[ok] if plates[o] == plates[s] and o != s]) for s in X]); bar = np.quantile(tared, .95)
rows = []
for case, (shift, d) in {"NULL": (0, 0), "SHIFT": (1, 0), "d0.02": (1, .02), "d0.05": (1, .05), "d0.10": (1, .10)}.items():
    v = []
    for s, b in X.items():
        a = rg.uniform(.3, .6) if shift else 0.0; L_ = lum.copy(); L_[lo_] = L_[lo_] + d * (1 - L_[lo_]); L_[hi_] = L_[hi_] * (1 - d)
        bt = np.clip((1 - a) * b + a * L_, 0, 1); ref = np.median([base[o] for o in X if plates[o] == plates[s] and o != s]); v.append(comp_of(bt) / ref)
    v = np.array(v); rows.append(dict(case=case, n=len(v), mean=round(v.mean(), 4), sd=round(v.std(ddof=1), 4), share_above_bar=round((v > bar).mean(), 3), bar=round(bar, 4)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "plant_real_normals_output.csv"), index=False)
