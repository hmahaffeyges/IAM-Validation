"""DEV-TISSUE-BREAST-01 synthetic test 7, power (2026-10-11; real normals only). Design: each tumour read against its own patient's normal on the
same plate, at the composition-blind sites (blind_sites_test.py), ratio R = Met-A(tumour) / Met-A(own normal). Stand-in pair: two different
normals j, k on one plate (conservative: a real pair shares the person). Fake tumour = normal j blended toward the primary mammary epithelial 450K
array (TEMPLATE, a ~ U(0.3, 0.6)) with a planted change d in that part. Per-pair bar = 95th percentile of R under NULL (j vs k, no blend).
Reported per case: share of pairs above the bar; and the group test over 56 pairs (sign test, one-sided, p < 0.05), detection share over 2,000
draws of 56 pairs. Writes power_pairs_output.csv. Usage: TEMPLATE=... python3 power_pairs.py POSTDIR BETADIR"""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = sys.argv[:3]
src = open(os.path.join(HERE, "blind_sites_test.py")).read(); exec(src[:src.index("tared = np.array")])
TPL = os.environ["TEMPLATE"]; lum = pd.read_parquet(TPL).beta.rename(lambda x: str(x)).reindex(Mu.index).fillna(Mu["luminal"]).values
rg = np.random.default_rng(20261011); pairs = [(j, k) for j, k in itertools.permutations(keep, 2) if plates[j] == plates[k]]
def R(case):
    out = []
    for j, k in pairs:
        sh, d = case; a = rg.uniform(.3, .6) if sh else 0.0; L_ = lum.copy(); L_[lo_] = L_[lo_] + d * (1 - L_[lo_]); L_[hi_] = L_[hi_] * (1 - d)
        out.append(H(np.clip((1 - a) * X[j] + a * L_, 0, 1)[S_]).mean() / base[k])
    return np.array(out)
null = R((0, 0)); bar = np.quantile(null, .95); rows = []
for name, case in {"NULL": (0, 0), "SHIFT": (1, 0), "d0.02": (1, .02), "d0.05": (1, .05), "d0.10": (1, .10)}.items():
    v = null if name == "NULL" else R(case); above = (v > bar)
    crit = int(binom.ppf(0.95, 56, 0.05)) + 1      # pairs above the bar needed for p < 0.05 when each null pair crosses with probability 0.05
    det = np.mean([above[rg.integers(len(v), size=56)].sum() >= crit for _ in range(2000)])
    rows.append(dict(case=name, pairs=len(v), share_above_bar=round(above.mean(), 3), bar=round(bar, 4), crit_of_56=crit, group_detection=round(det, 3))); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "power_pairs_output.csv"), index=False)
