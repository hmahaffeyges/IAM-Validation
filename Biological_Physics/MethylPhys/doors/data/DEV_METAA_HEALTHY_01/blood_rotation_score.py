"""Scores the rotation test of a Normal for each cell inside a whole blood (written and committed 2026-11-08 before its readings exist). 200 new
constructed bloods (blood_constructed.py MODE all: every one of the atlas's 146 blood-cell donor arrays in the pool, new seeds; shares of the 100
healthy bloods, each used twice). For each cell and each of its donors in turn: Normal from the bloods built with the cell's other donors,
median plus or minus t(0.975, d - 1) x sqrt(1 + 1/d) x their standard deviation, d the number of those donors (the interval counts people, not
bloods: the earlier test's error); the left-out donor's bloods are scored against it. Only where the Deconvolver found the cell; its shares
are tested separately. Bars: 1. at least 90 % of all scored readings inside Normal; 2. each of the six cells at least 85 %.
Usage: python3 blood_rotation_score.py FOLDER OUT.txt"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy.stats import t
src, ot = sys.argv[1:3]; R = []
for f in glob.glob(os.path.join(src, "**", "blood_constructed_all_*.json"), recursive=True): R += json.load(open(f))
SIX = ["neutrophils", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells"]; rows = []
for r in R:
    got = {c["cell"]: c for c in r.get("cells") or []}
    for c in SIX: rows.append(dict(blood=r["blood"], cell=c, array=r["arrays"][c], met_a=got.get(c, {}).get("met_a")))
Y = pd.DataFrame(rows).drop_duplicates(["blood", "cell"]); L = [f"rotation test: {Y.blood.nunique()} constructed bloods read"]; ok2 = True; TI = TN = 0
for c in SIX:
    g = Y[(Y.cell == c)].dropna(subset=["met_a"]); ins = n = 0
    for d in g.array.unique():
        ref, test = g[g.array != d], g[g.array == d]; nd = ref.array.nunique()
        if nd < 3: continue
        k = t.ppf(0.975, nd - 1) * np.sqrt(1 + 1 / nd); lo, hi = ref.met_a.median() - k * ref.met_a.std(), ref.met_a.median() + k * ref.met_a.std()
        ins += int(test.met_a.between(lo, hi).sum()); n += len(test)
    ok2 &= n > 0 and ins >= 0.85 * n; TI += ins; TN += n
    L.append(f"  {c:12} donors {g.array.nunique():3} | readings {n:3} | inside {ins} ({100 * ins / max(n, 1):.1f} %) | median {g.met_a.median():.4f}")
ok1 = TN > 0 and TI >= 0.9 * TN
L += [f"bar 1 (at least 90 % inside): {TI} of {TN}, {'met' if ok1 else 'not met'}", f"bar 2 (each cell at least 85 %): {'met' if ok2 else 'not met'}", f"test: {'met' if ok1 and ok2 else 'not met'}"]
Y.to_csv(ot.replace(".txt", ".csv"), index=False); open(ot, "w").write("\n".join(L) + "\n"); print("\n".join(L))
