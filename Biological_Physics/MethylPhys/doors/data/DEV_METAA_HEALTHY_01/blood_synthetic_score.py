"""Scores the constructed bloods (blood_constructed.py: modes normal and new) and their synthetic (blood_synthetic.py: modes donors and new); the
first mode gives Normal. Scores the constructed-blood synthetic (written and committed 2026-11-08 before it runs). For each of the six cells: its Normal from the 100
donor bloods (its Met-A where the IAM-Deconvolver found it: median plus or minus t(0.975, n - 1) x sqrt(1 + 1/n) x the standard deviation; a cell
found in fewer than 3 donor bloods has no width). Bars (false-fail rates for a 95 % Normal computed first): 1. of the new bloods, at least 90 % of
all their cell readings with a width inside Normal (600 readings: under 0.1 %); 2. each of the six cells at least 85 % of its readings inside
(under 0.1 %); 3. every new blood's six cells found, each within 5 points of its built share.
Usage: python3 blood_synthetic_score.py FOLDER OUT.txt"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy.stats import t
src, ot = sys.argv[1:3]; R = []
for f in glob.glob(os.path.join(src, "**", "blood_synthetic_*.json"), recursive=True) + glob.glob(os.path.join(src, "**", "blood_constructed_*.json"), recursive=True): R += json.load(open(f))
for r in R: r["mode"] = "donors" if r["mode"] == "normal" else r["mode"]
SIX = ["neutrophils", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells"]; rows = []
for r in R:
    got = {c["cell"]: c for c in r.get("cells") or []}   # the cells found by their own names; a subtype found instead of its parent counts as not found
    for c in SIX: rows.append(dict(mode=r["mode"], blood=r["blood"], cell=c, built=r["built"][c], found=got.get(c, {}).get("share"), met_a=got.get(c, {}).get("met_a")))
X = pd.DataFrame(rows).drop_duplicates(["mode", "blood", "cell"]); L = [f"constructed bloods read: donors {X[X['mode'] == 'donors'].blood.nunique()}, new {X[X['mode'] == 'new'].blood.nunique()}"]
tot_in = tot = 0; ok2 = True
for c in SIX:
    dn = X[(X["mode"] == "donors") & (X.cell == c)].met_a.dropna(); nw = X[(X["mode"] == "new") & (X.cell == c)].met_a.dropna()
    if len(dn) < 3: L.append(f"  {c:12} no width (found in {len(dn)} donor bloods)"); continue
    k = t.ppf(0.975, len(dn) - 1) * np.sqrt(1 + 1 / len(dn)); lo, hi = dn.median() - k * dn.std(), dn.median() + k * dn.std(); ins = int(((nw >= lo) & (nw <= hi)).sum())
    tot_in += ins; tot += len(nw); ok2 &= len(nw) > 0 and ins >= 0.85 * len(nw)
    L.append(f"  {c:12} Normal {lo:.4f} to {hi:.4f} (donor bloods {len(dn)}, median {dn.median():.4f}) | new inside {ins} of {len(nw)}")
n = X[X["mode"] == "new"]; ok3 = bool(n.found.notna().all() and ((n.found - n.built).abs() <= 0.05).all()) and n.blood.nunique() == 100
ok1 = tot > 0 and tot_in >= 0.9 * tot
L += [f"bar 1 (at least 90 % of the new bloods' cell readings inside Normal): {tot_in} of {tot}, {'met' if ok1 else 'not met'}", f"bar 2 (each cell at least 85 %): {'met' if ok2 else 'not met'}",
      f"bar 3 (every new blood's six cells found within 5 points of the built share): {'met' if ok3 else 'not met'} (largest miss {(n.found - n.built).abs().max():.4f})",
      f"test: {'met' if ok1 and ok2 and ok3 else 'not met'}"]
X.to_csv(ot.replace(".txt", ".csv"), index=False); open(ot, "w").write("\n".join(L) + "\n"); print("\n".join(L))
