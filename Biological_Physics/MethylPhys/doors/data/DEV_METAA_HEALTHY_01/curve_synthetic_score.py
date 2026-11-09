"""Scores curve_synthetic.py (written with it, before it runs). Bars (the instrument's job: a bent array reads as the same blood unbent):
 1. shares: each bent blood read with the calibration lies at most half as far from the unbent reading (total variation over all cells found)
    as the same bent blood read without it, in at least 10 of the 12 bent readings;
 2. Met-A: each of the six cells' Met-A in the bent bloods read with the calibration within 0.005 of the unbent reading, median over the 12;
 3. the planted 3 % neutrophil loss survives: its change (planted minus healthy) read bent with the calibration within 0.005 of the change
    read unbent without it, in at least 5 of the 6 bloods.
Usage: curve_synthetic_score.py FOLDER OUT.txt"""
import os, sys, json, glob
import numpy as np, pandas as pd
R = [r for f in glob.glob(os.path.join(sys.argv[1], "curve_synthetic_p*.json")) for r in json.load(open(f))]
SIX = ["neutrophils", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells"]
def key(lab, cal, plant): return {(r["blood"]): r for r in R if r["lab"] == lab and r["cal"] == cal and r["plant"] == plant and r["rc"] == 0}
def sh(r): return {c["cell"]: c["share"] or 0 for c in r.get("cells") or []}
def ma(r, c): return next((x.get("met_a") for x in r.get("cells") or [] if x["cell"] == c), None)
def tv(p, q): return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))
U = key("none", False, False); L = [f"{len(R)} readings, {sum(r['rc'] == 0 for r in R)} read"]; b1 = []; b2 = []
for lab in ("A", "C"):
    raw, cal = key(lab, False, False), key(lab, True, False)
    for bl in sorted(U):
        if bl in raw and bl in cal:
            dr, dc = tv(sh(raw[bl]), sh(U[bl])), tv(sh(cal[bl]), sh(U[bl])); b1.append(dc <= 0.5 * dr)
            L.append(f"blood {bl} lab {lab}: share distance from unbent, raw {dr:.4f}, calibrated {dc:.4f}; GMP raw {sh(raw[bl]).get('GMP (bone marrow)', 0):.4f}, calibrated {sh(cal[bl]).get('GMP (bone marrow)', 0):.4f}")
            for c in SIX:
                u, k = ma(U[bl], c), ma(cal[bl], c)
                if u is not None and k is not None: b2.append(abs(k - u))
P0, PC = key("none", False, True), key("A", True, True); CA = key("A", True, False); b3 = []
for bl in sorted(U):
    if bl in P0 and bl in PC and bl in CA and None not in (ma(P0[bl], "neutrophils"), ma(U[bl], "neutrophils"), ma(PC[bl], "neutrophils"), ma(CA[bl], "neutrophils")):
        d0 = ma(P0[bl], "neutrophils") - ma(U[bl], "neutrophils"); dc = ma(PC[bl], "neutrophils") - ma(CA[bl], "neutrophils"); b3.append(abs(dc - d0) <= 0.005)
        L.append(f"blood {bl}: planted loss read unbent {d0:+.4f}, bent and calibrated {dc:+.4f}")
m1 = sum(b1) >= 10 and len(b1) == 12; m2 = len(b2) > 0 and float(np.median(b2)) <= 0.005; m3 = sum(b3) >= 5 and len(b3) == 6
L += [f"bar 1 (shares): {sum(b1)} of {len(b1)}: {'met' if m1 else 'not met'}", f"bar 2 (Met-A): median {np.median(b2) if b2 else float('nan'):.4f}: {'met' if m2 else 'not met'}",
      f"bar 3 (planted loss survives): {sum(b3)} of {len(b3)}: {'met' if m3 else 'not met'}", "MET" if (m1 and m2 and m3) else "NOT MET"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
