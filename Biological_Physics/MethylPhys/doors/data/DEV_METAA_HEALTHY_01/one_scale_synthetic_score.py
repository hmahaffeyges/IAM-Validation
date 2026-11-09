"""Scores one_scale_synthetic.py (written with it, before it runs). Bars (the instrument's job: a laboratory's array line does not move a
healthy cell's reading): 1. one scale: each of the six cells' prediction-corrected Met-A (ratio) under the worst line within 0.005 of the no-line
reading, median over the six bloods' cells; 2. the planted 3 % neutrophil loss read under the worst line within 0.005 of the loss read with no
line, in 5 of the 6 bloods. Reported beside them, not scored: the uncorrected Met-A (A) on both paths (the old path's rows carry no prediction; fixed 2026-11-09 after the first scoring raised on it). Usage: one_scale_synthetic_score.py FOLDER OUT.txt"""
import os, sys, json, glob
import numpy as np
R = [r for f in glob.glob(os.path.join(sys.argv[1], "one_scale_synthetic_p*.json")) for r in json.load(open(f)) if r["rc"] == 0]
SIX = ["neutrophils", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells"]
def get(lab, plant): return {r["blood"]: r for r in R if r["lab"] == lab and r["plant"] == plant}
FIELD = "ratio"
def v(r, path, c): return next((x.get(FIELD) for x in r.get(path) or [] if x["cell"] == c), None)
L = [f"{len(R)} readings"]; out = {}
for path, FIELD in (("one_scale_cells", "ratio"), ("one_scale_cells", "A"), ("selftare_cells", "A")):
    N0 = get("none", False); d1 = {}
    for lab in ("typical", "worst"):
        X = get(lab, False); d1[lab] = [abs(v(X[b], path, c) - v(N0[b], path, c)) for b in N0 if b in X for c in SIX if None not in (v(X[b], path, c), v(N0[b], path, c))]
    P0, PW = get("none", True), get("worst", True); d2 = []
    for b in N0:
        if b in P0 and b in PW and b in get("worst", False):
            a0 = v(P0[b], path, "neutrophils"); h0 = v(N0[b], path, "neutrophils"); aw = v(PW[b], path, "neutrophils"); hw = v(get("worst", False)[b], path, "neutrophils")
            if None not in (a0, h0, aw, hw): d2.append(abs((aw - hw) - (a0 - h0)))
    out[(path, FIELD)] = (d1, d2)
    L.append(f"{path} {FIELD}: typical line moves a cell by median {np.median(d1['typical']):.4f} (max {max(d1['typical']):.4f}); worst line median {np.median(d1['worst']):.4f} (max {max(d1['worst']):.4f}); planted loss changed by the worst line, median {np.median(d2):.4f}")
d1, d2 = out[("one_scale_cells", "ratio")]; m1 = len(d1["worst"]) > 0 and float(np.median(d1["worst"])) <= 0.005; m2 = sum(x <= 0.005 for x in d2) >= 5
L += [f"bar 1 (one scale, worst line, median within 0.005): {'met' if m1 else 'not met'}", f"bar 2 (planted loss within 0.005, 5 of 6): {sum(x <= 0.005 for x in d2)} of {len(d2)}: {'met' if m2 else 'not met'}", "MET" if (m1 and m2) else "NOT MET"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
