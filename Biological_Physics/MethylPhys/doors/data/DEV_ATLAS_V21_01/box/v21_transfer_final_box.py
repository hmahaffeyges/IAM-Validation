#!/usr/bin/env python3
"""Atlas v2.1 readiness L22, second design (2026-10-14), committed BEFORE it runs. The author chose (after L22 was NOT MET and the diagnosis):
one bounded transfer for every WGBS source, the measured cell-to-cell scatter carried into the error bars, no cell removed.
WHY a second design: L22 found the transfer built on 17 cells missed 5 unseen cells by 0.026; the diagnosis showed the miss is cell-to-cell
difference between the sequencing and array cohorts (as wide inside the 17 as outside), not laboratory offsets. Disclosed: all 22 shared cells
have now been seen in aggregate, so the test below is leave-one-cell-out, not fresh data.
DESIGN. g22: the same method as L21 (L20 bins, >= 500 pairs per knot, pool-adjacent-violators, linear between knots, held at the ends) on all
22 Loyfer-shared cells of the final table (run12's own pairs). Loyfer, Tian and ENCODE all use h(m) = g22(m), no extra layer.
SCATTER: for each left-out cell, g built on the other 21; the cell's weighted mean residual in each of four ranges of x (< 0.15, 0.15-0.45,
0.45-0.85, >= 0.85); s_R = the standard deviation of those means across the 22 cells. In the fit, s(m) is interpolated between the range
midpoints (0.075, 0.30, 0.65, 0.925; held at the ends) and a sequencing row's variance gains n * s(m)^2, n the number of that source's samples
for that cell and locus, so the source's mean carries s(m)^2 however many samples it has (the effect of one shared cell-level bias).
BARS (MET only if all hold):
 T1 leave-one-cell-out: in every L20 bin with >= 500 pooled pairs, the pair-weighted mean over cells of the left-out residuals within +/-0.02.
 T2 g22 within [0, 1] and non-decreasing.
 T3 each Tian and ENCODE cell's mean |residual| against g22 (bins with >= 200 pairs) is at most the 95th percentile of the 22 Loyfer cells'
    leave-one-out mean |residual|; a source that misses enters marked SHAPE NOT CONFIRMED (kept in, the truth recorded), and is not a stop.
Writes transfer_v21_final.json, transfer_v21_final_loco.csv, transfer_v21_final_summary.txt."""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
FILES = ["v21_transfer_final_box.py", "v21_twin_terms_box.py", "transfer_bounded_v21.py", "final_table_v21.py", "publish_code.py", "terms_final_v21.json",
         "roster_samples.csv", "roster.csv", "hsc_manifest.csv", "twin_family_thresholds_v1.json", "v21_samples_all.csv", "atlas_v21_build_samples.csv",
         "conditions_v21.json", "twin_sample_level_v21.csv", "stage1_v21_manifest.csv"]
publish_code("results/ATLAS_V21/transfer_final/code/run1/", FILES, job="atlas v2.1 L22 second design (one curve, scatter carried)", commit=os.environ.get("REPO_COMMIT"))
src = open("v21_twin_terms_box.py").read()
exec(src[src.index("import os, sys, json, itertools"):src.index("sys.path.insert(0, os.getcwd())")])
exec(src[src.index("BUCKET = "):src.index("# ---------------- 09:")])
import final_table_v21 as FT, transfer_bounded_v21 as TB
F, _ = FT.build(".")
r12 = src[src.index("# ---------------- 12:"):src.index('S2 = pd.read_csv("roster_samples.csv")')]
h1 = "        X = pd.concat(pairs.values()); a, b = deming(X.x.values, X.y.values); bs = []"
PDUMP = {}; exec(r12.replace(h1, h1 + "\n        PDUMP[k] = {c: p.copy() for c, p in pairs.items()}"), globals())
O = run12(F, list(SUBV2) + sorted(set(F[F.why != "v2"].source)))
P = PDUMP["Loyfer2023"]; cells = sorted(P); log("Loyfer-shared cells", len(cells))
kx, ky = TB.build(pd.concat(P.values())); g = lambda x, kx=kx, ky=ky: np.interp(x, kx, ky)
T2 = bool((ky >= 0).all() and (ky <= 1).all() and (np.diff(ky) >= 0).all())
RANGES = [(0, 0.15), (0.15, 0.45), (0.45, 0.85), (0.85, 1.0001)]; MID = [0.075, 0.30, 0.65, 0.925]
rows, cellR = [], []
for c in cells:
    bx, by = TB.build(pd.concat([P[o] for o in cells if o != c])); X = P[c].assign(bin=pd.cut(P[c].x, TB.EDGES, include_lowest=True))
    X["r"] = X.y - np.interp(X.x, bx, by)
    T = X.groupby("bin", observed=True).agg(n=("x", "size"), r=("r", "mean")).reset_index(); T["cell"] = c; rows.append(T)
    cellR.append(dict(cell=c, mean_abs=float(np.average(T[T.n >= 200].r.abs(), weights=T[T.n >= 200].n)),
                      **{f"R{i}": float(X[(X.x >= lo) & (X.x < hi)].r.mean()) for i, (lo, hi) in enumerate(RANGES)}))
L = pd.concat(rows); L["bin"] = L["bin"].astype(str); L.to_csv("transfer_v21_final_loco.csv", index=False)
pooled = L.groupby("bin").apply(lambda t: pd.Series({"n": t.n.sum(), "r": np.average(t.r, weights=t.n)}), include_groups=False)
pooled = pooled[pooled.n >= 500]; T1 = bool((pooled.r.abs() <= 0.02).all()); T1w = float(pooled.r.abs().max())
CR = pd.DataFrame(cellR); s = [float(CR[f"R{i}"].std(ddof=1)) for i in range(4)]; p95 = float(np.percentile(CR.mean_abs, 95))
seq = {"Loyfer2023": {"form": "bounded", "status": "MEASURED (22 cells; leave-one-cell-out T1)"}}
T3 = {}
for k in ("Tian2023", "ENCODE"):
    if k not in PDUMP: continue
    vals = []
    for c, X in PDUMP[k].items():
        X = X.assign(bin=pd.cut(X.x, TB.EDGES, include_lowest=True), r=X.y - g(X.x)); T = X.groupby("bin", observed=True).agg(n=("x", "size"), r=("r", "mean"))
        T = T[T.n >= 200]; vals.append((c, float(np.average(T.r.abs(), weights=T.n))))
    ok = all(v <= p95 for _, v in vals); T3[k] = vals
    seq[k] = {"form": "bounded", "status": ("MEASURED" if ok else "SHAPE NOT CONFIRMED") + f" (cells {vals}; Loyfer leave-one-out p95 {p95:.4f})"}
TF = json.load(open("terms_final_v21.json"))
out = {"rule": "v21_transfer_final_box.py docstring", "knots_x": kx.tolist(), "knots_y": ky.tolist(), "cells": cells, "scatter_mid": MID, "scatter_s": s,
       "sequencing": seq, "array": TF["array"], "final_table_sha256": TF["final_table_sha256"],
       "fit_rule": "sequencing row: mean g(m); variance (slope^2 x donor var [/3 pooled]) + h(1-h)/cov + n x s(m)^2 + floor; slope (g(m+0.01)-g(m-0.01))/0.02",
       "checks": {"T1_worst": T1w, "T1": T1, "T2": T2, "T3": {k: [v for _, v in vv] for k, vv in T3.items()}, "p95": p95}}
ok = T1 and T2; out["L22"] = "MET" if ok else "NOT MET"
json.dump(out, open("transfer_v21_final.json", "w"), indent=1)
S = [f"Atlas v2.1 readiness L22 second design, run once {time.strftime('%Y-%m-%d')} (v21_transfer_final_box.py)",
     f"g22 on {len(cells)} Loyfer-shared cells, {sum(len(p) for p in P.values()):,} pairs, {len(kx)} knots, range [{ky.min():.4f}, {ky.max():.4f}]",
     f"T1 leave-one-cell-out pooled bias: {len(pooled)} bins, worst {T1w:.4f} (bar 0.02) -> {'met' if T1 else 'NOT met'}",
     f"T2 bounds and monotone -> {'met' if T2 else 'NOT met'}",
     f"cell-to-cell scatter s by range (<0.15, 0.15-0.45, 0.45-0.85, >=0.85): {[round(v, 4) for v in s]}",
     *[f"T3 {k}: {seq[k]['status']}" for k in T3],
     "array laboratory offsets and the twin recheck: as in L22's first run (terms_final_v21.json), unchanged",
     f"VERDICT: L22 {'MET' if ok else 'NOT MET'}"]
open("transfer_v21_final_summary.txt", "w").write("\n".join(S) + "\n"); print("\n".join(S), flush=True)
for f in ("transfer_v21_final.json", "transfer_v21_final_loco.csv", "transfer_v21_final_summary.txt"): s3.upload_file(f, BUCKET, "results/ATLAS_V21/transfer_final/" + f)
open("DONE", "w").write("ok\n")
