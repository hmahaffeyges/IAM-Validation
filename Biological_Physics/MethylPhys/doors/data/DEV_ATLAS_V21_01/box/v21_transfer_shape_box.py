#!/usr/bin/env python3
"""Readiness L20 (2026-10-14), committed BEFORE it runs: is the array-to-sequencing transfer a straight line across the whole range?
v2's Loyfer transfer is y = a + b x (a = -0.090, b = 1.103; x = our array scale, y = Loyfer's sequencing beta), fitted by Deming regression
on the cells both measured. The line predicts negative sequencing readings for x below 0.082 and readings above 1 for x above 0.988, where
real readings cannot go. The decision 16 synthetic test showed that readings floored at 0 bias the terms; this asks whether real data bend.
CODE: the twin/terms job's own loaders and its source-term step (run12) are executed from v21_twin_terms_box.py unchanged, on v2's own sample
table exactly as its reproduction gate builds it; one line is added inside run12 to keep the pairs it already builds for Loyfer.
RULE (fixed now): bin the pairs by x (0.01 wide in [0, 0.15] and [0.85, 1], 0.05 between). The straight line HOLDS if in every bin with at
least 500 pairs the mean residual y - (a + b x) is within +/- 0.02 (twice v2's 0.01 measurement floor). Otherwise it BENDS, and a transfer that
cannot leave [0, 1] is designed and tested on synthetic data before launch. Writes transfer_shape_v21.csv and transfer_shape_v21_summary.txt."""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
publish_code("results/ATLAS_V21/transfer/code/", ["v21_transfer_shape_box.py", "v21_twin_terms_box.py", "publish_code.py", "roster_samples.csv", "roster.csv", "hsc_manifest.csv", "twin_family_thresholds_v1.json"],
             job="atlas v2.1 transfer shape check (L20)", commit=os.environ.get("REPO_COMMIT"))
src = open("v21_twin_terms_box.py").read()
exec(src[src.index("import os, sys, json, itertools"):src.index("sys.path.insert(0, os.getcwd())")])
exec(src[src.index("BUCKET = "):src.index("# ---------------- 09:")])
r12 = src[src.index("# ---------------- 12:"):src.index('S2 = pd.read_csv("roster_samples.csv")')]   # run12 only, nothing after it
assert "def run12" in r12 and "run12(" not in r12.split("def run12", 1)[1].replace("def run12(", "")
hook = "        X = pd.concat(pairs.values()); a, b = deming(X.x.values, X.y.values); bs = []"
assert r12.count(hook) == 1
XDUMP = {}
exec(r12.replace(hook, hook + "\n        XDUMP[k] = (X.copy(), a, b)"), globals())
lines = src.split("\n"); grab = lambda start: next(l for l in lines if l.startswith(start))
for st in ('S2 = pd.read_csv("roster_samples.csv")', "H = pd.read_csv(\"hsc_manifest.csv\")", "S12 = pd.concat([S2,", "adm = RO.index["): exec(grab(st))
t0 = time.time(); O2 = run12(S12, list(SUBV2)); log("run12 done", round(time.time() - t0), "s | Loyfer a, b", O2["sequencing"]["Loyfer2023"]["a"], O2["sequencing"]["Loyfer2023"]["b"])
X, a, b = XDUMP["Loyfer2023"]
edges = np.unique(np.round(np.concatenate([np.arange(0, 0.15, 0.01), np.arange(0.15, 0.85, 0.05), np.arange(0.85, 1.0001, 0.01)]), 4))
X["bin"] = pd.cut(X.x, edges, include_lowest=True); X["res"] = X.y - (a + b * X.x)
T = X.groupby("bin", observed=True).agg(n=("x", "size"), x_median=("x", "median"), y_median=("y", "median"), y_share_at_or_below_0_01=("y", lambda v: float((v <= 0.01).mean())),
                                         mean_residual=("res", "mean"), line_at_median=("x", lambda v: a + b * float(np.median(v)))).reset_index()
T["bin"] = T["bin"].astype(str); T.to_csv("transfer_shape_v21.csv", index=False)
big = T[T.n >= 500]; worst = big.loc[big.mean_residual.abs().idxmax()]
verdict = "HOLDS" if (big.mean_residual.abs() <= 0.02).all() else "BENDS"
L = [f"Atlas v2.1 readiness L20, transfer shape, run once {time.strftime('%Y-%m-%d')} (v21_transfer_shape_box.py)",
     f"Loyfer transfer reproduced: a {a:.6f}, b {b:.6f}, pairs {len(X):,} over {len(O2['sequencing']['Loyfer2023']['overlap_cells'])} shared cells",
     f"bins with >= 500 pairs: {len(big)}; worst mean residual {worst.mean_residual:+.4f} in bin {worst.bin} (n {int(worst.n)})",
     f"bins beyond +/-0.02: {big[big.mean_residual.abs() > 0.02].bin.tolist()}", f"VERDICT: the straight line {verdict}"]
open("transfer_shape_v21_summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)
for f in ("transfer_shape_v21.csv", "transfer_shape_v21_summary.txt"): s3.upload_file(f, BUCKET, "results/ATLAS_V21/transfer/" + f)
open("DONE", "w").write("ok\n")
