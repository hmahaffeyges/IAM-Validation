#!/usr/bin/env python3
"""Atlas v2.1 readiness L22 (2026-10-14), committed BEFORE it runs: the laboratory terms the fit will use, on the FINAL build table
(final_table_v21.py) and the bounded transfer (L21). The terms of twin run 2 were computed on a straight line and on a table that still held
sets L8 later removed. Uses the twin/terms job's loaders, run09, run10 and run12 unchanged (executed from v21_twin_terms_box.py), with hooks
that only keep what those functions already build.

PART A, twin recheck. Removing samples from a cell can only narrow its sample range, so v2's non-overlap count of separating sites can only
rise and a not-twin pair at sample level cannot become a twin by losing samples; the profile r can still move. So: cells whose sample set
differs from twin run 2's (S12b) are rechecked: profile r against every admitted cell (run09's profiles and cross-platform map), and v2's
sample-level test (run10) on every such pair with r >= 0.975. RULE: no sample-level twin among admitted cells. Otherwise STOP.
PART B, run12 on the final table (record): array-laboratory offsets d, centered as v2 did; the straight-line sequencing fits are recorded
only for comparison.
PART C, out-of-sample test of the bounded transfer g (transfer_bounded_v21.json, built on L20's 17 cells): on the Loyfer-shared cells of the
final table that are NOT among those 17, every bin with >= 500 pairs within +/-0.02 of g (L20's bins). RULE: met, or STOP.
PART D, other sequencing sources take Loyfer's array-to-sequencing shape with their own linear terms on top (v2's precedent: "ENCODE would
take Loyfer's WGBS transfer"): y = alpha + beta * g(m). Directly identified sources: alpha, beta by Deming regression of y on g(x) over run12's
own pairs; via-Loyfer sources: run12's own sequencing-to-Loyfer line. Same bar (+/-0.02, bins >= 500) -> MEASURED; otherwise the source
enters marked SHAPE NOT CONFIRMED with its bin table (v2's rule: keep it in and record the truth).
PART E, array laboratories identified only through Loyfer: per shared cell, d solves median(y_Loyfer - g(x_lab - d)) = 0 (bisection; g is
non-decreasing); d = median over cells, not centered (as v2). Laboratories with no shared cell: ASSUMED 0 (as v2).
FIT RULE recorded with the terms: a sequencing row's mean is h(m) (g, or alpha + beta g); its variance slope is the centered difference
(h(m + 0.01) - h(m - 0.01)) / 0.02, continuous in m, replacing v2's constant b.
Writes terms_final_v21.json, terms_final_v21_bins.csv, terms_final_v21_twins.csv, terms_final_v21_summary.txt."""
import os, sys, json, time, hashlib
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
FILES = ["v21_terms_final_box.py", "v21_twin_terms_box.py", "transfer_bounded_v21.py", "transfer_bounded_v21.json", "final_table_v21.py", "publish_code.py",
         "roster_samples.csv", "roster.csv", "hsc_manifest.csv", "twin_family_thresholds_v1.json", "v21_samples_all.csv", "atlas_v21_build_samples.csv",
         "conditions_v21.json", "twin_sample_level_v21.csv", "stage1_v21_manifest.csv"]
publish_code("results/ATLAS_V21/terms_final/code/run1/", FILES, job="atlas v2.1 final terms (L22)", commit=os.environ.get("REPO_COMMIT"))
src = open("v21_twin_terms_box.py").read()
exec(src[src.index("import os, sys, json, itertools"):src.index("sys.path.insert(0, os.getcwd())")])
exec(src[src.index("BUCKET = "):src.index("# ---------------- 09:")])
import final_table_v21 as FT, transfer_bounded_v21 as TB
F, FL = FT.build("."); F.to_csv("final_samples_v21.csv", index=False); FSHA = hashlib.sha256(open("final_samples_v21.csv", "rb").read()).hexdigest()
for l in FL: log(l)
lines = src.split("\n"); grab = lambda st: next(l for l in lines if l.startswith(st))
for st in ('S2 = pd.read_csv("roster_samples.csv")', 'H = pd.read_csv("hsc_manifest.csv")', "S12 = pd.concat([S2,", "adm = RO.index["): exec(grab(st))
S21 = pd.read_csv("v21_samples_all.csv", keep_default_na=False, dtype={"path": str}); S21["qc"] = S21.qc.astype(str).eq("True")
add = S21[S21.path != ""]; new_cells = sorted(set(add[add.qc].cell) - set(RO.index)); cnt = add[add.qc].groupby("cell").size()
ent = lambda c: (str(RO.loc[c, "entry"]).startswith("ADMIT") if c in RO.index else int(cnt.get(c, 0)) >= 2)
adm21 = adm + [c for c in new_cells if ent(c)]
S12b = pd.concat([S12.assign(path=""), add[add.qc & add.cell.isin(adm21)]], ignore_index=True)       # twin run 2's terms table
key = lambda T: T.groupby("cell")["sample"].apply(lambda s: tuple(sorted(map(str, s)))).to_dict()
k0, k1 = key(S12b), key(F); changed = sorted(c for c in k1 if k0.get(c) != k1[c])
log("cells whose sample set changed:", changed)
# ---- PART A
r09 = src[src.index("def run09"):src.index("# ---------------- 10:")]
hook = "    admitted = [c for c in admitted_order if entry_ok(c) and c in prof]"; assert r09.count(hook) == 1
PROF = {}
exec(r09.replace(hook, "    PROF.update(prof); return None\n" + hook), globals())
exec(src[src.index("def run10"):src.index("# ---------------- 12:")], globals())
run09(F, [], lambda c: True)
rows = []
for a in changed:
    if a not in PROF: continue
    pa, xa = PROF[a]
    for b, (pb, xb) in PROF.items():
        if b == a: continue
        j = xa.dropna().index.intersection(xb.dropna().index)
        if len(j) < 1000: continue
        u, v = xa.loc[j].values, xb.loc[j].values
        if pa != pb:
            if pa == "array": s_, i_ = np.polyfit(u, v, 1); u = i_ + s_ * u
            else: s_, i_ = np.polyfit(v, u, 1); v = i_ + s_ * v
        rows.append(dict(a=a, b=b, r=round(float(np.corrcoef(u, v)[0, 1]), 4)))
RA = pd.DataFrame(rows, columns=["a", "b", "r"]); PA = run10(F, RA, faithful=False) if (RA.r >= 0.975).any() else pd.DataFrame(columns=["a", "b", "twin"])
PA.to_csv("terms_final_v21_twins.csv", index=False)
twins_A = PA[PA.twin.astype(str) == "True"][["a", "b"]].values.tolist() if len(PA) else []
A_ok = not twins_A; log("PART A pairs tested", len(PA), "| twins", twins_A)
# ---- PART B (run12 with hooks that keep its own pairs and means)
r12 = src[src.index("# ---------------- 12:"):src.index('S2 = pd.read_csv("roster_samples.csv")')]
h1 = "        X = pd.concat(pairs.values()); a, b = deming(X.x.values, X.y.values); bs = []"; assert r12.count(h1) == 1
h2 = next(l for l in r12.split("\n") if l.startswith("    cells = sorted({k[0] for k in M})"))
PDUMP, MD = {}, {}
r12 = r12.replace(h1, h1 + "\n        PDUMP[k] = {c: p.copy() for c, p in pairs.items()}").replace(h2, h2 + "\n    MD.update(M=M, arr_mean=arr_mean, cells=cells, ARR=ARR, SEQ=SEQ)")
exec(r12, globals())
SUBORDER = list(SUBV2) + sorted(set(F[F.why != "v2"].source))
O = run12(F, SUBORDER); json.dump(O, open("terms_final_v21_run12.json", "w"), indent=1); log("run12 done")
# ---- PART C
TJ = json.load(open("transfer_bounded_v21.json")); kx, ky = np.array(TJ["knots_x"]), np.array(TJ["knots_y"])
g = lambda x: np.interp(x, kx, ky)
L = PDUMP["Loyfer2023"]; oos = sorted(set(L) - set(TJ["cells"]))
bins_rows = []
def bintab(X, pred, label):
    T = TB.bins(X); T["pred"] = pred(T.xm.values); T["resid"] = T.ym - T.pred; T["source"] = label; bins_rows.append(T.assign(bin=T.bin.astype(str))); return T
if oos:
    TC = bintab(pd.concat([L[c] for c in oos]), g, "Loyfer2023 out-of-sample"); C_ok = bool((TC.resid.abs() <= TB.TOL).all()); C_worst = float(TC.resid.abs().max()); C_bins = len(TC)
else: C_ok, C_worst, C_bins = None, None, 0
log("PART C cells", oos, "| bins", C_bins, "| worst", C_worst, "| met", C_ok)
# ---- PART D
def deming(x, y):
    sx, sy = np.var(x), np.var(y); sxy = np.cov(x, y)[0, 1]; b = (sy - sx + np.sqrt((sy - sx) ** 2 + 4 * sxy ** 2)) / (2 * sxy); return float(np.mean(y) - b * np.mean(x)), float(b)
TRN = {"Loyfer2023": {"form": "bounded", "knots_x": kx.tolist(), "knots_y": ky.tolist(), "status": "MEASURED (L21; out-of-sample check PART C)"}}
for k, v in O["sequencing"].items():
    if k == "Loyfer2023": continue
    if v.get("status") == "IDENTIFIED":
        X = pd.concat(PDUMP[k].values()); al, be = deming(g(X.x.values), X.y.values); how = "direct: Deming of y on g(x)"
    elif v.get("status") == "IDENTIFIED VIA LOYFER":
        al, be = v["a_vs_loyfer"], v["b_vs_loyfer"]; X = None; how = "via Loyfer: run12's sequencing-to-Loyfer line on top of g"
    else:
        TRN[k] = {"form": "composed", "alpha": 0.0, "beta": 1.0, "status": "ASSUMED (no shared cell): takes Loyfer's transfer unchanged"}; continue
    st = "MEASURED"
    if X is not None:
        T = bintab(X, lambda x, al=al, be=be: al + be * g(x), k); okk = bool((T.resid.abs() <= TB.TOL).all()) and len(T) > 0
        Tl = TB.bins(X); lin = float((Tl.ym - (v["a"] + v["b"] * Tl.xm)).abs().max())
        st = ("MEASURED" if okk else "SHAPE NOT CONFIRMED") + f" ({len(T)} bins, worst {T.resid.abs().max():.4f}; straight line worst {lin:.4f})"
    TRN[k] = {"form": "composed", "alpha": al, "beta": be, "how": how, "status": st}
    log("PART D", k, TRN[k]["status"])
# ---- PART E
M, AM = MD["M"], MD["arr_mean"]
ARRT = {}
for k, v in O["array"].items():
    if "d" in v: ARRT[k] = {"d": v["d"], "status": f"MEASURED ({v['overlap_cells']} shared cells, centered)"}; continue
    ov = [c for c in MD["cells"] if (c, k) in M and (c, "Loyfer2023") in M]
    if not ov: ARRT[k] = {"d": 0.0, "status": "ASSUMED 0 (no shared cell with another array laboratory or Loyfer)"}; continue
    ds = []
    for c in ov:
        Z = pd.DataFrame({"x": M[(c, k)], "y": M[(c, "Loyfer2023")]}).dropna(); lo, hi = -0.3, 0.3
        f = lambda d: float(np.median(Z.y.values - g(Z.x.values - d)))
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid) < 0: lo = mid
            else: hi = mid
        ds.append((lo + hi) / 2)
    ARRT[k] = {"d": float(np.median(ds)), "status": f"MEASURED VIA LOYFER on the bounded transfer ({ov}); not centered", "straight_line_d": v.get("d_via_loyfer")}
    log("PART E", k, ARRT[k])
ok = A_ok and C_ok is not False
out = {"rule": "v21_terms_final_box.py docstring", "final_table_sha256": FSHA, "sequencing": TRN, "array": ARRT,
       "variance_slope": "(h(m + 0.01) - h(m - 0.01)) / 0.02 for a sequencing row with mean h(m)", "checks": {"A_no_new_twin": A_ok, "A_pairs_tested": int(len(PA)),
       "C_out_of_sample_cells": oos, "C_bins": C_bins, "C_worst": C_worst, "C_met": C_ok}, "L22": "MET" if ok else "NOT MET"}
json.dump(out, open("terms_final_v21.json", "w"), indent=1)
pd.concat(bins_rows).to_csv("terms_final_v21_bins.csv", index=False) if bins_rows else None
S = [f"Atlas v2.1 readiness L22, final terms, run once {time.strftime('%Y-%m-%d')} (v21_terms_final_box.py)", *FL, f"final table sha256 {FSHA}",
     f"PART A twin recheck: changed cells {changed}; pairs tested at sample level {len(PA)}; twins {twins_A} -> {'met' if A_ok else 'NOT MET'}",
     f"PART C out-of-sample transfer: cells {oos}; {C_bins} bins; worst {C_worst} -> {C_ok}",
     *[f"PART D {k}: alpha {v.get('alpha')}, beta {v.get('beta')} | {v['status']}" for k, v in TRN.items() if k != "Loyfer2023"],
     *[f"PART E {k}: d {v['d']:.5f} | {v['status']}" for k, v in ARRT.items()],
     f"VERDICT: L22 {'MET' if ok else 'NOT MET'}"]
open("terms_final_v21_summary.txt", "w").write("\n".join(S) + "\n"); print("\n".join(S), flush=True)
for f in ("terms_final_v21.json", "terms_final_v21_summary.txt", "terms_final_v21_bins.csv", "terms_final_v21_twins.csv", "terms_final_v21_run12.json", "final_samples_v21.csv"):
    if os.path.exists(f): s3.upload_file(f, BUCKET, "results/ATLAS_V21/terms_final/" + f)
open("DONE", "w").write("ok\n")
