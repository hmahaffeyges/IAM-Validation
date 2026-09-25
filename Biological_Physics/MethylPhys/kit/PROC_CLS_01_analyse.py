#!/usr/bin/env python3
"""PROC-CLS-01 step 2: the spectra against the pre-registered bars.

Correlations are reported for EVERY band power, the total and the ratio - not only for whichever one passes.
The pre-registration named "the adopted summary statistic" without fixing which it would be, so the honest
course is to show them all and let the reader see that nothing was picked after the fact.
"""
import json

import numpy as np

BANDS = [(2, 8), (9, 24), (25, 64), (65, 128), (129, 191), (192, 255)]
LAB = ["GSE87571", "GSE42861", "GSE111629", "GSE125105"]

Z = np.load("handoff/cls01_spectra.npz", allow_pickle=True)
M = json.load(open("handoff/cls01_meta.json"))["arrays"]
B = {r["gsm"]: r for r in json.load(open("handoff/band01_arrays.json"))["arrays"]}
gsms = [g for g in M if g + "_bp" in Z.files]
bp = np.array([Z[g + "_bp"] for g in gsms])
nul = np.array([Z[g + "_null"] for g in gsms])
fsky = np.array([M[g]["f_sky"] for g in gsms])
lab = np.array([M[g]["gse"] for g in gsms])
print("arrays: %d | bands: %s" % (len(gsms), BANDS))

# ---------------------------------------------------------------- B1
ok = fsky >= 0.5
print("\nB1 computable with f_sky >= 0.5: %.1f %% of arrays (median f_sky %.3f, range %.3f-%.3f)"
      % (100 * ok.mean(), np.median(fsky), fsky.min(), fsky.max()))
print("   bar >= 95 %% -> %s" % ("MET" if ok.mean() >= 0.95 else "NOT MET"))

# ---------------------------------------------------------------- B2 is there structure
print("\nB2 spectrum vs its own within-mask permutation null (20 shuffles per array):")
outside = np.zeros(len(BANDS))
for b in range(len(BANDS)):
    lo = np.percentile(nul[:, :, b], 2.5, axis=1)
    hi = np.percentile(nul[:, :, b], 97.5, axis=1)
    out = (bp[:, b] < lo) | (bp[:, b] > hi)
    outside[b] = out.mean()
    ratio = np.median(bp[:, b] / np.median(nul[:, :, b], axis=1))
    print("   l %3d-%3d  outside its null on %5.1f %% of arrays   median C_b / null %.3f"
          % (BANDS[b][0], BANDS[b][1], 100 * out.mean(), ratio))
b2met = outside.max() >= 0.95
print("   bar: at least one band >= 95 %% -> %s" % ("MET" if b2met else "NOT MET"))

# ---------------------------------------------------------------- B3 reproducibility
print("\nB3 leave-one-laboratory-out coverage of the p10-p90 band powers:")
cov_by_lab = {}
for held in LAB:
    tr = lab != held
    te = lab == held
    per = []
    for b in range(len(BANDS)):
        lo, hi = np.percentile(bp[tr, b], 10), np.percentile(bp[tr, b], 90)
        per.append(float(((bp[te, b] >= lo) & (bp[te, b] <= hi)).mean()))
    cov_by_lab[held] = per
    print("   held out %-10s mean %.3f  per band %s" % (held, np.mean(per), [round(x, 2) for x in per]))
means = {k: float(np.mean(v)) for k, v in cov_by_lab.items()}
worst = min(min(v) for v in cov_by_lab.values())
b3met = all(0.70 <= m <= 0.90 for m in means.values()) and worst >= 0.60
print("   bar: mean in [0.70,0.90] every fold and no band < 0.60 (worst band %.3f) -> %s"
      % (worst, "MET" if b3met else "NOT MET"))

# ---------------------------------------------------------------- B4 / B5 confounds
zi = []
keep = []
for i, g in enumerate(gsms):
    r = B.get(g)
    a = (r or {}).get("immune", {}).get("A_abs")
    if a is not None:
        zi.append(abs(a - 1.0))
        keep.append(i)
zi = np.array(zi)
keep = np.array(keep)
tot = bp.sum(axis=1)
ratio = bp[:, -1] / bp[:, 0]
print("\nB4 / B5 confounds, for every candidate statistic (n=%d with an immune reading):" % len(keep))
print("   %-22s %10s %10s" % ("statistic", "r vs |z_imm|", "r vs f_sky"))
cands = [("band %d-%d" % BANDS[b], bp[:, b]) for b in range(len(BANDS))]
cands += [("total power", tot), ("high/low ratio", ratio)]
res = {}
for name, v in cands:
    r_imm = float(np.corrcoef(v[keep], zi)[0, 1])
    r_fs = float(np.corrcoef(v, fsky)[0, 1])
    res[name] = (r_imm, r_fs)
    flag = "" if (abs(r_imm) < 0.9 and abs(r_fs) < 0.5) else "   <- fails a confound bar"
    print("   %-22s %10.3f %10.3f%s" % (name, r_imm, r_fs, flag))

json.dump({"b1_fsky_pass": float(ok.mean()), "fsky": {"median": float(np.median(fsky)),
                                                      "min": float(fsky.min()), "max": float(fsky.max())},
           "b2_outside_null": {("%d-%d" % BANDS[b]): float(outside[b]) for b in range(len(BANDS))},
           "b3_coverage": cov_by_lab, "b3_means": means, "b3_worst_band": float(worst),
           "b4_b5": {k: {"r_vs_z_immune": v[0], "r_vs_fsky": v[1]} for k, v in res.items()},
           "n_arrays": len(gsms), "bands": BANDS},
          open("handoff/cls01_results.json", "w"), indent=1)
print("\nwrote handoff/cls01_results.json")
