#!/usr/bin/env python3
"""PROC-BAND-01 step 2: score the joint haematopoietic-progenitor surface against the pre-registered bars.

Every construction here is the published one, not a new invention:
  age curve   decade medians of A_mapped with the 40s decade as zero, exactly as reference_age_curve_v1.json
              is shaped (c(40) = 0.0, the other decades offsets from it)
  laboratory zero   lab_zero.compute_lab_zero - median(A - c(age)) - 1.0 over a healthy panel of >= 40
  band        pooled p10-p90 of A_abs, which is identity_band_v3's own definition, with mu = 1.000 and
              sigma = (p90 - p10) / (2 x 1.2816)
  tail        fraction of healthy arrays with |z| > 1.96, which is what tail_p95 means in the band file

Nothing is fitted on the arrays that judge it: in each fold the curve and the band come from three
laboratories, and the held-out laboratory computes its own zero from its own panel - which is exactly what a
new laboratory does when it joins.
"""
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, "iamrepo/Biological_Physics/MethylPhys/chain")
import cpg_conductor as C                                             # noqa: E402

lz = C._load_module("lab_zero", C._find("lab_zero.py"))
D = json.load(open("handoff/band01_arrays.json"))
A = D["arrays"]
LABS = sorted({r["gse"] for r in A})
Z = 1.2815515655446004                                                # the 80 % interval's half-width in sigma


def curve_from(rows):
    """Decade offsets with the 40s as zero - the shape reference_age_curve_v1.json has."""
    by = {}
    for r in rows:
        if r["age"] is None or r["joint"]["A_mapped"] is None:
            continue
        by.setdefault(int(r["age"] // 10 * 10), []).append(r["joint"]["A_mapped"])
    med = {d: statistics.median(v) for d, v in by.items() if len(v) >= 8}
    if not med:
        return None
    base = med.get(40, statistics.median(list(med.values())))
    return {d: round(m - base, 4) for d, m in sorted(med.items())}


def zero_for(rows, curve):
    a = [r["joint"]["A_mapped"] for r in rows if r["age"] is not None and r["joint"]["A_mapped"] is not None]
    g = [r["age"] for r in rows if r["age"] is not None and r["joint"]["A_mapped"] is not None]
    return lz.compute_lab_zero(a, g, curve)


def abs_joint(r, curve, z_lab):
    if r["age"] is None or r["joint"]["A_mapped"] is None:
        return None
    return r["joint"]["A_mapped"] - lz.age_reference(r["age"], curve) - z_lab


usable = [r for r in A if r["age"] is not None]
print("arrays: %d | with an age: %d" % (len(A), len(usable)))

# ---------------------------------------------------------------- B1 presence
pres = [r for r in A if (r["joint"].get("fraction") or 0) >= 0.01]
b1 = len(pres) / len(A)
jf = [r["joint"].get("fraction") or 0 for r in A]
print("\nB1 presence: joint fraction >= 0.01 in %.1f %% of %d arrays (median fraction %.4f, min %.4f)"
      % (100 * b1, len(A), statistics.median(jf), min(jf)))
print("   bar >= 95 %% -> %s" % ("MET" if b1 >= 0.95 else "NOT MET"))

# ---------------------------------------------------------------- B2 leave-one-laboratory-out
print("\nB2 leave-one-laboratory-out (nominal 0.80, bar [0.70, 0.90]):")
b2 = {}
for held in LABS:
    tr = [r for r in usable if r["gse"] != held]
    te = [r for r in usable if r["gse"] == held]
    cv = curve_from(tr)
    if cv is None:
        print("   %s: no curve from the other three" % held)
        continue
    zt = {g: zero_for([r for r in tr if r["gse"] == g], cv) for g in {r["gse"] for r in tr}}
    tr_abs = [abs_joint(r, cv, zt[r["gse"]]) for r in tr]
    tr_abs = [x for x in tr_abs if x is not None]
    p10, p90 = float(np.percentile(tr_abs, 10)), float(np.percentile(tr_abs, 90))
    z_held = zero_for(te, cv)                                     # as a new laboratory would, from its own panel
    te_abs = [x for x in (abs_joint(r, cv, z_held) for r in te) if x is not None]
    cov = sum(p10 <= x <= p90 for x in te_abs) / len(te_abs)
    b2[held] = cov
    print("   held out %-10s n=%3d  band [%.4f, %.4f] from %d arrays  coverage %.3f  %s"
          % (held, len(te_abs), p10, p90, len(tr_abs), cov, "ok" if 0.70 <= cov <= 0.90 else "OUTSIDE"))
b2met = all(0.70 <= c <= 0.90 for c in b2.values())
print("   bar: every fold in [0.70, 0.90] -> %s" % ("MET" if b2met else "NOT MET"))

# ---------------------------------------------------------------- the pooled band (built once, all four)
cv_all = curve_from(usable)
z_all = {g: zero_for([r for r in usable if r["gse"] == g], cv_all) for g in LABS}
for r in usable:
    r["_ja"] = abs_joint(r, cv_all, z_all[r["gse"]])
ja = [r["_ja"] for r in usable if r["_ja"] is not None]
p10, p50, p90 = (float(np.percentile(ja, q)) for q in (10, 50, 90))
sig = (p90 - p10) / (2 * Z)
print("\npooled joint band: p10 %.4f  p50 %.4f  p90 %.4f  sigma %.4f  (n=%d)" % (p10, p50, p90, sig, len(ja)))
print("   age curve:", cv_all)
print("   laboratory zeros:", {k: round(v, 4) for k, v in z_all.items()})

# ---------------------------------------------------------------- B3 false alarm
print("\nB3 false alarm, fraction with |z| > 1.96:")
tails = {}
for g in LABS:
    v = [r["_ja"] for r in usable if r["gse"] == g and r["_ja"] is not None]
    t = sum(abs((x - 1.0) / sig) > 1.96 for x in v) / len(v)
    tails[g] = t
    print("   %-10s n=%3d  tail %.4f  %s" % (g, len(v), t, "ok" if t <= 0.10 else "OVER"))
pooled_tail = sum(abs((x - 1.0) / sig) > 1.96 for x in ja) / len(ja)
b3met = all(t <= 0.10 for t in tails.values()) and pooled_tail <= 0.06
print("   pooled %.4f  | bar: each <= 0.10 and pooled <= 0.06 -> %s" % (pooled_tail, "MET" if b3met else "NOT MET"))

# ---------------------------------------------------------------- B4 does it add an axis
pair = [(r["immune"].get("A_abs"), r["_ja"]) for r in usable
        if r["immune"].get("A_abs") is not None and r["_ja"] is not None]
ia = np.array([p[0] for p in pair])
jaa = np.array([p[1] for p in pair])
rr = float(np.corrcoef(ia, jaa)[0, 1])
print("\nB4 independence: Pearson r(immune A_abs, joint A_abs) = %.4f on %d arrays" % (rr, len(pair)))
print("   bar |r| < 0.9 -> %s" % ("MET" if abs(rr) < 0.9 else "NOT MET"))

# ---------------------------------------------------------------- B5 the two-axis threshold
imm_sig = (float(np.percentile(ia, 90)) - float(np.percentile(ia, 10))) / (2 * Z)
zi = (ia - 1.0) / imm_sig
zj = (jaa - 1.0) / sig
Xz = np.vstack([zi, zj])
cov = np.cov(Xz)
inv = np.linalg.inv(cov)
d2 = np.einsum("ij,jk,ik->i", Xz.T, inv, Xz.T)
d = np.sqrt(d2)
meas = float(np.percentile(d, 95))
chi2 = float(np.sqrt(5.991464547107979))
print("\nB5 two-axis distance: measured p95 %.4f | chi-square table (k=2) %.4f | disagreement %.1f %%"
      % (meas, chi2, 100 * abs(meas - chi2) / chi2))
print("   immune sigma %.4f | joint sigma %.4f | correlation in the covariance %.4f"
      % (imm_sig, sig, cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])))

# ---------------------------------------------------------------- B6 the commissioned reading has not moved
imm_med = float(np.median(ia))
imm_tails = {g: sum(abs((r["immune"]["A_abs"] - 1.0) / imm_sig) > 1.96
                    for r in usable if r["gse"] == g and r["immune"].get("A_abs") is not None)
                / max(1, sum(1 for r in usable if r["gse"] == g and r["immune"].get("A_abs") is not None))
             for g in LABS}
print("\nB6 the commissioned immune reading, measured on this path:")
print("   median A_abs %.6f (the band's own mu is 1.000) | per-laboratory tails %s"
      % (imm_med, {k: round(v, 4) for k, v in imm_tails.items()}))
print("   band file records tails 0.0441 / 0.0647 / 0.0984 / 0.0561 on the FULL cohorts")

json.dump({"b1": b1, "b2": b2, "b3": {"per_lab": tails, "pooled": pooled_tail}, "b4": rr,
           "b5": {"measured_p95": meas, "chi2": chi2, "immune_sigma": imm_sig, "joint_sigma": sig,
                  "corr": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))},
           "b6": {"immune_median_A_abs": imm_med, "immune_tails": imm_tails},
           "band": {"p10": p10, "p50": p50, "p90": p90, "sigma": sig, "n": len(ja)},
           "age_curve": cv_all, "lab_zeros": {k: round(v, 6) for k, v in z_all.items()}},
          open("handoff/band01_results.json", "w"), indent=1)
print("\nwrote handoff/band01_results.json")
