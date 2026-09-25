#!/usr/bin/env python3
"""PROC-LABBAND-01: are the four laboratories' healthy widths different, stable, and worth adopting?

Reads the 318 immune A'' values PROC-BAND-01 published. Nothing is recalibrated: the laboratory zero already
centres each cohort at 1.000, so only the WIDTH is in question.

sigma is measured the way the band file measures it - (p90 - p10) / (2 x 1.2816) - rather than as a standard
deviation, so a per-laboratory width is the same kind of object as the pooled one it would replace, and a
single outlying array cannot move it.
"""
import json

import numpy as np

Z = 1.2815515655446004
RNG = np.random.default_rng(20260925)
D = json.load(open("handoff/band01_arrays.json"))["arrays"]
rows = [r for r in D if r["immune"].get("A_abs") is not None]
LAB = sorted({r["gse"] for r in rows})
A = {g: np.array([r["immune"]["A_abs"] for r in rows if r["gse"] == g]) for g in LAB}
allA = np.concatenate([A[g] for g in LAB])


def sigma(v):
    return (np.percentile(v, 90) - np.percentile(v, 10)) / (2 * Z)


pooled = sigma(allA)
sig = {g: float(sigma(A[g])) for g in LAB}
print("arrays: %d | pooled sigma %.5f" % (len(allA), pooled))
for g in LAB:
    print("   %-10s n=%3d  sigma %.5f  (%.2fx pooled)  median A'' %.4f"
          % (g, len(A[g]), sig[g], sig[g] / pooled, np.median(A[g])))

# ---------------------------------------------------------------- B1 do the widths differ
ratio = max(sig.values()) / min(sig.values())
sizes = [len(A[g]) for g in LAB]
null = np.empty(5000)
for k in range(5000):
    perm = RNG.permutation(allA)
    i, s = 0, []
    for n in sizes:
        s.append(sigma(perm[i:i + n]))
        i += n
    null[k] = max(s) / min(s)
p = float((null >= ratio).mean())
b1 = ratio > 1.20 and p < 0.01
print("\nB1 widths differ: max/min sigma ratio %.3f (bar > 1.20) | permutation p = %.4f (bar < 0.01, %d shuffles)"
      % (ratio, p, len(null)))
print("   null ratio median %.3f, 99th pct %.3f -> %s" % (np.median(null), np.percentile(null, 99),
                                                          "MET" if b1 else "NOT MET"))

# ---------------------------------------------------------------- B2 is a width from 80 arrays stable
print("\nB2 split-half stability of sigma, 200 splits per laboratory:")
b2 = True
stab = {}
for g in LAB:
    v = A[g]
    rs = []
    for _ in range(200):
        idx = RNG.permutation(len(v))
        h = len(v) // 2
        sa, sb = sigma(v[idx[:h]]), sigma(v[idx[h:2 * h]])
        rs.append(sa / sb)
    rs = np.array(rs)
    med, lo, hi = float(np.median(rs)), float(np.percentile(rs, 5)), float(np.percentile(rs, 95))
    stab[g] = {"median": med, "p5": lo, "p95": hi}
    ok = 0.80 <= med <= 1.25 and lo >= 0.65 and hi <= 1.55
    b2 &= ok
    print("   %-10s median %.3f  90%% spread [%.3f, %.3f]  %s" % (g, med, lo, hi, "ok" if ok else "OUTSIDE"))
print("   -> %s" % ("MET" if b2 else "NOT MET"))

# ---------------------------------------------------------------- B3 out-of-sample tail
print("\nB3 out-of-sample tail beyond |z| > 1.96 (fit sigma on half, measure on the other half):")
BANDFILE = {"GSE111629": 0.0441, "GSE125105": 0.0647, "GSE42861": 0.0984, "GSE87571": 0.0561}
b3 = True
tails = {}
for g in LAB:
    v = A[g]
    ts, tp = [], []
    for _ in range(200):
        idx = RNG.permutation(len(v))
        h = len(v) // 2
        fit, test = v[idx[:h]], v[idx[h:]]
        s = sigma(fit)
        ts.append(float(np.mean(np.abs((test - 1.0) / s) > 1.96)))
        tp.append(float(np.mean(np.abs((test - 1.0) / pooled) > 1.96)))
    med, medp = float(np.median(ts)), float(np.median(tp))
    tails[g] = {"per_lab": med, "pooled_same_split": medp, "band_file_full_cohort": BANDFILE.get(g)}
    ok = 0.03 <= med <= 0.07
    b3 &= ok
    print("   %-10s per-lab %.4f   pooled on the same halves %.4f   band file (full cohort) %.4f   %s"
          % (g, med, medp, BANDFILE.get(g, float("nan")), "ok" if ok else "OUTSIDE"))
print("   bar: every laboratory in [0.03, 0.07] -> %s" % ("MET" if b3 else "NOT MET"))

# ---------------------------------------------------------------- B4 sensitivity
print("\nB4 sensitivity: a +2 sigma_pooled departure injected into every healthy reading:")
b4 = True
det = {}
for g in LAB:
    v = A[g] + 2 * pooled
    d_lab = float(np.mean(np.abs((v - 1.0) / sig[g]) > 1.96))
    d_pool = float(np.mean(np.abs((v - 1.0) / pooled) > 1.96))
    det[g] = {"per_lab": d_lab, "pooled": d_pool}
    ok = d_lab >= 0.80
    b4 &= ok
    print("   %-10s detected per-lab %.3f   (pooled would detect %.3f)   %s"
          % (g, d_lab, d_pool, "ok" if ok else "BELOW 0.80"))
print("   -> %s" % ("MET" if b4 else "NOT MET"))

json.dump({"pooled_sigma": float(pooled), "sigma": sig, "b1": {"ratio": float(ratio), "p": p, "met": bool(b1)},
           "b2": {"per_lab": stab, "met": bool(b2)}, "b3": {"tails": tails, "met": bool(b3)},
           "b4": {"detection": det, "met": bool(b4)}, "n": {g: int(len(A[g])) for g in LAB}},
          open("handoff/labband01.json", "w"), indent=1)
print("\nwrote handoff/labband01.json")
