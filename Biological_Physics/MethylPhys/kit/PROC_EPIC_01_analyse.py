#!/usr/bin/env python3
"""PROC-EPIC-01 step 2: score the held-out EPIC-Italy arrays against the pre-registered bars.

Every quantity is the one the pre-registration named, computed on the held-out samples only. The compared
quantity is A' = A_mapped - c(age) with the commissioned age curve, so the missing EPIC-Italy laboratory
zero - a constant per laboratory - cancels.

Nothing here is refitted and no threshold moves: the bars were fixed in PROC_EPIC_01_PREREG.md before the
first array was scored.
"""
import json

import numpy as np

RNG = np.random.default_rng(20260926)
S = json.load(open("handoff/epic01_scored.json"))
ho = [r for r in S if r.get("held_out") and r.get("A_prime") is not None]

CTRL = [r for r in ho if r["icd"] is None and r["sex"] == "F"]          # 163 female controls
BR = [r for r in ho if r["icd"] == "C50"]
CR = [r for r in ho if r["icd"] in ("C18", "C19", "C20")]
a = lambda rows: np.array([r["A_prime"] for r in rows], dtype=float)


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else 0.0


def perm_p(case, ctrl, n=5000):
    """One-sided on ELEVATION, the direction the pre-registration fixed. Labels shuffled, not values."""
    obs = cohen_d(case, ctrl)
    pool = np.concatenate([case, ctrl])
    k = len(case)
    hits = 0
    for _ in range(n):
        RNG.shuffle(pool)
        if cohen_d(pool[:k], pool[k:]) >= obs:
            hits += 1
    return obs, (hits + 1) / (n + 1)


ctrl = a(CTRL)
print("held-out arrays with a reading: %d breast, %d colorectal, %d female controls"
      % (len(BR), len(CR), len(CTRL)))
print("control A': median %.4f  sd %.4f" % (float(np.median(ctrl)), float(ctrl.std(ddof=1))))

res = {}

# ---------------------------------------------------------------- B1 the primary test
b1 = [r for r in BR if r["ttd"] is not None and r["ttd"] > 10]
d1, p1 = perm_p(a(b1), ctrl)
res["b1"] = {"n": len(b1), "d": d1, "p": p1, "median_A_prime": float(np.median(a(b1))),
             "met": bool(d1 >= 0.5 and p1 < 0.01)}
print("\nB1  breast >10y (n=%d) vs controls: d=%+.3f  p=%.4f  ->  %s"
      % (len(b1), d1, p1, "MET" if res["b1"]["met"] else "FAILED (bar: d>=0.5 and p<0.01, elevation)"))

# ---------------------------------------------------------------- B2 the temporal shape
strata = {"0-2y": [r for r in BR if r["ttd"] is not None and r["ttd"] <= 2],
          "2-5y": [r for r in BR if r["ttd"] is not None and 2 < r["ttd"] <= 5],
          "5-8y": [r for r in BR if r["ttd"] is not None and 5 < r["ttd"] <= 8],
          ">8y": [r for r in BR if r["ttd"] is not None and r["ttd"] > 8]}
ds = {k: (cohen_d(a(v), ctrl) if len(v) > 2 else None) for k, v in strata.items()}
res["b2"] = {"d_by_stratum": {k: (None if v is None else round(v, 4)) for k, v in ds.items()},
             "n_by_stratum": {k: len(v) for k, v in strata.items()},
             "met": bool(ds[">8y"] is not None and ds["0-2y"] is not None and ds[">8y"] - ds["0-2y"] > 0)}
print("B2  d by lead time: " + "  ".join("%s n=%d d=%s" % (k, len(strata[k]),
      ("%+.3f" % ds[k]) if ds[k] is not None else "--") for k in ("0-2y", "2-5y", "5-8y", ">8y")))
print("    d(>8y) - d(0-2y) = %s  ->  %s"
      % (("%+.3f" % (ds[">8y"] - ds["0-2y"])) if res["b2"]["met"] is not None else "--",
         "MET" if res["b2"]["met"] else "FAILED"))

# ---------------------------------------------------------------- B3 the cross-cancer arm
c3 = [r for r in CR if r["ttd"] is not None and r["ttd"] > 5]
d3, p3 = perm_p(a(c3), ctrl)
res["b3"] = {"n": len(c3), "d": d3, "p": p3, "met": bool(p3 < 0.05)}
print("B3  colorectal >5y (n=%d) vs controls: d=%+.3f  p=%.4f  ->  %s"
      % (len(c3), d3, p3, "MET" if res["b3"]["met"] else "FAILED (bar: p<0.05, elevation)"))

# ---------------------------------------------------------------- B4 the negative control
half = []
for _ in range(2000):
    idx = RNG.permutation(len(ctrl))
    h = len(ctrl) // 2
    half.append(abs(cohen_d(ctrl[idx[:h]], ctrl[idx[h:]])))
med = float(np.median(half))
res["b4"] = {"median_abs_d": med, "p95": float(np.percentile(half, 95)), "met": bool(med < 0.20)}
print("B4  controls split at random 2,000x: median |d| = %.4f (p95 %.4f)  ->  %s"
      % (med, res["b4"]["p95"], "MET" if res["b4"]["met"] else "FAILED (bar: < 0.20)"))

# ---------------------------------------------------------------- B5 is the guard confounded with disease?
def withheld(rows):
    n = sum(1 for r in rows if r.get("composition_verified") is False)
    return n, len(rows), (n / len(rows) if rows else 0.0)
wc = withheld(BR + CR)
wk = withheld(CTRL)
ratio = (wc[2] / wk[2]) if wk[2] > 0 else (float("inf") if wc[2] > 0 else 1.0)
res["b5"] = {"cases": {"withheld": wc[0], "n": wc[1], "rate": wc[2]},
             "controls": {"withheld": wk[0], "n": wk[1], "rate": wk[2]},
             "ratio": ratio, "met": bool(0.5 <= ratio <= 2.0)}
print("B5  composition guard withheld: cases %d/%d (%.3f), controls %d/%d (%.3f), ratio %.2f  ->  %s"
      % (wc[0], wc[1], wc[2], wk[0], wk[1], wk[2], ratio, "MET" if res["b5"]["met"] else "FAILED (bar: <2x)"))

# ---------------------------------------------------------------- B6 the instrument has not moved
# The pre-registration named PROC-E2E-01's sealed file. That file records exit codes, timings, age and
# series for the eleven commissioning arrays and NO per-array readings, so the bar AS WRITTEN cannot be
# run - naming a source without checking it holds the quantity was my error in the pre-registration.
# What is run instead, and reported as a substitution rather than as the pre-registered bar: the same
# check against the readings PROC-BAND-01 did publish (318 arrays, four laboratories).
try:
    b6 = json.load(open("handoff/epic01_b6.json"))
    res["b6"] = {"substituted": True, "source": b6["source"], "n": b6["n"],
                 "max_abs_delta_A_mapped": b6["max_abs_delta_A_mapped"],
                 "met": bool(b6["max_abs_delta_A_mapped"] == 0.0), "note": b6["note"]}
    print("B6  instrument unchanged on %d published arrays: max |delta A_mapped| = %.3e  ->  %s"
          % (b6["n"], b6["max_abs_delta_A_mapped"],
             "MET (substituted source - see note)" if res["b6"]["met"] else "FAILED"))
except FileNotFoundError:
    res["b6"] = {"met": None, "note": "NOT ASSESSED - the recomputation had not completed"}
    print("B6  NOT ASSESSED - recomputation incomplete; not reported as a pass")

json.dump(res, open("handoff/epic01_results.json", "w"), indent=1)
print("\nwrote handoff/epic01_results.json")
