#!/usr/bin/env python3
"""PROC-TISSUE-01 step 2: score the bars fixed in PROC_TISSUE_01_PREREG.md.

B6 is evaluated FIRST and deliberately, because the pre-registered decision rule makes it gating: if the
median epithelial fraction is below 0.50 the specimens are bulk tissue, and the rule says the composition
finding is reported INSTEAD of the ordering. Everything after B6 is therefore computed and printed, but it
is descriptive - not the procedure's claim - unless B6 passes.

Three discriminators the author's own record demands, because DISC-BLADDER-003 documents that bulk-WGBS
atlases on MUCOSAL substrates inflate cross-tile A-scores from substrate mismatch alone:
  * is the displacement in ONE class or spread across all of them (biology moves one; mismatch moves all)
  * does it survive matching on age, sex and colon side (biology does; demography does not)
  * how does it compare with VAL-062's +0.724 on the same statistic (implausibly large means artifact)
"""
import collections
import json
import statistics

import numpy as np

RNG = np.random.default_rng(20260926)
D = json.load(open("handoff/tissue01_scored.json"))
SC = D["scored_class"]
R = [r for r in D["rows"] if r.get("A_abs") is not None]
VAL062 = 0.724

G = collections.defaultdict(list)
for r in R:
    G[r["group"]].append(r)
print("scored class (chosen from healthy mucosae alone): %s" % SC)
print("groups: %s\n" % {k: len(v) for k, v in G.items()})


def d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else 0.0


def perm_p(x, y, n=5000):
    obs = abs(d(x, y))
    pool = np.concatenate([x, y])
    k = 0
    for _ in range(n):
        RNG.shuffle(pool)
        if abs(d(pool[:len(x)], pool[len(x):])) >= obs:
            k += 1
    return (k + 1) / (n + 1)


res = {"scored_class": SC, "n": {k: len(v) for k, v in G.items()}}

# ---------------------------------------------------------------- B6 first: are these what they claim?
epi = {k: float(np.median([r["epithelial_fraction"] for r in v])) for k, v in G.items()}
imm = {k: float(np.median([r["immune_fraction"] for r in v])) for k, v in G.items()}
res["b6"] = {"median_epithelial_fraction": epi, "median_immune_fraction": imm,
             "met": bool(min(epi.values()) > 0.50)}
print("B6  median EPITHELIAL fraction by group: %s" % {k: round(v, 3) for k, v in epi.items()})
print("    median IMMUNE fraction by group:     %s" % {k: round(v, 3) for k, v in imm.items()})
print("    -> %s (bar: epithelial > 0.50 in the tissue groups)\n"
      % ("MET" if res["b6"]["met"] else "FAILED - these are BULK MUCOSA, not sorted epithelium"))

A = {k: np.array([r["A_abs"] for r in v], float) for k, v in G.items()}
med = {k: float(np.median(v)) for k, v in A.items()}
print("A_abs medians: %s" % {k: round(v, 4) for k, v in med.items()})

# ---------------------------------------------------------------- B1 the ordering
order = ["mucosa", "adjacent_normal", "tumour"]
inc = all(med[order[i]] < med[order[i + 1]] for i in range(len(order) - 1))
res["b1"] = {"medians": med, "order": order, "met": bool(inc)}
print("B1  %s  ->  %s" % (" < ".join("%s %.4f" % (o, med[o]) for o in order), "MET" if inc else "FAILED"))

# ---------------------------------------------------------------- B2 field effect, as written
dd = d(A["adjacent_normal"], A["mucosa"])
pp = perm_p(A["adjacent_normal"], A["mucosa"])
res["b2"] = {"d": dd, "p": pp, "met": bool(dd >= 0.5 and pp < 0.01)}
print("B2  adjacent normal vs healthy mucosa: d=%+.3f  p=%.4f  ->  %s"
      % (dd, pp, "MET" if res["b2"]["met"] else "FAILED"))

# ---------------------------------------------------------------- B3 the disease contrast
d3 = d(A["tumour"], A["mucosa"])
p3 = perm_p(A["tumour"], A["mucosa"])
res["b3"] = {"d": d3, "p": p3, "met": bool(d3 >= 1.0 and p3 < 0.001)}
print("B3  tumour vs healthy mucosa:          d=%+.3f  p=%.4f  ->  %s"
      % (d3, p3, "MET" if res["b3"]["met"] else "FAILED"))

# ---------------------------------------------------------------- B4 a null that does not know the answer
h = A["mucosa"]
ds = []
for _ in range(2000):
    p = RNG.permutation(len(h))
    ds.append(abs(d(h[p[:len(h) // 2]], h[p[len(h) // 2:]])))
res["b4"] = {"median_abs_d": float(np.median(ds)), "p95": float(np.percentile(ds, 95)),
             "met": bool(np.median(ds) < 0.20)}
print("B4  healthy split at random 2,000x: median |d| = %.4f (p95 %.4f)  ->  %s"
      % (res["b4"]["median_abs_d"], res["b4"]["p95"], "MET" if res["b4"]["met"] else "FAILED"))

# ---------------------------------------------------------------- B5 age, and the matched comparison
ages = {k: statistics.median([r["age_num"] for r in v if r.get("age_num")]) for k, v in G.items()}
gap = abs(ages["adjacent_normal"] - ages["mucosa"])
print("\nB5  median ages: %s | gap %.0f y -> matching %s"
      % ({k: round(v) for k, v in ages.items()}, gap, "REQUIRED" if gap > 5 else "not required"))


def matched(a, b, keys=("gender", "location"), tol=5.0):
    """One-to-one nearest-age match within identical sex and colon side. Declared beyond the bar."""
    pool = list(b)
    out_a, out_b = [], []
    for r in sorted(a, key=lambda x: x.get("age_num") or 0):
        cand = [s for s in pool if all(s.get(k) == r.get(k) for k in keys)
                and s.get("age_num") and r.get("age_num") and abs(s["age_num"] - r["age_num"]) <= tol]
        if cand:
            s = min(cand, key=lambda x: abs(x["age_num"] - r["age_num"]))
            pool.remove(s)
            out_a.append(r)
            out_b.append(s)
    return out_a, out_b


ma, mb = matched(G["adjacent_normal"], G["mucosa"])
if len(ma) >= 10:
    xa = np.array([r["A_abs"] for r in ma]); xb = np.array([r["A_abs"] for r in mb])
    dm = d(xa, xb); pm = perm_p(xa, xb)
    res["b2_matched"] = {"n_pairs": len(ma), "d": dm, "p": pm}
    print("    MATCHED on sex, colon side and age (+/-5y): %d pairs, d=%+.3f, p=%.4f" % (len(ma), dm, pm))
else:
    res["b2_matched"] = {"n_pairs": len(ma), "note": "too few matched pairs to test"}
    print("    matched comparison not possible: only %d pairs" % len(ma))

# ---------------------------------------------------------------- the within-patient contrast
pairs = collections.defaultdict(dict)
for r in R:
    if r["group"] in ("tumour", "adjacent_normal") and r.get("patient"):
        pairs[r["patient"]][r["group"]] = r["A_abs"]
both = [(v["tumour"], v["adjacent_normal"]) for v in pairs.values() if len(v) == 2]
if both:
    diff = np.array([t - n for t, n in both])
    res["within_patient"] = {"n": len(both), "median_delta": float(np.median(diff)),
                             "mean_delta": float(diff.mean()),
                             "frac_positive": float((diff > 0).mean()),
                             "d_paired": float(diff.mean() / diff.std(ddof=1))}
    print("\nWITHIN PATIENT (tumour - own adjacent normal), %d patients: median %+.4f, %.0f%% positive, paired d=%+.3f"
          % (len(both), np.median(diff), 100 * (diff > 0).mean(), res["within_patient"]["d_paired"]))

# ---------------------------------------------------------------- one class, or all of them?
classes = sorted(R[0]["A_all_classes"])
spread = {}
for c in classes:
    a = np.array([r["A_all_classes"][c] for r in G["adjacent_normal"]], float)
    m = np.array([r["A_all_classes"][c] for r in G["mucosa"]], float)
    t = np.array([r["A_all_classes"][c] for r in G["tumour"]], float)
    spread[c] = {"adj_vs_healthy_d": d(a, m), "tumour_vs_healthy_d": d(t, m)}
res["class_spread"] = spread
print("\nIS THE DISPLACEMENT IN ONE CLASS OR ALL? (biology moves one; substrate mismatch moves all)")
print("   %-12s %18s %20s" % ("class", "adj vs healthy d", "tumour vs healthy d"))
for c in classes:
    print("   %-12s %18.3f %20.3f" % (c, spread[c]["adj_vs_healthy_d"], spread[c]["tumour_vs_healthy_d"]))
big = [c for c in classes if abs(spread[c]["adj_vs_healthy_d"]) >= 0.5]
res["classes_displaced"] = big
print("   classes displaced by |d| >= 0.5 in adjacent normal: %d of %d  %s" % (len(big), len(classes), big))
print("   VAL-062 reference on the same statistic: +%.3f. Effects far above it are the DISC-BLADDER-003 artifact." % VAL062)

json.dump(res, open("handoff/tissue01_results.json", "w"), indent=1)
print("\nwrote handoff/tissue01_results.json")
