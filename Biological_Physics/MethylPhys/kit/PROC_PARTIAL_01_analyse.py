#!/usr/bin/env python3
"""PROC-PARTIAL-01 step 2: score the recovery against the bars fixed in PROC_PARTIAL_01_PREREG.md.

No threshold moves here. SIGMA is the immune band's own width, named in the pre-registration so that
"close enough" is a measured healthy spread rather than a tolerance invented for this run.
"""
import collections
import json

import numpy as np

D = json.load(open("handoff/partial01.json"))
SIG = D["sigma"]
rows = D["rows"]
by = collections.defaultdict(list)
for r in rows:
    by[(r["class"], r["f"])].append(r)

print("bar scale: 1 sigma = %.4f, 2 sigma = %.4f (the immune band's own width)\n" % (SIG, 2 * SIG))
res = {"per_cell": {}, "qualifying": {}}
head = "%-11s %5s %6s %9s %9s %9s %9s %8s" % ("class", "f", "n", "med|err|", "p10-p90", "IQR", "naive|err|", "beats")
print(head)
print("-" * len(head))

for cls in ("secretory", "terminal", "stromal"):
    for f in sorted({r["f"] for r in rows if r["class"] == cls}):
        g = by[(cls, f)]
        if len(g) < 30:
            continue
        ah = np.array([r["A_hat"] for r in g])
        at = g[0]["A_true"]
        err = np.abs(ah - at)
        nerr = np.abs(np.array([r["A_naive"] for r in g]) - at)
        spread = float(np.percentile(ah, 90) - np.percentile(ah, 10))
        iqr = float(np.percentile(ah, 75) - np.percentile(ah, 25))
        cell = {"n": len(g), "median_abs_err": float(np.median(err)), "p10_p90": spread, "iqr": iqr,
                "median_naive_abs_err": float(np.median(nerr)),
                "beats_naive": bool(np.median(err) < np.median(nerr)),
                "b1": bool(np.median(err) <= SIG and spread <= 2 * SIG),
                "b2": bool(iqr <= SIG)}
        res["per_cell"]["%s@%.2f" % (cls, f)] = cell
        print("%-11s %5.2f %6d %9.4f %9.4f %9.4f %9.4f %8s"
              % (cls, f, len(g), cell["median_abs_err"], spread, iqr, cell["median_naive_abs_err"],
                 "yes" if cell["beats_naive"] else "NO"))

# ---------------------------------------------------------------- B1/B2/B3 the qualifying fraction
for cls in ("secretory", "terminal", "stromal"):
    qual = [f for f in sorted({r["f"] for r in rows if r["class"] == cls})
            if res["per_cell"].get("%s@%.2f" % (cls, f), {}).get("b1")
            and res["per_cell"].get("%s@%.2f" % (cls, f), {}).get("b2")
            and res["per_cell"].get("%s@%.2f" % (cls, f), {}).get("beats_naive")
            and f <= 0.20]
    res["qualifying"][cls] = qual[0] if qual else None

print("\nB1+B2+B3  smallest fraction meeting accuracy, host-independence and beating the naive score:")
for cls, q in res["qualifying"].items():
    print("   %-11s %s" % (cls, ("f = %.2f" % q) if q else "NONE at f <= 0.20"))

# ---------------------------------------------------------------- B4 it must refuse when it cannot see
F0 = json.load(open("handoff/foreign01.json"))["rows"]
unspiked = [r for r in F0 if float(r["f"]) == 0]
b4 = {}
for cls, q in res["qualifying"].items():
    if q is None:
        continue
    fired = 0
    for r in unspiked:
        fr = r["fr"] if isinstance(r["fr"], dict) else json.loads(r["fr"].replace("'", '"'))
        if float(fr.get(cls, 0.0)) >= q:
            fired += 1
    b4[cls] = {"would_report": fired, "n": len(unspiked), "rate": fired / max(len(unspiked), 1)}
res["b4"] = b4
print("\nB4  unspiked healthy hosts that would still produce a reading under the qualifying fraction:")
for cls, v in b4.items():
    print("   %-11s %d of %d (%.3f)" % (cls, v["would_report"], v["n"], v["rate"]))
if not b4:
    print("   not applicable - no fraction qualified, so no reporting floor exists to test")

# ---------------------------------------------------------------- B5 nothing in service moved
b6 = json.load(open("handoff/epic01_b6.json"))
res["b5"] = {"max_abs_delta_A_mapped": b6["max_abs_delta_A_mapped"], "n": b6["n"],
             "met": bool(b6["max_abs_delta_A_mapped"] == 0.0),
             "note": "this procedure is an offline estimator and changes nothing in the chain; "
                     "the figure is the recomputation run today on PROC-BAND-01's published arrays"}
print("\nB5  immune A unchanged on %d published arrays: max |delta| = %.3e"
      % (b6["n"], b6["max_abs_delta_A_mapped"]))

json.dump(res, open("handoff/partial01_results.json", "w"), indent=1)
print("\nwrote handoff/partial01_results.json")
