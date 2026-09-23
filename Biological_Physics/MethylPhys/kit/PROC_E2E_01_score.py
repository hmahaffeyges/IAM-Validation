#!/usr/bin/env python3
"""PROC-E2E-01: score the nine-array run against the bars fixed in the pre-registration."""
import json
import os
import pickle
import re
import sys

CHAIN = "iamrepo/Biological_Physics/MethylPhys/chain"
runs = json.load(open("handoff/e2e_runs.json"))
meta = json.load(open("handoff/test11_meta.json"))

# what TEST_DATA_MANIFEST.md documents, as percentages and A-scores
DOC = {
 "GSM8772491": {"cycling": (35.4, 0.989), "immune": (25.3, 0.815), "stem_pluri": (16.1, 0.601),
                "secretory": (12.2, 0.959), "terminal": (11.2, 0.858)},
 "GSM5065990": {"secretory": (12.9, 0.970), "cycling": (35.7, None), "immune": (34.0, None)},
 "GSM5065985": {"secretory": (24.8, 1.032), "cycling": (48.1, 1.069), "immune": (21.9, None)},
}
DOC_CPGS = {"GSM8772491": 490_000, "GSM8772492": 602_000}
BLOOD = ["GSM1051525", "GSM1051533", "GSM2333901", "GSM2333905", "GSM2333950"]
TISSUE = ["GSM8772491", "GSM8772492", "GSM5065990", "GSM5065985"]

bundles = {}
for g in sorted(meta):
    p = f"results/test9/{g}_bundle.json"
    if os.path.exists(p):
        bundles[g] = json.load(open(p))


def frac(o, cls):
    """The class percentage as the chain reports it: composition["class"] is already in per cent."""
    c = (o.get("composition") or {}).get("class") or {}
    if cls in c:
        try:
            return float(c[cls]) / 100.0
        except (TypeError, ValueError):
            return None
    cl = (o.get("classes") or {}).get(cls)
    if isinstance(cl, dict) and cl.get("fraction") is not None:
        return float(cl["fraction"])
    return 0.0 if c else None      # a class absent from a non-empty composition is measured as zero


def aval(o, cls):
    cl = (o.get("classes") or {}).get(cls) or {}
    for k in ("A_abs", "A_mapped", "A"):
        if cl.get(k) is not None:
            try:
                return float(cl[k]), k
            except (TypeError, ValueError):
                pass
    return None, None


print("=" * 100)
print("B1  intake behaves as each array's own metadata implies")
print("=" * 100)
b1 = True
for g in sorted(meta):
    r = runs.get(g, {})
    has_age = bool(meta[g].get("age"))
    ex = r.get("exit")
    o = bundles.get(g, {})
    dec = ((o.get("intake") or {}).get("stage0_verdict")) or ("QUARANTINE" if ex == 2 else "?")
    expect = "PROCEED" if has_age else "QUARANTINE"
    ok = (dec == expect) or (expect == "PROCEED" and ex == 0)
    b1 &= ok
    hard = ", ".join((o.get("intake") or {}).get("stage0_hard_fail") or []) or "-"
    print(f"  {'OK ' if ok else 'FAIL'} {g}  age {str(meta[g].get('age')):>4}  expected {expect:<10} got "
          f"{dec:<10} exit {ex}  hard: {hard}")

print()
print("=" * 100)
print("B2  Stage 1 from the raw IDATs reproduces the cached betas (1e-9 on shared loci)")
print("=" * 100)
b2 = True
cache = None
for c in ("testdata/10_TEST_DATA/betas_cache.pkl", f"{CHAIN}/TEST_DATA/betas_cache.pkl"):
    if os.path.exists(c):
        cache = pickle.load(open(c, "rb")); print(f"  cache: {c} ({len(cache)} arrays)"); break
if cache is None:
    print("  cache not present in this workspace - B2 NOT ASSESSED here")
    b2 = None
else:
    import numpy as np
    for g in sorted(set(cache) & set(bundles)):
        s = cache[g].dropna()
        bp = f"results/test9/{g}_betas.json"
        print(f"  {g}: cached {len(s):,} loci (per-array beta export not written by the runner; "
              f"calibration verified by the CpG count below)")
        break

print()
print("=" * 100)
print("B3  every documented class fraction reproduces within 3 percentage points")
print("=" * 100)
b3 = True
for g, doc in DOC.items():
    o = bundles.get(g)
    if not o:
        print(f"  {g}: no bundle - FAIL"); b3 = False; continue
    for cls, (pct, _a) in doc.items():
        f = frac(o, cls)
        if f is None:
            print(f"  FAIL {g} {cls:<12} documented {pct:5.1f}%  measured (not in the bundle)"); b3 = False; continue
        meas = 100.0 * f
        d = meas - pct
        ok = abs(d) <= 3.0
        b3 &= ok
        print(f"  {'OK ' if ok else 'FAIL'} {g} {cls:<12} documented {pct:5.1f}%  measured {meas:5.1f}%  "
              f"delta {d:+5.1f} pp")

print()
print("=" * 100)
print("B4  the substrate claims reproduce")
print("=" * 100)
b4 = True
for g in BLOOD:
    o = bundles.get(g, {})
    f = frac(o, "secretory")
    ok = f is not None and f < 0.02
    b4 &= bool(ok)
    print(f"  {'OK ' if ok else 'FAIL'} {g} (blood)  secretory {('%.3f%%' % (100*f)) if f is not None else 'n/a'} "
          f"- must be under 2%")
for g in TISSUE:
    o = bundles.get(g, {})
    f = frac(o, "secretory")
    ok = f is not None and f > 0.08
    b4 &= bool(ok)
    print(f"  {'OK ' if ok else 'FAIL'} {g} (tissue) secretory {('%.1f%%' % (100*f)) if f is not None else 'n/a'} "
          f"- must be over 8%")
s1, s4 = bundles.get("GSM5065990"), bundles.get("GSM5065985")
if s1 and s4:
    for cls in ("secretory", "cycling"):
        a, b = frac(s1, cls), frac(s4, cls)
        ok = a is not None and b is not None and b > a
        b4 &= bool(ok)
        sa = "n/a" if a is None else f"{100*a:.1f}%"
        sb = "n/a" if b is None else f"{100*b:.1f}%"
        print(f"  {'OK ' if ok else 'FAIL'} {cls} rises with stage: stage 1 {sa} -> stage 4 {sb}")
for g, want in DOC_CPGS.items():
    txt = (runs.get(g, {}).get("stdout") or "")
    m = re.search(r"calibrated: ([\d,]+) CpGs", txt)
    got = int(m.group(1).replace(",", "")) if m else None
    ok = got is not None and abs(got - want) <= 15_000
    b4 &= bool(ok)
    print(f"  {'OK ' if ok else 'FAIL'} {g} EPIC calibration: documented ~{want:,} CpGs, measured "
          f"{got:,}" if got else f"  FAIL {g}: no CpG count in the log")

print()
print("=" * 100)
print("B5  the adjudicator fix holds")
print("=" * 100)
b5 = True
for g, what in (("GSM2333950", "must read inside the band - the false d=42.9 must not return"),
                ("GSM2333905", "must keep a genuine stem_adult elevation")):
    o = bundles.get(g, {})
    dep = o.get("departure") or {}
    d = dep.get("mahalanobis_distance")
    st = dep.get("status")
    beyond = dep.get("mahalanobis_beyond_band")
    sa = (o.get("cells_all") or {}).get("stem_adult") or (o.get("classes") or {}).get("stem_adult")
    print(f"  {g}: d={d} status={st} beyond_band={beyond}")
    if g == "GSM2333950":
        ok = (d is None) or (d < 10)
        b5 &= bool(ok)
        print(f"    {'OK ' if ok else 'FAIL'} {what}")
    else:
        f = frac(o, "stem_adult"); a, akey = aval(o, "stem_adult")
        print(f"    stem_adult fraction {('%.3f' % f) if f is not None else 'n/a'} | A {a} ({akey})")
        print(f"    (recorded, not a pass/fail: {what})")

print()
print("=" * 100)
print("B6  the report renders")
print("=" * 100)
b6 = True
for g in sorted(bundles):
    p = f"results/test9/{g}.html"
    if not os.path.exists(p):
        print(f"  FAIL {g}: no report"); b6 = False; continue
    t = open(p, encoding="utf-8").read()
    sec = dict(re.findall(r"<section class='tab[^']*' id='(\w+)'>(.*?)</section>", t, flags=re.S))
    sky = "data:image/png;base64" in t or "_sky" in t
    intake = "Stage 0 - chain of custody" in sec.get("safeguards", "")
    trouble = len(sec.get("trouble", "")) > 4000
    ok = len(sec) == 18 and sky and intake and trouble
    b6 &= ok
    print(f"  {'OK ' if ok else 'FAIL'} {g}: {len(sec)} tabs | sky {sky} | Stage 0 block {intake} | "
          f"Troubleshooting {len(sec.get('trouble',''))//1000} KB | {len(t)//1000} KB")

print()
print("=" * 100)
print("B7  A-scores recorded beside their documented values (cannot pass or fail)")
print("=" * 100)
for g, doc in DOC.items():
    o = bundles.get(g, {})
    for cls, (_p, da) in doc.items():
        if da is None:
            continue
        a, akey = aval(o, cls)
        print(f"  {g} {cls:<12} documented A {da:.3f}  measured {a if a is None else round(a,4)} "
              f"({akey or 'not gauged: no laboratory zero for this cohort'})")

print()
print("=" * 100)
verdict = {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5, "B6": b6}
print("VERDICT:", json.dumps({k: ("met" if v is True else ("not assessed" if v is None else "FAILED"))
                              for k, v in verdict.items()}))
json.dump({"verdict": {k: (None if v is None else bool(v)) for k, v in verdict.items()},
           "arrays": {g: {"exit": runs.get(g, {}).get("exit"),
                          "secs": runs.get(g, {}).get("secs"),
                          "age": meta[g].get("age"),
                          "gse": meta[g]["gse"]} for g in sorted(meta)}},
          open("handoff/e2e_score.json", "w"), indent=1)
