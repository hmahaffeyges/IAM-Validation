"""Scores the self-calibration synthetic, third form (written before any of its libraries was read; development; the same rows and bars as the first, with Met-A as the observed disorder minus the disorder the library's measured end error adds in expectation under counting, run_sample.py --dev-seq-self-cal-disorder). Each bundle's top cell found carries
self_calibration (A with the calibration in met_a_each_cell's A, A_without the same reading without it).
Bar 1 (end errors absorbed): for each library, |A(c1..c3) - A(c0)| <= 0.01 with the calibration, all nine.
Bar 2 (the synthetic can see its plant): without the calibration, the median |A(c1..c3) - A(c0)| >= 0.02; if not, the synthetic is
uninformative (neither met nor not met).
Bar 3 (a real change is kept): the uniform change (c4) moves A with the calibration by at least 80 % of what it moves A without it, all three.
Usage: selfcal_synthetic_score.py BUNDLE_DIR OUT.txt"""
import sys, os, json, glob
rows = {}
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    B = json.load(open(f)); n = os.path.basename(f)[:-5]; g, c = n.rsplit("_", 1)
    cf = sorted((B.get("cells_found") or {}).get("cells") or [], key=lambda x: -(x.get("share") or 0))
    if not cf: rows[(g, c)] = None; continue
    top = cf[0]; e = [r for r in (B.get("met_a_each_cell") or {}).get("cells", []) if r["cell"] == top["cell"]]
    sc = (e[0] if e else {}).get("self_calibration") or {}
    rows[(g, c)] = dict(cell=top["cell"], share=top["share"], A=(e[0] if e else {}).get("A"), A_without=sc.get("A_without"), sc=sc)
L = [f"{g} {c}: {v and v['cell']} share {v and v['share']} A {v and v['A']} A_without {v and v['A_without']}" for (g, c), v in sorted(rows.items())]
libs = sorted({g for g, _ in rows}); b1, b2, b3 = [], [], []
for g in libs:
    r0 = rows.get((g, "c0"))
    if not r0 or r0["A"] is None or r0["A_without"] is None: continue
    for c in ("c1", "c2", "c3"):
        r = rows.get((g, c))
        if r and r["A"] is not None: b1.append(abs(r["A"] - r0["A"]) <= 0.01); b2.append(abs(r["A_without"] - r0["A_without"]))
    r4 = rows.get((g, "c4"))
    if r4 and r4["A"] is not None:
        dw = r4["A_without"] - r0["A_without"]; ds = r4["A"] - r0["A"]; b3.append(abs(dw) > 0 and ds / dw >= 0.8)
import statistics as st
m2 = st.median(b2) if b2 else None
L += [f"bar 1 (end errors absorbed, |dA| <= 0.01 with the calibration): {sum(b1)} of {len(b1)} -> {'met' if b1 and all(b1) and len(b1) == 9 else 'not met'}",
      f"bar 2 (plant visible without the calibration, median |dA| >= 0.02): {m2 and round(m2, 4)} -> {'informative' if m2 and m2 >= 0.02 else 'uninformative'}",
      f"bar 3 (a uniform change kept, >= 80 %): {sum(b3)} of {len(b3)} -> {'met' if b3 and all(b3) and len(b3) == 3 else 'not met'}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
