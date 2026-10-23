"""DEV-EACH-CELL-01 exploration 8 (written and committed 2026-10-23 before it runs; exploration, not sealed; plan 39.5 and 39.10). How much each
cell's Met-A in whole blood varies between healthy people, and so how large a loss must be before it shows. Every reading is run_sample.py's
(--cell-panel atlas --dev-metaa-allsites) on every fifth of Group-A's 278 healthy bloods (one laboratory series; spread_08_samples.csv). This
scorer only summarizes what the bundles hold: for each cell and each reading (identity sites; every site), the number read, the median A, the
relative spread between people (SD over median) and the median move a 1 % loss makes, in units of that spread.
Usage: python3 spread_08_score.py ROWS SAMPLES OUT"""
import sys, os, glob, json, numpy as np
out = sys.argv[3]; B = [p for p in glob.glob(os.path.join(out, "pass1", "*.json")) if not p.endswith(("_planted.json", "_values.json"))]
L = [f"DEV-EACH-CELL-01 exploration 8: each cell's Met-A in {len(B)} healthy whole bloods of one laboratory series, all by run_sample.py (pass 1, untared)"]
for key, name in (("met_a_each_cell", "identity sites"), ("met_a_each_cell_allsites", "every site")):
    V = {}
    for p in B:
        for c in ((json.load(open(p)).get(key) or {}).get("cells") or []):
            if c.get("A") is not None: V.setdefault(c["cell"], []).append((float(c["A"]), c.get("shift_per_1pct_loss")))
    for cell, v in sorted(V.items(), key=lambda x: -len(x[1])):
        a = np.array([x[0] for x in v]); s = np.array([x[1] for x in v if x[1] is not None], dtype=float)
        if len(a) < 5: continue
        med = float(np.median(a)); rsd = float(np.std(a, ddof=1) / med); mv = float(np.median(s)) if len(s) else None
        L.append(f"{name:<15} {cell:<14} n {len(a):3} | median A {med:.4f} | spread between people {100 * rsd:.2f} % | a 1 % loss moves "
                 + (f"{100 * mv / med:.2f} % = {mv / (rsd * med):.2f} spreads" if mv else "not recorded"))
open(os.path.join(out, "spread_08_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
