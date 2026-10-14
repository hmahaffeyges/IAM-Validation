#!/usr/bin/env python3
"""Readiness L21 real check (2026-10-14), committed BEFORE it runs. The rule is transfer_bounded_v21.py's docstring (committed 504ffb8a):
the bounded monotone transfer is built on half of v2's 17 Loyfer-shared cells and checked on the other half, both ways; ADOPTED only if
every bin with >= 500 pairs in the checked half is within +/-0.02 in both directions and the synthetic test (shipped file) reads MET.
CODE: the pairs are those L20 used, built by the twin/terms job's own run12 on v2's sample table, unchanged; one line kept inside run12
saves the per-cell pairs it already builds. Writes transfer_bounded_v21.json (knots) and transfer_bounded_v21_summary.txt."""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
publish_code("results/ATLAS_V21/transfer_bounded/code/run1/", ["v21_transfer_bounded_box.py", "transfer_bounded_v21.py", "transfer_bounded_v21_synthetic.txt",
             "v21_twin_terms_box.py", "publish_code.py", "roster_samples.csv", "roster.csv", "hsc_manifest.csv", "twin_family_thresholds_v1.json"],
             job="atlas v2.1 bounded transfer real check (L21)", commit=os.environ.get("REPO_COMMIT"))
src = open("v21_twin_terms_box.py").read()
exec(src[src.index("import os, sys, json, itertools"):src.index("sys.path.insert(0, os.getcwd())")])
exec(src[src.index("BUCKET = "):src.index("# ---------------- 09:")])
r12 = src[src.index("# ---------------- 12:"):src.index('S2 = pd.read_csv("roster_samples.csv")')]
assert "def run12" in r12
hook = "        X = pd.concat(pairs.values()); a, b = deming(X.x.values, X.y.values); bs = []"
assert r12.count(hook) == 1
PDUMP = {}
exec(r12.replace(hook, hook + "\n        PDUMP[k] = {c: p.copy() for c, p in pairs.items()}"), globals())
lines = src.split("\n"); grab = lambda start: next(l for l in lines if l.startswith(start))
for st in ('S2 = pd.read_csv("roster_samples.csv")', "H = pd.read_csv(\"hsc_manifest.csv\")", "S12 = pd.concat([S2,", "adm = RO.index["): exec(grab(st))
t0 = time.time(); O2 = run12(S12, list(SUBV2)); log("run12 done", round(time.time() - t0), "s")
import transfer_bounded_v21 as TB
P = PDUMP["Loyfer2023"]; n = sum(len(p) for p in P.values())
assert len(P) == 17 and abs(n - 767672) < 50, ("not L20's pairs", len(P), n)
syn = open("transfer_bounded_v21_synthetic.txt").read(); SYN = "SYNTHETIC: MET" in syn
cr = TB.cross(P); kx, ky = TB.build(pd.concat(P.values()))
ok = cr["A_on_B"]["ok"] and cr["B_on_A"]["ok"] and SYN
json.dump({"form": "monotone piecewise-linear, PAV on L20 bins (transfer_bounded_v21.py)", "knots_x": [float(v) for v in kx], "knots_y": [float(v) for v in ky],
           "pairs": n, "cells": sorted(P), "halves": cr["halves"], "adopted": bool(ok)}, open("transfer_bounded_v21.json", "w"), indent=1)
L = [f"Atlas v2.1 readiness L21, bounded transfer real check, run once {time.strftime('%Y-%m-%d')} (v21_transfer_bounded_box.py)",
     f"pairs {n:,} over 17 cells (L20's); halves A {len(cr['halves']['A'])} cells, B {len(cr['halves']['B'])} cells",
     f"built on A, checked on B: {cr['A_on_B']['bins']} bins, worst {cr['A_on_B']['worst']:.4f}, {cr['A_on_B']['beyond']} beyond +/-0.02",
     f"built on B, checked on A: {cr['B_on_A']['bins']} bins, worst {cr['B_on_A']['worst']:.4f}, {cr['B_on_A']['beyond']} beyond +/-0.02",
     f"synthetic test (shipped file): {'MET' if SYN else 'NOT MET'}",
     f"final curve on all 17 cells: {len(kx)} knots, range [{ky.min():.4f}, {ky.max():.4f}]",
     f"VERDICT: {'ADOPTED' if ok else 'NOT ADOPTED'}"]
open("transfer_bounded_v21_summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)
for f in ("transfer_bounded_v21.json", "transfer_bounded_v21_summary.txt"): s3.upload_file(f, BUCKET, "results/ATLAS_V21/transfer_bounded/" + f)
open("DONE", "w").write("ok\n")
