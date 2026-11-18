#!/usr/bin/env python3
"""Overhaul test T1 (register OVERHAUL plan; chain audit's provisional finding that a whole-blood molecule file is read as one non-granulocyte cell and that
the mixed-specimen route is reached only with an operator-named cell). A cosmologist's injection test: plant a signal whose answer is known. Runs ON A GITHUB
MACHINE. Builds one whole-blood-like molecule file from five spent IAM Sequencing Atlas libraries (GSE186458; in the atlas, so the shares are read against
the atlas's own donors: a ROUTE test, not an accuracy test), mixed by molecules in known shares, then runs the ONE BUTTON twice with every default:
(a) --specimen "whole blood", no cell named; (b) --specimen "sorted cells", no cell named (the joint run's call). Prints the route, the Deconvolver's shares,
and what IAM-A and Met-A read. Prediction written in the register before the run."""
import os, sys, json, gzip, random, subprocess, tempfile, time, collections
IF = "Biological_Physics/MethylPhys/chain/MethylPhys_Interface"; CH = "Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH); import stage_q0_intake as Q0
MIX = {"GSM5652313": ("granulocytes", 0.60), "GSM5652279": ("cd4 t cells", 0.15), "GSM5652282": ("cd8 t cells", 0.10),
       "GSM5652302": ("monocytes", 0.10), "GSM5652316": ("b cells", 0.05)}
os.makedirs("xout", exist_ok=True); W = tempfile.mkdtemp(prefix="mix")
import numpy as np, pandas as pd
CHUNK = 2_000_000
def chunks(f):
    for d in pd.read_csv(f, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chr", "idx", "pat", "n"], dtype={"chr": str, "idx": np.int64, "pat": str, "n": np.int64}, chunksize=CHUNK):
        yield d
NTOT = 10_000_000                                       # molecules in the planted mixture (each library holds far more; thinning only)
rng = np.random.default_rng(7); got = {}; parts = []; mol = {}
for g, (_, s) in MIX.items():                           # one library on disk at a time
    f = Q0.geo_pat(g, W)["path"]; mol[g] = int(sum(d["n"].sum() for d in chunks(f)))
    p = s * NTOT / mol[g]; assert p <= 1, (g, mol[g]); k = 0; tp = os.path.join(W, f"{g}.thin.tsv")
    with open(tp, "w") as fo:
        for d in chunks(f):
            d["n"] = rng.binomial(d["n"].to_numpy(), p); d = d[d["n"] > 0]; k += int(d["n"].sum())
            d.to_csv(fo, sep="\t", header=False, index=False)
    os.remove(f); got[g] = k; parts.append(tp)
    print(f"::notice::T1 source {g} {MIX[g][0]}: {mol[g]:,} molecules, kept {k:,}")
mp = os.path.join(W, "MIXT1.pat.gz")
subprocess.run(f"sort -k2,2n -k3,3 -S 1G -T {W} {' '.join(parts)} | gzip -1 > {mp}", shell=True, check=True)
for tp in parts: os.remove(tp)
tot = sum(got.values()); planted = {MIX[g][0]: round(got[g] / tot, 4) for g in MIX}
print("::notice::T1 planted shares (by molecules): " + json.dumps(planted) + f" | total {tot:,}")
res = {"planted": planted, "runs": {}}
for tag, spec in (("a_whole_blood", "whole blood"), ("b_sorted_cells", "sorted cells")):
    b = os.path.join(W, f"{tag}.json"); t0 = time.time()
    with open(os.path.join("xout", f"t1_{tag}_log.txt"), "w") as fh:
        rc = subprocess.run(["python3", "run_sample.py", "--pat", mp, "--specimen", spec, "--id", f"MIXT1_{tag}", "--out", os.path.join(W, f"{tag}.html"),
                             "--bundle", b], cwd=IF, stdout=fh, stderr=subprocess.STDOUT).returncode
    B = json.load(open(b)) if rc == 0 and os.path.exists(b) else {}
    if os.path.exists(b): subprocess.run(["cp", b, os.path.join("xout", f"t1_{tag}_bundle.json")])
    r_shares = B.get("cell_shares"); r_found = B.get("cells_found"); r_mix = B.get("iam_a_mix")
    ia = B.get("iam_a") or {}; dc = B.get("deconvolver", B.get("cell_shares", {}))
    top = sorted(dc.items(), key=lambda kv: -kv[1])[:6] if isinstance(dc, dict) and dc and all(isinstance(v, (int, float)) for v in dc.values()) else str(dc)[:200]
    tail = open(os.path.join("xout", f"t1_{tag}_log.txt")).read().splitlines()[-6:]
    r = {"rc": rc, "minutes": round((time.time() - t0) / 60, 1), "route": B.get("route") or B.get("specimen_route") or ia.get("route"),
         "iam_a": {k: ia.get(k) for k in ("A", "A_derived", "cell", "state", "refusal", "refusal_code", "mixed", "cells")},
         "deconvolver_top": top, "met_a_cells": [(m.get("cell"), m.get("share"), m.get("met_a")) for m in (B.get("met_a_each_cell") or [])][:6] if isinstance(B.get("met_a_each_cell"), list) else str(B.get("met_a_each_cell"))[:200],
         "cell_shares": r_shares, "cells_found": r_found, "iam_a_mix": r_mix, "bundle_keys": sorted(B)[:50], "log_tail": [x[:220] for x in tail]}
    res["runs"][tag] = r
    s = json.dumps(r, default=str)
    for i in range(0, min(len(s), 7600), 1900): print(f"::notice::T1 {tag} part {i // 1900 + 1}: " + s[i:i + 1900])
json.dump(res, open("xout/t1_mixture.json", "w"), indent=1, default=str)
